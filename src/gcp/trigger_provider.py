"""GCP trigger provider — deploys Jobs + Workflow + Eventarc.

Implements CloudTriggerProvider using:
- Cloud Run Jobs for script execution (reuses deployer.py)
- Workflows for orchestration
- Eventarc for event routing
- Pub/Sub for multi-trigger continuation
"""

import logging
from pathlib import Path
from typing import Optional

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import workflows_v1, pubsub_v1

from lamia_cloud.interfaces import CloudTriggerProvider
from lamia_cloud.types import TriggerDeploymentPlan, TriggerStage
from lamia_cloud.gcp.deployer import deploy, teardown, deployment_name
from lamia_cloud.gcp.workflow_generator import generate_workflow_yaml

logger = logging.getLogger(__name__)

TRIGGER_METHOD_TO_EVENTARC_TYPE = {
    "file_created": "google.cloud.storage.object.v1.finalized",
    "file_deleted": "google.cloud.storage.object.v1.deleted",
    "file_modified": "google.cloud.storage.object.v1.metadataUpdated",
    "email_received": "google.cloud.pubsub.topic.v1.messagePublished",
}


class GCPTriggerProvider(CloudTriggerProvider):
    """GCP implementation: Cloud Run Jobs + Workflows + Eventarc."""

    def __init__(self, *, project_id: str, location: str):
        self.project_id = project_id
        self.location = location

    @classmethod
    def from_config(cls, cloud_cfg: dict) -> "GCPTriggerProvider":
        project_id = cloud_cfg.get("project_id")
        if not project_id:
            raise ValueError("cloud.project_id is required for triggers.")
        location = cloud_cfg.get("location", "us-central1")
        return cls(project_id=project_id, location=location)

    def deploy(self, plan: TriggerDeploymentPlan) -> str:
        """Deploy all stages as Jobs, create Workflow, create Eventarc trigger."""
        total_stages = len(plan.stages)

        for stage in plan.stages:
            job_name = self._stage_job_name(plan.name, stage.stage_index, total_stages)
            logger.info(f"Deploying stage {stage.stage_index}: {job_name}")
            deploy(
                project_id=self.project_id,
                location=self.location,
                project_root=Path.cwd(),
                script_name=f"{plan.name}.lm",
                name=job_name,
                capabilities=plan.capabilities or None,
            )

        if total_stages > 1:
            self._create_pubsub_for_continuations(plan)

        workflow_yaml = generate_workflow_yaml(plan, self.project_id, self.location)
        workflow_name = f"lamia-trigger-{plan.name}"
        self._deploy_workflow(workflow_name, workflow_yaml)

        self._create_eventarc_trigger(plan)

        logger.info(f"Trigger deployed: {plan.name}")
        return workflow_name

    def undeploy(self, name: str) -> None:
        """Remove Workflow, Eventarc trigger, Pub/Sub resources, and Jobs."""
        workflow_name = f"lamia-trigger-{name}"
        self._delete_workflow(workflow_name)
        self._delete_eventarc_trigger(name)
        teardown(self.project_id, self.location, name)
        logger.info(f"Trigger undeployed: {name}")

    def list_deployments(self) -> list[dict]:
        """List workflows with lamia-trigger- prefix."""
        client = workflows_v1.WorkflowsClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        deployments = []
        try:
            for wf in client.list_workflows(parent=parent):
                if wf.name.split("/")[-1].startswith("lamia-trigger-"):
                    trigger_name = wf.name.split("/")[-1].replace("lamia-trigger-", "")
                    deployments.append({
                        "name": trigger_name,
                        "script": f"{trigger_name.replace('-', '_')}.lm",
                        "trigger_method": (wf.labels or {}).get("trigger-method", ""),
                        "last_run": (wf.labels or {}).get("last-run", "never"),
                        "last_status": (wf.labels or {}).get("last-status", "unknown"),
                    })
        except Exception as e:
            logger.warning(f"Failed to list workflows: {e}")
        return deployments

    def _stage_job_name(self, plan_name: str, stage_index: int, total_stages: int) -> str:
        if total_stages == 1:
            return plan_name
        return f"{plan_name}-stage-{stage_index}"

    def _deploy_workflow(self, name: str, source: str) -> None:
        client = workflows_v1.WorkflowsClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        full_name = f"{parent}/workflows/{name}"

        workflow = workflows_v1.Workflow(
            name=full_name,
            source_contents=source,
            labels={"lamia-managed": "true"},
        )

        try:
            operation = client.update_workflow(workflow=workflow)
            operation.result(timeout=120)
            logger.info(f"Updated workflow: {name}")
        except NotFound:
            operation = client.create_workflow(
                parent=parent,
                workflow=workflow,
                workflow_id=name,
            )
            operation.result(timeout=120)
            logger.info(f"Created workflow: {name}")

    def _delete_workflow(self, name: str) -> None:
        client = workflows_v1.WorkflowsClient()
        full_name = (
            f"projects/{self.project_id}/locations/{self.location}/workflows/{name}"
        )
        try:
            operation = client.delete_workflow(name=full_name)
            operation.result(timeout=60)
        except NotFound:
            pass

    def _create_eventarc_trigger(self, plan: TriggerDeploymentPlan) -> None:
        """Create Eventarc trigger for the first stage's event -> starts Workflow."""
        from google.cloud import eventarc_v1

        first_stage = plan.stages[0]
        event_type = TRIGGER_METHOD_TO_EVENTARC_TYPE.get(first_stage.trigger_method)
        if not event_type:
            raise ValueError(f"Unknown trigger method: {first_stage.trigger_method}")

        client = eventarc_v1.EventarcClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        trigger_id = f"lamia-trigger-{plan.name}"
        workflow_name = f"lamia-trigger-{plan.name}"

        event_filters = [
            eventarc_v1.EventFilter(attribute="type", value=event_type),
        ]

        bucket = first_stage.trigger_config.get("path", "")
        if bucket and "storage" in event_type:
            event_filters.append(
                eventarc_v1.EventFilter(attribute="bucket", value=bucket)
            )

        sa_email = f"lamia-runner@{self.project_id}.iam.gserviceaccount.com"

        trigger_obj = eventarc_v1.Trigger(
            name=f"{parent}/triggers/{trigger_id}",
            event_filters=event_filters,
            destination=eventarc_v1.Destination(
                workflow=(
                    f"projects/{self.project_id}/locations/{self.location}"
                    f"/workflows/{workflow_name}"
                ),
            ),
            service_account=sa_email,
            labels={"lamia-managed": "true"},
        )

        try:
            client.create_trigger(
                parent=parent,
                trigger=trigger_obj,
                trigger_id=trigger_id,
            )
            logger.info(f"Created Eventarc trigger: {trigger_id}")
        except AlreadyExists:
            client.update_trigger(trigger=trigger_obj)
            logger.info(f"Updated Eventarc trigger: {trigger_id}")

    def _delete_eventarc_trigger(self, name: str) -> None:
        from google.cloud import eventarc_v1

        client = eventarc_v1.EventarcClient()
        trigger_name = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/triggers/lamia-trigger-{name}"
        )
        try:
            client.delete_trigger(name=trigger_name)
        except NotFound:
            pass

    def _create_pubsub_for_continuations(self, plan: TriggerDeploymentPlan) -> None:
        """Create Pub/Sub topics + subscriptions for multi-trigger continuation stages."""
        publisher = pubsub_v1.PublisherClient()
        subscriber = pubsub_v1.SubscriberClient()

        for stage in plan.stages[1:]:
            topic_id = f"lamia-trigger-{plan.name}-stage-{stage.stage_index}"
            topic_path = f"projects/{self.project_id}/topics/{topic_id}"
            sub_path = f"projects/{self.project_id}/subscriptions/{topic_id}"

            try:
                publisher.create_topic(name=topic_path)
                logger.info(f"Created Pub/Sub topic: {topic_id}")
            except AlreadyExists:
                pass

            try:
                subscriber.create_subscription(name=sub_path, topic=topic_path)
                logger.info(f"Created Pub/Sub subscription: {topic_id}")
            except AlreadyExists:
                pass
