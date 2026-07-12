"""GCP trigger provider — deploys Jobs + Workflow + Eventarc.

Implements CloudTriggerProvider using:
- Cloud Run Jobs for script execution (reuses deployer.py)
- Workflows for orchestration (reactive or drain-based)
- Eventarc for event routing
- Pub/Sub for event accumulation (scheduled mode) and multi-trigger continuation
- Cloud Scheduler for drain activation (scheduled mode)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import eventarc_v1, monitoring_v3, scheduler_v1, workflows_v1, pubsub_v1

from lamia_cloud.interfaces import CloudTriggerProvider
from lamia_cloud.types import TriggerDeploymentPlan, TriggerStage
from lamia_cloud.gcp.deployer import deploy, teardown, deployment_name
from lamia_cloud.gcp.workflow_generator import (
    generate_workflow_yaml,
    generate_drain_workflow_yaml,
)

logger = logging.getLogger(__name__)

TRIGGER_METHOD_TO_EVENTARC_TYPE = {
    "file_created": "google.cloud.storage.object.v1.finalized",
    "file_deleted": "google.cloud.storage.object.v1.deleted",
    "file_modified": "google.cloud.storage.object.v1.metadataUpdated",
    "email_received": "google.cloud.pubsub.topic.v1.messagePublished",
}

STORAGE_CONFIG_TO_EVENTARC_ATTR = {
    "path": "bucket",
}

PUBSUB_FILTER_ATTRIBUTES = {
    "to": "recipient",
    "from_domain": "senderDomain",
    "subject_contains": "subjectContains",
    "label": "label",
}


def _build_eventarc_filters(
    trigger_method: str,
    trigger_config: dict,
    event_type: str,
) -> list:
    """Build Eventarc event_filters from trigger config kwargs.

    All string kwargs in trigger_config become infrastructure-level filters
    that prevent the script from launching for non-matching events.
    """
    filters = [
        eventarc_v1.EventFilter(attribute="type", value=event_type),
    ]

    if "storage" in event_type:
        for config_key, eventarc_attr in STORAGE_CONFIG_TO_EVENTARC_ATTR.items():
            value = trigger_config.get(config_key, "")
            if value:
                filters.append(
                    eventarc_v1.EventFilter(attribute=eventarc_attr, value=value)
                )

    return filters


def _build_pubsub_filter_expression(trigger_method: str, trigger_config: dict) -> str:
    """Build a Pub/Sub subscription filter expression from trigger config kwargs.

    Used for email and other Pub/Sub-routed triggers where Eventarc attribute
    filtering is insufficient. Multiple filters are AND-combined.
    """
    clauses = []
    attr_map = PUBSUB_FILTER_ATTRIBUTES if trigger_method == "email_received" else {}

    for config_key, pubsub_attr in attr_map.items():
        value = trigger_config.get(config_key, "")
        if value:
            clauses.append(f'attributes.{pubsub_attr} = "{value}"')

    return " AND ".join(clauses)


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
        """Deploy trigger infrastructure based on plan mode."""
        if plan.mode == "scheduled":
            return self._deploy_scheduled(plan)
        return self._deploy_reactive(plan)

    def _deploy_reactive(self, plan: TriggerDeploymentPlan) -> str:
        """Reactive mode: Eventarc -> Workflow -> Job per event."""
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
        self._create_dead_letter_topic(plan)

        logger.info(f"Trigger deployed (reactive): {plan.name}")
        return workflow_name

    def _deploy_scheduled(self, plan: TriggerDeploymentPlan) -> str:
        """Scheduled mode: events -> Pub/Sub accumulation, Scheduler -> drain workflow."""
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

        self._create_event_accumulation_pubsub(plan)

        if total_stages > 1:
            self._create_pubsub_for_continuations(plan)

        workflow_yaml = generate_drain_workflow_yaml(plan, self.project_id, self.location)
        workflow_name = f"lamia-trigger-{plan.name}"
        self._deploy_workflow(workflow_name, workflow_yaml)

        self._create_eventarc_to_pubsub(plan)

        self._create_cloud_scheduler(plan, workflow_name)
        self._create_dead_letter_topic(plan)

        logger.info(f"Trigger deployed (scheduled): {plan.name}")
        return workflow_name

    def undeploy(self, name: str) -> None:
        """Remove Workflow, Eventarc trigger, Pub/Sub resources, Scheduler, and Jobs."""
        workflow_name = f"lamia-trigger-{name}"
        self._delete_workflow(workflow_name)
        self._delete_eventarc_trigger(name)
        self._delete_cloud_scheduler(name)
        self._delete_accumulation_pubsub(name)
        self._delete_dead_letter_topic(name)
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
                    failed_count = self._get_failed_event_count(trigger_name)
                    deployments.append({
                        "name": trigger_name,
                        "script": f"{trigger_name.replace('-', '_')}.lm",
                        "trigger_method": (wf.labels or {}).get("trigger-method", ""),
                        "mode": (wf.labels or {}).get("trigger-mode", "reactive"),
                        "last_run": (wf.labels or {}).get("last-run", "never"),
                        "last_status": (wf.labels or {}).get("last-status", "unknown"),
                        "failed_event_count": failed_count,
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
        first_stage = plan.stages[0]
        event_type = TRIGGER_METHOD_TO_EVENTARC_TYPE.get(first_stage.trigger_method)
        if not event_type:
            raise ValueError(f"Unknown trigger method: {first_stage.trigger_method}")

        client = eventarc_v1.EventarcClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        trigger_id = f"lamia-trigger-{plan.name}"
        workflow_name = f"lamia-trigger-{plan.name}"

        event_filters = _build_eventarc_filters(
            first_stage.trigger_method,
            first_stage.trigger_config,
            event_type,
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

    def _create_eventarc_to_pubsub(self, plan: TriggerDeploymentPlan) -> None:
        """Create Eventarc trigger that routes events to Pub/Sub (scheduled mode)."""
        first_stage = plan.stages[0]
        event_type = TRIGGER_METHOD_TO_EVENTARC_TYPE.get(first_stage.trigger_method)
        if not event_type:
            raise ValueError(f"Unknown trigger method: {first_stage.trigger_method}")

        client = eventarc_v1.EventarcClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        trigger_id = f"lamia-trigger-{plan.name}"
        topic_name = f"projects/{self.project_id}/topics/lamia-trigger-{plan.name}-events"

        event_filters = _build_eventarc_filters(
            first_stage.trigger_method,
            first_stage.trigger_config,
            event_type,
        )

        sa_email = f"lamia-runner@{self.project_id}.iam.gserviceaccount.com"

        trigger_obj = eventarc_v1.Trigger(
            name=f"{parent}/triggers/{trigger_id}",
            event_filters=event_filters,
            destination=eventarc_v1.Destination(
                cloud_run=None,
                workflow=None,
            ),
            transport=eventarc_v1.Transport(
                pubsub=eventarc_v1.Pubsub(topic=topic_name),
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
            logger.info(f"Created Eventarc trigger -> Pub/Sub: {trigger_id}")
        except AlreadyExists:
            client.update_trigger(trigger=trigger_obj)
            logger.info(f"Updated Eventarc trigger -> Pub/Sub: {trigger_id}")

    def _delete_eventarc_trigger(self, name: str) -> None:
        client = eventarc_v1.EventarcClient()
        trigger_name = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/triggers/lamia-trigger-{name}"
        )
        try:
            client.delete_trigger(name=trigger_name)
        except NotFound:
            pass

    def _create_event_accumulation_pubsub(self, plan: TriggerDeploymentPlan) -> None:
        """Create Pub/Sub topic + subscription for event accumulation (scheduled mode)."""
        publisher = pubsub_v1.PublisherClient()
        subscriber = pubsub_v1.SubscriberClient()

        topic_id = f"lamia-trigger-{plan.name}-events"
        topic_path = f"projects/{self.project_id}/topics/{topic_id}"
        sub_path = f"projects/{self.project_id}/subscriptions/{topic_id}"

        try:
            publisher.create_topic(name=topic_path)
            logger.info(f"Created accumulation topic: {topic_id}")
        except AlreadyExists:
            pass

        first_stage = plan.stages[0]
        filter_expr = _build_pubsub_filter_expression(
            first_stage.trigger_method, first_stage.trigger_config
        )

        sub_kwargs = {
            "name": sub_path,
            "topic": topic_path,
            "ack_deadline_seconds": 600,
        }
        if filter_expr:
            sub_kwargs["filter"] = filter_expr

        try:
            subscriber.create_subscription(**sub_kwargs)
            logger.info(f"Created accumulation subscription: {topic_id}")
            if filter_expr:
                logger.info(f"  filter: {filter_expr}")
        except AlreadyExists:
            pass

    def _delete_accumulation_pubsub(self, name: str) -> None:
        """Delete accumulation Pub/Sub topic + subscription."""
        publisher = pubsub_v1.PublisherClient()
        subscriber = pubsub_v1.SubscriberClient()

        topic_id = f"lamia-trigger-{name}-events"
        topic_path = f"projects/{self.project_id}/topics/{topic_id}"
        sub_path = f"projects/{self.project_id}/subscriptions/{topic_id}"

        try:
            subscriber.delete_subscription(subscription=sub_path)
        except NotFound:
            pass
        try:
            publisher.delete_topic(topic=topic_path)
        except NotFound:
            pass

    def _create_dead_letter_topic(self, plan: TriggerDeploymentPlan) -> None:
        """Create dead-letter topic + subscription for failed events."""
        publisher = pubsub_v1.PublisherClient()
        subscriber = pubsub_v1.SubscriberClient()

        topic_id = f"lamia-trigger-{plan.name}-dead-letter"
        topic_path = f"projects/{self.project_id}/topics/{topic_id}"
        sub_path = f"projects/{self.project_id}/subscriptions/{topic_id}"

        try:
            publisher.create_topic(name=topic_path)
            logger.info(f"Created dead-letter topic: {topic_id}")
        except AlreadyExists:
            pass

        try:
            subscriber.create_subscription(
                name=sub_path,
                topic=topic_path,
                ack_deadline_seconds=600,
            )
            logger.info(f"Created dead-letter subscription: {topic_id}")
        except AlreadyExists:
            pass

    def _delete_dead_letter_topic(self, name: str) -> None:
        """Delete dead-letter topic + subscription."""
        publisher = pubsub_v1.PublisherClient()
        subscriber = pubsub_v1.SubscriberClient()

        topic_id = f"lamia-trigger-{name}-dead-letter"
        topic_path = f"projects/{self.project_id}/topics/{topic_id}"
        sub_path = f"projects/{self.project_id}/subscriptions/{topic_id}"

        try:
            subscriber.delete_subscription(subscription=sub_path)
        except NotFound:
            pass
        try:
            publisher.delete_topic(topic=topic_path)
        except NotFound:
            pass

    def get_failed_events(self, name: str) -> list[dict]:
        """Peek failed events without consuming them."""
        subscriber = pubsub_v1.SubscriberClient()
        sub_path = f"projects/{self.project_id}/subscriptions/lamia-trigger-{name}-dead-letter"

        events: list[dict] = []
        try:
            response = subscriber.pull(
                subscription=sub_path,
                max_messages=100,
                return_immediately=True,
            )
            for msg in response.received_messages:
                try:
                    payload = json.loads(msg.message.data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload = {"raw": msg.message.data.decode("utf-8", errors="replace")}
                events.append({
                    "payload": payload,
                    "timestamp": msg.message.publish_time.isoformat(),
                    "attempt_count": int(msg.message.attributes.get("retry_count", "0")) or None,
                })
            if response.received_messages:
                subscriber.modify_ack_deadline(
                    subscription=sub_path,
                    ack_ids=[m.ack_id for m in response.received_messages],
                    ack_deadline_seconds=0,
                )
        except NotFound:
            pass
        except Exception as e:
            logger.warning(f"Failed to peek failed events for {name}: {e}")
        return events

    def clear_failed_events(self, name: str) -> int:
        """Acknowledge and remove all failed events. Returns count removed."""
        subscriber = pubsub_v1.SubscriberClient()
        sub_path = f"projects/{self.project_id}/subscriptions/lamia-trigger-{name}-dead-letter"

        total = 0
        try:
            while True:
                response = subscriber.pull(
                    subscription=sub_path,
                    max_messages=100,
                    return_immediately=True,
                )
                if not response.received_messages:
                    break
                ack_ids = [m.ack_id for m in response.received_messages]
                subscriber.acknowledge(subscription=sub_path, ack_ids=ack_ids)
                total += len(ack_ids)
        except NotFound:
            pass
        except Exception as e:
            logger.warning(f"Failed to clear failed events for {name}: {e}")
        return total

    def _get_failed_event_count(self, name: str) -> int:
        """Get number of unprocessed failed events (via Cloud Monitoring metric)."""
        client = monitoring_v3.MetricServiceClient()
        project_path = f"projects/{self.project_id}"
        sub_id = f"lamia-trigger-{name}-dead-letter"

        query = (
            f'resource.type="pubsub_subscription" '
            f'AND resource.labels.subscription_id="{sub_id}" '
            f'AND metric.type="pubsub.googleapis.com/subscription/num_undelivered_messages"'
        )

        now = int(time.time())
        try:
            results = client.list_time_series(
                name=project_path,
                filter=query,
                interval=monitoring_v3.TimeInterval(
                    end_time={"seconds": now},
                    start_time={"seconds": now - 300},
                ),
                view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            )
            for ts in results:
                if ts.points:
                    return ts.points[0].value.int64_value
        except Exception:
            pass
        return 0

    def _create_pubsub_for_continuations(self, plan: TriggerDeploymentPlan) -> None:
        """Create Pub/Sub topics for multi-trigger continuation stages.

        Only topics are created at deploy time. Subscriptions are created
        per-execution by the workflow itself, ensuring full isolation between
        concurrent waits.
        """
        publisher = pubsub_v1.PublisherClient()

        for stage in plan.stages[1:]:
            topic_id = f"lamia-trigger-{plan.name}-stage-{stage.stage_index}"
            topic_path = f"projects/{self.project_id}/topics/{topic_id}"

            try:
                publisher.create_topic(name=topic_path)
                logger.info(f"Created Pub/Sub topic: {topic_id}")
            except AlreadyExists:
                pass

    def _create_cloud_scheduler(self, plan: TriggerDeploymentPlan, workflow_name: str) -> None:
        """Create Cloud Scheduler job that triggers the drain workflow at cron time."""
        client = scheduler_v1.CloudSchedulerClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        job_id = f"lamia-trigger-{plan.name}-scheduler"
        job_name = f"{parent}/jobs/{job_id}"

        workflow_path = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/workflows/{workflow_name}"
        )

        sa_email = f"lamia-runner@{self.project_id}.iam.gserviceaccount.com"

        http_target = scheduler_v1.HttpTarget(
            uri=f"https://workflowexecutions.googleapis.com/v1/{workflow_path}/executions",
            http_method=scheduler_v1.HttpMethod.POST,
            body=b'{"argument":"{\\"source\\":\\"scheduler\\"}"}',
            oauth_token=scheduler_v1.OAuthToken(
                service_account_email=sa_email,
                scope="https://www.googleapis.com/auth/cloud-platform",
            ),
        )

        job = scheduler_v1.Job(
            name=job_name,
            schedule=plan.cron,
            time_zone="UTC",
            http_target=http_target,
            description=f"Lamia trigger drain: {plan.name}",
        )

        try:
            client.create_job(parent=parent, job=job)
            logger.info(f"Created Cloud Scheduler: {job_id}")
        except AlreadyExists:
            client.update_job(job=job)
            logger.info(f"Updated Cloud Scheduler: {job_id}")

    def _delete_cloud_scheduler(self, name: str) -> None:
        """Delete Cloud Scheduler job if it exists."""
        client = scheduler_v1.CloudSchedulerClient()
        job_name = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/jobs/lamia-trigger-{name}-scheduler"
        )
        try:
            client.delete_job(name=job_name)
        except NotFound:
            pass
