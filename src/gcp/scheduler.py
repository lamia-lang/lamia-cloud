"""GCP Cloud Scheduler backend.

Orchestrates: Cloud Build -> Cloud Run -> Cloud Scheduler.
The user only provides project_id and location. Everything else is automated.
"""

import json
import logging
import time
from typing import Optional

from lamia_cloud.interfaces import CloudScheduler
from lamia_cloud.types import CloudScheduleJob, CloudJobStatus
from lamia_cloud.gcp.deployer import deploy, teardown

logger = logging.getLogger(__name__)


def _enable_apis(project_id: str) -> None:
    """Enable all required GCP APIs automatically.

    Service Usage API must be enabled first (bootstraps the rest).
    Most GCP projects have it enabled by default.
    """
    try:
        from google.cloud import service_usage_v1
        client = service_usage_v1.ServiceUsageClient()
        apis = [
            "serviceusage.googleapis.com",
            "cloudscheduler.googleapis.com",
            "cloudbuild.googleapis.com",
            "run.googleapis.com",
            "storage.googleapis.com",
            "aiplatform.googleapis.com",
            "iam.googleapis.com",
        ]
        for api in apis:
            service_name = f"projects/{project_id}/services/{api}"
            try:
                client.enable_service(request={"name": service_name})
            except Exception as e:
                if "SERVICE_DISABLED" in str(e) and "serviceusage" in api:
                    logger.warning(
                        f"Service Usage API not enabled. Run once:\n"
                        f"  gcloud services enable serviceusage.googleapis.com "
                        f"--project={project_id}"
                    )
                    return
    except ImportError:
        pass


class GCPCloudScheduler(CloudScheduler):
    """GCP Cloud Scheduler backend with automatic deployment."""

    def __init__(self, *, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        self._parent = f"projects/{project_id}/locations/{location}"

    @classmethod
    def name(cls) -> str:
        return "gcp-cloud"

    @classmethod
    def from_config(cls, cloud_cfg: dict) -> "GCPCloudScheduler":
        """Create instance from config.yaml cloud section."""
        project_id = cloud_cfg.get("project_id")
        if not project_id:
            raise ValueError("cloud.project_id is required in config.yaml.")

        location = cloud_cfg.get("location", "us-central1")

        _enable_apis(project_id)
        return cls(project_id=project_id, location=location)

    def _scheduler_client(self):
        from google.cloud import scheduler_v1
        return scheduler_v1.CloudSchedulerClient()

    def _job_name(self, job: CloudScheduleJob) -> str:
        return f"{self._parent}/jobs/lamia-{job.schedule_id}"

    def _build_scheduler_job(self, job: CloudScheduleJob, target_url: str):
        from google.cloud import scheduler_v1

        schedule = job.cron
        if schedule == "@reboot":
            schedule = "0 * * * *"

        body = json.dumps({
            "schedule_id": job.schedule_id,
            "script": job.script,
        }).encode()

        return scheduler_v1.Job(
            name=self._job_name(job),
            schedule=schedule,
            time_zone="UTC",
            http_target=scheduler_v1.HttpTarget(
                uri=target_url,
                http_method=scheduler_v1.HttpMethod.POST,
                headers={"Content-Type": "application/json"},
                body=body,
                oidc_token=scheduler_v1.OidcToken(
                    service_account_email=(
                        f"lamia-runner@{self.project_id}.iam.gserviceaccount.com"
                    ),
                    audience=target_url,
                ),
            ),
        )

    def install(self, job: CloudScheduleJob, lamia_bin: str) -> None:
        """Deploy the script to Cloud Run and create a Cloud Scheduler trigger."""
        logger.info(f"Deploying {job.script} to cloud...")

        service_url = deploy(
            project_id=self.project_id,
            location=self.location,
            project_root=job.project_root,
            script_name=job.script,
            schedule_id=job.schedule_id,
        )

        client = self._scheduler_client()
        scheduler_job = self._build_scheduler_job(job, service_url)

        from google.api_core.exceptions import AlreadyExists, NotFound

        for attempt in range(5):
            try:
                client.create_job(parent=self._parent, job=scheduler_job)
                logger.info(f"Created cloud schedule: {job.script} ({job.cron})")
                break
            except AlreadyExists:
                client.update_job(job=scheduler_job)
                logger.info(f"Updated cloud schedule: {job.script} ({job.cron})")
                break
            except NotFound:
                if attempt == 4:
                    raise RuntimeError(
                        f"Cloud Scheduler location '{self.location}' not ready after retries. "
                        f"The API may still be propagating — wait a minute and retry."
                    )
                logger.info(f"Cloud Scheduler not ready yet, retrying in 15s...")
                time.sleep(15)

    def uninstall(self, job: CloudScheduleJob) -> None:
        """Remove both the Cloud Scheduler job and the Cloud Run service."""
        client = self._scheduler_client()
        from google.api_core.exceptions import Aborted, NotFound

        for attempt in range(5):
            try:
                client.delete_job(name=self._job_name(job))
                logger.info(f"Deleted cloud schedule: {job.script}")
                break
            except NotFound:
                break
            except Aborted:
                if attempt == 4:
                    raise
                time.sleep(10)

        teardown(self.project_id, self.location, job.schedule_id)

    def is_installed(self, job: CloudScheduleJob) -> bool:
        client = self._scheduler_client()
        try:
            client.get_job(name=self._job_name(job))
            return True
        except Exception:
            return False

    def get_status(self, job: CloudScheduleJob) -> CloudJobStatus:
        client = self._scheduler_client()
        try:
            cloud_job = client.get_job(name=self._job_name(job))
            from google.cloud.scheduler_v1 import Job
            if cloud_job.state == Job.State.ENABLED:
                return CloudJobStatus.ACTIVE
            return CloudJobStatus.INACTIVE
        except Exception:
            return CloudJobStatus.UNKNOWN

    def get_installed_config(self, job: CloudScheduleJob) -> Optional[dict]:
        client = self._scheduler_client()
        try:
            cloud_job = client.get_job(name=self._job_name(job))
            return {
                "schedule": cloud_job.schedule,
                "state": cloud_job.state.name,
                "last_attempt_time": (
                    cloud_job.last_attempt_time.isoformat()
                    if cloud_job.last_attempt_time
                    else None
                ),
            }
        except Exception:
            return None
