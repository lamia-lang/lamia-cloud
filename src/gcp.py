"""GCP Cloud Scheduler backend.

Uses the google-cloud-scheduler Python SDK.
All config comes from config.yaml cloud section via the loader.
"""

import json
import logging
from typing import Optional

from lamia.scheduling.base import BaseScheduler, JobStatus, ScheduleJob

logger = logging.getLogger(__name__)


def _get_client():
    """Import and return the Cloud Scheduler client."""
    from google.cloud import scheduler_v1
    return scheduler_v1.CloudSchedulerClient()


def _enable_api_if_needed(project_id: str) -> None:
    """Attempt to enable Cloud Scheduler API automatically."""
    try:
        from google.cloud import service_usage_v1
        client = service_usage_v1.ServiceUsageClient()
        service_name = f"projects/{project_id}/services/cloudscheduler.googleapis.com"
        client.enable_service(request={"name": service_name})
    except Exception:
        pass


class GCPCloudScheduler(BaseScheduler):
    """GCP Cloud Scheduler backend."""

    def __init__(self, *, project_id: str, location: str, target_url: str):
        self.project_id = project_id
        self.location = location
        self.target_url = target_url
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
        target_url = cloud_cfg.get("target_url")
        if not target_url:
            raise ValueError("cloud.target_url is required in config.yaml.")

        _enable_api_if_needed(project_id)
        return cls(project_id=project_id, location=location, target_url=target_url)

    def _job_name(self, job: ScheduleJob) -> str:
        return f"{self._parent}/jobs/lamia-{job.schedule_id}"

    def _build_job_payload(self, job: ScheduleJob):
        from google.cloud import scheduler_v1

        schedule = job.cron
        if schedule == "@reboot":
            schedule = "0 * * * *"

        body = json.dumps({
            "schedule_id": job.schedule_id,
            "script": job.script,
            "project_root": str(job.project_root),
        }).encode()

        return scheduler_v1.Job(
            name=self._job_name(job),
            schedule=schedule,
            time_zone="UTC",
            http_target=scheduler_v1.HttpTarget(
                uri=self.target_url,
                http_method=scheduler_v1.HttpMethod.POST,
                headers={"Content-Type": "application/json"},
                body=body,
            ),
        )

    def install(self, job: ScheduleJob, lamia_bin: str) -> None:
        client = _get_client()
        job_payload = self._build_job_payload(job)

        try:
            client.create_job(parent=self._parent, job=job_payload)
            logger.info(f"Created cloud schedule: {job.script} ({job.cron})")
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                client.update_job(job=job_payload)
                logger.info(f"Updated cloud schedule: {job.script} ({job.cron})")
            else:
                raise RuntimeError(f"Failed to create cloud schedule: {e}") from e

    def uninstall(self, job: ScheduleJob) -> None:
        client = _get_client()
        try:
            client.delete_job(name=self._job_name(job))
            logger.info(f"Deleted cloud schedule: {job.script}")
        except Exception as e:
            if "NOT_FOUND" in str(e):
                return
            raise RuntimeError(f"Failed to delete cloud schedule: {e}") from e

    def is_installed(self, job: ScheduleJob) -> bool:
        client = _get_client()
        try:
            client.get_job(name=self._job_name(job))
            return True
        except Exception:
            return False

    def get_status(self, job: ScheduleJob) -> JobStatus:
        client = _get_client()
        try:
            cloud_job = client.get_job(name=self._job_name(job))
            from google.cloud.scheduler_v1 import Job
            if cloud_job.state == Job.State.ENABLED:
                return JobStatus.ACTIVE
            return JobStatus.INACTIVE
        except Exception:
            return JobStatus.UNKNOWN

    def get_installed_config(self, job: ScheduleJob) -> Optional[dict]:
        client = _get_client()
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
