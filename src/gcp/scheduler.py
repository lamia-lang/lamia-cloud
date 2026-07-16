"""GCP Cloud Scheduler backend using Cloud Run Jobs.

Orchestrates: Cloud Build -> Cloud Run Job -> Cloud Scheduler.
Cloud Scheduler triggers jobs via the Cloud Run Admin API (POST jobs/:run).
"""

import json
import logging
import time
from typing import Optional

from lamia_cloud.interfaces import CloudScheduler
from lamia_cloud.types import CloudScheduleJob, CloudJobStatus
from lamia_cloud.gcp.deployer import deploy, deployment_name, teardown, run_job, fetch_execution_logs, fetch_latest_logs

logger = logging.getLogger(__name__)


def _enable_apis(project_id: str) -> None:
    """Enable all required GCP APIs automatically."""
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
            "logging.googleapis.com",
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
    """GCP Cloud Scheduler backend with Cloud Run Jobs."""

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

    def _scheduler_job_name(self, job: CloudScheduleJob) -> str:
        return f"{self._parent}/jobs/lamia-{job.schedule_id}"

    def _build_scheduler_job(self, job: CloudScheduleJob, cr_job_name: str):
        """Build a Cloud Scheduler job that triggers a Cloud Run Job via the Run API."""
        from google.cloud import scheduler_v1

        schedule = job.cron
        if schedule == "@reboot":
            schedule = "0 * * * *"

        run_job_url = (
            f"https://{self.location}-run.googleapis.com/v2/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"jobs/{cr_job_name}:run"
        )

        sa_email = f"lamia-runner@{self.project_id}.iam.gserviceaccount.com"

        return scheduler_v1.Job(
            name=self._scheduler_job_name(job),
            schedule=schedule,
            time_zone="UTC",
            http_target=scheduler_v1.HttpTarget(
                uri=run_job_url,
                http_method=scheduler_v1.HttpMethod.POST,
                headers={"Content-Type": "application/json"},
                body=b"{}",
                oauth_token=scheduler_v1.OAuthToken(
                    service_account_email=sa_email,
                    scope="https://www.googleapis.com/auth/cloud-platform",
                ),
            ),
        )

    def install(self, job: CloudScheduleJob, lamia_bin: str) -> None:
        """Deploy the script as a Cloud Run Job and create a Cloud Scheduler trigger."""
        logger.info(f"Deploying {job.script} to cloud...")

        cr_job_name = deploy(
            project_id=self.project_id,
            location=self.location,
            project_root=job.project_root,
            script_name=job.script,
            name=job.schedule_id,
        )

        client = self._scheduler_client()
        scheduler_job = self._build_scheduler_job(job, cr_job_name)

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
                logger.info("Cloud Scheduler not ready yet, retrying in 15s...")
                time.sleep(15)

    def uninstall(self, job: CloudScheduleJob) -> None:
        """Remove both the Cloud Scheduler job and the Cloud Run Job."""
        client = self._scheduler_client()
        from google.api_core.exceptions import Aborted, NotFound

        for attempt in range(5):
            try:
                client.delete_job(name=self._scheduler_job_name(job))
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
            client.get_job(name=self._scheduler_job_name(job))
            return True
        except Exception:
            return False

    def get_status(self, job: CloudScheduleJob) -> CloudJobStatus:
        client = self._scheduler_client()
        try:
            cloud_job = client.get_job(name=self._scheduler_job_name(job))
            from google.cloud.scheduler_v1 import Job
            if cloud_job.state == Job.State.ENABLED:
                return CloudJobStatus.ACTIVE
            return CloudJobStatus.INACTIVE
        except Exception:
            return CloudJobStatus.UNKNOWN

    def pause(self, job: CloudScheduleJob) -> None:
        """Pause the Cloud Scheduler job (stops triggering)."""
        client = self._scheduler_client()
        client.pause_job(name=self._scheduler_job_name(job))
        logger.info(f"Paused cloud schedule: {job.script}")

    def resume(self, job: CloudScheduleJob) -> None:
        """Resume a paused Cloud Scheduler job."""
        client = self._scheduler_client()
        client.resume_job(name=self._scheduler_job_name(job))
        logger.info(f"Resumed cloud schedule: {job.script}")

    def get_installed_config(self, job: CloudScheduleJob) -> Optional[dict]:
        client = self._scheduler_client()
        try:
            cloud_job = client.get_job(name=self._scheduler_job_name(job))
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

    def fetch_logs(self, job: CloudScheduleJob) -> dict:
        """Fetch logs from the most recent execution.

        Returns {stdout, stderr, logs_url}.
        """
        cr_job_name = deployment_name(job.schedule_id)
        stdout, stderr, logs_url = fetch_latest_logs(
            project_id=self.project_id,
            location=self.location,
            target=cr_job_name,
        )
        return {"stdout": stdout, "stderr": stderr, "logs_url": logs_url}

    def run_once(self, job: CloudScheduleJob, verbose: bool = False) -> dict:
        """Deploy and invoke once without scheduling."""
        cr_job_name = deploy(
            project_id=self.project_id,
            location=self.location,
            project_root=job.project_root,
            script_name=job.script,
            name=job.schedule_id,
        )

        result = run_job(
            project_id=self.project_id,
            location=self.location,
            target=cr_job_name,
            verbose=verbose,
        )

        stdout, stderr = fetch_execution_logs(
            project_id=self.project_id,
            target=cr_job_name,
            execution_name=result.get("execution_name", ""),
        )
        result["stdout"] = stdout
        result["stderr"] = stderr

        return result
