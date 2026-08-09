"""Cloud Scheduler job construction shared by the schedule and trigger backends.

Both backends drive work from cron via a Cloud Scheduler job that POSTs to a
Google API as the Lamia runner service account. Only the target URI and request
body differ — schedules invoke a Cloud Run Job (`jobs:run`), triggers start a
drain Workflow execution.
"""

from google.cloud import scheduler_v1

DEFAULT_TIME_ZONE = "UTC"
OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Cloud Scheduler accepts only cron expressions, so boot-time scheduling
# degrades to the closest recurring equivalent.
REBOOT_FALLBACK_CRON = "0 * * * *"


def runner_service_account(project_id: str) -> str:
    """Email of the service account Cloud Scheduler authenticates as."""
    return f"lamia-runner@{project_id}.iam.gserviceaccount.com"


def schedule_job_id(schedule_id: str) -> str:
    """Cloud Scheduler job id backing a scheduled script."""
    return f"lamia-{schedule_id}"


def trigger_job_id(trigger_name: str) -> str:
    """Cloud Scheduler job id backing a scheduled-mode trigger's drain."""
    return f"lamia-trigger-{trigger_name}-scheduler"


def normalize_cron(cron: str) -> str:
    """Map cron aliases Cloud Scheduler rejects onto accepted expressions."""
    if cron == "@reboot":
        return REBOOT_FALLBACK_CRON
    return cron


def build_scheduler_job(
    *,
    name: str,
    project_id: str,
    cron: str,
    target_uri: str,
    body: bytes = b"{}",
    description: str = "",
    time_zone: str = DEFAULT_TIME_ZONE,
) -> scheduler_v1.Job:
    """Build a Cloud Scheduler job that POSTs JSON to a Google API endpoint.

    `name` is the full resource name (`projects/../locations/../jobs/<id>`).
    """
    return scheduler_v1.Job(
        name=name,
        schedule=normalize_cron(cron),
        time_zone=time_zone,
        description=description,
        http_target=scheduler_v1.HttpTarget(
            uri=target_uri,
            http_method=scheduler_v1.HttpMethod.POST,
            headers={"Content-Type": "application/json"},
            body=body,
            oauth_token=scheduler_v1.OAuthToken(
                service_account_email=runner_service_account(project_id),
                scope=OAUTH_SCOPE,
            ),
        ),
    )
