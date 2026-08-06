"""Tests for lamia_cloud.gcp.scheduler_job."""

from google.cloud import scheduler_v1

from lamia_cloud.gcp.scheduler_job import (
    OAUTH_SCOPE,
    REBOOT_FALLBACK_CRON,
    build_scheduler_job,
    normalize_cron,
    runner_service_account,
    schedule_job_id,
    trigger_job_id,
)


class TestNaming:
    def test_runner_service_account(self):
        assert runner_service_account("proj") == "lamia-runner@proj.iam.gserviceaccount.com"

    def test_schedule_job_id(self):
        assert schedule_job_id("task-abc1") == "lamia-task-abc1"

    def test_trigger_job_id(self):
        assert trigger_job_id("inbox") == "lamia-trigger-inbox-scheduler"


class TestNormalizeCron:
    def test_reboot_mapped_to_fallback(self):
        assert normalize_cron("@reboot") == REBOOT_FALLBACK_CRON

    def test_regular_cron_preserved(self):
        assert normalize_cron("30 8 * * 1") == "30 8 * * 1"


class TestBuildSchedulerJob:
    def _build(self, **overrides):
        kwargs = {
            "name": "projects/proj/locations/us-central1/jobs/lamia-task",
            "project_id": "proj",
            "cron": "0 * * * *",
            "target_uri": "https://run.googleapis.com/v2/jobs/lamia-task:run",
        }
        kwargs.update(overrides)
        return build_scheduler_job(**kwargs)

    def test_posts_json_to_target(self):
        job = self._build()

        assert job.http_target.uri == "https://run.googleapis.com/v2/jobs/lamia-task:run"
        assert job.http_target.http_method == scheduler_v1.HttpMethod.POST
        assert job.http_target.headers["Content-Type"] == "application/json"

    def test_authenticates_as_runner_service_account(self):
        job = self._build()

        assert (
            job.http_target.oauth_token.service_account_email
            == "lamia-runner@proj.iam.gserviceaccount.com"
        )
        assert job.http_target.oauth_token.scope == OAUTH_SCOPE

    def test_cron_is_normalized(self):
        job = self._build(cron="@reboot")

        assert job.schedule == REBOOT_FALLBACK_CRON

    def test_defaults_to_empty_json_body_and_utc(self):
        job = self._build()

        assert job.http_target.body == b"{}"
        assert job.time_zone == "UTC"

    def test_body_and_description_are_overridable(self):
        job = self._build(body=b'{"source":"scheduler"}', description="drain")

        assert job.http_target.body == b'{"source":"scheduler"}'
        assert job.description == "drain"
