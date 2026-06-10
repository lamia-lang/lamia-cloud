"""Integration test: deploy a scheduled .lm script to GCP and verify it works.

Prerequisites:
    - GCP project with billing enabled
    - Application Default Credentials configured (gcloud auth application-default login)
    - Set LAMIA_CLOUD_TEST_PROJECT env var to your GCP project ID

Run:
    LAMIA_CLOUD_TEST_PROJECT=my-project pytest tests/test_quickstart.py -v
"""

import os
import time
from pathlib import Path

import pytest

SKIP_REASON = "Set LAMIA_CLOUD_TEST_PROJECT to run integration tests"
PROJECT_ID = os.environ.get("LAMIA_CLOUD_TEST_PROJECT")


@pytest.fixture
def test_project(tmp_path):
    """Create a minimal lamia project with a trivial .lm script."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"cloud:\n"
        f"  provider: gcp\n"
        f"  project_id: {PROJECT_ID}\n"
        f"  location: us-central1\n"
    )

    script = tmp_path / "hello.lm"
    script.write_text(
        'def greet() -> str:\n'
        '    return "hello from lamia-cloud"\n'
        '\n'
        'print(greet())\n'
    )

    return tmp_path


@pytest.mark.skipif(not PROJECT_ID, reason=SKIP_REASON)
class TestQuickstart:
    def test_deploy_and_schedule(self, test_project):
        """Full cycle: deploy -> verify scheduler job exists -> remove."""
        from lamia.scheduling.base import ScheduleJob, generate_schedule_id
        from lamia_cloud import get_scheduler

        scheduler = get_scheduler(test_project)
        script_name = "hello.lm"
        schedule_id = generate_schedule_id(script_name, str(test_project))

        job = ScheduleJob(
            script=script_name,
            cron="0 12 * * *",
            schedule_id=schedule_id,
            catch_up=False,
            project_root=test_project,
        )

        # Deploy and schedule
        scheduler.install(job, lamia_bin="lamia")

        # Verify Cloud Scheduler job was created
        assert scheduler.is_installed(job)

        # Check status
        from lamia.scheduling.base import JobStatus
        status = scheduler.get_status(job)
        assert status == JobStatus.ACTIVE

        # Get config
        config = scheduler.get_installed_config(job)
        assert config is not None
        assert config["schedule"] == "0 12 * * *"

        # Cleanup
        scheduler.uninstall(job)
        assert not scheduler.is_installed(job)

    def test_get_scheduler_from_config(self, test_project):
        """Verify get_scheduler correctly loads from config.yaml."""
        from lamia_cloud import get_scheduler
        from lamia_cloud.gcp import GCPCloudScheduler

        scheduler = get_scheduler(test_project)
        assert isinstance(scheduler, GCPCloudScheduler)
        assert scheduler.project_id == PROJECT_ID
        assert scheduler.location == "us-central1"
