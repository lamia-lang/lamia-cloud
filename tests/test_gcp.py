"""Tests for lamia_cloud.gcp.scheduler."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from google.cloud.scheduler_v1 import Job

from lamia_cloud.gcp import GCPCloudScheduler
from lamia_cloud.types import CloudJobStatus, CloudScheduleJob


def _make_scheduler() -> GCPCloudScheduler:
    return GCPCloudScheduler(project_id="proj", location="us-central1")


def _make_job(cron: str = "0 * * * *") -> CloudScheduleJob:
    return CloudScheduleJob(
        script="task.lm",
        cron=cron,
        schedule_id="task-abc1",
    )


class TestBuildSchedulerJob:
    def test_reboot_cron_mapped_to_hourly(self):
        scheduler = _make_scheduler()
        job = _make_job(cron="@reboot")

        scheduler_job = scheduler._build_scheduler_job(job, "lamia-task-abc1")

        assert scheduler_job.schedule == "0 * * * *"

    def test_regular_cron_preserved(self):
        scheduler = _make_scheduler()
        job = _make_job(cron="30 8 * * 1")

        scheduler_job = scheduler._build_scheduler_job(job, "lamia-task-abc1")

        assert scheduler_job.schedule == "30 8 * * 1"


class TestGCPCloudSchedulerGetStatus:
    def test_enabled_maps_to_active(self):
        scheduler = _make_scheduler()
        cloud_job = MagicMock()
        cloud_job.state = Job.State.ENABLED
        mock_client = MagicMock()
        mock_client.get_job.return_value = cloud_job
        scheduler._scheduler_client = MagicMock(return_value=mock_client)

        status = scheduler.get_status(_make_job())

        assert status == CloudJobStatus.ACTIVE

    def test_paused_maps_to_inactive(self):
        scheduler = _make_scheduler()
        cloud_job = MagicMock()
        cloud_job.state = Job.State.PAUSED
        mock_client = MagicMock()
        mock_client.get_job.return_value = cloud_job
        scheduler._scheduler_client = MagicMock(return_value=mock_client)

        status = scheduler.get_status(_make_job())

        assert status == CloudJobStatus.INACTIVE

    def test_exception_maps_to_unknown(self):
        scheduler = _make_scheduler()
        mock_client = MagicMock()
        mock_client.get_job.side_effect = RuntimeError("not found")
        scheduler._scheduler_client = MagicMock(return_value=mock_client)

        status = scheduler.get_status(_make_job())

        assert status == CloudJobStatus.UNKNOWN


class TestGCPCloudSchedulerGetInstalledConfig:
    def test_returns_schedule_state_and_last_attempt(self):
        scheduler = _make_scheduler()
        last_attempt = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        cloud_job = MagicMock()
        cloud_job.schedule = "0 12 * * *"
        cloud_job.state.name = "ENABLED"
        cloud_job.last_attempt_time = last_attempt
        mock_client = MagicMock()
        mock_client.get_job.return_value = cloud_job
        scheduler._scheduler_client = MagicMock(return_value=mock_client)

        config = scheduler.get_installed_config(_make_job())

        assert config == {
            "schedule": "0 12 * * *",
            "state": "ENABLED",
            "last_attempt_time": "2026-07-01T00:00:00+00:00",
        }

    def test_returns_none_on_exception(self):
        scheduler = _make_scheduler()
        mock_client = MagicMock()
        mock_client.get_job.side_effect = RuntimeError("missing")
        scheduler._scheduler_client = MagicMock(return_value=mock_client)

        assert scheduler.get_installed_config(_make_job()) is None


class TestGCPCloudScheduler:
    def test_from_config_missing_project_id_raises(self):
        with pytest.raises(ValueError, match="cloud.project_id is required"):
            GCPCloudScheduler.from_config({"provider": "gcp", "location": "us-central1"})

    @patch("lamia_cloud.gcp.scheduler._enable_apis")
    def test_from_config_valid(self, mock_enable):
        scheduler = GCPCloudScheduler.from_config({
            "provider": "gcp",
            "project_id": "proj",
            "location": "europe-west1",
        })
        assert scheduler.project_id == "proj"
        assert scheduler.location == "europe-west1"

    @patch("lamia_cloud.gcp.scheduler._enable_apis")
    def test_from_config_defaults_location(self, mock_enable):
        scheduler = GCPCloudScheduler.from_config({
            "provider": "gcp",
            "project_id": "proj",
        })
        assert scheduler.location == "us-central1"

    def test_name(self):
        assert GCPCloudScheduler.name() == "gcp-cloud"
