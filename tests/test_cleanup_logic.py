"""Tests for auto-cleanup business logic.

Covers edge cases in resource label management and stale resource cleanup.
"""

import logging
from datetime import date, timedelta
from unittest.mock import patch

import pytest

import lamia_cloud.gcp.deployer as deployer_module
from lamia_cloud.contracts import (
    LABEL_LAST_USED,
    LABEL_MANAGED,
    LABEL_RESOURCE_TYPE,
    STALE_RESOURCE_DAYS,
)
from lamia_cloud.gcp.deployer import (
    cleanup_stale_resources,
    run_job,
)


def _managed_job(name, days_old, resource_type="one-shot"):
    """Helper: build a managed-job dict with lamia-last-used N days ago."""
    used = (date.today() - timedelta(days=days_old)).strftime("%Y%m%d")
    labels = {LABEL_MANAGED: "true", LABEL_LAST_USED: used}
    if resource_type:
        labels[LABEL_RESOURCE_TYPE] = resource_type
    return {
        "name": name,
        "full_name": f"projects/p/locations/us-central1/jobs/{name}",
        "labels": labels,
        "create_time": None,
        "update_time": None,
    }


class TestCleanupAbortsWhenReferencesUnknown:
    """If the Cloud Scheduler API is unreachable, cleanup must not delete
    anything — it cannot distinguish 'no references' from 'API down'."""

    @patch("lamia_cloud.gcp.deployer.run_v2")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    def test_returns_empty_when_reference_check_fails(self, mock_list, mock_run_v2):
        mock_list.return_value = [_managed_job("lamia-daily-report", 45)]

        with patch(
            "lamia_cloud.gcp.deployer._referenced_job_names", return_value=None
        ):
            cleaned = cleanup_stale_resources("proj", "us-central1")

        assert cleaned == []


class TestTriggerStageJobsSkippedByCleanup:
    """Cloud Run Jobs used as trigger stages don't carry the
    lamia-resource-type=one-shot label, so cleanup skips them."""

    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_trigger_stage_crj_not_deleted(self, mock_run_v2, mock_list, mock_ref):
        mock_list.return_value = [
            _managed_job("lamia-trigger-pricing-stage0", 45, resource_type=""),
        ]
        mock_ref.return_value = set()

        cleaned = cleanup_stale_resources("proj", "us-central1")

        assert cleaned == []


class TestLabelRefreshLogsOnFailure:
    """_touch_last_used must log a warning if the GCP API call fails,
    so operators can detect stale labels before cleanup acts on them."""

    def test_run_job_logs_warning_when_label_refresh_fails(self, monkeypatch, caplog):
        from google.cloud import run_v2
        from google.protobuf import timestamp_pb2

        execution = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-test/executions/exec-ok",
            succeeded_count=1,
            start_time=timestamp_pb2.Timestamp(seconds=200),
            completion_time=timestamp_pb2.Timestamp(seconds=210),
        )

        class FakeOperation:
            def result(self):
                return execution

        class FakeClient:
            def get_job(self, request):
                raise Exception("Transient GCP 503")

            def update_job(self, job):
                pass

            def run_job(self, request):
                return FakeOperation()

        monkeypatch.setattr(deployer_module.run_v2, "JobsClient", lambda: FakeClient())

        with caplog.at_level(logging.WARNING):
            result = run_job("p", "us-central1", "lamia-test")

        assert result["exit_code"] == 0
        assert any("Failed to update" in r.message for r in caplog.records)


class TestPreLabelJobsKeptByDesign:
    """Jobs deployed before the label feature have no lamia-resource-type.
    They are intentionally kept (clutter over data loss)."""

    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_job_without_resource_type_label_not_cleaned(
        self, mock_run_v2, mock_list, mock_ref
    ):
        mock_list.return_value = [{
            "name": "lamia-ancient",
            "full_name": "projects/p/locations/us-central1/jobs/lamia-ancient",
            "labels": {LABEL_MANAGED: "true"},
            "create_time": date.today() - timedelta(days=90),
            "update_time": date.today() - timedelta(days=90),
        }]
        mock_ref.return_value = set()

        cleaned = cleanup_stale_resources("proj", "us-central1")

        assert cleaned == []


class TestFailingScheduleProtected:
    """A schedule failing for 30+ days is still protected by the
    Cloud Scheduler reference check."""

    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_failing_schedule_not_deleted(self, mock_run_v2, mock_list, mock_ref):
        mock_list.return_value = [_managed_job("lamia-failing-daily", 45)]
        mock_ref.return_value = {"lamia-failing-daily"}

        cleaned = cleanup_stale_resources("proj", "us-central1")

        assert cleaned == []
