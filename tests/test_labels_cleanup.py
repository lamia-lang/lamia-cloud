"""Tests for GCP label helpers and auto-cleanup logic."""

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lamia_cloud.contracts import (
    LABEL_DEPLOY_MODE,
    LABEL_LAST_USED,
    LABEL_MANAGED,
    LABEL_PROJECT_HASH,
    LABEL_RESOURCE_TYPE,
    LABEL_SCRIPT,
    STALE_RESOURCE_DAYS,
    sanitize_label_value,
)
from lamia_cloud.gcp.deployer import (
    _project_hash,
    _today_label,
    build_resource_labels,
    cleanup_stale_resources,
    list_managed_jobs,
)


class TestSanitizeLabelValue:
    def test_lowercase(self):
        assert sanitize_label_value("Pricing_Reply") == "pricing_reply"

    def test_replaces_dots(self):
        assert sanitize_label_value("pricing_reply.lm") == "pricing_reply_lm"

    def test_replaces_slashes(self):
        assert sanitize_label_value("github.com/team/project") == "github_com_team_project"

    def test_truncates_to_63(self):
        long = "a" * 100
        assert len(sanitize_label_value(long)) == 63

    def test_preserves_dashes_and_underscores(self):
        assert sanitize_label_value("my-script_v2") == "my-script_v2"

    def test_empty_string(self):
        assert sanitize_label_value("") == ""


class TestTodayLabel:
    def test_format(self):
        result = _today_label()
        assert len(result) == 8
        assert result.isdigit()


class TestProjectHash:
    def test_length(self):
        result = _project_hash(Path("/Users/sergey/project"))
        assert len(result) == 8
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        p = Path("/Users/sergey/project")
        assert _project_hash(p) == _project_hash(p)

    def test_different_paths_different_hashes(self):
        assert _project_hash(Path("/a")) != _project_hash(Path("/b"))


class TestBuildResourceLabels:
    def test_basic_labels(self):
        labels = build_resource_labels("pricing.lm", Path("/project"))
        assert labels[LABEL_MANAGED] == "true"
        assert labels[LABEL_SCRIPT] == "pricing_lm"
        assert len(labels[LABEL_PROJECT_HASH]) == 8
        assert len(labels[LABEL_LAST_USED]) == 8
        assert labels[LABEL_DEPLOY_MODE] == "local"
        assert labels[LABEL_RESOURCE_TYPE] == "one-shot"
        assert "lamia-repo-url" not in labels

    def test_git_mode_includes_repo_url(self):
        labels = build_resource_labels(
            "pricing.lm", Path("/project"),
            deploy_mode="git", repo_url="github.com/team/repo",
        )
        assert labels[LABEL_DEPLOY_MODE] == "git"
        assert labels["lamia-repo-url"] == "github_com_team_repo"


class TestListManagedJobs:
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_filters_by_label(self, mock_run_v2):
        managed_job = MagicMock()
        managed_job.name = "projects/p/locations/l/jobs/lamia-abc123"
        managed_job.labels = {"lamia-managed": "true", "lamia-last-used": "20260801"}
        managed_job.create_time = None
        managed_job.update_time = None

        unmanaged_job = MagicMock()
        unmanaged_job.name = "projects/p/locations/l/jobs/other-job"
        unmanaged_job.labels = {}

        mock_run_v2.JobsClient.return_value.list_jobs.return_value = [
            managed_job, unmanaged_job,
        ]

        result = list_managed_jobs("my-project", "us-central1")
        assert len(result) == 1
        assert result[0]["name"] == "lamia-abc123"


class TestCleanupStaleResources:
    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_deletes_stale_unreferenced(self, mock_run_v2, mock_list, mock_ref):
        stale_date = (date.today() - timedelta(days=45)).strftime("%Y%m%d")
        mock_list.return_value = [{
            "name": "lamia-old",
            "full_name": "projects/p/locations/l/jobs/lamia-old",
            "labels": {
                "lamia-last-used": stale_date,
                "lamia-managed": "true",
                LABEL_RESOURCE_TYPE: "one-shot",
            },
            "create_time": None,
            "update_time": None,
        }]
        mock_ref.return_value = set()

        cleaned = cleanup_stale_resources("my-project", "us-central1")
        assert cleaned == ["lamia-old"]
        mock_run_v2.JobsClient.return_value.delete_job.assert_called_once()

    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_skips_referenced_jobs(self, mock_run_v2, mock_list, mock_ref):
        stale_date = (date.today() - timedelta(days=45)).strftime("%Y%m%d")
        mock_list.return_value = [{
            "name": "lamia-scheduled",
            "full_name": "projects/p/locations/l/jobs/lamia-scheduled",
            "labels": {
                "lamia-last-used": stale_date,
                "lamia-managed": "true",
                LABEL_RESOURCE_TYPE: "one-shot",
            },
            "create_time": None,
            "update_time": None,
        }]
        mock_ref.return_value = {"lamia-scheduled"}

        cleaned = cleanup_stale_resources("my-project", "us-central1")
        assert cleaned == []
        mock_run_v2.JobsClient.return_value.delete_job.assert_not_called()

    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_skips_recent_jobs(self, mock_run_v2, mock_list, mock_ref):
        recent_date = (date.today() - timedelta(days=5)).strftime("%Y%m%d")
        mock_list.return_value = [{
            "name": "lamia-recent",
            "full_name": "projects/p/locations/l/jobs/lamia-recent",
            "labels": {
                "lamia-last-used": recent_date,
                "lamia-managed": "true",
                LABEL_RESOURCE_TYPE: "one-shot",
            },
            "create_time": None,
            "update_time": None,
        }]
        mock_ref.return_value = set()

        cleaned = cleanup_stale_resources("my-project", "us-central1")
        assert cleaned == []
        mock_run_v2.JobsClient.return_value.delete_job.assert_not_called()

    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_deletes_at_cutoff_boundary(self, mock_run_v2, mock_list, mock_ref):
        cutoff_date = (date.today() - timedelta(days=STALE_RESOURCE_DAYS)).strftime("%Y%m%d")
        mock_list.return_value = [{
            "name": "lamia-boundary",
            "full_name": "projects/p/locations/l/jobs/lamia-boundary",
            "labels": {
                "lamia-last-used": cutoff_date,
                "lamia-managed": "true",
                LABEL_RESOURCE_TYPE: "one-shot",
            },
            "create_time": None,
            "update_time": None,
        }]
        mock_ref.return_value = set()

        cleaned = cleanup_stale_resources("my-project", "us-central1")
        assert cleaned == ["lamia-boundary"]
        mock_run_v2.JobsClient.return_value.delete_job.assert_called_once()

    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_skips_when_last_used_missing(self, mock_run_v2, mock_list, mock_ref):
        mock_list.return_value = [{
            "name": "lamia-nodate",
            "full_name": "projects/p/locations/l/jobs/lamia-nodate",
            "labels": {"lamia-managed": "true"},
            "create_time": None,
            "update_time": None,
        }]
        mock_ref.return_value = set()

        cleaned = cleanup_stale_resources("my-project", "us-central1")
        assert cleaned == []
        mock_run_v2.JobsClient.return_value.delete_job.assert_not_called()

    @patch("lamia_cloud.gcp.deployer._referenced_job_names")
    @patch("lamia_cloud.gcp.deployer.list_managed_jobs")
    @patch("lamia_cloud.gcp.deployer.run_v2")
    def test_skips_when_last_used_malformed(self, mock_run_v2, mock_list, mock_ref):
        mock_list.return_value = [{
            "name": "lamia-bad-date",
            "full_name": "projects/p/locations/l/jobs/lamia-bad-date",
            "labels": {
                "lamia-last-used": "bad-date",
                "lamia-managed": "true",
                LABEL_RESOURCE_TYPE: "one-shot",
            },
            "create_time": None,
            "update_time": None,
        }]
        mock_ref.return_value = set()

        cleaned = cleanup_stale_resources("my-project", "us-central1")
        assert cleaned == []
        mock_run_v2.JobsClient.return_value.delete_job.assert_not_called()
