"""Tests for lamia_cloud.gcp."""

import pytest
from unittest.mock import patch, MagicMock

from lamia_cloud.gcp import GCPCloudScheduler


class TestGCPCloudScheduler:
    def test_from_config_missing_project_id_raises(self):
        with pytest.raises(ValueError, match="cloud.project_id is required"):
            GCPCloudScheduler.from_config({"provider": "gcp", "location": "us-central1"})

    def test_from_config_missing_target_url_raises(self):
        with pytest.raises(ValueError, match="cloud.target_url is required"):
            GCPCloudScheduler.from_config({
                "provider": "gcp",
                "project_id": "proj",
                "location": "us-central1",
            })

    @patch("lamia_cloud.gcp._enable_api_if_needed")
    def test_from_config_valid(self, mock_enable):
        scheduler = GCPCloudScheduler.from_config({
            "provider": "gcp",
            "project_id": "proj",
            "location": "europe-west1",
            "target_url": "https://example.com/run",
        })
        assert scheduler.project_id == "proj"
        assert scheduler.location == "europe-west1"
        assert scheduler.target_url == "https://example.com/run"

    def test_name(self):
        assert GCPCloudScheduler.name() == "gcp-cloud"
