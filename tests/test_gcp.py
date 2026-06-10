"""Tests for lamia_cloud.gcp."""

import pytest
from unittest.mock import patch, MagicMock

from lamia_cloud.gcp import GCPCloudScheduler


class TestGCPCloudScheduler:
    def test_from_config_missing_project_id_raises(self):
        with pytest.raises(ValueError, match="cloud.project_id is required"):
            GCPCloudScheduler.from_config({"provider": "gcp", "location": "us-central1"})

    @patch("lamia_cloud.gcp._enable_apis")
    def test_from_config_valid(self, mock_enable):
        scheduler = GCPCloudScheduler.from_config({
            "provider": "gcp",
            "project_id": "proj",
            "location": "europe-west1",
        })
        assert scheduler.project_id == "proj"
        assert scheduler.location == "europe-west1"

    @patch("lamia_cloud.gcp._enable_apis")
    def test_from_config_defaults_location(self, mock_enable):
        scheduler = GCPCloudScheduler.from_config({
            "provider": "gcp",
            "project_id": "proj",
        })
        assert scheduler.location == "us-central1"

    def test_name(self):
        assert GCPCloudScheduler.name() == "gcp-cloud"
