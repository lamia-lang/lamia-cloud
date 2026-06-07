"""Tests for lamia_cloud.loader."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lamia_cloud.loader import get_scheduler


class TestGetScheduler:
    def test_missing_config_yaml_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No config.yaml found"):
            get_scheduler(tmp_path)

    def test_missing_cloud_section_raises(self, tmp_path):
        (tmp_path / "config.yaml").write_text("llm:\n  model: gpt-4\n")
        with pytest.raises(ValueError, match="must contain a 'cloud' section"):
            get_scheduler(tmp_path)

    def test_missing_provider_raises(self, tmp_path):
        (tmp_path / "config.yaml").write_text("cloud:\n  project_id: x\n")
        with pytest.raises(ValueError, match="cloud.provider is required"):
            get_scheduler(tmp_path)

    def test_unsupported_provider_raises(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "cloud:\n  provider: aws\n  project_id: x\n"
        )
        with pytest.raises(ValueError, match="Unsupported cloud provider 'aws'"):
            get_scheduler(tmp_path)

    @patch("lamia_cloud.gcp.GCPCloudScheduler.from_config")
    def test_valid_gcp_config_returns_scheduler(self, mock_from_config, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "cloud:\n"
            "  provider: gcp\n"
            "  project_id: my-project\n"
            "  location: us-central1\n"
            "  target_url: https://example.com/schedule\n"
        )
        mock_scheduler = MagicMock()
        mock_from_config.return_value = mock_scheduler

        result = get_scheduler(tmp_path)

        assert result is mock_scheduler
        mock_from_config.assert_called_once_with({
            "provider": "gcp",
            "project_id": "my-project",
            "location": "us-central1",
            "target_url": "https://example.com/schedule",
        })
