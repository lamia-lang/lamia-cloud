"""Tests for lamia_cloud.gcp.deployer (packaging logic, no GCP required)."""

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import lamia_cloud.gcp.deployer as deployer_module
from lamia_cloud.contracts import FileSyncEntry
from lamia_cloud.gcp.deployer import (
    _REQUIRED_GCP_APIS,
    _extract_capability_flags,
    compute_resource_tier,
    create_source_tarball,
    ensure_apis_enabled,
    package_deployment,
    sync_files_to_bucket,
)


class TestPackageDeployment:
    def test_creates_staging_with_dockerfile(self, tmp_path):
        script = tmp_path / "hello.lm"
        script.write_text('def greet() -> str:\n    return "hi"\n')

        staging = package_deployment(tmp_path, "hello.lm", "abc123")

        assert (staging / "Dockerfile").exists()
        assert (staging / "requirements.txt").exists()
        assert (staging / "project" / "hello.lm").exists()

    def test_requirements_includes_lamia(self, tmp_path):
        script = tmp_path / "hello.lm"
        script.write_text('print("hi")')

        staging = package_deployment(tmp_path, "hello.lm", "abc123")
        reqs = (staging / "requirements.txt").read_text()
        assert "lamia-lang" in reqs

    def test_preserves_existing_requirements(self, tmp_path):
        script = tmp_path / "hello.lm"
        script.write_text('print("hi")')
        (tmp_path / "requirements.txt").write_text("requests>=2.0\npandas\n")

        staging = package_deployment(tmp_path, "hello.lm", "abc123")
        reqs = (staging / "requirements.txt").read_text()
        assert "lamia-lang" in reqs
        assert "requests>=2.0" in reqs
        assert "pandas" in reqs

    def test_copies_project_files(self, tmp_path):
        (tmp_path / "hello.lm").write_text('print("hi")')
        (tmp_path / "config.yaml").write_text("llm:\n  model: gpt-4\n")
        (tmp_path / "helpers.py").write_text("x = 1")
        (tmp_path / "data.json").write_text("{}")

        staging = package_deployment(tmp_path, "hello.lm", "abc123")
        project = staging / "project"
        assert (project / "hello.lm").exists()
        assert (project / "config.yaml").exists()
        assert (project / "helpers.py").exists()
        assert (project / "data.json").exists()


    def test_env_files_excluded_from_deployment(self, tmp_path):
        """SECURITY: .env files must never be baked into Docker images."""
        (tmp_path / "hello.lm").write_text('print("hi")')
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-secret-value\n")
        (tmp_path / "config.yaml").write_text("llm:\n  model: gpt-4\n")

        staging = package_deployment(tmp_path, "hello.lm", "abc123")
        project = staging / "project"
        assert not (project / ".env").exists()
        assert (project / "hello.lm").exists()
        assert (project / "config.yaml").exists()

    def test_dockerfile_default_cmd_no_files(self, tmp_path):
        (tmp_path / "hello.lm").write_text('print("hi")')

        staging = package_deployment(tmp_path, "hello.lm", "abc123", uses_files=False)
        content = (staging / "Dockerfile").read_text()
        assert "cd /app/project && lamia ${LAMIA_SCRIPT}" in content
        assert "/mnt/lamia-files" not in content

    def test_dockerfile_cmd_uses_fuse_mount_when_files_used(self, tmp_path):
        (tmp_path / "hello.lm").write_text('file.read("data.txt")')

        staging = package_deployment(tmp_path, "hello.lm", "abc123", uses_files=True)
        content = (staging / "Dockerfile").read_text()
        assert "cd /mnt/lamia-files" in content
        assert "lamia /app/project/${LAMIA_SCRIPT}" in content


class TestCreateSourceTarball:
    def test_produces_valid_gzip_tarball(self, tmp_path):
        script = tmp_path / "hello.lm"
        script.write_text('print("hi")')

        staging = package_deployment(tmp_path, "hello.lm", "abc123")
        tarball_bytes = create_source_tarball(staging)

        assert len(tarball_bytes) > 0
        buf = io.BytesIO(tarball_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
            assert "Dockerfile" in names
            assert "requirements.txt" in names


class TestResourceTierCalculation:
    def test_default_tier_is_smallest(self):
        assert compute_resource_tier() == ("512Mi", "1")

    def test_llm_only_tier(self):
        assert compute_resource_tier(uses_llm=True) == ("1Gi", "1")

    def test_files_only_tier(self):
        assert compute_resource_tier(uses_files=True) == ("1Gi", "1")

    def test_file_context_only_tier(self):
        assert compute_resource_tier(uses_file_context=True) == ("1Gi", "1")

    def test_browser_tier_dominates(self):
        assert compute_resource_tier(uses_browser=True) == ("4Gi", "2")
        assert compute_resource_tier(
            uses_llm=True,
            uses_browser=True,
            uses_files=True,
            uses_file_context=True,
        ) == ("4Gi", "2")


class TestCapabilityContract:
    def test_extract_capability_flags_accepts_valid_payload(self):
        payload = {
            "uses_llm": True,
            "uses_browser": False,
            "uses_files": True,
            "uses_file_context": False,
        }

        assert _extract_capability_flags(payload) == {
            "uses_llm": True,
            "uses_browser": False,
            "uses_files": True,
            "uses_file_context": False,
        }

    def test_extract_capability_flags_raises_on_missing_fields(self):
        payload = {
            "uses_llm": True,
            "uses_browser": False,
            "uses_files": True,
        }

        with pytest.raises(
            ValueError,
            match=(
                r"missing fields \[uses_file_context\].*"
                r"update BOTH the producer capability payload schema and "
                r"lamia_cloud\.contracts\.SCRIPT_CAPABILITY_FIELDS"
            ),
        ):
            _extract_capability_flags(payload)


class _FakeBlob:
    def __init__(self, key, existing):
        self.key = key
        self._existing = existing
        self.metadata = existing.get(key, {}).get("metadata")
        self._existing_store = existing

    def exists(self):
        return self.key in self._existing_store

    def reload(self):
        if self.exists():
            self.metadata = self._existing_store[self.key]["metadata"]

    def upload_from_filename(self, path):
        self._existing_store[self.key] = {"metadata": self.metadata, "path": path}


class _FakeBucket:
    def __init__(self, existing):
        self._existing = existing

    def blob(self, key):
        return _FakeBlob(key, self._existing)


class _FakeStorageClient:
    def __init__(self, existing):
        self._existing = existing

    def bucket(self, _bucket_name):
        return _FakeBucket(self._existing)


class TestIncrementalFileSync:
    def test_sync_uploads_new_files_and_skips_unchanged(self, tmp_path, monkeypatch):
        local = tmp_path / "data.txt"
        local.write_text("hello")

        existing = {}
        monkeypatch.setattr(
            deployer_module.storage,
            "Client",
            lambda project: _FakeStorageClient(existing),
        )

        plan = [FileSyncEntry(raw_path="data.txt", resolved_path=str(local), bucket_key="data.txt")]

        first = sync_files_to_bucket("proj", "bucket", plan)
        assert first["uploaded"] == 1
        assert first["skipped"] == 0

        second = sync_files_to_bucket("proj", "bucket", plan)
        assert second["uploaded"] == 0
        assert second["skipped"] == 1

    def test_sync_warns_on_overwrite(self, tmp_path, monkeypatch):
        local = tmp_path / "data.txt"
        local.write_text("new-content")

        existing = {
            "data.txt": {"metadata": {"lamia-sha256": "oldhash"}, "path": "old"}
        }
        monkeypatch.setattr(
            deployer_module.storage,
            "Client",
            lambda project: _FakeStorageClient(existing),
        )

        plan = [FileSyncEntry(raw_path="data.txt", resolved_path=str(local), bucket_key="data.txt")]
        result = sync_files_to_bucket("proj", "bucket", plan)
        assert result["uploaded"] == 1
        assert len(result["overwrite_warnings"]) == 1


class TestEnsureApisEnabled:
    @patch("lamia_cloud.gcp.deployer.service_usage_v1")
    def test_enables_all_required_apis(self, mock_service_usage):
        mock_client = MagicMock()
        mock_service_usage.ServiceUsageClient.return_value = mock_client

        ensure_apis_enabled("my-project")

        mock_service_usage.ServiceUsageClient.assert_called_once()
        enabled_services = [
            call.kwargs["request"]["name"]
            for call in mock_client.enable_service.call_args_list
        ]
        assert enabled_services == [
            f"projects/my-project/services/{api}" for api in _REQUIRED_GCP_APIS
        ]

    def test_no_crash_when_service_usage_not_importable(self, monkeypatch):
        monkeypatch.setattr(deployer_module, "service_usage_v1", None)

        ensure_apis_enabled("my-project")

    @patch("lamia_cloud.gcp.deployer.service_usage_v1")
    def test_idempotent_when_called_twice(self, mock_service_usage):
        mock_client = MagicMock()
        mock_service_usage.ServiceUsageClient.return_value = mock_client

        ensure_apis_enabled("my-project")
        ensure_apis_enabled("my-project")

        assert mock_client.enable_service.call_count == len(_REQUIRED_GCP_APIS) * 2

    @patch("lamia_cloud.gcp.deployer.service_usage_v1")
    def test_warns_when_service_usage_api_disabled(self, mock_service_usage, caplog):
        mock_client = MagicMock()
        mock_service_usage.ServiceUsageClient.return_value = mock_client
        mock_client.enable_service.side_effect = Exception(
            "SERVICE_DISABLED: serviceusage.googleapis.com"
        )

        with caplog.at_level("WARNING"):
            ensure_apis_enabled("my-project")

        assert "Service Usage API not enabled" in caplog.text
        assert mock_client.enable_service.call_count == 1
