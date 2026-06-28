"""Tests for lamia_cloud.gcp.deployer (packaging logic, no GCP required)."""

import io
import tarfile
from pathlib import Path

import pytest

from lamia_cloud.gcp.deployer import (
    _extract_capability_flags,
    compute_resource_tier,
    create_source_tarball,
    package_deployment,
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
