"""Tests for lamia_cloud.gcp.deployer (packaging, build, deploy, run)."""

import io
import tarfile
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import lamia_cloud.gcp.deployer as deployer_module
from lamia_cloud.contracts import FileSyncEntry
from lamia_cloud.gcp.deployer import (
    _REQUIRED_GCP_APIS,
    _extract_capability_flags,
    _execution_from_operation,
    _result_from_execution,
    _cloud_logging_url,
    _memory_to_mib,
    collect_project_files,
    compute_resource_tier,
    create_source_tarball,
    deployment_name,
    fetch_execution_logs,
    ensure_apis_enabled,
    package_deployment,
    run_job,
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
        assert "lamia-cloud" in reqs

    def test_preserves_existing_requirements(self, tmp_path):
        script = tmp_path / "hello.lm"
        script.write_text('print("hi")')
        (tmp_path / "requirements.txt").write_text("requests>=2.0\npandas\n")

        staging = package_deployment(tmp_path, "hello.lm", "abc123")
        reqs = (staging / "requirements.txt").read_text()
        assert "lamia-lang" in reqs
        assert "lamia-cloud" in reqs
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


class TestPackageDeploymentGitMode:
    """Git mode: tarball contains only Dockerfile + requirements.txt, no project files."""

    def test_git_mode_no_project_directory(self, tmp_path):
        (tmp_path / "hello.lm").write_text('print("hi")')

        staging = package_deployment(
            tmp_path, "hello.lm", "abc123", deploy_mode="git",
        )

        assert (staging / "Dockerfile").exists()
        assert (staging / "requirements.txt").exists()
        assert not (staging / "project").exists()

    def test_git_mode_dockerfile_identical_to_local(self, tmp_path):
        (tmp_path / "hello.lm").write_text('print("hi")')

        local_staging = package_deployment(tmp_path, "hello.lm", "abc1")
        git_staging = package_deployment(
            tmp_path, "hello.lm", "abc2", deploy_mode="git",
        )

        local_df = (local_staging / "Dockerfile").read_text()
        git_df = (git_staging / "Dockerfile").read_text()
        assert local_df == git_df

    def test_git_mode_requirements_same_as_local(self, tmp_path):
        (tmp_path / "hello.lm").write_text('print("hi")')
        (tmp_path / "requirements.txt").write_text("requests>=2.0\n")

        local_staging = package_deployment(tmp_path, "hello.lm", "a1")
        git_staging = package_deployment(
            tmp_path, "hello.lm", "a2", deploy_mode="git",
        )

        assert (local_staging / "requirements.txt").read_text() == \
               (git_staging / "requirements.txt").read_text()

    def test_git_mode_tarball_is_smaller(self, tmp_path):
        (tmp_path / "hello.lm").write_text('print("hi")')
        (tmp_path / "big_data.json").write_text("{}" * 10000)

        local_staging = package_deployment(tmp_path, "hello.lm", "a1")
        git_staging = package_deployment(
            tmp_path, "hello.lm", "a2", deploy_mode="git",
        )

        local_tarball = create_source_tarball(local_staging)
        git_tarball = create_source_tarball(git_staging)
        assert len(git_tarball) < len(local_tarball)


class TestSubmitBuildGitMode:
    """submit_build() adds a git clone step when repo_url is provided."""

    @patch("lamia_cloud.gcp.deployer.cloudbuild_v1")
    def test_local_mode_single_docker_step(self, mock_cb):
        mock_client = MagicMock()
        mock_cb.CloudBuildClient.return_value = mock_client
        op = MagicMock()
        op.result.return_value = MagicMock(
            status=mock_cb.Build.Status.SUCCESS,
        )
        mock_client.create_build.return_value = op

        from lamia_cloud.gcp.deployer import submit_build
        submit_build("proj", "gs://bucket/src.tar.gz", "img:latest")

        step_calls = mock_cb.BuildStep.call_args_list
        assert len(step_calls) == 1
        assert step_calls[0].kwargs["name"] == "gcr.io/cloud-builders/docker"

    @patch("lamia_cloud.gcp.deployer.cloudbuild_v1")
    def test_git_mode_prepends_clone_step(self, mock_cb):
        mock_client = MagicMock()
        mock_cb.CloudBuildClient.return_value = mock_client
        op = MagicMock()
        op.result.return_value = MagicMock(
            status=mock_cb.Build.Status.SUCCESS,
        )
        mock_client.create_build.return_value = op

        from lamia_cloud.gcp.deployer import submit_build
        submit_build(
            "proj", "gs://bucket/src.tar.gz", "img:latest",
            repo_url="https://github.com/lamia-lang/lamia",
        )

        step_calls = mock_cb.BuildStep.call_args_list
        assert len(step_calls) == 2
        assert step_calls[0].kwargs["name"] == "gcr.io/cloud-builders/git"
        assert "https://github.com/lamia-lang/lamia" in step_calls[0].kwargs["args"]
        assert step_calls[1].kwargs["name"] == "gcr.io/cloud-builders/docker"

    @patch("lamia_cloud.gcp.deployer.cloudbuild_v1")
    def test_git_clone_pins_main_branch(self, mock_cb):
        """S3: Clone step must specify --branch main to prevent ref confusion."""
        mock_client = MagicMock()
        mock_cb.CloudBuildClient.return_value = mock_client
        op = MagicMock()
        op.result.return_value = MagicMock(
            status=mock_cb.Build.Status.SUCCESS,
        )
        mock_client.create_build.return_value = op

        from lamia_cloud.gcp.deployer import submit_build
        submit_build(
            "proj", "gs://bucket/src.tar.gz", "img:latest",
            repo_url="https://github.com/acme/app",
        )

        clone_args = mock_cb.BuildStep.call_args_list[0].kwargs["args"]
        assert "--branch" in clone_args
        assert "main" in clone_args


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


class TestMemoryToMib:
    def test_gibibytes(self):
        assert _memory_to_mib("4Gi") == 4096

    def test_mebibytes(self):
        assert _memory_to_mib("512Mi") == 512

    def test_gigabytes(self):
        assert _memory_to_mib("2G") == 2048

    def test_megabytes(self):
        assert _memory_to_mib("1024M") == 1024

    def test_garbage_input_defaults_to_512(self):
        assert _memory_to_mib("not-a-memory-value") == 512


class TestResourceTierCalculation:
    def test_default_tier_is_smallest(self):
        assert compute_resource_tier() == ("512Mi", "1")

    def test_llm_only_tier(self):
        assert compute_resource_tier(uses_llm=True) == ("1Gi", "1")

    def test_files_only_tier(self):
        assert compute_resource_tier(uses_files=True) == ("1Gi", "1")

    def test_file_context_only_tier(self):
        assert compute_resource_tier(uses_file_context=True) == ("1Gi", "1")

    def test_browser_tier(self):
        assert compute_resource_tier(uses_browser=True) == ("4Gi", "2")

    def test_combined_flags_browser_dominates(self):
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

    def test_extract_capability_flags_raises_on_non_dict(self):
        with pytest.raises(ValueError, match="expected dict-like mapping"):
            _extract_capability_flags("not-a-dict")


class TestCollectProjectFiles:
    def test_collects_supported_files_and_excludes_env(self, tmp_path):
        (tmp_path / "script.lm").write_text("def run(): pass")
        (tmp_path / "helpers.py").write_text("x = 1")
        (tmp_path / "config.yaml").write_text("cloud:\n  project_id: proj")
        (tmp_path / ".env").write_text("SECRET=leak")
        subdir = tmp_path / "lib"
        subdir.mkdir()
        (subdir / "util.py").write_text("def util(): pass")

        collected = {f.name for f in collect_project_files(tmp_path)}

        assert collected == {"script.lm", "helpers.py", "config.yaml", "util.py"}
        assert ".env" not in collected


class TestDeploymentName:
    def test_prepends_lamia_prefix(self):
        assert deployment_name("hello") == "lamia-hello"


class TestCloudLoggingUrl:
    NAME = "projects/p/locations/us-central1/jobs/j/executions/exec-123"

    def _url(self, **kwargs):
        return _cloud_logging_url(
            project_id="my-project",
            target="lamia-hello",
            execution_name=self.NAME,
            **kwargs,
        )

    def test_builds_filtered_console_url(self):
        url = self._url()
        assert url.startswith("https://console.cloud.google.com/logs/query;")
        assert "project=my-project" in url
        assert "lamia-hello" in url
        assert "exec-123" in urllib.parse.unquote(url)

    def test_filter_is_fully_encoded(self):
        """A raw '/' would end the path segment and truncate the filter."""
        url = self._url()
        query = url.split(";query=", 1)[1].split("?", 1)[0]
        assert "/" not in query
        assert "run.googleapis.com%2Fexecution_name" in query

    def test_includes_execution_region(self):
        assert 'resource.labels.location="us-central1"' in urllib.parse.unquote(self._url())

    def test_region_omitted_when_name_has_none(self):
        url = _cloud_logging_url(
            project_id="my-project", target="lamia-hello", execution_name="exec-123",
        )
        assert "resource.labels.location" not in urllib.parse.unquote(url)

    def test_time_range_spans_the_execution(self):
        start = datetime(2026, 8, 13, 8, 51, 41, tzinfo=timezone.utc)
        end = datetime(2026, 8, 13, 8, 53, 48, tzinfo=timezone.utc)

        window = urllib.parse.unquote(self._url(start_time=start, end_time=end))
        window = window.split(";timeRange=", 1)[1].split("?", 1)[0]
        begins, _, finishes = window.partition("/")

        assert begins == "2026-08-13T08:51:41Z"
        assert finishes == "2026-08-13T08:58:48Z", "tail buffer for late log delivery"

    def test_no_time_range_without_start(self):
        assert ";timeRange=" not in self._url()


class TestFetchExecutionLogs:
    """Container output only: audit logs share the execution's resource labels."""

    @staticmethod
    def _entry(payload, stream="stdout", severity="INFO"):
        entry = MagicMock()
        entry.payload = payload
        entry.severity = severity
        entry.log_name = f"projects/proj/logs/run.googleapis.com%2F{stream}"
        return entry

    def _fetch(self, mock_client_cls, entries):
        mock_client = MagicMock()
        mock_client.list_entries.return_value = entries
        mock_client_cls.return_value = mock_client
        with patch("lamia_cloud.gcp.deployer.time.sleep"):
            result = fetch_execution_logs(
                project_id="proj",
                target="lamia-task",
                execution_name="projects/p/locations/l/jobs/j/executions/exec-1",
            )
        return result, mock_client

    @patch("lamia_cloud.gcp.deployer.cloud_logging.Client")
    def test_splits_by_log_stream_not_severity(self, mock_client_cls):
        entries = [
            self._entry("hello stdout"),
            self._entry("something failed", stream="stderr", severity="ERROR"),
            self._entry("watch out", severity="WARNING"),
        ]

        (stdout, stderr), _ = self._fetch(mock_client_cls, entries)

        assert stdout == "hello stdout\nwatch out"
        assert stderr == "something failed"

    @patch("lamia_cloud.gcp.deployer.cloud_logging.Client")
    def test_filter_restricts_to_container_streams(self, mock_client_cls):
        _, mock_client = self._fetch(mock_client_cls, [])

        filter_arg = mock_client.list_entries.call_args.kwargs["filter_"]
        assert 'resource.labels.job_name="lamia-task"' in filter_arg
        assert 'execution_name"="exec-1"' in filter_arg
        assert "run.googleapis.com%2Fstdout" in filter_arg
        assert "run.googleapis.com%2Fstderr" in filter_arg
        assert "logName=(" in filter_arg

    @patch("lamia_cloud.gcp.deployer.cloud_logging.Client")
    def test_structured_payload_uses_message_field(self, mock_client_cls):
        entries = [self._entry({"message": "structured line", "severity": "INFO"})]

        (stdout, _), _ = self._fetch(mock_client_cls, entries)

        assert stdout == "structured line"

    @patch("lamia_cloud.gcp.deployer.cloud_logging.Client")
    def test_structured_payload_without_message_is_json(self, mock_client_cls):
        entries = [self._entry({"foo": "bar"})]

        (stdout, _), _ = self._fetch(mock_client_cls, entries)

        assert stdout == '{"foo": "bar"}'
        assert "OrderedDict" not in stdout

    @patch("lamia_cloud.gcp.deployer.cloud_logging.Client")
    def test_empty_payloads_are_dropped(self, mock_client_cls):
        entries = [self._entry(None), self._entry(""), self._entry("kept")]

        (stdout, _), _ = self._fetch(mock_client_cls, entries)

        assert stdout == "kept"

    @patch("lamia_cloud.gcp.deployer.time.sleep")
    @patch("lamia_cloud.gcp.deployer.cloud_logging.Client")
    def test_retries_when_logging_has_not_ingested_yet(self, mock_client_cls, mock_sleep):
        mock_client = MagicMock()
        late = [self._entry("late but here")]
        mock_client.list_entries.side_effect = [[], late, late]
        mock_client_cls.return_value = mock_client

        stdout, stderr = fetch_execution_logs(
            project_id="proj",
            target="lamia-task",
            execution_name="projects/p/locations/l/jobs/j/executions/exec-1",
        )

        assert stdout == "late but here"
        assert mock_client.list_entries.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("lamia_cloud.gcp.deployer.time.sleep")
    @patch("lamia_cloud.gcp.deployer.cloud_logging.Client")
    def test_does_not_return_partial_result_while_logs_are_still_growing(
        self, mock_client_cls, mock_sleep
    ):
        line1 = [self._entry("line1")]
        line1_and_2 = [self._entry("line1"), self._entry("line2")]
        mock_client = MagicMock()
        mock_client.list_entries.side_effect = [line1, line1_and_2, line1_and_2]
        mock_client_cls.return_value = mock_client

        stdout, _ = fetch_execution_logs(
            project_id="proj",
            target="lamia-task",
            execution_name="projects/p/locations/l/jobs/j/executions/exec-1",
        )

        assert stdout == "line1\nline2"
        assert mock_client.list_entries.call_count == 3

    @patch("lamia_cloud.gcp.deployer.time.sleep")
    @patch("lamia_cloud.gcp.deployer.cloud_logging.Client")
    def test_gives_up_empty_after_max_attempts(self, mock_client_cls, mock_sleep):
        mock_client = MagicMock()
        mock_client.list_entries.return_value = []
        mock_client_cls.return_value = mock_client

        stdout, stderr = fetch_execution_logs(
            project_id="proj",
            target="lamia-task",
            execution_name="projects/p/locations/l/jobs/j/executions/exec-1",
        )

        assert (stdout, stderr) == ("", "")
        assert mock_client.list_entries.call_count == 5
        assert mock_sleep.call_count == 4


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


class TestRunJob:
    def test_run_job_updates_last_used_when_stale(self, monkeypatch):
        from google.cloud import run_v2
        from google.protobuf import timestamp_pb2

        execution = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-test/executions/exec-ok",
            succeeded_count=1,
            start_time=timestamp_pb2.Timestamp(seconds=200, nanos=0),
            completion_time=timestamp_pb2.Timestamp(seconds=205, nanos=0),
        )

        job_obj = type("JobObj", (), {"labels": {"lamia-last-used": "20240101"}})()

        class FakeOperation:
            def result(self):
                return execution

        class FakeClient:
            def __init__(self):
                self.updated = 0

            def get_job(self, request):
                return job_obj

            def update_job(self, job):
                self.updated += 1
                return None

            def run_job(self, request):
                return FakeOperation()

        fake_client = FakeClient()
        monkeypatch.setattr(deployer_module.run_v2, "JobsClient", lambda: fake_client)

        result = run_job("p", "us-central1", "lamia-test")
        assert result["exit_code"] == 0
        assert fake_client.updated == 1
        assert len(job_obj.labels["lamia-last-used"]) == 8

    def test_run_job_does_not_update_last_used_when_already_today(self, monkeypatch):
        from google.cloud import run_v2
        from google.protobuf import timestamp_pb2

        execution = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-test/executions/exec-ok",
            succeeded_count=1,
            start_time=timestamp_pb2.Timestamp(seconds=200, nanos=0),
            completion_time=timestamp_pb2.Timestamp(seconds=205, nanos=0),
        )

        today = deployer_module._today_label()
        job_obj = type("JobObj", (), {"labels": {"lamia-last-used": today}})()

        class FakeOperation:
            def result(self):
                return execution

        class FakeClient:
            def __init__(self):
                self.updated = 0

            def get_job(self, request):
                return job_obj

            def update_job(self, job):
                self.updated += 1
                return None

            def run_job(self, request):
                return FakeOperation()

        fake_client = FakeClient()
        monkeypatch.setattr(deployer_module.run_v2, "JobsClient", lambda: fake_client)

        result = run_job("p", "us-central1", "lamia-test")
        assert result["exit_code"] == 0
        assert fake_client.updated == 0

    def test_run_job_returns_failure_on_aborted(self, monkeypatch):
        from google.api_core.exceptions import Aborted
        from google.cloud import run_v2
        from google.protobuf import timestamp_pb2

        start = timestamp_pb2.Timestamp(seconds=100, nanos=0)
        end = timestamp_pb2.Timestamp(seconds=110, nanos=0)
        execution_meta = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-test/executions/exec-123",
            succeeded_count=0,
            failed_count=1,
            start_time=start,
            completion_time=end,
        )

        class FakeOperation:
            def result(self):
                raise Aborted("The container exited with an error")

            @property
            def metadata(self):
                return execution_meta

            @property
            def operation(self):
                return type("Op", (), {"HasField": lambda self, f: False})()

        class FakeClient:
            def run_job(self, request):
                return FakeOperation()

        monkeypatch.setattr(deployer_module.run_v2, "JobsClient", lambda: FakeClient())

        result = run_job("p", "us-central1", "lamia-test")

        assert result["exit_code"] == 1
        assert result["execution_name"].endswith("/executions/exec-123")
        assert result["elapsed_seconds"] == 10.0
        assert "exec-123" in result["logs_url"]
        assert "console.cloud.google.com/logs" in result["logs_url"]

    def test_run_job_recovers_logs_on_non_aborted_failure(self, monkeypatch):
        """Timeouts and cancellations also leave an Execution worth reporting."""
        from google.api_core.exceptions import DeadlineExceeded
        from google.cloud import run_v2
        from google.protobuf import timestamp_pb2

        execution_meta = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-test/executions/exec-slow",
            succeeded_count=0,
            failed_count=1,
            start_time=timestamp_pb2.Timestamp(seconds=100),
            completion_time=timestamp_pb2.Timestamp(seconds=130),
        )

        class FakeOperation:
            def result(self):
                raise DeadlineExceeded("Execution timed out")

            @property
            def metadata(self):
                return execution_meta

            @property
            def operation(self):
                return type("Op", (), {"HasField": lambda self, f: False})()

        monkeypatch.setattr(
            deployer_module.run_v2,
            "JobsClient",
            lambda: type("C", (), {"run_job": lambda self, request: FakeOperation()})(),
        )

        result = run_job("p", "us-central1", "lamia-test")

        assert result["exit_code"] == 1
        assert result["execution_name"].endswith("/executions/exec-slow")
        assert result["elapsed_seconds"] == 30.0

    def test_run_job_reraises_when_no_execution_exists(self, monkeypatch):
        """API errors that never produced an Execution must not be swallowed."""
        from google.api_core.exceptions import PermissionDenied

        class FakeOperation:
            def result(self):
                raise PermissionDenied("caller lacks run.jobs.run")

            @property
            def metadata(self):
                return None

            @property
            def operation(self):
                return type("Op", (), {"HasField": lambda self, f: False})()

        monkeypatch.setattr(
            deployer_module.run_v2,
            "JobsClient",
            lambda: type("C", (), {"run_job": lambda self, request: FakeOperation()})(),
        )

        with pytest.raises(PermissionDenied):
            run_job("p", "us-central1", "lamia-test")

    def test_run_job_success_path(self, monkeypatch):
        from google.cloud import run_v2
        from google.protobuf import timestamp_pb2

        start = timestamp_pb2.Timestamp(seconds=200, nanos=0)
        end = timestamp_pb2.Timestamp(seconds=205, nanos=500000000)
        execution = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-test/executions/exec-ok",
            succeeded_count=1,
            start_time=start,
            completion_time=end,
        )

        class FakeOperation:
            def result(self):
                return execution

        class FakeClient:
            def run_job(self, request):
                return FakeOperation()

        monkeypatch.setattr(deployer_module.run_v2, "JobsClient", lambda: FakeClient())

        result = run_job("p", "us-central1", "lamia-test")

        assert result["exit_code"] == 0
        assert result["elapsed_seconds"] == 5.5

    def test_execution_from_operation_reads_metadata(self):
        from google.cloud import run_v2

        execution_meta = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-test/executions/exec-meta",
        )

        class FakeOperation:
            @property
            def metadata(self):
                return execution_meta

            @property
            def operation(self):
                return type("Op", (), {"HasField": lambda self, f: False})()

        extracted = _execution_from_operation(FakeOperation())
        assert extracted.name.endswith("/executions/exec-meta")

    def test_result_from_execution_builds_logs_url(self):
        from google.cloud import run_v2

        execution = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-hello/executions/exec-1",
            succeeded_count=0,
        )
        result = _result_from_execution("my-project", "lamia-hello", execution)
        assert result["exit_code"] == 1
        assert result["execution_name"] == execution.name
        assert "my-project" in result["logs_url"]
        assert "lamia-hello" in result["logs_url"]
        assert result["pending_seconds"] is None
        assert result["running_seconds"] is None

    def test_result_from_execution_splits_pending_and_running_time(self):
        from datetime import datetime, timedelta, timezone
        from google.cloud import run_v2

        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        started = created + timedelta(seconds=80)
        completed = started + timedelta(seconds=20)

        execution = run_v2.Execution(
            name="projects/p/locations/us-central1/jobs/lamia-hello/executions/exec-1",
            succeeded_count=1,
            create_time=created,
            completion_time=completed,
            conditions=[run_v2.Condition(type_="Started", last_transition_time=started)],
        )
        result = _result_from_execution("my-project", "lamia-hello", execution)
        assert result["pending_seconds"] == 80.0
        assert result["running_seconds"] == 20.0

class TestEnsureApisEnabled:
    @staticmethod
    def _service(api: str, enabled: bool, enabled_state):
        service = MagicMock()
        service.name = f"projects/my-project/services/{api}"
        service.state = enabled_state if enabled else object()
        return service

    @patch("lamia_cloud.gcp.deployer.service_usage_v1")
    def test_skips_enable_when_all_already_enabled(self, mock_service_usage):
        mock_client = MagicMock()
        mock_service_usage.ServiceUsageClient.return_value = mock_client
        enabled_state = mock_service_usage.State.ENABLED
        mock_client.batch_get_services.return_value = MagicMock(
            services=[
                self._service(api, True, enabled_state) for api in _REQUIRED_GCP_APIS
            ]
        )

        ensure_apis_enabled("my-project")

        mock_client.batch_get_services.assert_called_once()
        mock_client.batch_enable_services.assert_not_called()

    @patch("lamia_cloud.gcp.deployer.service_usage_v1")
    def test_enables_only_the_apis_not_yet_enabled(self, mock_service_usage):
        mock_client = MagicMock()
        mock_service_usage.ServiceUsageClient.return_value = mock_client
        enabled_state = mock_service_usage.State.ENABLED
        already_enabled = _REQUIRED_GCP_APIS[0]
        mock_client.batch_get_services.return_value = MagicMock(
            services=[
                self._service(api, api == already_enabled, enabled_state)
                for api in _REQUIRED_GCP_APIS
            ]
        )

        ensure_apis_enabled("my-project")

        service_ids = mock_client.batch_enable_services.call_args.kwargs["request"][
            "service_ids"
        ]
        assert already_enabled not in service_ids
        assert set(service_ids) == set(_REQUIRED_GCP_APIS) - {already_enabled}

    @patch("lamia_cloud.gcp.deployer.service_usage_v1")
    def test_enables_all_required_apis_when_check_fails(self, mock_service_usage):
        mock_client = MagicMock()
        mock_service_usage.ServiceUsageClient.return_value = mock_client
        mock_client.batch_get_services.side_effect = Exception("boom")

        ensure_apis_enabled("my-project")

        service_ids = mock_client.batch_enable_services.call_args.kwargs["request"][
            "service_ids"
        ]
        assert set(service_ids) == set(_REQUIRED_GCP_APIS)

    @patch("lamia_cloud.gcp.deployer.service_usage_v1")
    def test_warns_when_service_usage_api_disabled(self, mock_service_usage, caplog):
        mock_client = MagicMock()
        mock_service_usage.ServiceUsageClient.return_value = mock_client
        mock_client.batch_get_services.side_effect = Exception("boom")
        mock_client.batch_enable_services.side_effect = Exception(
            "SERVICE_DISABLED: serviceusage.googleapis.com"
        )

        with caplog.at_level("WARNING"):
            ensure_apis_enabled("my-project")

        assert "Service Usage API not enabled" in caplog.text
        mock_client.batch_enable_services.assert_called_once()


class TestGCPDeployerTimeoutConfig:
    def test_from_config_uses_default_timeout(self):
        deployer = deployer_module.GCPDeployer.from_config(
            {"project_id": "my-project", "location": "us-central1"}
        )
        assert deployer.task_timeout_seconds == 3600

    def test_from_config_accepts_custom_timeout(self):
        deployer = deployer_module.GCPDeployer.from_config(
            {
                "project_id": "my-project",
                "location": "us-central1",
                "resources": {"task_timeout_seconds": 7200},
            }
        )
        assert deployer.task_timeout_seconds == 7200

    def test_from_config_rejects_out_of_range_timeout(self):
        with pytest.raises(ValueError, match="task_timeout_seconds must be between"):
            deployer_module.GCPDeployer.from_config(
                {
                    "project_id": "my-project",
                    "location": "us-central1",
                    "resources": {"task_timeout_seconds": 700000},
                }
            )
