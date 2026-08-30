"""Deploys a lamia project to Cloud Run as a Job via Cloud Build.

Flow:
1. Package the .lm script + project files into a staging directory
2. Add Dockerfile + requirements.txt
3. Upload to GCS as source tarball
4. Submit Cloud Build to build the container
5. Deploy the container as a Cloud Run Job (with Vertex AI IAM for LLM access)

LLM authentication uses Vertex AI — the Cloud Run Job service account gets
roles/aiplatform.user, so no API keys are needed at runtime.
"""

import hashlib
import io
import json
import logging
import re
import shutil
import tarfile
import tempfile
import time
import urllib.parse
from collections.abc import Mapping
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from google.api_core import protobuf_helpers
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import (
    iam_admin_v1,
    logging as cloud_logging,
    resourcemanager_v3,
    run_v2,
    service_usage_v1,
    storage,
)
from google.cloud.devtools import cloudbuild_v1
from google.iam.v1 import policy_pb2

from lamia_cloud.gcp.secrets import (
    cleanup_secrets,
    secret_env_vars,
    sync_secrets,
)

from lamia_cloud.contracts import (
    CLOUD_TASK_TIMEOUT_DEFAULT_SECONDS,
    CLOUD_TASK_TIMEOUT_MAX_SECONDS,
    CLOUD_TASK_TIMEOUT_MIN_SECONDS,
    LABEL_DEPLOY_MODE,
    LABEL_LAST_USED,
    LABEL_MANAGED,
    LABEL_PROJECT_HASH,
    LABEL_RESOURCE_TYPE,
    LABEL_SCRIPT,
    SCRIPT_CAPABILITY_FIELDS,
    SOURCE_HASH_LABEL,
    STALE_RESOURCE_DAYS,
    sanitize_label_value,
)
from lamia_cloud.file_sync import file_sha256
from lamia_cloud.gcp.llm.vertex import get_verified_vertex_models, remember_verified_vertex_models

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
FILES_MOUNT_PATH = "/mnt/lamia-files"

# Cloud Run writes container output to these two log streams.  Audit logs
# carry the same resource labels, so the stream is what separates script
# output from Cloud Run's own bookkeeping.
STDOUT_LOG_ID = "run.googleapis.com%2Fstdout"
STDERR_LOG_ID = "run.googleapis.com%2Fstderr"

# Logs can land slightly after an execution reports completion.
LOG_WINDOW_TAIL = timedelta(minutes=5)

# Cloud Logging ingestion lags a completed execution by a few seconds,
# most noticeably right after a fresh deploy. Retry before giving up empty.
LOG_FETCH_MAX_ATTEMPTS = 5
LOG_FETCH_RETRY_DELAY_SECONDS = 3.0

_REQUIRED_GCP_APIS = (
    "serviceusage.googleapis.com",
    "cloudscheduler.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "secretmanager.googleapis.com",
)

# Enforced at image build time, so it also covers git mode, where project
# files are cloned by Cloud Build instead of passing through
# collect_project_files.  Both forms are needed: a bare pattern matches only
# the build context root, while `**/` reaches the cloned project directory.
DOCKERIGNORE_CONTENT = """\
.*
**/.*
*.pem
**/*.pem
*.key
**/*.key
"""


def _today_label() -> str:
    """Return today's date as a GCP-label-safe string: YYYYMMDD."""
    return date.today().strftime("%Y%m%d")


def _project_hash(project_root: Path) -> str:
    """8-char hex hash of the project root path for label identification."""
    return hashlib.sha256(str(project_root).encode()).hexdigest()[:8]


def build_resource_labels(
    script_name: str,
    project_root: Path,
    deploy_mode: str = "local",
    repo_url: str | None = None,
) -> dict[str, str]:
    """Build the full set of lamia metadata labels for a Cloud Run Job."""
    labels = {
        LABEL_MANAGED: "true",
        LABEL_SCRIPT: sanitize_label_value(script_name),
        LABEL_PROJECT_HASH: _project_hash(project_root),
        LABEL_LAST_USED: _today_label(),
        LABEL_DEPLOY_MODE: deploy_mode,
        LABEL_RESOURCE_TYPE: "one-shot",
    }
    if repo_url:
        labels["lamia-repo-url"] = sanitize_label_value(repo_url)
    return labels


def _touch_last_used(client, project_id: str, location: str, target: str) -> None:
    """Update the lamia-last-used label on a Cloud Run Job to today."""
    try:
        resource = f"projects/{project_id}/locations/{location}/jobs/{target}"
        job = client.get_job(request={"name": resource})
        if job.labels is None:
            job.labels = {}
        today = _today_label()
        if job.labels.get(LABEL_LAST_USED) != today:
            job.labels[LABEL_LAST_USED] = today
            client.update_job(job=job)
    except Exception as exc:
        logger.warning(f"Failed to update {LABEL_LAST_USED} for {target}: {exc}")


def ensure_apis_enabled(project_id: str) -> None:
    """Enable any of the required GCP APIs that aren't already enabled."""
    client = service_usage_v1.ServiceUsageClient()
    parent = f"projects/{project_id}"
    names = [f"{parent}/services/{api}" for api in _REQUIRED_GCP_APIS]

    try:
        response = client.batch_get_services(request={"parent": parent, "names": names})
        disabled = [
            service.name.rsplit("/", 1)[-1]
            for service in response.services
            if service.state != service_usage_v1.State.ENABLED
        ]
    except Exception:
        disabled = list(_REQUIRED_GCP_APIS)

    if not disabled:
        return

    try:
        client.batch_enable_services(request={"parent": parent, "service_ids": disabled})
    except Exception as e:
        if "SERVICE_DISABLED" in str(e) and "serviceusage.googleapis.com" in disabled:
            logger.warning(
                f"Service Usage API not enabled. Run once:\n"
                f"  gcloud services enable serviceusage.googleapis.com "
                f"--project={project_id}"
            )


def compute_resource_tier(
    uses_llm: bool = False,
    uses_browser: bool = False,
    uses_files: bool = False,
    uses_file_context: bool = False,
) -> tuple[str, str]:
    """Compute (memory, cpu) for a Cloud Run Job based on script capabilities.

    GCP-specific: respects Cloud Run's CPU/memory coupling rules.

    Lamia scripts execute sequentially — LLM, browser, and file operations
    never run concurrently within a single container. Memory is allocated for
    the peak consumer (browser > LLM > files), not the sum.
    """
    memory_mib = 512

    if uses_files or uses_file_context:
        memory_mib = max(memory_mib, 1024)
    if uses_llm:
        memory_mib = max(memory_mib, 1024)
    if uses_browser:
        memory_mib = max(memory_mib, 4096)

    if memory_mib <= 512:
        return ("512Mi", "1")
    elif memory_mib <= 1024:
        return ("1Gi", "1")
    elif memory_mib <= 2048:
        return ("2Gi", "1")
    elif memory_mib <= 4096:
        return ("4Gi", "2")
    elif memory_mib <= 8192:
        return ("8Gi", "2")
    elif memory_mib <= 16384:
        return ("16Gi", "4")
    else:
        return ("32Gi", "8")


def _get_existing_resources(
    project_id: str, location: str, job_name: str
) -> Optional[tuple[str, str]]:
    """Read current memory/cpu from an existing Cloud Run Job.

    Returns (memory, cpu) or None if the job doesn't exist.
    """
    try:
        client = run_v2.JobsClient()
        name = f"projects/{project_id}/locations/{location}/jobs/{job_name}"
        job = client.get_job(request={"name": name})
        containers = job.template.template.containers
        if containers:
            limits = containers[0].resources.limits or {}
            return (limits.get("memory", "512Mi"), limits.get("cpu", "1"))
    except Exception:
        pass
    return None


def _memory_to_mib(mem: str) -> int:
    """Convert memory string like '4Gi' or '512Mi' to MiB integer."""
    mem = mem.strip()
    if mem.endswith("Gi"):
        return int(float(mem[:-2]) * 1024)
    if mem.endswith("Mi"):
        return int(float(mem[:-2]))
    if mem.endswith("G"):
        return int(float(mem[:-1]) * 1024)
    if mem.endswith("M"):
        return int(float(mem[:-1]))
    return 512


def _extract_capability_flags(capabilities) -> dict[str, bool]:
    """Extract and validate capability flags from metadata object.

    This is an explicit contract boundary between lamia core (AST analyzer)
    and cloud providers. If fields are renamed on either side, deployment
    fails fast with a clear error.
    """
    if not isinstance(capabilities, Mapping):
        raise ValueError("Invalid script capability payload: expected dict-like mapping.")

    missing = [field for field in SCRIPT_CAPABILITY_FIELDS if field not in capabilities]
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(
            "Invalid script capability payload: missing fields "
            f"[{missing_csv}]. If you changed capability field names, update BOTH "
            "the producer capability payload schema and "
            "lamia_cloud.contracts.SCRIPT_CAPABILITY_FIELDS."
        )

    return {field: bool(capabilities[field]) for field in SCRIPT_CAPABILITY_FIELDS}


def get_deployed_source_hash(project_id: str, location: str, target: str) -> Optional[str]:
    """Read source hash label from deployed Cloud Run Job."""
    try:
        client = run_v2.JobsClient()
        resource = f"projects/{project_id}/locations/{location}/jobs/{target}"
        job = client.get_job(request={"name": resource})
        return (job.labels or {}).get(SOURCE_HASH_LABEL)
    except Exception:
        return None


def set_deployed_source_hash(project_id: str, location: str, target: str, hash_val: str) -> None:
    """Set source hash label on deployed Cloud Run Job."""
    try:
        client = run_v2.JobsClient()
        resource = f"projects/{project_id}/locations/{location}/jobs/{target}"
        job = client.get_job(request={"name": resource})
        if job.labels is None:
            job.labels = {}
        job.labels[SOURCE_HASH_LABEL] = hash_val
        client.update_job(job=job)
    except Exception:
        pass


def deployment_name(name: str) -> str:
    return f"lamia-{name}"


def _image_name(project_id: str, name: str) -> str:
    ts = int(time.time())
    return f"gcr.io/{project_id}/lamia-{name}:{ts}"


def collect_project_files(project_root: Path) -> list[Path]:
    """Collect .lm files, config.yaml, and supporting Python files from the project.

    SECURITY: nothing whose name starts with a dot is collected.
    Files needed at runtime belong in ``with files(...)`` instead.
    """
    files = []
    for pattern in ("*.lm", "*.py", "*.yaml", "*.yml", "*.json", "*.txt", "*.csv"):
        files.extend(project_root.glob(pattern))
    for subdir in project_root.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for pattern in ("**/*.lm", "**/*.py", "**/*.yaml", "**/*.json"):
                files.extend(subdir.glob(pattern))
    return [f for f in files if not _has_hidden_part(f, project_root)]


def _has_hidden_part(path: Path, project_root: Path) -> bool:
    """True if any path segment below *project_root* starts with a dot.

    Checks every component (directories and filename) for a leading dot.
    Paths not under *project_root* are always treated as hidden (excluded).
    """
    root_parts = project_root.parts
    path_parts = path.parts
    if path_parts[:len(root_parts)] != root_parts:
        return True
    return any(part.startswith(".") for part in path_parts[len(root_parts):])


def package_deployment(
    project_root: Path,
    script_name: str,
    name: str,
    uses_files: bool = False,
    deploy_mode: str = "local",
) -> Path:
    """Create a staging directory with everything needed for Cloud Build.

    In local mode the full project tree is included in the tarball.
    In git mode only the Dockerfile and requirements.txt are included;
    project files are cloned by a Cloud Build step before the Docker build.
    """
    staging = Path(tempfile.mkdtemp(prefix="lamia-deploy-"))

    if deploy_mode != "git":
        project_dest = staging / "project"
        project_dest.mkdir()
        for f in collect_project_files(project_root):
            rel = f.relative_to(project_root)
            dest = project_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

    (staging / ".dockerignore").write_text(DOCKERIGNORE_CONTENT)

    dockerfile_dest = staging / "Dockerfile"
    dockerfile_content = (TEMPLATES_DIR / "Dockerfile").read_text()
    if uses_files:
        cmd = (
            'CMD ["sh", "-c", '
            f'"mkdir -p {FILES_MOUNT_PATH}/${{LAMIA_FILES_NS}} '
            f'&& cd {FILES_MOUNT_PATH}/${{LAMIA_FILES_NS}} '
            '&& lamia /app/project/${LAMIA_SCRIPT} ${LAMIA_EXTRA_ARGS:-}"]'
        )
    else:
        cmd = 'CMD ["sh", "-c", "cd /app/project && lamia ${LAMIA_SCRIPT} ${LAMIA_EXTRA_ARGS:-}"]'
    dockerfile_dest.write_text(dockerfile_content + cmd + "\n")

    requirements = staging / "requirements.txt"
    project_requirements = project_root / "requirements.txt"
    if project_requirements.exists():
        reqs = project_requirements.read_text()
    else:
        reqs = ""
    if "lamia-lang" not in reqs:
        reqs = "lamia-lang\n" + reqs
    if "lamia-cloud" not in reqs:
        reqs += "lamia-cloud\n"
    if "google-auth" not in reqs:
        reqs += "google-auth\n"
    requirements.write_text(reqs)

    return staging


def create_source_tarball(staging_dir: Path) -> bytes:
    """Create a gzipped tarball from the staging directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in staging_dir.iterdir():
            tar.add(item, arcname=item.name)
    return buf.getvalue()


def upload_source(project_id: str, tarball: bytes, name: str) -> str:
    """Upload source tarball to GCS and return the gs:// URI."""
    bucket_name = f"{project_id}_cloudbuild"
    blob_name = f"lamia-source/{name}.tar.gz"

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        bucket = client.create_bucket(bucket_name, location="us")

    blob = bucket.blob(blob_name)
    blob.upload_from_string(tarball, content_type="application/gzip")

    return f"gs://{bucket_name}/{blob_name}"


def ensure_files_bucket(project_id: str, location: str) -> str:
    """Ensure filesystem bucket exists (bucket name == project_id)."""
    bucket_name = project_id
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        bucket = client.create_bucket(bucket_name, location=location)
        logger.info(f"Created files bucket: {bucket_name}")
    return bucket_name


def sync_files_to_bucket(
    project_id: str,
    bucket_name: str,
    entries: list,
) -> dict:
    """Incrementally sync planned files to GCS and report overwrites."""
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    uploaded = 0
    skipped = 0
    overwrite_warnings: list[str] = []
    total = len(entries)

    for i, entry in enumerate(entries, 1):
        local_path = entry.resolved_path
        key = entry.bucket_key
        local_sha = file_sha256(local_path)

        blob = bucket.blob(key)
        if blob.exists():
            blob.reload()
            if (blob.metadata or {}).get("lamia-sha256") == local_sha:
                skipped += 1
                logger.info(f"  [{i}/{total}] Skipped (unchanged): {key}")
                continue
            overwrite_warnings.append(f"Remote file will be updated: gs://{bucket_name}/{key}")

        logger.info(f"  [{i}/{total}] Uploading: {key}")
        blob.metadata = {"lamia-sha256": local_sha}
        blob.upload_from_filename(local_path)
        uploaded += 1
        logger.info(f"  [{i}/{total}] Uploaded: {key}")

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "overwrite_warnings": overwrite_warnings,
    }


def sync_runtime_files(
    project_id: str,
    location: str,
    entries: list,
    files_namespace: str = "",
) -> dict:
    """Sync runtime file references for a remote invocation.

    When *files_namespace* is set, every bucket key is prefixed with it
    so that different scripts/triggers get isolated subdirectories.
    """
    if not entries:
        return {"uploaded": 0, "skipped": 0, "overwrite_warnings": []}
    files_bucket = ensure_files_bucket(project_id, location)
    if files_namespace:
        from lamia_cloud.contracts import FileSyncEntry
        entries = [
            FileSyncEntry(
                raw_path=e.raw_path,
                resolved_path=e.resolved_path,
                bucket_key=f"{files_namespace}/{e.bucket_key}",
            )
            for e in entries
        ]
    return sync_files_to_bucket(
        project_id=project_id,
        bucket_name=files_bucket,
        entries=entries,
    )


def submit_build(
    project_id: str,
    source_uri: str,
    image_name: str,
    repo_url: str | None = None,
) -> None:
    """Submit a Cloud Build to build the container image.

    When *repo_url* is provided (git mode), a shallow clone step runs first
    so that project files are available in the ``project/`` directory before
    the Docker build.  The uploaded tarball only needs the Dockerfile and
    requirements.txt in this case.
    """
    client = cloudbuild_v1.CloudBuildClient()

    steps: list[cloudbuild_v1.BuildStep] = []
    if repo_url:
        steps.append(
            cloudbuild_v1.BuildStep(
                name="gcr.io/cloud-builders/git",
                args=["clone", "--depth=1", "--branch", "main", repo_url, "project"],
            )
        )
    steps.append(
        cloudbuild_v1.BuildStep(
            name="gcr.io/cloud-builders/docker",
            args=["build", "-t", image_name, "."],
        )
    )

    build = cloudbuild_v1.Build(
        source=cloudbuild_v1.Source(
            storage_source=cloudbuild_v1.StorageSource(
                bucket=source_uri.split("/")[2],
                object_="/".join(source_uri.split("/")[3:]),
            )
        ),
        steps=steps,
        images=[image_name],
    )

    operation = client.create_build(project_id=project_id, build=build)
    logger.info("Cloud Build submitted, waiting for completion...")
    result = operation.result(timeout=600)

    if result.status != cloudbuild_v1.Build.Status.SUCCESS:
        raise RuntimeError(
            f"Cloud Build failed with status {result.status.name}: "
            f"{result.status_detail}"
        )
    logger.info(f"Cloud Build succeeded: {image_name}")


def deploy_job(
    project_id: str,
    location: str,
    job_name: str,
    image_name: str,
    script_name: str,
    memory: str = "512Mi",
    cpu: str = "1",
    files_bucket: Optional[str] = None,
    files_namespace: str = "",
    extra_labels: dict[str, str] | None = None,
    exec_service_account: Optional[str] = None,
    task_timeout_seconds: int = CLOUD_TASK_TIMEOUT_DEFAULT_SECONDS,
    secret_keys: Optional[list[str]] = None,
    secrets_namespace: str = "",
) -> None:
    """Deploy (or update) a Cloud Run Job.

    The job runs the lamia CLI directly — no HTTP handler.
    ``extra_labels`` are merged with the default ``lamia-managed`` label.

    When *exec_service_account* is provided (git/CI mode), the job runs
    with minimal permissions.  Otherwise falls back to the shared
    ``lamia-runner`` SA for backward compatibility with local deploys.

    ``secret_keys`` are referenced from Secret Manager, so their values are
    injected at runtime rather than stored in the job spec.
    """
    client = run_v2.JobsClient()
    parent = f"projects/{project_id}/locations/{location}"
    full_name = f"{parent}/jobs/{job_name}"

    if exec_service_account:
        service_account = exec_service_account
    else:
        service_account = _ensure_service_account(project_id)

    env_vars = [
        run_v2.EnvVar(name="LAMIA_SCRIPT", value=script_name),
        run_v2.EnvVar(name="GOOGLE_CLOUD_PROJECT", value=project_id),
    ]
    if files_namespace:
        env_vars.append(run_v2.EnvVar(name="LAMIA_FILES_NS", value=files_namespace))
    if secret_keys and secrets_namespace:
        env_vars.extend(secret_env_vars(secrets_namespace, secret_keys))

    container = run_v2.Container(
        image=image_name,
        env=env_vars,
        resources=run_v2.ResourceRequirements(
            limits={"memory": memory, "cpu": cpu},
        ),
    )

    volumes = []
    if files_bucket:
        container.volume_mounts = [
            run_v2.VolumeMount(
                name="lamia-files",
                mount_path=FILES_MOUNT_PATH,
            )
        ]
        volumes = [
            run_v2.Volume(
                name="lamia-files",
                gcs=run_v2.GCSVolumeSource(
                    bucket=files_bucket,
                    read_only=False,
                ),
            )
        ]

    merged_labels = {LABEL_MANAGED: "true"}
    if extra_labels:
        merged_labels.update(extra_labels)

    job = run_v2.Job(
        template=run_v2.ExecutionTemplate(
            template=run_v2.TaskTemplate(
                containers=[container],
                volumes=volumes,
                service_account=service_account,
                max_retries=0,
                timeout={"seconds": task_timeout_seconds},
            ),
        ),
        labels=merged_labels,
    )

    try:
        job.name = full_name
        operation = client.update_job(job=job)
        operation.result(timeout=300)
        logger.info(f"Updated Cloud Run Job: {job_name}")
    except NotFound:
        job.name = ""
        operation = client.create_job(parent=parent, job=job, job_id=job_name)
        operation.result(timeout=300)
        logger.info(f"Created Cloud Run Job: {job_name}")

    _allow_scheduler_job_invocation(project_id, location, job_name)


def _execution_from_operation(operation) -> Optional[run_v2.Execution]:
    """Extract Execution metadata from a completed run_job LRO.

    When a container crashes, operation.result() raises, but the LRO metadata
    still contains the Execution resource with its name and timing.
    """
    metadata = operation.metadata
    if metadata and getattr(metadata, "name", None):
        return metadata

    op = operation.operation
    if op.HasField("response"):
        execution = protobuf_helpers.from_any_pb(run_v2.Execution, op.response)
        if execution.name:
            return execution

    return None


def _result_from_execution(
    project_id: str,
    target: str,
    execution: run_v2.Execution,
) -> dict:
    """Build the run_job result dict from an Execution resource."""
    exit_code = 0 if execution.succeeded_count > 0 else 1
    elapsed = 0.0
    if execution.completion_time and execution.start_time:
        elapsed = (execution.completion_time - execution.start_time).total_seconds()

    started_time = None
    for condition in execution.conditions:
        if condition.type_ == "Started" and condition.last_transition_time:
            started_time = condition.last_transition_time
            break

    pending_seconds = None
    running_seconds = None
    if started_time and execution.create_time:
        pending_seconds = (started_time - execution.create_time).total_seconds()
    if started_time and execution.completion_time:
        running_seconds = (execution.completion_time - started_time).total_seconds()

    logs_url = _cloud_logging_url(
        project_id,
        target,
        execution.name,
        start_time=execution.start_time or None,
        end_time=execution.completion_time or None,
    )

    return {
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "pending_seconds": pending_seconds,
        "running_seconds": running_seconds,
        "logs_url": logs_url,
        "execution_name": execution.name,
    }


def run_job(
    project_id: str,
    location: str,
    target: str,
    verbose: bool = False,
) -> dict:
    """Execute the remote target and wait for completion.

    Returns dict with exit_code, logs_url, and elapsed_seconds.
    """
    client = run_v2.JobsClient()
    name = f"projects/{project_id}/locations/{location}/jobs/{target}"

    _touch_last_used(client, project_id, location, target)

    overrides = None
    if verbose:
        overrides = run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    env=[run_v2.EnvVar(name="LAMIA_EXTRA_ARGS", value="--verbose")]
                )
            ]
        )

    request = run_v2.RunJobRequest(name=name)
    if overrides:
        request.overrides = overrides

    operation = client.run_job(request=request)
    try:
        execution = operation.result()
    except GoogleAPICallError:
        execution = _execution_from_operation(operation)
        if execution is None:
            raise
        return _result_from_execution(project_id, target, execution)

    return _result_from_execution(project_id, target, execution)


def _execution_id(execution_name: str) -> str:
    return execution_name.rsplit("/", 1)[-1] if "/" in execution_name else execution_name


def _location_from_execution_name(execution_name: str) -> str:
    """Read the region out of a fully qualified execution resource name."""
    parts = execution_name.split("/")
    if "locations" in parts:
        index = parts.index("locations")
        if index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _log_time_range(start_time, end_time) -> str:
    """Logs Explorer time window covering the execution, or '' without a start."""
    if not start_time:
        return ""
    end = (end_time or start_time + timedelta(hours=1)) + LOG_WINDOW_TAIL
    return f"{_log_timestamp(start_time)}/{_log_timestamp(end)}"


def _log_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cloud_logging_url(
    project_id: str,
    target: str,
    execution_name: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> str:
    """Build a Logs Explorer URL filtered to this execution.

    Terms are newline separated and fully percent-encoded: the filter travels
    in a path segment, so an unescaped '/' would truncate it.
    """
    terms = [
        'resource.type="cloud_run_job"',
        f'resource.labels.job_name="{target}"',
    ]
    location = _location_from_execution_name(execution_name)
    if location:
        terms.append(f'resource.labels.location="{location}"')
    terms.append(
        f'labels."run.googleapis.com/execution_name"="{_execution_id(execution_name)}"'
    )

    url = (
        "https://console.cloud.google.com/logs/query;"
        f"query={urllib.parse.quote(chr(10).join(terms), safe='')}"
    )
    window = _log_time_range(start_time, end_time)
    if window:
        url += f";timeRange={urllib.parse.quote(window, safe='')}"
    return f"{url}?project={project_id}"


def fetch_execution_logs(
    project_id: str,
    target: str,
    execution_name: str,
) -> tuple[str, str]:
    """Fetch stdout and stderr from Cloud Logging for a completed execution.

    Cloud Logging has no signal for "ingestion is complete" -- entries can
    keep arriving over several seconds after an execution finishes. Polls
    until two consecutive reads agree (or the attempt budget runs out)
    instead of trusting the first sign of content, since a growing result
    means more is still coming.

    Returns (stdout, stderr) as strings.
    """
    client = cloud_logging.Client(project=project_id)

    stdout_log = f"projects/{project_id}/logs/{STDOUT_LOG_ID}"
    stderr_log = f"projects/{project_id}/logs/{STDERR_LOG_ID}"
    filter_str = (
        f'resource.type="cloud_run_job" '
        f'resource.labels.job_name="{target}" '
        f'labels."run.googleapis.com/execution_name"="{_execution_id(execution_name)}" '
        f'logName=("{stdout_log}" OR "{stderr_log}")'
    )

    def _query() -> tuple[list[str], list[str]]:
        stdout_lines = []
        stderr_lines = []
        for entry in client.list_entries(filter_=filter_str, order_by="timestamp asc"):
            text = _log_entry_text(entry)
            if not text:
                continue
            if (entry.log_name or "").endswith(STDERR_LOG_ID):
                stderr_lines.append(text)
            else:
                stdout_lines.append(text)
        return stdout_lines, stderr_lines

    previous = None
    for attempt in range(LOG_FETCH_MAX_ATTEMPTS):
        current = _query()
        is_last_attempt = attempt == LOG_FETCH_MAX_ATTEMPTS - 1
        has_content = current != ([], [])
        if is_last_attempt or (has_content and current == previous):
            stdout_lines, stderr_lines = current
            return "\n".join(stdout_lines), "\n".join(stderr_lines)
        previous = current
        time.sleep(LOG_FETCH_RETRY_DELAY_SECONDS)

    return "", ""


def _log_entry_text(entry) -> str:
    """Render a log entry payload as the line the script actually emitted."""
    payload = entry.payload
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        message = payload.get("message")
        if isinstance(message, str):
            return message
        return json.dumps(dict(payload), default=str, sort_keys=True)
    return str(payload)


def _is_execution_completed(execution: run_v2.Execution) -> bool:
    return execution.completion_time is not None


def fetch_latest_logs(
    project_id: str,
    location: str,
    target: str,
) -> tuple[str, str, str]:
    """Fetch logs from the most recent completed execution of a Cloud Run Job.

    Returns (stdout, stderr, logs_url). Raises if no executions exist.
    """
    client = run_v2.ExecutionsClient()
    parent = f"projects/{project_id}/locations/{location}/jobs/{target}"

    seen_any = False
    latest_completed = None
    for execution in client.list_executions(request={"parent": parent, "page_size": 1}):
        seen_any = True
        if _is_execution_completed(execution):
            latest_completed = execution
            break

    if not seen_any:
        raise ValueError(f"No executions found for job {target}")
    if latest_completed is None:
        raise ValueError(f"No completed executions found for job {target}")

    logs_url = _cloud_logging_url(
        project_id,
        target,
        latest_completed.name,
        start_time=latest_completed.start_time or None,
        end_time=latest_completed.completion_time or None,
    )
    stdout, stderr = fetch_execution_logs(project_id, target, latest_completed.name)
    return stdout, stderr, logs_url



def _ensure_service_account(project_id: str) -> str:
    """Create lamia-runner service account with required permissions.

    Grants:
    - roles/aiplatform.user — Vertex AI model access
    - roles/storage.objectViewer — read cached model maps / regions from GCS
    - roles/run.developer — allows Cloud Scheduler to run jobs
    """
    sa_email = f"lamia-runner@{project_id}.iam.gserviceaccount.com"
    iam_client = iam_admin_v1.IAMClient()

    try:
        iam_client.get_service_account(
            request={"name": f"projects/{project_id}/serviceAccounts/{sa_email}"}
        )
    except Exception as e:
        if "NOT_FOUND" in str(e):
            iam_client.create_service_account(
                request={
                    "name": f"projects/{project_id}",
                    "account_id": "lamia-runner",
                    "service_account": {"display_name": "Lamia Cloud Runner"},
                }
            )
            logger.info(f"Created service account: {sa_email}")
        else:
            raise

    rm_client = resourcemanager_v3.ProjectsClient()
    resource = f"projects/{project_id}"
    policy = rm_client.get_iam_policy(request={"resource": resource})

    project_number = _get_project_number(project_id)
    scheduler_sa = f"service-{project_number}@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
    member = f"serviceAccount:{sa_email}"

    required_bindings = {
        "roles/aiplatform.user": [member],
        "roles/storage.objectViewer": [member],
        "roles/iam.serviceAccountTokenCreator": [f"serviceAccount:{scheduler_sa}"],
        "roles/run.developer": [f"serviceAccount:{scheduler_sa}"],
    }

    changed = False
    for role, members in required_bindings.items():
        for m in members:
            already = any(
                b.role == role and m in b.members for b in policy.bindings
            )
            if not already:
                policy.bindings.append(
                    policy_pb2.Binding(role=role, members=[m])
                )
                logger.info(f"Granted {role} to {m}")
                changed = True

    if changed:
        rm_client.set_iam_policy(request={"resource": resource, "policy": policy})

    return sa_email


def _allow_scheduler_job_invocation(project_id: str, location: str, job_name: str) -> None:
    """Grant Cloud Scheduler permission to invoke the Cloud Run Job."""
    client = run_v2.JobsClient()
    resource = f"projects/{project_id}/locations/{location}/jobs/{job_name}"

    try:
        policy = client.get_iam_policy(request={"resource": resource})
    except Exception:
        policy = policy_pb2.Policy()

    invoker_role = "roles/run.invoker"
    scheduler_sa = f"service-{_get_project_number(project_id)}@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
    member = f"serviceAccount:{scheduler_sa}"

    for binding in policy.bindings:
        if binding.role == invoker_role and member in binding.members:
            return

    policy.bindings.append(
        policy_pb2.Binding(role=invoker_role, members=[member])
    )
    client.set_iam_policy(request={"resource": resource, "policy": policy})


def _get_project_number(project_id: str) -> str:
    """Get the project number from project ID."""
    client = resourcemanager_v3.ProjectsClient()
    project = client.get_project(name=f"projects/{project_id}")
    return project.name.split("/")[1]


def deploy(
    project_id: str,
    location: str,
    project_root: Path,
    script_name: str,
    name: str,
    capabilities=None,
    uses_files: bool = False,
    deploy_mode: str = "local",
    repo_url: str | None = None,
    task_timeout_seconds: int = CLOUD_TASK_TIMEOUT_DEFAULT_SECONDS,
    files_namespace: str = "",
    secret_keys: Optional[list[str]] = None,
    secrets_namespace: str = "",
) -> str:
    """Full deploy pipeline. Returns the deployment name.

    If capabilities (ScriptCapabilities dataclass) is provided, resource tier
    is computed from it. If the job already exists with a higher tier (e.g.
    elevated by a previous OOM recovery), the higher tier is preserved.
    """
    job_name = deployment_name(name)
    image = _image_name(project_id, name)

    if capabilities is not None:
        flags = _extract_capability_flags(capabilities)
        memory, cpu = compute_resource_tier(
            uses_llm=flags["uses_llm"],
            uses_browser=flags["uses_browser"],
            uses_files=flags["uses_files"],
            uses_file_context=flags["uses_file_context"],
        )
    else:
        memory, cpu = ("1Gi", "1")

    existing = _get_existing_resources(project_id, location, job_name)
    if existing:
        existing_mib = _memory_to_mib(existing[0])
        computed_mib = _memory_to_mib(memory)
        if existing_mib > computed_mib:
            memory, cpu = existing

    logger.info(f"Packaging {script_name} for deployment...")
    staging = package_deployment(
        project_root,
        script_name,
        name,
        uses_files=uses_files,
        deploy_mode=deploy_mode,
    )

    try:
        logger.info("Creating source tarball...")
        tarball = create_source_tarball(staging)

        logger.info("Uploading source to GCS...")
        source_uri = upload_source(project_id, tarball, name)

        logger.info("Submitting Cloud Build...")
        submit_build(
            project_id, source_uri, image,
            repo_url=repo_url if deploy_mode == "git" else None,
        )

        files_bucket = None
        ns = files_namespace or name
        if uses_files:
            files_bucket = ensure_files_bucket(project_id, location)

        resource_labels = build_resource_labels(
            script_name, project_root,
            deploy_mode=deploy_mode, repo_url=repo_url,
        )

        run_sa = None
        if repo_url:
            run_sa = exec_sa_email(project_id, repo_url)

        logger.info(f"Deploying Cloud Run Job ({memory}, {cpu} vCPU)...")
        deploy_job(
            project_id,
            location,
            job_name,
            image,
            script_name,
            memory=memory,
            cpu=cpu,
            files_bucket=files_bucket,
            files_namespace=ns,
            extra_labels=resource_labels,
            exec_service_account=run_sa,
            task_timeout_seconds=task_timeout_seconds,
            secret_keys=secret_keys,
            secrets_namespace=secrets_namespace,
        )

        return job_name
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def teardown(project_id: str, location: str, name: str) -> None:
    """Remove the deployed Cloud Run resource."""
    client = run_v2.JobsClient()
    job_name = deployment_name(name)
    full_name = f"projects/{project_id}/locations/{location}/jobs/{job_name}"

    try:
        client.delete_job(name=full_name)
        logger.info(f"Deleted Cloud Run Job: {job_name}")
    except NotFound:
        pass
    except Exception as e:
        if "NOT_FOUND" not in str(e):
            raise


def list_managed_jobs(project_id: str, location: str) -> list[dict]:
    """List all Cloud Run Jobs with the lamia-managed label.

    Returns a list of dicts with keys: name, labels, create_time, update_time.
    """
    client = run_v2.JobsClient()
    parent = f"projects/{project_id}/locations/{location}"
    results = []
    for job in client.list_jobs(parent=parent):
        if (job.labels or {}).get(LABEL_MANAGED) == "true":
            results.append({
                "name": job.name.rsplit("/", 1)[-1],
                "full_name": job.name,
                "labels": dict(job.labels or {}),
                "create_time": job.create_time,
                "update_time": job.update_time,
            })
    return results


def _referenced_job_names(project_id: str, location: str) -> set[str] | None:
    """Collect Cloud Run Job names referenced by active Cloud Scheduler jobs.

    Returns None when the Cloud Scheduler API cannot be queried — callers
    must treat None as "unknown" and skip any deletions.
    """
    try:
        from google.cloud import scheduler_v1
        client = scheduler_v1.CloudSchedulerClient()
        parent = f"projects/{project_id}/locations/{location}"
        referenced: set[str] = set()
        for sched_job in client.list_jobs(parent=parent):
            target = getattr(sched_job, "http_target", None)
            if target and target.uri:
                for part in target.uri.split("/"):
                    if part.startswith("lamia-"):
                        referenced.add(part)
                        break
        return referenced
    except Exception:
        logger.warning("Cloud Scheduler API unreachable — skipping reference check")
        return None


def cleanup_stale_resources(
    project_id: str,
    location: str,
    max_age_days: int = STALE_RESOURCE_DAYS,
) -> list[str]:
    """Delete Cloud Run Jobs inactive for more than max_age_days.

    Skips jobs referenced by Cloud Scheduler or trigger workflows.
    Returns list of cleaned-up job names.
    """
    cutoff = date.today() - timedelta(days=max_age_days)
    managed = list_managed_jobs(project_id, location)
    referenced = _referenced_job_names(project_id, location)

    if referenced is None:
        return []

    cleaned = []
    client = run_v2.JobsClient()
    for job in managed:
        if job["name"] in referenced:
            continue
        if job["labels"].get(LABEL_RESOURCE_TYPE, "") != "one-shot":
            continue
        last_used_raw = job["labels"].get(LABEL_LAST_USED, "")
        if not last_used_raw:
            continue
        try:
            last_used = datetime.strptime(last_used_raw, "%Y%m%d").date()
        except ValueError:
            logger.warning(
                f"Skipping cleanup for {job['name']}: invalid {LABEL_LAST_USED}={last_used_raw!r}"
            )
            continue
        if last_used <= cutoff:
            try:
                client.delete_job(name=job["full_name"])
                age = (date.today() - last_used).days
                logger.info(f"Cleaned up stale resource: {job['name']} (unused for {age} days)")
                cleaned.append(job["name"])
            except Exception as exc:
                logger.warning(f"Failed to clean up {job['name']}: {exc}")
    return cleaned


# ---------------------------------------------------------------------------
# CloudDeployer implementation (wraps the module-level functions above)
# ---------------------------------------------------------------------------

from lamia_cloud.gcp.connect import exec_sa_email
from lamia_cloud.interfaces import CloudDeployer


class GCPDeployer(CloudDeployer):
    """GCP implementation of CloudDeployer using Cloud Build + Cloud Run Jobs."""

    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        task_timeout_seconds: int = CLOUD_TASK_TIMEOUT_DEFAULT_SECONDS,
    ):
        self.project_id = project_id
        self.location = location
        self.task_timeout_seconds = task_timeout_seconds

    @classmethod
    def from_config(cls, cloud_cfg: dict) -> "GCPDeployer":
        project_id = cloud_cfg.get("project_id")
        if not project_id:
            raise ValueError("cloud.project_id is required in config.yaml.")
        location = cloud_cfg.get("location", "us-central1")
        resources_cfg = cloud_cfg.get("resources", {})
        timeout = resources_cfg.get(
            "task_timeout_seconds",
            CLOUD_TASK_TIMEOUT_DEFAULT_SECONDS,
        )
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            raise ValueError("cloud.resources.task_timeout_seconds must be an integer.")
        if timeout < CLOUD_TASK_TIMEOUT_MIN_SECONDS or timeout > CLOUD_TASK_TIMEOUT_MAX_SECONDS:
            raise ValueError(
                f"cloud.resources.task_timeout_seconds must be between "
                f"{CLOUD_TASK_TIMEOUT_MIN_SECONDS} and {CLOUD_TASK_TIMEOUT_MAX_SECONDS}."
            )
        return cls(
            project_id=project_id,
            location=location,
            task_timeout_seconds=timeout,
        )

    def ensure_apis_enabled(self) -> None:
        ensure_apis_enabled(self.project_id)

    def deployment_name(self, name: str) -> str:
        return deployment_name(name)

    def collect_project_files(self, project_root: Path) -> list[Path]:
        return collect_project_files(project_root)

    def get_deployed_source_hash(self, target: str) -> Optional[str]:
        return get_deployed_source_hash(self.project_id, self.location, target)

    def set_deployed_source_hash(self, target: str, hash_val: str) -> None:
        set_deployed_source_hash(self.project_id, self.location, target, hash_val)

    def get_verified_model_access(self) -> set:
        return get_verified_vertex_models(self.project_id)

    def remember_verified_model_access(self, models: set) -> None:
        remember_verified_vertex_models(self.project_id, models)

    def sync_runtime_files(self, entries: list, files_namespace: str = "") -> dict:
        return sync_runtime_files(
            project_id=self.project_id,
            location=self.location,
            entries=entries,
            files_namespace=files_namespace,
        )

    def sync_secrets(self, secrets: dict, namespace: str) -> list:
        return sync_secrets(
            project_id=self.project_id,
            secrets=secrets,
            namespace=namespace,
            service_account=_ensure_service_account(self.project_id),
        )

    def cleanup_secrets(self, namespace: str) -> list:
        return cleanup_secrets(
            project_id=self.project_id,
            location=self.location,
            namespace=namespace,
        )

    def deploy(
        self,
        project_root: Path,
        script_name: str,
        name: str,
        capabilities=None,
        uses_files: bool = False,
        deploy_mode: str = "local",
        repo_url: str | None = None,
        files_namespace: str = "",
        secret_keys: Optional[list[str]] = None,
        secrets_namespace: str = "",
    ) -> str:
        return deploy(
            project_id=self.project_id,
            location=self.location,
            project_root=project_root,
            script_name=script_name,
            name=name,
            capabilities=capabilities,
            uses_files=uses_files,
            deploy_mode=deploy_mode,
            repo_url=repo_url,
            task_timeout_seconds=self.task_timeout_seconds,
            files_namespace=files_namespace,
            secret_keys=secret_keys,
            secrets_namespace=secrets_namespace,
        )

    def run_job(self, target: str, verbose: bool = False) -> dict:
        return run_job(
            project_id=self.project_id,
            location=self.location,
            target=target,
            verbose=verbose,
        )

    def fetch_execution_logs(self, target: str, execution_name: str = "") -> tuple[str, str]:
        return fetch_execution_logs(
            project_id=self.project_id,
            target=target,
            execution_name=execution_name,
        )

    def teardown(self, name: str) -> None:
        teardown(self.project_id, self.location, name)

    def list_managed_jobs(self) -> list[dict]:
        return list_managed_jobs(self.project_id, self.location)

    def cleanup_stale_resources(self, max_age_days: int = STALE_RESOURCE_DAYS) -> list[str]:
        return cleanup_stale_resources(self.project_id, self.location, max_age_days=max_age_days)

