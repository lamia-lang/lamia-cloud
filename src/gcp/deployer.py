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

import io
import logging
import shutil
import tarfile
import tempfile
import time
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from google.api_core.exceptions import NotFound
from google.cloud import iam_admin_v1, logging as cloud_logging, resourcemanager_v3, run_v2, storage
from google.cloud.devtools import cloudbuild_v1
from google.iam.v1 import policy_pb2

from lamia_cloud.contracts import SCRIPT_CAPABILITY_FIELDS, SOURCE_HASH_LABEL
from lamia_cloud.file_sync import file_sha256

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
FILES_MOUNT_PATH = "/mnt/lamia-files"


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

    SECURITY: .env files are explicitly excluded — secrets must never be baked
    into Docker image layers.
    """
    files = []
    for pattern in ("*.lm", "*.py", "*.yaml", "*.yml", "*.json", "*.txt", "*.csv"):
        files.extend(project_root.glob(pattern))
    files = [f for f in files if f.name != ".env"]
    for subdir in project_root.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for pattern in ("**/*.lm", "**/*.py", "**/*.yaml", "**/*.json"):
                files.extend(subdir.glob(pattern))
    return files


def package_deployment(
    project_root: Path,
    script_name: str,
    name: str,
    uses_files: bool = False,
) -> Path:
    """Create a staging directory with everything needed for Cloud Build."""
    staging = Path(tempfile.mkdtemp(prefix="lamia-deploy-"))

    project_dest = staging / "project"
    project_dest.mkdir()
    for f in collect_project_files(project_root):
        rel = f.relative_to(project_root)
        dest = project_dest / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

    dockerfile_dest = staging / "Dockerfile"
    dockerfile_content = (TEMPLATES_DIR / "Dockerfile").read_text()
    if uses_files:
        cmd = f'CMD ["sh", "-c", "cd {FILES_MOUNT_PATH} && lamia /app/project/${{LAMIA_SCRIPT}} ${{LAMIA_EXTRA_ARGS:-}}"]'
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
) -> dict:
    """Sync runtime file references for a remote invocation."""
    if not entries:
        return {"uploaded": 0, "skipped": 0, "overwrite_warnings": []}
    files_bucket = ensure_files_bucket(project_id, location)
    return sync_files_to_bucket(
        project_id=project_id,
        bucket_name=files_bucket,
        entries=entries,
    )


def submit_build(
    project_id: str,
    source_uri: str,
    image_name: str,
) -> None:
    """Submit a Cloud Build to build the container image."""
    client = cloudbuild_v1.CloudBuildClient()

    build = cloudbuild_v1.Build(
        source=cloudbuild_v1.Source(
            storage_source=cloudbuild_v1.StorageSource(
                bucket=source_uri.split("/")[2],
                object_="/".join(source_uri.split("/")[3:]),
            )
        ),
        steps=[
            cloudbuild_v1.BuildStep(
                name="gcr.io/cloud-builders/docker",
                args=["build", "-t", image_name, "."],
            )
        ],
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
) -> None:
    """Deploy (or update) a Cloud Run Job.

    The job runs the lamia CLI directly — no HTTP handler.
    """
    client = run_v2.JobsClient()
    parent = f"projects/{project_id}/locations/{location}"
    full_name = f"{parent}/jobs/{job_name}"

    service_account = _ensure_service_account(project_id)

    container = run_v2.Container(
        image=image_name,
        env=[
            run_v2.EnvVar(name="LAMIA_SCRIPT", value=script_name),
            run_v2.EnvVar(name="GOOGLE_CLOUD_PROJECT", value=project_id),
        ],
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

    job = run_v2.Job(
        template=run_v2.ExecutionTemplate(
            template=run_v2.TaskTemplate(
                containers=[container],
                volumes=volumes,
                service_account=service_account,
                max_retries=0,
                timeout={"seconds": 3600},
            ),
        ),
        labels={"lamia-managed": "true"},
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
    execution = operation.result()

    exit_code = 0 if execution.succeeded_count > 0 else 1
    elapsed = 0.0
    if execution.completion_time and execution.start_time:
        elapsed = (execution.completion_time - execution.start_time).total_seconds()

    logs_url = _cloud_logging_url(project_id, target, execution.name)

    return {
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "logs_url": logs_url,
        "execution_name": execution.name,
    }


def _cloud_logging_url(project_id: str, target: str, execution_name: str) -> str:
    """Build a Cloud Logging URL filtered to this execution."""
    execution_id = execution_name.rsplit("/", 1)[-1] if "/" in execution_name else execution_name
    query = (
        f'resource.type="cloud_run_job" '
        f'resource.labels.job_name="{target}" '
        f'labels."run.googleapis.com/execution_name"="{execution_id}"'
    )
    return (
        f"https://console.cloud.google.com/logs/query;"
        f"query={urllib.parse.quote(query)}?project={project_id}"
    )


def fetch_execution_logs(
    project_id: str,
    target: str,
    execution_name: str,
) -> tuple[str, str]:
    """Fetch stdout and stderr from Cloud Logging for a completed execution.

    Returns (stdout, stderr) as strings.
    """
    client = cloud_logging.Client(project=project_id)

    execution_id = execution_name.rsplit("/", 1)[-1] if "/" in execution_name else execution_name
    filter_str = (
        f'resource.type="cloud_run_job" '
        f'resource.labels.job_name="{target}" '
        f'labels."run.googleapis.com/execution_name"="{execution_id}"'
    )

    stdout_lines = []
    stderr_lines = []
    for entry in client.list_entries(filter_=filter_str, order_by="timestamp asc"):
        text = entry.payload if isinstance(entry.payload, str) else str(entry.payload)
        if entry.severity and entry.severity.upper() in ("ERROR", "CRITICAL", "WARNING"):
            stderr_lines.append(text)
        else:
            stdout_lines.append(text)

    return "\n".join(stdout_lines), "\n".join(stderr_lines)


def _ensure_service_account(project_id: str) -> str:
    """Create lamia-runner service account with required permissions.

    Grants:
    - roles/aiplatform.user — Vertex AI model access
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
    )

    try:
        logger.info("Creating source tarball...")
        tarball = create_source_tarball(staging)

        logger.info("Uploading source to GCS...")
        source_uri = upload_source(project_id, tarball, name)

        logger.info("Submitting Cloud Build...")
        submit_build(project_id, source_uri, image)

        files_bucket = None
        if uses_files:
            files_bucket = ensure_files_bucket(project_id, location)

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
