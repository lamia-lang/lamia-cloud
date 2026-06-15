"""Deploys a lamia project to Cloud Run via Cloud Build.

Flow:
1. Package the .lm script + project files into a staging directory
2. Add Dockerfile + main.py handler + requirements.txt
3. Upload to GCS as source tarball
4. Submit Cloud Build to build the container
5. Deploy the container to Cloud Run (with Vertex AI IAM for LLM access)
6. Return the Cloud Run service URL

LLM authentication uses Vertex AI — the Cloud Run service account gets
roles/aiplatform.user, so no API keys are needed at runtime.
"""

import io
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# templates/ lives at the lamia_cloud package root, one level above this gcp/ subpackage
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _service_name(schedule_id: str) -> str:
    return f"lamia-{schedule_id}"


def _image_name(project_id: str, schedule_id: str) -> str:
    return f"gcr.io/{project_id}/lamia-{schedule_id}"


def _collect_project_files(project_root: Path) -> list[Path]:
    """Collect .lm files, config.yaml, and supporting Python files from the project.

    SECURITY: .env files are explicitly excluded — secrets must never be baked
    into Docker image layers. API keys are injected via Cloud Run env vars at
    deploy time (see deploy_cloud_run).
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
    schedule_id: str,
) -> Path:
    """Create a staging directory with everything needed for the Cloud Build."""
    staging = Path(tempfile.mkdtemp(prefix="lamia-deploy-"))

    project_dest = staging / "project"
    project_dest.mkdir()
    for f in _collect_project_files(project_root):
        rel = f.relative_to(project_root)
        dest = project_dest / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

    shutil.copy2(TEMPLATES_DIR / "Dockerfile", staging / "Dockerfile")
    shutil.copy2(TEMPLATES_DIR / "main.py", staging / "main.py")

    requirements = staging / "requirements.txt"
    project_requirements = project_root / "requirements.txt"
    if project_requirements.exists():
        reqs = project_requirements.read_text()
    else:
        reqs = ""
    if "lamia-lang" not in reqs:
        reqs = "lamia-lang\n" + reqs
    requirements.write_text(reqs)

    return staging


def create_source_tarball(staging_dir: Path) -> bytes:
    """Create a gzipped tarball from the staging directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in staging_dir.iterdir():
            tar.add(item, arcname=item.name)
    return buf.getvalue()


def upload_source(project_id: str, tarball: bytes, schedule_id: str) -> str:
    """Upload source tarball to GCS and return the gs:// URI."""
    from google.cloud import storage

    bucket_name = f"{project_id}_cloudbuild"
    blob_name = f"lamia-source/{schedule_id}.tar.gz"

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        bucket = client.create_bucket(bucket_name, location="us")

    blob = bucket.blob(blob_name)
    blob.upload_from_string(tarball, content_type="application/gzip")

    return f"gs://{bucket_name}/{blob_name}"


def submit_build(
    project_id: str,
    source_uri: str,
    image_name: str,
) -> None:
    """Submit a Cloud Build to build the container image."""
    from google.cloud.devtools import cloudbuild_v1

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
    logger.info(f"Cloud Build submitted, waiting for completion...")
    result = operation.result(timeout=600)

    if result.status != cloudbuild_v1.Build.Status.SUCCESS:
        raise RuntimeError(
            f"Cloud Build failed with status {result.status.name}: "
            f"{result.status_detail}"
        )
    logger.info(f"Cloud Build succeeded: {image_name}")


def deploy_cloud_run(
    project_id: str,
    location: str,
    service_name: str,
    image_name: str,
    script_name: str,
) -> str:
    """Deploy (or update) a Cloud Run service. Returns the service URL.

    LLM auth is handled by Vertex AI — the service account gets
    roles/aiplatform.user so no API keys are injected.
    """
    from google.cloud import run_v2

    client = run_v2.ServicesClient()
    parent = f"projects/{project_id}/locations/{location}"
    full_name = f"{parent}/services/{service_name}"

    service_account = _ensure_service_account(project_id)

    env_vars = [
        run_v2.EnvVar(name="LAMIA_SCRIPT", value=script_name),
        run_v2.EnvVar(name="GOOGLE_CLOUD_PROJECT", value=project_id),
    ]

    container = run_v2.Container(
        image=image_name,
        env=env_vars,
        resources=run_v2.ResourceRequirements(
            limits={"memory": "512Mi", "cpu": "1"},
        ),
    )

    service = run_v2.Service(
        template=run_v2.RevisionTemplate(
            containers=[container],
            service_account=service_account,
            max_instance_request_concurrency=1,
            timeout={"seconds": 540},
        ),
        ingress=run_v2.IngressTraffic.INGRESS_TRAFFIC_INTERNAL_ONLY,
    )

    try:
        operation = client.update_service(
            service=service,
            request={"service": service, "name": full_name},
        )
        result = operation.result(timeout=300)
    except Exception as e:
        if "NOT_FOUND" in str(e):
            service.name = full_name
            operation = client.create_service(
                parent=parent,
                service=service,
                service_id=service_name,
            )
            result = operation.result(timeout=300)
        else:
            raise

    url = result.uri
    logger.info(f"Cloud Run deployed: {url}")

    _allow_scheduler_invocation(project_id, location, service_name)

    return url


def _ensure_service_account(project_id: str) -> str:
    """Create lamia-runner service account with Vertex AI access if it doesn't exist."""
    from google.cloud import iam_admin_v1
    from google.cloud import resourcemanager_v3
    from google.iam.v1 import iam_policy_pb2, policy_pb2

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

    vertex_role = "roles/aiplatform.user"
    member = f"serviceAccount:{sa_email}"

    already_granted = False
    for binding in policy.bindings:
        if binding.role == vertex_role and member in binding.members:
            already_granted = True
            break

    if not already_granted:
        policy.bindings.append(
            policy_pb2.Binding(role=vertex_role, members=[member])
        )
        rm_client.set_iam_policy(request={"resource": resource, "policy": policy})
        logger.info(f"Granted {vertex_role} to {sa_email}")

    return sa_email


def _allow_scheduler_invocation(project_id: str, location: str, service_name: str) -> None:
    """Grant Cloud Scheduler permission to invoke the Cloud Run service."""
    from google.cloud import run_v2
    from google.iam.v1 import iam_policy_pb2, policy_pb2

    client = run_v2.ServicesClient()
    resource = f"projects/{project_id}/locations/{location}/services/{service_name}"

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
    from google.cloud import resourcemanager_v3

    client = resourcemanager_v3.ProjectsClient()
    project = client.get_project(name=f"projects/{project_id}")
    return project.name.split("/")[1]


def deploy(
    project_id: str,
    location: str,
    project_root: Path,
    script_name: str,
    schedule_id: str,
) -> str:
    """Full deploy pipeline. Returns the Cloud Run service URL."""
    service_name = _service_name(schedule_id)
    image = _image_name(project_id, schedule_id)

    logger.info(f"Packaging {script_name} for deployment...")
    staging = package_deployment(project_root, script_name, schedule_id)

    try:
        logger.info("Creating source tarball...")
        tarball = create_source_tarball(staging)

        logger.info("Uploading source to GCS...")
        source_uri = upload_source(project_id, tarball, schedule_id)

        logger.info("Submitting Cloud Build...")
        submit_build(project_id, source_uri, image)

        logger.info("Deploying to Cloud Run with Vertex AI access...")
        url = deploy_cloud_run(project_id, location, service_name, image, script_name)

        return url
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def teardown(project_id: str, location: str, schedule_id: str) -> None:
    """Remove the Cloud Run service for a schedule."""
    from google.cloud import run_v2

    client = run_v2.ServicesClient()
    service_name = _service_name(schedule_id)
    full_name = f"projects/{project_id}/locations/{location}/services/{service_name}"

    try:
        client.delete_service(name=full_name)
        logger.info(f"Deleted Cloud Run service: {service_name}")
    except Exception as e:
        if "NOT_FOUND" not in str(e):
            raise
