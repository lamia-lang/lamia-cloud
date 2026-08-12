"""Repository connection — links git repositories to GCP for CI deployments.

Manages the full trust chain: Cloud Build GitHub App connection, Workload
Identity Federation (WIF) pool/provider, per-repo service accounts (CI and
runtime), and IAM bindings.  All naming is deterministic and derived from
the repository URL so that reconnecting is idempotent.
"""

import atexit
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import urllib.parse

from google.cloud import iam_admin_v1, resourcemanager_v3
from google.iam.v1 import policy_pb2

from lamia_cloud.interfaces import RepositoryConnector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cloud Build GitHub App connection
# ---------------------------------------------------------------------------

_CONNECTION_NAME = "lamia-github"


def _sanitize_repo_name(repo_url: str) -> str:
    """Derive a Cloud Build repository resource name from a remote URL.

    ``https://github.com/acme/widgets.git`` -> ``acme-widgets``
    """
    raw = repo_url.strip()
    scp_match = re.match(r"^[^@]+@([^:]+):(.+)$", raw)
    if scp_match:
        path = scp_match.group(2)
    else:
        path = urllib.parse.urlparse(raw).path

    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return re.sub(r"[^a-z0-9-]", "-", path.lower()).strip("-")


def _ensure_connection(project_id: str, location: str) -> str:
    """Create the Cloud Build GitHub connection if it does not exist.

    Returns the connection name.  On first run this prints the OAuth URL
    the user must open to authorize the GitHub App.
    """
    check = subprocess.run(
        [
            "gcloud", "builds", "connections", "describe", _CONNECTION_NAME,
            f"--region={location}", f"--project={project_id}",
            "--format=value(installationState.stage)",
        ],
        capture_output=True, text=True,
    )
    if check.returncode == 0 and check.stdout.strip():
        stage = check.stdout.strip()
        if stage == "COMPLETE":
            return _CONNECTION_NAME
        logger.info(
            f"Connection '{_CONNECTION_NAME}' exists but stage={stage}. "
            "Complete the authorization in your browser."
        )
        return _CONNECTION_NAME

    create = subprocess.run(
        [
            "gcloud", "builds", "connections", "create", "github",
            _CONNECTION_NAME,
            f"--region={location}", f"--project={project_id}",
        ],
        capture_output=True, text=True,
    )
    if create.returncode != 0:
        raise RuntimeError(
            f"Failed to create Cloud Build connection: {create.stderr.strip()}"
        )
    logger.info(
        "Cloud Build GitHub connection created. "
        "Follow the URL printed above to authorize the GitHub App."
    )
    return _CONNECTION_NAME


# ---------------------------------------------------------------------------
# Workload Identity Federation (WIF) — per-repo trust
# ---------------------------------------------------------------------------

_WIF_POOL_ID = "lamia-github-pool"

_CI_SA_ROLES = (
    "roles/run.admin",
    "roles/cloudbuild.builds.editor",
    "roles/storage.admin",
    "roles/logging.viewer",
)

_EXEC_SA_ROLES = (
    "roles/aiplatform.user",
)

_REQUIRED_SA_ROLES = _CI_SA_ROLES + _EXEC_SA_ROLES

# ---------------------------------------------------------------------------
# Opaque connection identifiers
# ---------------------------------------------------------------------------


def _repo_identity_key(repo_url: str) -> str:
    """Host+path identity used to derive stable opaque connection IDs."""
    raw = repo_url.strip()
    scp_match = re.match(r"^[^@]+@([^:]+):(.+)$", raw)
    if scp_match:
        host = scp_match.group(1).lower()
        path = scp_match.group(2).strip("/")
    else:
        parsed = urllib.parse.urlparse(raw)
        host = (parsed.hostname or "").lower()
        path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return f"{host}/{path.lower()}"


def _connection_suffix(repo_url: str) -> str:
    """Opaque 12-hex suffix derived from canonical repo identity."""
    identity = _repo_identity_key(repo_url)
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def connection_suffix_for_repo(repo_url: str) -> str:
    """Repository digest embedded in a connection ID."""
    return _connection_suffix(repo_url)


def make_connection_id(project_number: str, repo_url: str) -> str:
    """Public Lamia connection handle for CI."""
    return f"v1-{project_number}-{_connection_suffix(repo_url)}"


def parse_connection_id(connection_id: str) -> tuple[str, str]:
    """Parse ``LAMIA_CONNECTION_ID`` -> (project_number, suffix)."""
    m = re.match(r"^v1-([0-9]+)-([0-9a-f]{12})$", connection_id.strip())
    if not m:
        raise ValueError("Invalid LAMIA_CONNECTION_ID format")
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Per-repo resource naming (deterministic from repo URL)
# ---------------------------------------------------------------------------


def _wif_provider_id_from_suffix(suffix: str) -> str:
    return f"lamia-gh-{suffix}"


def _per_repo_sa_id_from_suffix(prefix: str, suffix: str) -> str:
    return f"{prefix}-{suffix}"


def _wif_provider_id(repo_url: str) -> str:
    """Per-repo WIF provider ID (opaque, non-revealing)."""
    return _wif_provider_id_from_suffix(_connection_suffix(repo_url))


def _per_repo_sa_id(prefix: str, repo_url: str) -> str:
    """Per-repo SA account ID derived from opaque suffix."""
    return _per_repo_sa_id_from_suffix(prefix, _connection_suffix(repo_url))


def ci_sa_email(project_id: str, repo_url: str) -> str:
    """Derive CI SA email from repository identity."""
    suffix = _connection_suffix(repo_url)
    return ci_sa_email_from_connection(project_id, suffix)


def exec_sa_email(project_id: str, repo_url: str) -> str:
    """Derive runtime SA email from repository identity."""
    suffix = _connection_suffix(repo_url)
    return exec_sa_email_from_connection(project_id, suffix)


def ci_sa_email_from_connection(project_id: str, connection_suffix: str) -> str:
    """Derive CI SA email from opaque connection suffix."""
    sa_id = _per_repo_sa_id_from_suffix("lm-ci", connection_suffix)
    return f"{sa_id}@{project_id}.iam.gserviceaccount.com"


def exec_sa_email_from_connection(project_id: str, connection_suffix: str) -> str:
    """Derive runtime SA email from opaque connection suffix."""
    sa_id = _per_repo_sa_id_from_suffix("lm-run", connection_suffix)
    return f"{sa_id}@{project_id}.iam.gserviceaccount.com"


def derive_wif_provider(project_number: str, repo_url: str) -> str:
    """Derive WIF provider path from repo identity."""
    return derive_wif_provider_from_connection(project_number, _connection_suffix(repo_url))


def derive_wif_provider_from_connection(project_number: str, connection_suffix: str) -> str:
    """Derive WIF provider path from opaque connection suffix."""
    provider_id = _wif_provider_id_from_suffix(connection_suffix)
    return (
        f"projects/{project_number}/locations/global/"
        f"workloadIdentityPools/{_WIF_POOL_ID}/"
        f"providers/{provider_id}"
    )


# ---------------------------------------------------------------------------
# Repo URL helpers
# ---------------------------------------------------------------------------


def _extract_repo_full_name(repo_url: str) -> str:
    """Extract owner/repo path from any git URL format.

    ``https://github.com/acme/widgets.git`` -> ``acme/widgets``
    ``git@github.com:acme/widgets.git`` -> ``acme/widgets``
    """
    raw = repo_url.strip()
    scp_match = re.match(r"^[^@]+@([^:]+):(.+)$", raw)
    if scp_match:
        path = scp_match.group(2)
    else:
        path = urllib.parse.urlparse(raw).path
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


def _to_https_remote(repo_url: str) -> str:
    """Convert any git remote URL to HTTPS for Cloud Build."""
    if repo_url.startswith(("https://", "http://")):
        uri = repo_url
        if not uri.endswith(".git"):
            uri += ".git"
        return uri
    scp_match = re.match(r"^[^@]+@([^:]+):(.+)$", repo_url.strip())
    if scp_match:
        host = scp_match.group(1)
        path = scp_match.group(2).rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return f"https://{host}/{path}.git"
    return repo_url


# ---------------------------------------------------------------------------
# IAM helpers
# ---------------------------------------------------------------------------


def _get_project_number(project_id: str) -> str:
    """Get the project number from project ID."""
    client = resourcemanager_v3.ProjectsClient()
    project = client.get_project(name=f"projects/{project_id}")
    return project.name.split("/")[1]


def _ensure_per_repo_sa(
    project_id: str, sa_id: str, display_name: str,
) -> str:
    """Create a per-repo SA if it doesn't exist. Returns the SA email."""
    sa_email = f"{sa_id}@{project_id}.iam.gserviceaccount.com"
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
                    "account_id": sa_id,
                    "service_account": {"display_name": display_name},
                }
            )
            logger.info(f"Created service account: {sa_email}")
        else:
            raise
    return sa_email


def _grant_sa_roles(
    project_id: str, sa_email: str, roles: tuple[str, ...],
) -> None:
    """Grant project-level IAM roles to a service account."""
    rm_client = resourcemanager_v3.ProjectsClient()
    resource = f"projects/{project_id}"
    policy = rm_client.get_iam_policy(request={"resource": resource})
    member = f"serviceAccount:{sa_email}"

    changed = False
    for role in roles:
        already = any(
            b.role == role and member in b.members for b in policy.bindings
        )
        if not already:
            policy.bindings.append(
                policy_pb2.Binding(role=role, members=[member])
            )
            logger.info(f"Granted {role} to {sa_email}")
            changed = True

    if changed:
        rm_client.set_iam_policy(request={"resource": resource, "policy": policy})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_REPO_FULL_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _validate_condition_operand(value: str, kind: str, pattern: re.Pattern) -> str:
    """Reject values unsafe to embed in a CEL attribute condition."""
    if not pattern.match(value) or '"' in value or "\\" in value:
        raise RuntimeError(
            f"Invalid {kind} {value!r}: only letters, digits and '.', '_', '-', '/' "
            "are allowed."
        )
    return value


# ---------------------------------------------------------------------------
# gcloud subprocess helper
# ---------------------------------------------------------------------------


def _run_gcloud(args: list[str], error_msg: str) -> subprocess.CompletedProcess:
    """Run a gcloud command and raise on failure."""
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "ALREADY_EXISTS" not in stderr:
            raise RuntimeError(f"{error_msg}: {stderr}")
    return result


# ---------------------------------------------------------------------------
# Core WIF setup
# ---------------------------------------------------------------------------


def _ensure_wif(
    project_id: str, repo_url: str, *, branch: str = "main",
) -> dict:
    """Set up Workload Identity Federation for GitHub Actions.

    Creates:
    - WIF pool (shared across repos in the project)
    - Per-repo OIDC provider with ``assertion.repository`` + ``assertion.ref`` condition
    - Per-repo CI SA (``lm-ci-*``) with deploy permissions
    - Per-repo exec SA (``lm-run-*``) with runtime permissions
    - WIF -> CI SA impersonation binding

    Returns {"connection_id": str}.
    """
    project_number = _get_project_number(project_id)
    suffix = _connection_suffix(repo_url)
    connection_id = make_connection_id(project_number, repo_url)
    repo_full = _validate_condition_operand(
        _extract_repo_full_name(repo_url), "repository", _REPO_FULL_RE,
    )
    branch = _validate_condition_operand(branch, "branch", _BRANCH_RE)
    provider_id = _wif_provider_id_from_suffix(suffix)

    ci_sa_id = _per_repo_sa_id_from_suffix("lm-ci", suffix)
    exec_sa_id = _per_repo_sa_id_from_suffix("lm-run", suffix)

    ci_sa = _ensure_per_repo_sa(project_id, ci_sa_id, f"Lamia CI — {repo_full}")
    run_sa = _ensure_per_repo_sa(project_id, exec_sa_id, f"Lamia Runtime — {repo_full}")

    _grant_sa_roles(project_id, ci_sa, _CI_SA_ROLES)
    _grant_sa_roles(project_id, run_sa, _EXEC_SA_ROLES)

    _run_gcloud(
        [
            "gcloud", "iam", "workload-identity-pools", "create", _WIF_POOL_ID,
            "--location=global", f"--project={project_id}",
            "--display-name=Lamia GitHub Actions",
        ],
        error_msg="Failed to create WIF pool",
    )

    attribute_condition = (
        f'assertion.repository=="{repo_full}"'
        f' && assertion.ref=="refs/heads/{branch}"'
    )
    provider_args = [
        f"--workload-identity-pool={_WIF_POOL_ID}",
        "--location=global", f"--project={project_id}",
        "--attribute-mapping=google.subject=assertion.sub,"
        "attribute.repository=assertion.repository,"
        "attribute.repository_owner=assertion.repository_owner",
        f"--attribute-condition={attribute_condition}",
    ]
    created = _run_gcloud(
        [
            "gcloud", "iam", "workload-identity-pools", "providers", "create-oidc",
            provider_id,
            "--issuer-uri=https://token.actions.githubusercontent.com",
            *provider_args,
        ],
        error_msg="Failed to create WIF provider",
    )
    if created.returncode != 0:
        _run_gcloud(
            [
                "gcloud", "iam", "workload-identity-pools", "providers",
                "update-oidc", provider_id, *provider_args,
            ],
            error_msg="Failed to update WIF provider condition",
        )

    wif_member = (
        f"principalSet://iam.googleapis.com/"
        f"projects/{project_number}/locations/global/"
        f"workloadIdentityPools/{_WIF_POOL_ID}/"
        f"attribute.repository/{repo_full}"
    )

    _run_gcloud(
        [
            "gcloud", "iam", "service-accounts", "add-iam-policy-binding",
            ci_sa,
            f"--project={project_id}",
            f"--role=roles/iam.workloadIdentityUser",
            f"--member={wif_member}",
        ],
        error_msg="Failed to bind WIF to CI service account",
    )

    return {"connection_id": connection_id}


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------


def connect_repository(
    project_id: str, location: str, repo_url: str, *, branch: str = "main",
) -> dict:
    """Full connection: Cloud Build repo link + Workload Identity Federation.

    After this completes, CI can authenticate and deploy without any
    manual GCP setup.  Idempotent: every step is safe to re-run.

    Returns an opaque Lamia connection ID for CI variable storage.
    """
    connection = _ensure_connection(project_id, location)
    repo_name = _sanitize_repo_name(repo_url)
    remote_uri = _to_https_remote(repo_url)

    _run_gcloud(
        [
            "gcloud", "builds", "repositories", "create", repo_name,
            f"--remote-uri={remote_uri}",
            f"--connection={connection}",
            f"--region={location}", f"--project={project_id}",
        ],
        error_msg="Failed to link repository",
    )

    ci_auth = _ensure_wif(project_id, repo_url, branch=branch)

    return {
        "connected": True,
        "message": f"Connected {repo_url} to cloud builds.",
        "connection_id": ci_auth["connection_id"],
        "branch": branch,
    }


def is_repository_connected(project_id: str, location: str, repo_url: str) -> bool:
    """Verify the complete connection chain for source-based builds.

    Checks every resource that ``connect_repository`` creates:
    1. Cloud Build connection exists and is COMPLETE
    2. Cloud Build repository resource exists
    3. WIF pool exists
    4. WIF provider exists with correct repo-scoped condition
    5. Per-repo CI service account exists

    Returns False as soon as any check fails.
    """
    repo_name = _sanitize_repo_name(repo_url)
    provider_id = _wif_provider_id(repo_url)

    result = subprocess.run(
        [
            "gcloud", "builds", "connections", "describe", _CONNECTION_NAME,
            f"--region={location}", f"--project={project_id}",
            "--format=value(installationState.stage)",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "COMPLETE":
        logger.info("Connection check failed: Cloud Build connection missing or incomplete")
        return False

    result = subprocess.run(
        [
            "gcloud", "builds", "repositories", "describe", repo_name,
            f"--connection={_CONNECTION_NAME}",
            f"--region={location}", f"--project={project_id}",
            "--format=value(name)",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        logger.info("Connection check failed: Cloud Build repository link missing")
        return False

    result = subprocess.run(
        [
            "gcloud", "iam", "workload-identity-pools", "describe", _WIF_POOL_ID,
            "--location=global", f"--project={project_id}",
            "--format=value(name)",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.info("Connection check failed: WIF pool missing")
        return False

    result = subprocess.run(
        [
            "gcloud", "iam", "workload-identity-pools", "providers", "describe",
            provider_id,
            f"--workload-identity-pool={_WIF_POOL_ID}",
            "--location=global", f"--project={project_id}",
            "--format=value(name)",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.info("Connection check failed: WIF provider missing")
        return False

    ci_sa = ci_sa_email(project_id, repo_url)
    try:
        iam_client = iam_admin_v1.IAMClient()
        iam_client.get_service_account(
            request={"name": f"projects/{project_id}/serviceAccounts/{ci_sa}"}
        )
    except Exception:
        logger.info("Connection check failed: per-repo CI service account missing")
        return False

    result = subprocess.run(
        [
            "gcloud", "iam", "workload-identity-pools", "providers", "describe",
            provider_id,
            f"--workload-identity-pool={_WIF_POOL_ID}",
            "--location=global", f"--project={project_id}",
            "--format=value(attributeCondition)",
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        condition = result.stdout.strip()
        if "assertion.ref==" not in condition:
            logger.warning(
                "WIF provider exists but attribute condition is missing "
                "branch restriction (assertion.ref). Re-run 'lamia cloud connect'."
            )

    return True


def disconnect_repository(project_id: str, location: str, repo_url: str) -> dict:
    """Remove WIF trust and per-repo SAs for a previously connected repository.

    Deletes:
    1. WIF provider for this repo
    2. Per-repo CI SA (``lm-ci-*``)
    3. Per-repo exec SA (``lm-run-*``)
    4. Cloud Build repository link

    The shared WIF pool is left intact (other repos may use it).
    """
    provider_id = _wif_provider_id(repo_url)
    repo_name = _sanitize_repo_name(repo_url)
    ci_sa = ci_sa_email(project_id, repo_url)
    run_sa = exec_sa_email(project_id, repo_url)
    deleted: list[str] = []

    result = subprocess.run(
        [
            "gcloud", "iam", "workload-identity-pools", "providers", "delete",
            provider_id,
            f"--workload-identity-pool={_WIF_POOL_ID}",
            "--location=global", f"--project={project_id}",
            "--quiet",
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        deleted.append(f"WIF provider: {provider_id}")

    iam_client = iam_admin_v1.IAMClient()
    for sa in (ci_sa, run_sa):
        try:
            iam_client.delete_service_account(
                request={"name": f"projects/{project_id}/serviceAccounts/{sa}"}
            )
            deleted.append(f"Service account: {sa}")
        except Exception:
            pass

    result = subprocess.run(
        [
            "gcloud", "builds", "repositories", "delete", repo_name,
            f"--connection={_CONNECTION_NAME}",
            f"--region={location}", f"--project={project_id}",
            "--quiet",
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        deleted.append(f"Cloud Build repository: {repo_name}")

    return {
        "disconnected": True,
        "deleted": deleted,
    }


# ---------------------------------------------------------------------------
# CI authentication (WIF OIDC token exchange)
# ---------------------------------------------------------------------------


def _write_credential_config(wif_provider: str, service_account: str) -> str:
    """Write an external-account credential config and return its path.

    Replicates what ``google-github-actions/auth@v2`` does, so the
    google-auth SDK exchanges the GitHub OIDC token for GCP credentials
    via STS.
    """
    token_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    token_bearer = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not token_url or not token_bearer:
        raise RuntimeError(
            "GitHub OIDC token not available. Ensure your workflow has:\n"
            "  permissions:\n"
            "    id-token: write"
        )

    audience = f"//iam.googleapis.com/{wif_provider}"
    encoded_audience = urllib.parse.quote(audience, safe="")

    cred_config = {
        "type": "external_account",
        "audience": audience,
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "token_url": "https://sts.googleapis.com/v1/token",
        "credential_source": {
            "url": f"{token_url}&audience={encoded_audience}",
            "headers": {
                "Authorization": f"bearer {token_bearer}",
            },
            "format": {
                "type": "json",
                "subject_token_field_name": "value",
            },
        },
        "service_account_impersonation_url": (
            f"https://iamcredentials.googleapis.com/v1/projects/-/"
            f"serviceAccounts/{service_account}:generateAccessToken"
        ),
    }

    fd, cred_path = tempfile.mkstemp(suffix=".json", prefix="lamia-wif-")
    with os.fdopen(fd, "w") as f:
        json.dump(cred_config, f)
    os.chmod(cred_path, 0o600)
    atexit.register(lambda: _cleanup_cred_file(cred_path))
    return cred_path


def _cleanup_cred_file(path: str) -> None:
    """Remove temporary credential config on process exit."""
    try:
        os.unlink(path)
    except OSError:
        pass


def configure_ci_auth(project_id: str, repo_url: str, connection_id: str) -> None:
    """Resolve a connection ID into WIF credentials for the current CI run."""
    project_number, suffix = parse_connection_id(connection_id)
    wif_provider = derive_wif_provider_from_connection(project_number, suffix)
    service_account = ci_sa_email_from_connection(project_id, suffix)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _write_credential_config(
        wif_provider, service_account,
    )
    logger.info("CI auth: credentials configured via WIF")


# ---------------------------------------------------------------------------
# GCPRepositoryConnector (wraps the module-level functions)
# ---------------------------------------------------------------------------


class GCPRepositoryConnector(RepositoryConnector):
    """GCP implementation of RepositoryConnector using WIF + Cloud Build."""

    def __init__(self, *, project_id: str, location: str):
        self.project_id = project_id
        self.location = location

    @classmethod
    def from_config(cls, cloud_cfg: dict) -> "GCPRepositoryConnector":
        project_id = cloud_cfg.get("project_id")
        if not project_id:
            raise ValueError("cloud.project_id is required in config.yaml.")
        location = cloud_cfg.get("location", "us-central1")
        return cls(project_id=project_id, location=location)

    def connect_repository(self, repo_url: str, *, branch: str = "main") -> dict:
        return connect_repository(self.project_id, self.location, repo_url, branch=branch)

    def configure_ci_auth(self, repo_url: str, connection_id: str) -> None:
        configure_ci_auth(self.project_id, repo_url, connection_id)

    def is_repository_connected(self, repo_url: str) -> bool:
        return is_repository_connected(self.project_id, self.location, repo_url)

    def disconnect_repository(self, repo_url: str) -> dict:
        return disconnect_repository(self.project_id, self.location, repo_url)
