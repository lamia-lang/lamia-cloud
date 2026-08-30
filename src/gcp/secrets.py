"""Secret Manager operations backing CloudDeployer's secret methods.

Values are stored as Secret Manager secrets and referenced by Cloud Run Jobs
through SecretKeySelector, so they are injected as environment variables at
runtime and never appear in a job's own configuration.
"""

import logging
import re

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import run_v2, secretmanager
from google.iam.v1 import policy_pb2

logger = logging.getLogger(__name__)

SECRET_ACCESSOR_ROLE = "roles/secretmanager.secretAccessor"
_SECRET_ID_ALLOWED = re.compile(r"[^a-zA-Z0-9_-]")


def secret_id(namespace: str, key: str) -> str:
    """Return the Secret Manager id holding *key* for *namespace*."""
    safe_key = _SECRET_ID_ALLOWED.sub("-", key)
    return f"lamia-{namespace}-{safe_key}"


def sync_secrets(
    project_id: str,
    secrets: dict[str, str],
    namespace: str,
    service_account: str,
) -> list[str]:
    """Create or update *secrets* and let *service_account* read them.

    A new version is added only when the value differs from the current one,
    so repeated deploys of unchanged secrets do not accumulate versions.
    """
    if not secrets:
        return []

    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project_id}"
    synced: list[str] = []

    for key, value in sorted(secrets.items()):
        name = secret_id(namespace, key)
        resource = f"{parent}/secrets/{name}"
        try:
            _ensure_secret(client, parent, name)
            if _current_value(client, resource) != value:
                client.add_secret_version(
                    request={
                        "parent": resource,
                        "payload": {"data": value.encode()},
                    }
                )
            _grant_accessor(client, resource, service_account)
            synced.append(key)
        except Exception as exc:
            logger.warning(f"Failed to sync secret {key}: {exc}")

    return synced


def secret_env_vars(namespace: str, keys: list[str]) -> list:
    """Return Cloud Run env vars that reference stored secrets."""
    return [
        run_v2.EnvVar(
            name=key,
            value_source=run_v2.EnvVarSource(
                secret_key_ref=run_v2.SecretKeySelector(
                    secret=secret_id(namespace, key),
                    version="latest",
                ),
            ),
        )
        for key in keys
    ]


def cleanup_secrets(
    project_id: str,
    location: str,
    namespace: str,
    exclude_job: str = "",
) -> list[str]:
    """Delete *namespace* secrets no remaining Cloud Run Job references.

    *exclude_job* is the job being torn down; it is ignored when collecting
    still-referenced secrets so its own references do not keep them alive.
    """
    client = secretmanager.SecretManagerServiceClient()
    prefix = f"lamia-{namespace}-"
    parent = f"projects/{project_id}"

    try:
        stored = [
            secret.name.rsplit("/", 1)[-1]
            for secret in client.list_secrets(request={"parent": parent})
            if secret.name.rsplit("/", 1)[-1].startswith(prefix)
        ]
    except Exception as exc:
        logger.warning(f"Could not list secrets for cleanup: {exc}")
        return []

    in_use = referenced_secrets(project_id, location, exclude_job=exclude_job)
    deleted: list[str] = []

    for name in stored:
        if name in in_use:
            continue
        try:
            client.delete_secret(request={"name": f"{parent}/secrets/{name}"})
            deleted.append(name[len(prefix):])
        except NotFound:
            pass
        except Exception as exc:
            logger.warning(f"Failed to delete secret {name}: {exc}")

    return deleted


def referenced_secrets(
    project_id: str, location: str, exclude_job: str = ""
) -> set[str]:
    """Return secret ids referenced by lamia-managed Cloud Run Jobs."""
    client = run_v2.JobsClient()
    parent = f"projects/{project_id}/locations/{location}"
    referenced: set[str] = set()

    try:
        jobs = list(client.list_jobs(parent=parent))
    except Exception as exc:
        logger.warning(f"Could not list jobs while checking secret usage: {exc}")
        return referenced

    for job in jobs:
        if exclude_job and job.name.rsplit("/", 1)[-1] == exclude_job:
            continue
        for container in job.template.template.containers:
            for env in container.env:
                ref = env.value_source.secret_key_ref.secret
                if ref:
                    referenced.add(ref.rsplit("/", 1)[-1])

    return referenced


def _ensure_secret(client, parent: str, name: str) -> None:
    try:
        client.create_secret(
            request={
                "parent": parent,
                "secret_id": name,
                "secret": {"replication": {"automatic": {}}},
            }
        )
    except AlreadyExists:
        pass


def _current_value(client, resource: str) -> str | None:
    try:
        response = client.access_secret_version(
            request={"name": f"{resource}/versions/latest"}
        )
    except Exception:
        return None
    return response.payload.data.decode()


def _grant_accessor(client, resource: str, service_account: str) -> None:
    member = f"serviceAccount:{service_account}"
    policy = client.get_iam_policy(request={"resource": resource})

    for binding in policy.bindings:
        if binding.role == SECRET_ACCESSOR_ROLE and member in binding.members:
            return

    policy.bindings.append(
        policy_pb2.Binding(role=SECRET_ACCESSOR_ROLE, members=[member])
    )
    client.set_iam_policy(request={"resource": resource, "policy": policy})
