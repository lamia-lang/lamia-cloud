"""Tests for lamia_cloud.gcp.secrets (Secret Manager sync and cleanup)."""

from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import AlreadyExists

from lamia_cloud.gcp.deployer import _REQUIRED_GCP_APIS
from lamia_cloud.gcp.secrets import (
    SECRET_ACCESSOR_ROLE,
    cleanup_secrets,
    referenced_secrets,
    secret_env_vars,
    secret_id,
    sync_secrets,
)

NAMESPACE = "abc123def456"
RUNNER_SA = "lamia-runner@demo-project.iam.gserviceaccount.com"


def make_client(current_value=None):
    """Secret Manager client whose latest version holds *current_value*."""
    client = MagicMock()
    if current_value is None:
        client.access_secret_version.side_effect = Exception("no version")
    else:
        client.access_secret_version.return_value.payload.data = current_value.encode()
    policy = MagicMock()
    policy.bindings = []
    client.get_iam_policy.return_value = policy
    return client


def make_secret(name):
    """Secret Manager secret resource with *name* as its short id."""
    secret = MagicMock()
    secret.name = f"projects/demo/secrets/{name}"
    return secret


def make_job(name, secret_ids=(), plain_env=()):
    """Cloud Run Job referencing *secret_ids* plus any plain env vars."""
    job = MagicMock()
    job.name = f"projects/demo/locations/us-central1/jobs/{name}"
    env = []
    for sid in secret_ids:
        var = MagicMock()
        var.value_source.secret_key_ref.secret = sid
        env.append(var)
    for plain in plain_env:
        var = MagicMock()
        var.value_source.secret_key_ref.secret = ""
        var.name = plain
        env.append(var)
    container = MagicMock()
    container.env = env
    job.template.template.containers = [container]
    return job


class TestSecretId:
    def test_namespaces_the_key(self):
        assert secret_id(NAMESPACE, "BRIGHTDATA_API_KEY") == (
            f"lamia-{NAMESPACE}-BRIGHTDATA_API_KEY"
        )

    def test_replaces_characters_the_provider_rejects(self):
        assert "." not in secret_id(NAMESPACE, "SOME.KEY")

    def test_different_namespaces_never_collide(self):
        assert secret_id("one", "KEY") != secret_id("two", "KEY")


class TestSyncSecrets:
    def test_no_secrets_makes_no_calls(self):
        with patch("lamia_cloud.gcp.secrets.secretmanager") as sm:
            assert sync_secrets("demo", {}, NAMESPACE, RUNNER_SA) == []
            sm.SecretManagerServiceClient.assert_not_called()

    def test_creates_secret_and_adds_first_version(self):
        client = make_client()
        with patch(
            "lamia_cloud.gcp.secrets.secretmanager.SecretManagerServiceClient",
            return_value=client,
        ):
            synced = sync_secrets(
                "demo", {"BRIGHTDATA_API_KEY": "bd-1"}, NAMESPACE, RUNNER_SA
            )

        assert synced == ["BRIGHTDATA_API_KEY"]
        client.create_secret.assert_called_once()
        client.add_secret_version.assert_called_once()

    def test_existing_secret_is_reused(self):
        client = make_client(current_value="old")
        client.create_secret.side_effect = AlreadyExists("exists")
        with patch(
            "lamia_cloud.gcp.secrets.secretmanager.SecretManagerServiceClient",
            return_value=client,
        ):
            synced = sync_secrets("demo", {"KEY": "new"}, NAMESPACE, RUNNER_SA)

        assert synced == ["KEY"]
        client.add_secret_version.assert_called_once()

    def test_unchanged_value_adds_no_new_version(self):
        client = make_client(current_value="same")
        client.create_secret.side_effect = AlreadyExists("exists")
        with patch(
            "lamia_cloud.gcp.secrets.secretmanager.SecretManagerServiceClient",
            return_value=client,
        ):
            synced = sync_secrets("demo", {"KEY": "same"}, NAMESPACE, RUNNER_SA)

        assert synced == ["KEY"]
        client.add_secret_version.assert_not_called()

    def test_grants_the_runner_read_access(self):
        client = make_client()
        with patch(
            "lamia_cloud.gcp.secrets.secretmanager.SecretManagerServiceClient",
            return_value=client,
        ):
            sync_secrets("demo", {"KEY": "value"}, NAMESPACE, RUNNER_SA)

        policy = client.set_iam_policy.call_args.kwargs["request"]["policy"]
        binding = policy.bindings[0]
        assert binding.role == SECRET_ACCESSOR_ROLE
        assert f"serviceAccount:{RUNNER_SA}" in binding.members

    def test_existing_grant_is_not_rewritten(self):
        client = make_client()
        binding = MagicMock()
        binding.role = SECRET_ACCESSOR_ROLE
        binding.members = [f"serviceAccount:{RUNNER_SA}"]
        client.get_iam_policy.return_value.bindings = [binding]
        with patch(
            "lamia_cloud.gcp.secrets.secretmanager.SecretManagerServiceClient",
            return_value=client,
        ):
            sync_secrets("demo", {"KEY": "value"}, NAMESPACE, RUNNER_SA)

        client.set_iam_policy.assert_not_called()

    def test_one_failure_does_not_abort_the_rest(self):
        client = make_client()
        client.add_secret_version.side_effect = [Exception("boom"), MagicMock()]
        with patch(
            "lamia_cloud.gcp.secrets.secretmanager.SecretManagerServiceClient",
            return_value=client,
        ):
            synced = sync_secrets(
                "demo", {"A_KEY": "1", "B_KEY": "2"}, NAMESPACE, RUNNER_SA
            )

        assert synced == ["B_KEY"]


class TestSecretEnvVars:
    def test_references_the_secret_instead_of_its_value(self):
        env = secret_env_vars(NAMESPACE, ["BRIGHTDATA_API_KEY"])
        assert len(env) == 1
        assert env[0].name == "BRIGHTDATA_API_KEY"
        assert env[0].value_source.secret_key_ref.secret == secret_id(
            NAMESPACE, "BRIGHTDATA_API_KEY"
        )
        assert env[0].value == ""

    def test_tracks_the_latest_version(self):
        env = secret_env_vars(NAMESPACE, ["KEY"])
        assert env[0].value_source.secret_key_ref.version == "latest"

    def test_no_keys_yields_no_env_vars(self):
        assert secret_env_vars(NAMESPACE, []) == []


class TestReferencedSecrets:
    def test_collects_secrets_across_jobs(self):
        client = MagicMock()
        client.list_jobs.return_value = [
            make_job("job-a", secret_ids=["lamia-ns-A_KEY"]),
            make_job("job-b", secret_ids=["lamia-ns-B_KEY"]),
        ]
        with patch(
            "lamia_cloud.gcp.secrets.run_v2.JobsClient", return_value=client
        ):
            assert referenced_secrets("demo", "us-central1") == {
                "lamia-ns-A_KEY",
                "lamia-ns-B_KEY",
            }

    def test_excluded_job_does_not_keep_its_secrets_alive(self):
        client = MagicMock()
        client.list_jobs.return_value = [make_job("doomed", secret_ids=["lamia-ns-KEY"])]
        with patch(
            "lamia_cloud.gcp.secrets.run_v2.JobsClient", return_value=client
        ):
            assert referenced_secrets("demo", "us-central1", exclude_job="doomed") == set()

    def test_plain_env_vars_are_ignored(self):
        client = MagicMock()
        client.list_jobs.return_value = [make_job("job", plain_env=["LAMIA_SCRIPT"])]
        with patch(
            "lamia_cloud.gcp.secrets.run_v2.JobsClient", return_value=client
        ):
            assert referenced_secrets("demo", "us-central1") == set()


class TestCleanupSecrets:
    def cleanup(self, stored, jobs, exclude_job=""):
        sm_client = MagicMock()
        sm_client.list_secrets.return_value = [make_secret(name) for name in stored]
        jobs_client = MagicMock()
        jobs_client.list_jobs.return_value = jobs

        with patch(
            "lamia_cloud.gcp.secrets.secretmanager.SecretManagerServiceClient",
            return_value=sm_client,
        ), patch(
            "lamia_cloud.gcp.secrets.run_v2.JobsClient", return_value=jobs_client
        ):
            deleted = cleanup_secrets("demo", "us-central1", NAMESPACE, exclude_job)
        return deleted, sm_client

    def test_deletes_secrets_nothing_references(self):
        deleted, client = self.cleanup(
            stored=[f"lamia-{NAMESPACE}-ORPHAN_KEY"], jobs=[]
        )
        assert deleted == ["ORPHAN_KEY"]
        client.delete_secret.assert_called_once()

    def test_keeps_secrets_another_deployment_still_uses(self):
        deleted, client = self.cleanup(
            stored=[f"lamia-{NAMESPACE}-SHARED_KEY"],
            jobs=[make_job("survivor", secret_ids=[f"lamia-{NAMESPACE}-SHARED_KEY"])],
        )
        assert deleted == []
        client.delete_secret.assert_not_called()

    def test_ignores_secrets_from_other_namespaces(self):
        deleted, client = self.cleanup(stored=["lamia-otherns-KEY"], jobs=[])
        assert deleted == []
        client.delete_secret.assert_not_called()

    def test_torn_down_job_does_not_protect_its_own_secrets(self):
        deleted, _ = self.cleanup(
            stored=[f"lamia-{NAMESPACE}-KEY"],
            jobs=[make_job("doomed", secret_ids=[f"lamia-{NAMESPACE}-KEY"])],
            exclude_job="doomed",
        )
        assert deleted == ["KEY"]


class TestApiEnablement:
    def test_secret_manager_api_is_required(self):
        assert "secretmanager.googleapis.com" in _REQUIRED_GCP_APIS
