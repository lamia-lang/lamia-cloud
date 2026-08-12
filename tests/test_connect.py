"""Tests for lamia_cloud.gcp.connect (repository connection, WIF, per-repo SAs)."""

from unittest.mock import MagicMock, patch

import pytest

from lamia_cloud.gcp.connect import (
    _CI_SA_ROLES,
    _EXEC_SA_ROLES,
    _REQUIRED_SA_ROLES,
    _extract_repo_full_name,
    _per_repo_sa_id,
    _sanitize_repo_name,
    _wif_provider_id,
    ci_sa_email,
    connect_repository,
    derive_wif_provider,
    exec_sa_email,
    is_repository_connected,
)


class TestSanitizeRepoName:
    def test_https_url(self):
        assert _sanitize_repo_name("https://github.com/acme/widgets.git") == "acme-widgets"

    def test_scp_url(self):
        assert _sanitize_repo_name("git@github.com:acme/widgets.git") == "acme-widgets"

    def test_nested_path(self):
        assert _sanitize_repo_name("https://gitlab.com/org/group/repo.git") == "org-group-repo"


class TestConnectRepository:
    @patch("lamia_cloud.gcp.connect._ensure_wif")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    @patch("lamia_cloud.gcp.connect._ensure_connection", return_value="lamia-github")
    def test_creates_repository_and_wif(self, mock_conn, mock_run, mock_wif):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_wif.return_value = {
            "connection_id": "v1-123456-abc123def456",
        }
        result = connect_repository("proj", "us-central1", "https://github.com/acme/widgets.git")
        assert result["connected"] is True
        assert result["connection_id"] == "v1-123456-abc123def456"
        mock_wif.assert_called_once()

    @patch("lamia_cloud.gcp.connect._ensure_wif")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    @patch("lamia_cloud.gcp.connect._ensure_connection", return_value="lamia-github")
    def test_already_exists_treated_as_success(self, mock_conn, mock_run, mock_wif):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="ALREADY_EXISTS")
        mock_wif.return_value = {
            "connection_id": "v1-123456-abc123def456",
        }
        result = connect_repository("proj", "us-central1", "https://github.com/acme/widgets.git")
        assert result["connected"] is True


class TestExtractRepoFullName:
    def test_https_url(self):
        assert _extract_repo_full_name("https://github.com/acme/widgets.git") == "acme/widgets"

    def test_scp_url(self):
        assert _extract_repo_full_name("git@github.com:acme/widgets.git") == "acme/widgets"

    def test_nested_gitlab(self):
        assert _extract_repo_full_name("https://gitlab.com/org/group/repo.git") == "org/group/repo"

    def test_no_trailing_git(self):
        assert _extract_repo_full_name("https://github.com/acme/widgets") == "acme/widgets"


class TestWifProviderNaming:
    def test_per_repo_provider_id(self):
        pid = _wif_provider_id("https://github.com/acme/widgets.git")
        assert pid.startswith("lamia-gh-")
        assert len(pid) <= 32

    def test_different_repos_get_different_providers(self):
        p1 = _wif_provider_id("https://github.com/acme/widgets.git")
        p2 = _wif_provider_id("https://github.com/acme/gadgets.git")
        assert p1 != p2

    def test_long_repo_name_fits_32_char_limit(self):
        pid = _wif_provider_id(
            "https://github.com/some-very-long-org/some-very-long-repo-name.git"
        )
        assert len(pid) <= 32
        assert pid.startswith("lamia-gh-")

    def test_provider_id_starts_with_letter(self):
        pid = _wif_provider_id("https://github.com/a/b.git")
        assert pid[0].isalpha()


class TestWifBranchRestriction:
    """WIF condition must restrict to refs/heads/{branch}."""

    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_wif_condition_includes_ref_restriction(
        self, mock_run, mock_pn, mock_roles, mock_sa,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        from lamia_cloud.gcp.connect import _ensure_wif
        _ensure_wif("proj", "https://github.com/acme/widgets.git")

        provider_create_calls = [
            c for c in mock_run.call_args_list
            if "create-oidc" in str(c)
        ]
        assert len(provider_create_calls) == 1
        cmd_str = str(provider_create_calls[0])
        assert 'assertion.ref==' in cmd_str
        assert 'refs/heads/main' in cmd_str
        assert 'assertion.repository==' in cmd_str

    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_custom_branch(self, mock_run, mock_pn, mock_roles, mock_sa):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        from lamia_cloud.gcp.connect import _ensure_wif
        _ensure_wif("proj", "https://github.com/acme/widgets.git", branch="master")

        provider_create_calls = [
            c for c in mock_run.call_args_list
            if "create-oidc" in str(c)
        ]
        cmd_str = str(provider_create_calls[0])
        assert 'refs/heads/master' in cmd_str

    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_returns_wif_and_sa(
        self, mock_run, mock_pn, mock_roles, mock_sa,
    ):
        """_ensure_wif returns an opaque connection ID."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        from lamia_cloud.gcp.connect import _ensure_wif
        result = _ensure_wif("proj", "https://github.com/acme/widgets.git")

        assert "connection_id" in result
        assert result["connection_id"].startswith("v1-123456-")

    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_subprocess_failure_raises(self, mock_run, mock_pn, mock_roles, mock_sa):
        """V3: Subprocess failures must raise, not silently continue."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="PERMISSION_DENIED",
        )
        from lamia_cloud.gcp.connect import _ensure_wif
        with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
            _ensure_wif("proj", "https://github.com/acme/widgets.git")


class TestWifProviderReconnect:
    """An existing provider must have its condition rewritten, not skipped."""

    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_existing_provider_condition_is_updated(
        self, mock_run, mock_pn, mock_roles, mock_sa,
    ):
        def responses(args, **kwargs):
            if "create-oidc" in args:
                return MagicMock(returncode=1, stdout="", stderr="ALREADY_EXISTS")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = responses
        from lamia_cloud.gcp.connect import _ensure_wif
        _ensure_wif("proj", "https://github.com/acme/widgets.git", branch="release")

        update_calls = [
            c for c in mock_run.call_args_list if "update-oidc" in str(c)
        ]
        assert len(update_calls) == 1, "reconnect must rewrite the condition"
        assert "refs/heads/release" in str(update_calls[0])

    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_fresh_provider_is_not_updated(
        self, mock_run, mock_pn, mock_roles, mock_sa,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        from lamia_cloud.gcp.connect import _ensure_wif
        _ensure_wif("proj", "https://github.com/acme/widgets.git")

        assert not [c for c in mock_run.call_args_list if "update-oidc" in str(c)]


class TestWifConditionOperandValidation:
    """Operands are interpolated into CEL, so they must be constrained."""

    @pytest.mark.parametrize(
        "branch",
        [
            'main" || true || "',
            'main"',
            "main && assertion.repository!=x",
            "main\\",
            "",
        ],
    )
    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_branch_injection_rejected(
        self, mock_run, mock_pn, mock_roles, mock_sa, branch,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        from lamia_cloud.gcp.connect import _ensure_wif
        with pytest.raises(RuntimeError, match="Invalid branch"):
            _ensure_wif(
                "proj", "https://github.com/acme/widgets.git", branch=branch,
            )

    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_no_gcloud_call_before_rejection(
        self, mock_run, mock_pn, mock_roles, mock_sa,
    ):
        """Validation must precede any resource creation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        from lamia_cloud.gcp.connect import _ensure_wif
        with pytest.raises(RuntimeError):
            _ensure_wif("proj", "https://github.com/acme/widgets.git", branch='x" || "')

        assert not [c for c in mock_run.call_args_list if "create-oidc" in str(c)]

    @pytest.mark.parametrize("branch", ["main", "release/2.0", "feat_x-1.2"])
    @patch("lamia_cloud.gcp.connect._ensure_per_repo_sa", return_value="sa@proj.iam")
    @patch("lamia_cloud.gcp.connect._grant_sa_roles")
    @patch("lamia_cloud.gcp.connect._get_project_number", return_value="123456")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_ordinary_branch_names_accepted(
        self, mock_run, mock_pn, mock_roles, mock_sa, branch,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        from lamia_cloud.gcp.connect import _ensure_wif
        _ensure_wif("proj", "https://github.com/acme/widgets.git", branch=branch)

        create = [c for c in mock_run.call_args_list if "create-oidc" in str(c)]
        assert f"refs/heads/{branch}" in str(create[0])


class TestPerRepoSa:
    """V1: Per-repo service account naming."""

    def test_ci_sa_email(self):
        email = ci_sa_email("my-proj", "https://github.com/acme/widgets.git")
        assert email.startswith("lm-ci-")
        assert email.endswith("@my-proj.iam.gserviceaccount.com")

    def test_exec_sa_email(self):
        email = exec_sa_email("my-proj", "https://github.com/acme/widgets.git")
        assert email.startswith("lm-run-")
        assert email.endswith("@my-proj.iam.gserviceaccount.com")

    def test_different_repos_get_different_sas(self):
        sa1 = ci_sa_email("proj", "https://github.com/acme/widgets.git")
        sa2 = ci_sa_email("proj", "https://github.com/acme/gadgets.git")
        assert sa1 != sa2

    def test_sa_id_max_length(self):
        long_url = "https://github.com/very-long-org-name/very-long-repo-name-that-exceeds-limits.git"
        sa_id = _per_repo_sa_id("lm-ci", long_url)
        assert len(sa_id) <= 30
        assert sa_id[0].isalpha()

    def test_sa_id_does_not_end_with_dash(self):
        sa_id = _per_repo_sa_id("lm-ci", "https://github.com/a/b-.git")
        assert not sa_id.endswith("-")


class TestDeriveWifProvider:
    """V8: WIF provider path derived by convention."""

    def test_derives_full_path(self):
        path = derive_wif_provider("123456", "https://github.com/acme/widgets.git")
        assert path.startswith("projects/123456/")
        assert "lamia-github-pool" in path
        assert "lamia-gh-" in path

    def test_no_config_needed(self):
        """Same inputs always produce same output -- no config lookup."""
        a = derive_wif_provider("123", "https://github.com/org/repo.git")
        b = derive_wif_provider("123", "https://github.com/org/repo.git")
        assert a == b


class TestSaRoleSeparation:
    """V2: CI roles and runtime roles must be separate."""

    def test_ci_roles_have_deploy_permissions(self):
        assert "roles/run.admin" in _CI_SA_ROLES
        assert "roles/cloudbuild.builds.editor" in _CI_SA_ROLES
        assert "roles/storage.admin" in _CI_SA_ROLES

    def test_exec_roles_are_minimal(self):
        assert "roles/aiplatform.user" in _EXEC_SA_ROLES
        assert "roles/run.admin" not in _EXEC_SA_ROLES
        assert "roles/storage.admin" not in _EXEC_SA_ROLES
        assert "roles/cloudbuild.builds.editor" not in _EXEC_SA_ROLES

    def test_required_sa_roles_is_union(self):
        assert set(_REQUIRED_SA_ROLES) == set(_CI_SA_ROLES) | set(_EXEC_SA_ROLES)


class TestRequiredSaRoles:
    def test_includes_minimum_roles_for_ci(self):
        assert "roles/run.admin" in _REQUIRED_SA_ROLES
        assert "roles/cloudbuild.builds.editor" in _REQUIRED_SA_ROLES
        assert "roles/storage.admin" in _REQUIRED_SA_ROLES
        assert "roles/logging.viewer" in _REQUIRED_SA_ROLES
        assert "roles/aiplatform.user" in _REQUIRED_SA_ROLES

    def test_no_overly_broad_roles(self):
        for role in _REQUIRED_SA_ROLES:
            assert role != "roles/editor"
            assert role != "roles/owner"


class TestIsRepositoryConnected:
    """Full-chain verification: all 5 checks must pass."""

    @patch("lamia_cloud.gcp.connect.iam_admin_v1")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_all_checks_pass(self, mock_run, mock_iam):
        def gcloud_ok(*args, **kwargs):
            cmd = args[0]
            if "connections" in cmd and "describe" in cmd:
                return MagicMock(returncode=0, stdout="COMPLETE\n")
            return MagicMock(returncode=0, stdout="found\n")
        mock_run.side_effect = gcloud_ok
        mock_iam.IAMClient.return_value.get_service_account.return_value = MagicMock()

        assert is_repository_connected(
            "proj", "us-central1", "https://github.com/acme/widgets.git"
        ) is True
        assert mock_run.call_count == 5

    @patch("lamia_cloud.gcp.connect.iam_admin_v1")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_missing_connection_returns_false(self, mock_run, mock_iam):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
        assert is_repository_connected(
            "proj", "us-central1", "https://github.com/acme/widgets.git"
        ) is False
        mock_run.assert_called_once()

    @patch("lamia_cloud.gcp.connect.iam_admin_v1")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_incomplete_connection_returns_false(self, mock_run, mock_iam):
        mock_run.return_value = MagicMock(returncode=0, stdout="PENDING_USER_SETUP\n")
        assert is_repository_connected(
            "proj", "us-central1", "https://github.com/acme/widgets.git"
        ) is False

    @patch("lamia_cloud.gcp.connect.iam_admin_v1")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_missing_wif_pool_returns_false(self, mock_run, mock_iam):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            cmd = args[0]
            if call_count[0] == 1:
                return MagicMock(returncode=0, stdout="COMPLETE\n")
            if call_count[0] == 2:
                return MagicMock(returncode=0, stdout="found\n")
            if "workload-identity-pools" in cmd and "providers" not in cmd:
                return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
            return MagicMock(returncode=0, stdout="found\n")

        mock_run.side_effect = side_effect
        assert is_repository_connected(
            "proj", "us-central1", "https://github.com/acme/widgets.git"
        ) is False

    @patch("lamia_cloud.gcp.connect.iam_admin_v1")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_missing_wif_provider_returns_false(self, mock_run, mock_iam):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                if call_count[0] == 1:
                    return MagicMock(returncode=0, stdout="COMPLETE\n")
                return MagicMock(returncode=0, stdout="found\n")
            return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")

        mock_run.side_effect = side_effect
        assert is_repository_connected(
            "proj", "us-central1", "https://github.com/acme/widgets.git"
        ) is False

    @patch("lamia_cloud.gcp.connect.iam_admin_v1")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_missing_service_account_returns_false(self, mock_run, mock_iam):
        def gcloud_ok(*args, **kwargs):
            cmd = args[0]
            if "connections" in cmd and "describe" in cmd:
                return MagicMock(returncode=0, stdout="COMPLETE\n")
            return MagicMock(returncode=0, stdout="found\n")

        mock_run.side_effect = gcloud_ok
        mock_iam.IAMClient.return_value.get_service_account.side_effect = (
            Exception("NOT_FOUND")
        )

        assert is_repository_connected(
            "proj", "us-central1", "https://github.com/acme/widgets.git"
        ) is False

    @patch("lamia_cloud.gcp.connect.iam_admin_v1")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_checks_per_repo_ci_sa_not_shared(self, mock_run, mock_iam):
        """V1: Must check per-repo lm-ci-* SA, not shared lamia-runner."""
        def gcloud_ok(*args, **kwargs):
            cmd = args[0]
            if "connections" in cmd and "describe" in cmd:
                return MagicMock(returncode=0, stdout="COMPLETE\n")
            return MagicMock(returncode=0, stdout="found\n")

        mock_run.side_effect = gcloud_ok
        mock_iam.IAMClient.return_value.get_service_account.return_value = MagicMock()

        is_repository_connected(
            "proj", "us-central1", "https://github.com/acme/widgets.git"
        )

        sa_request = mock_iam.IAMClient.return_value.get_service_account.call_args
        sa_name = sa_request[1]["request"]["name"] if sa_request[1] else sa_request[0][0]["name"]
        assert "lm-ci-" in sa_name
        assert "lamia-runner" not in sa_name

    @patch("lamia_cloud.gcp.connect.iam_admin_v1")
    @patch("lamia_cloud.gcp.connect.subprocess.run")
    def test_warns_when_wif_condition_missing_ref(self, mock_run, mock_iam, caplog):
        """V6: Must warn if WIF provider lacks branch restriction."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            cmd = args[0]
            if call_count[0] == 1 and "connections" in cmd:
                return MagicMock(returncode=0, stdout="COMPLETE\n")
            if "attributeCondition" in str(cmd):
                return MagicMock(
                    returncode=0,
                    stdout='assertion.repository=="acme/widgets"\n',
                )
            return MagicMock(returncode=0, stdout="found\n")

        mock_run.side_effect = side_effect
        mock_iam.IAMClient.return_value.get_service_account.return_value = MagicMock()

        import logging
        with caplog.at_level(logging.WARNING):
            is_repository_connected(
                "proj", "us-central1", "https://github.com/acme/widgets.git"
            )
        assert any("assertion.ref" in r.message for r in caplog.records)
