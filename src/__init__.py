"""lamia-cloud — lamia cloud services package.

Public API:
    get_cloud_llm() -> CloudLLM
    get_scheduler(project_root) -> CloudScheduler
    get_deployer(project_root) -> CloudDeployer
    get_trigger_provider(project_root) -> CloudTriggerProvider
    is_on_cloud() -> bool
    Types: CloudLLMRequest, CloudLLMResponse, CloudScheduleJob, CloudJobStatus,
           TriggerStage, TriggerDeploymentPlan
"""
from lamia_cloud.interfaces import CloudDeployer, CloudLLM, CloudScheduler, CloudTriggerProvider, RepositoryConnector
from lamia_cloud.types import (
    CloudLLMRequest,
    CloudLLMResponse,
    CloudScheduleJob,
    CloudJobStatus,
    TriggerStage,
    TriggerDeploymentPlan,
)
from lamia_cloud.gcp import VertexLLM, is_on_gcp
from lamia_cloud.loader import get_connector, get_deployer, get_scheduler, get_trigger_provider
from lamia_cloud.gcp.deployer import ensure_apis_enabled
from lamia_cloud.gcp.connect import (
    ci_sa_email,
    ci_sa_email_from_connection,
    connection_suffix_for_repo,
    derive_wif_provider,
    derive_wif_provider_from_connection,
    exec_sa_email,
    exec_sa_email_from_connection,
    parse_connection_id,
)

_llm_instance: CloudLLM = None


def get_cloud_llm(region: str = "") -> CloudLLM:
    """Return the cloud LLM instance.

    Only GCP is supported, so we return VertexLLM directly without
    conditional provider checking.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = VertexLLM(region=region or _detect_region())
    return _llm_instance


def _detect_region() -> str:
    """Auto-detect GCP region from Cloud Run metadata server."""
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/region",
            headers={"Metadata-Flavor": "Google"},
        )
        resp = urllib.request.urlopen(req, timeout=2)
        full = resp.read().decode().strip()
        return full.rsplit("/", 1)[-1]
    except Exception:
        return "us-central1"


def is_on_cloud() -> bool:
    """Check if running in a cloud environment."""
    return is_on_gcp()


__all__ = [
    "CloudDeployer",
    "CloudLLM",
    "CloudScheduler",
    "CloudTriggerProvider",
    "CloudLLMRequest",
    "CloudLLMResponse",
    "CloudScheduleJob",
    "CloudJobStatus",
    "TriggerStage",
    "TriggerDeploymentPlan",
    "get_cloud_llm",
    "get_connector",
    "get_deployer",
    "get_scheduler",
    "get_trigger_provider",
    "RepositoryConnector",
    "is_on_cloud",
    "ci_sa_email",
    "ci_sa_email_from_connection",
    "derive_wif_provider",
    "derive_wif_provider_from_connection",
    "ensure_apis_enabled",
    "exec_sa_email",
    "exec_sa_email_from_connection",
    "connection_suffix_for_repo",
    "parse_connection_id",
]
