"""lamia-cloud — lamia cloud services package.

Public API:
    get_cloud_llm() -> CloudLLM
    get_scheduler(project_root) -> CloudScheduler
    is_on_cloud() -> bool
    Types: CloudLLMRequest, CloudLLMResponse, CloudScheduleJob, CloudJobStatus
"""
from lamia_cloud.interfaces import CloudLLM, CloudScheduler
from lamia_cloud.types import CloudLLMRequest, CloudLLMResponse, CloudScheduleJob, CloudJobStatus
from lamia_cloud.gcp import VertexLLM, is_on_gcp
from lamia_cloud.loader import get_scheduler

_llm_instance: CloudLLM = None


def get_cloud_llm() -> CloudLLM:
    """Return the cloud LLM instance.

    Only GCP is supported, so we return VertexLLM directly without
    conditional provider checking.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = VertexLLM()  # only GCP supported
    return _llm_instance


def is_on_cloud() -> bool:
    """Check if running in a cloud environment."""
    return is_on_gcp()


__all__ = [
    "CloudLLM",
    "CloudScheduler",
    "CloudLLMRequest",
    "CloudLLMResponse",
    "CloudScheduleJob",
    "CloudJobStatus",
    "get_cloud_llm",
    "get_scheduler",
    "is_on_cloud",
]
