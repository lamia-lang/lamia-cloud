"""Config loader and cloud service factories.

Reads the cloud section from config.yaml and returns provider-agnostic
instances of CloudScheduler, CloudDeployer, and CloudTriggerProvider.
"""

import yaml
from pathlib import Path

from lamia_cloud.interfaces import CloudDeployer, CloudLLM, CloudScheduler, CloudTriggerProvider, RepositoryConnector
from lamia_cloud.gcp import GCPCloudScheduler, GCPDeployer, GCPRepositoryConnector, GCPTriggerProvider, VertexLLM

_SCHEDULERS = {
    "gcp": GCPCloudScheduler,
}

_DEPLOYERS = {
    "gcp": GCPDeployer,
}

_LLMS = {
    "gcp": VertexLLM,
}

_CONNECTORS = {
    "gcp": GCPRepositoryConnector,
}

_TRIGGER_PROVIDERS = {
    "gcp": GCPTriggerProvider,
}


def _load_cloud_cfg(project_root: Path) -> dict:
    """Read and validate the cloud section from config.yaml."""
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise ValueError(
            f"No config.yaml found at {project_root}. "
            f"Cloud features require a 'cloud' section in config.yaml."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "cloud" not in cfg:
        raise ValueError(
            "config.yaml must contain a 'cloud' section for cloud features."
        )

    cloud_cfg = cfg["cloud"]
    if not cloud_cfg.get("provider"):
        raise ValueError("cloud.provider is required in config.yaml.")

    return cloud_cfg


def _resolve_provider(registry: dict, cloud_cfg: dict, kind: str):
    """Look up a provider class from a registry, raising on unknown providers."""
    provider = cloud_cfg["provider"]
    cls = registry.get(provider)
    if not cls:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unsupported cloud provider '{provider}' for {kind}. Supported: {supported}"
        )
    return cls


def get_scheduler(project_root: Path) -> CloudScheduler:
    """Return a configured CloudScheduler for the project."""
    cloud_cfg = _load_cloud_cfg(project_root)
    cls = _resolve_provider(_SCHEDULERS, cloud_cfg, "scheduler")
    return cls.from_config(cloud_cfg)


def get_connector(project_root: Path) -> RepositoryConnector:
    """Return a configured RepositoryConnector for the project."""
    cloud_cfg = _load_cloud_cfg(project_root)
    cls = _resolve_provider(_CONNECTORS, cloud_cfg, "connector")
    return cls.from_config(cloud_cfg)


def get_deployer(project_root: Path) -> CloudDeployer:
    """Return a configured CloudDeployer for the project."""
    cloud_cfg = _load_cloud_cfg(project_root)
    cls = _resolve_provider(_DEPLOYERS, cloud_cfg, "deployer")
    return cls.from_config(cloud_cfg)


def get_trigger_provider(project_root: Path) -> CloudTriggerProvider:
    """Return a configured CloudTriggerProvider for the project."""
    cloud_cfg = _load_cloud_cfg(project_root)
    cls = _resolve_provider(_TRIGGER_PROVIDERS, cloud_cfg, "trigger provider")
    return cls.from_config(cloud_cfg)


def get_llm_router(project_root: Path) -> CloudLLM:
    """Return a CloudLLM configured for the project at `project_root`."""
    cloud_cfg = _load_cloud_cfg(project_root)
    cls = _resolve_provider(_LLMS, cloud_cfg, "LLM")
    return cls.from_config(cloud_cfg)
