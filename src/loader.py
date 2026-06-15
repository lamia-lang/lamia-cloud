"""Config loader and scheduler factory.

Reads the cloud section from config.yaml and returns a configured CloudScheduler.
"""

import yaml
from pathlib import Path

from lamia_cloud.interfaces import CloudScheduler
from lamia_cloud.gcp import GCPCloudScheduler

_PROVIDERS = {
    "gcp": GCPCloudScheduler,
}


def get_scheduler(project_root: Path) -> CloudScheduler:
    """Load cloud config and return a configured scheduler instance.

    Only GCP is supported, so we return GCPCloudScheduler directly without
    conditional provider checking beyond validating the config value.

    Raises ValueError if config is missing or invalid.
    """
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise ValueError(
            f"No config.yaml found at {project_root}. "
            f"Cloud scheduling requires a 'cloud' section in config.yaml."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "cloud" not in cfg:
        raise ValueError(
            "config.yaml must contain a 'cloud' section for remote scheduling."
        )

    cloud_cfg = cfg["cloud"]
    provider = cloud_cfg.get("provider")
    if not provider:
        raise ValueError("cloud.provider is required in config.yaml.")

    scheduler_cls = _PROVIDERS.get(provider)
    if not scheduler_cls:
        supported = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(
            f"Unsupported cloud provider '{provider}'. Supported: {supported}"
        )

    # Only GCP is supported — returns GCPCloudScheduler without further branching
    return scheduler_cls.from_config(cloud_cfg)
