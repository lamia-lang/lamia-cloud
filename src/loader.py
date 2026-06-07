"""Config loader and scheduler factory.

Reads the cloud section from config.yaml and returns a configured scheduler.
"""

import yaml
from pathlib import Path

from lamia.scheduling.base import BaseScheduler

from lamia_cloud.gcp import GCPCloudScheduler

_PROVIDERS = {
    "gcp": GCPCloudScheduler,
}


def get_scheduler(project_root: Path) -> BaseScheduler:
    """Load cloud config and return a configured scheduler instance.

    This is the entry point called by lamia's CLI.
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

    return scheduler_cls.from_config(cloud_cfg)
