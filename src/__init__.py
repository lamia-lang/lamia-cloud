"""lamia-cloud — Cloud scheduling backend for Lamia.

Public API:
    get_scheduler(project_root) -> BaseScheduler
"""

from lamia_cloud.gcp import GCPCloudScheduler
from lamia_cloud.loader import get_scheduler

__all__ = ["get_scheduler", "GCPCloudScheduler"]
