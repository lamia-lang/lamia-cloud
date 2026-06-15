"""Cloud interfaces — the public API of lamia-cloud.

These abstract base classes are what external consumers (e.g., lamia's cloud
adapter) implement against. Two separate abstractions: CloudLLM (text
generation) and CloudScheduler (remote jobs).
"""
from abc import ABC, abstractmethod
from typing import Optional

from lamia_cloud.types import CloudLLMRequest, CloudLLMResponse, CloudScheduleJob, CloudJobStatus


class CloudLLM(ABC):
    """Abstract cloud LLM that provides text generation via a cloud endpoint."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this cloud LLM is accessible in the current environment."""
        ...

    @abstractmethod
    async def generate(self, request: CloudLLMRequest) -> CloudLLMResponse:
        """Generate text via the cloud LLM endpoint."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the LLM client."""
        ...


class CloudScheduler(ABC):
    """Abstract cloud scheduler that manages remote job execution."""

    @classmethod
    @abstractmethod
    def from_config(cls, cloud_cfg: dict) -> "CloudScheduler":
        """Create instance from config.yaml cloud section."""
        ...

    @abstractmethod
    def install(self, job: CloudScheduleJob, lamia_bin: str) -> None:
        """Deploy and schedule a job in the cloud."""
        ...

    @abstractmethod
    def uninstall(self, job: CloudScheduleJob) -> None:
        """Remove a scheduled job from the cloud."""
        ...

    @abstractmethod
    def is_installed(self, job: CloudScheduleJob) -> bool:
        """Check if a job is currently scheduled."""
        ...

    @abstractmethod
    def get_status(self, job: CloudScheduleJob) -> CloudJobStatus:
        """Get the current status of a scheduled job."""
        ...

    @abstractmethod
    def get_installed_config(self, job: CloudScheduleJob) -> Optional[dict]:
        """Return the currently installed config, or None if not installed."""
        ...
