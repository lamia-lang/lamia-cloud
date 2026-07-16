"""Cloud interfaces — the public API of lamia-cloud.

These abstract base classes are what external consumers (e.g., lamia's cloud
adapter) implement against. Three separate abstractions: CloudLLM (text
generation), CloudScheduler (remote jobs), and CloudTriggerProvider (event-driven).
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from lamia_cloud.types import (
    CloudLLMRequest,
    CloudLLMResponse,
    CloudScheduleJob,
    CloudJobStatus,
    TriggerDeploymentPlan,
)


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

    @abstractmethod
    def get_last_execution_status(self, job: CloudScheduleJob) -> Optional[dict]:
        """Return last completed Cloud Run Job execution status, or None."""
        ...

    @abstractmethod
    def pause(self, job: CloudScheduleJob) -> None:
        """Pause a scheduled job (stops triggering without removing it)."""
        ...

    @abstractmethod
    def resume(self, job: CloudScheduleJob) -> None:
        """Resume a previously paused job."""
        ...

    @abstractmethod
    def run_once(self, job: CloudScheduleJob, verbose: bool = False) -> dict:
        """Deploy and invoke once without scheduling. Returns {exit_code, stdout, stderr, logs_url}."""
        ...


class CloudTriggerProvider(ABC):
    """Abstract cloud trigger provider for event-driven multi-stage execution.

    Orchestrates: event ingress -> orchestration engine -> Cloud Run Job execution.
    GCP: Eventarc -> Workflows -> Cloud Run Jobs.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, cloud_cfg: dict) -> "CloudTriggerProvider":
        """Create instance from config.yaml cloud section."""
        ...

    @abstractmethod
    def deploy(self, plan: "TriggerDeploymentPlan") -> str:
        """Deploy all stages + orchestration + event routing. Returns deployment_id."""
        ...

    @abstractmethod
    def undeploy(self, name: str) -> None:
        """Remove all stages, orchestration, and event routing."""
        ...

    @abstractmethod
    def list_deployments(self) -> list[dict]:
        """List deployed triggers with last execution status.

        Each dict must include at minimum:
            name, script, trigger_method, mode, last_run, last_status,
            failed_event_count (int)
        """
        ...

    @abstractmethod
    def get_failed_events(self, name: str) -> List[dict]:
        """Retrieve failed events that could not be processed.

        Returns a list of event payloads (dicts) that exhausted retries.
        Messages are peeked (not consumed) so they remain available for
        replay or manual inspection.
        """
        ...

    @abstractmethod
    def clear_failed_events(self, name: str) -> int:
        """Acknowledge and remove all failed events. Returns count removed."""
        ...
