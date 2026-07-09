"""lamia-cloud types — independent of lamia's internal types.

These types define the contract between lamia-cloud and any consumer.
lamia-cloud NEVER imports from the lamia package.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional


# ─── LLM types ───────────────────────────────────────────────────────────────

@dataclass
class CloudLLMRequest:
    """Request to generate text via a cloud LLM provider."""
    prompt: str
    model: str
    provider: str
    max_tokens: int = 1000
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    response_schema: Optional[dict] = None


@dataclass
class CloudLLMResponse:
    """Response from a cloud LLM provider."""
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    raw: Any = None


# ─── Scheduling types ────────────────────────────────────────────────────────

class CloudJobStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


@dataclass
class CloudScheduleJob:
    """A scheduled job as understood by lamia-cloud."""
    script: str
    cron: str
    schedule_id: str
    catch_up: bool = True
    project_root: Path = field(default_factory=Path)


# ─── Trigger types ────────────────────────────────────────────────────────────

@dataclass
class TriggerStage:
    """One stage of a triggered script (code between trigger boundaries)."""
    stage_index: int
    trigger_method: str
    trigger_config: dict = field(default_factory=dict)
    output_bindings: List[str] = field(default_factory=list)
    script_source: str = ""


@dataclass
class TriggerDeploymentPlan:
    """Provider-agnostic description of a triggered script deployment.

    mode:
        "reactive" -- Eventarc directly invokes workflow on each event (always-on).
        "scheduled" -- Events accumulate in a queue; a scheduler drains them at cron time.
    """
    name: str
    stages: List[TriggerStage] = field(default_factory=list)
    capabilities: dict = field(default_factory=dict)
    mode: str = "reactive"
    cron: Optional[str] = None
