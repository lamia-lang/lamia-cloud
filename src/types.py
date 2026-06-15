"""lamia-cloud types — independent of lamia's internal types.

These types define the contract between lamia-cloud and any consumer.
lamia-cloud NEVER imports from the lamia package.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


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
