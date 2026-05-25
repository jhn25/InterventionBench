"""Core data models for AgentSteer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal


# StrEnum was added in Python 3.11. Provide a 3.10-compatible fallback so
# scripts running under the OSWorld venv (Python 3.10) can import this module.
try:
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:
    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass

from pydantic import BaseModel, Field

# Stratum classification thresholds
_EASY_THRESHOLD = 0.7
_MEDIUM_THRESHOLD = 0.3


class BaseTask(BaseModel):
    """A single base task drawn from one of the underlying benchmarks."""

    domain: str
    task_id: str
    spec: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    """One step in an agent's execution trajectory."""

    role: Literal["thought", "action", "observation", "user", "system", "assistant", "tool"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Full execution trajectory for one (task, model, condition) tuple."""

    episode_id: str
    task_ref: str
    condition: Literal["A", "B", "A->B"]
    model: str
    steps: list[TrajectoryStep]
    success: bool
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class Stratum(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @classmethod
    def classify(cls, success_rate: float) -> Stratum:
        if success_rate >= _EASY_THRESHOLD:
            return cls.EASY
        if success_rate >= _MEDIUM_THRESHOLD:
            return cls.MEDIUM
        return cls.HARD
