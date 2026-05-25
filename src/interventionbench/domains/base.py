"""Abstract DomainAdapter interface that every domain must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from interventionbench.data.models import BaseTask, Trajectory
from interventionbench.models.base import ModelClient


@dataclass
class RunResult:
    trajectory: Trajectory
    success: bool
    score: float


class DomainAdapter(ABC):
    domain_name: str

    @abstractmethod
    def list_candidate_tasks(self, *, limit: int | None = None) -> list[BaseTask]: ...

    @abstractmethod
    def run_baseline(self, task: BaseTask, model_client: ModelClient) -> RunResult: ...
