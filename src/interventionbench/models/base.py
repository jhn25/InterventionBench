"""Abstract model client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    content: str
    usage: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict | None = None,
    ) -> ModelResponse: ...
