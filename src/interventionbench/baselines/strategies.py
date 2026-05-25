"""Intervention strategies for Phase 2 baselines.

DESIGN PRINCIPLE (revised):
- The intervention message itself is IDENTICAL across all baselines (= Phase 1's text).
- Phase 2 baselines differ only in agent capabilities (system prompts, tools, retries).
- This isolates the effect of the agent FRAMEWORK from the intervention WORDING.

So `make_intervention_message` returns the same text for all strategies.
The framework-specific behavior comes from prefix_messages + tools provided by
each Runner subclass.
"""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from interventionbench.data.models import ChatMessage
    from interventionbench.models.base import ModelClient


# Phase 1 intervention text — used uniformly by all baselines
def _phase1_inject(intervention_text: str) -> str:
    return f"User says: {intervention_text}. Please continue with this update."


# System prompts for framework baselines (used as prefix_messages by Runners)
PS_PREFIX = (
    "Let's first understand the task, extract the key constraints and "
    "deliverables, and devise a complete plan. Then, let's carry out the "
    "plan step by step, paying attention to each constraint, and complete "
    "the task."
)

MEMGPT_SYSTEM = (
    "You operate with a core memory: an always-visible scratchpad of facts "
    "you must keep in mind. Use core_memory_append(name, content) and "
    "core_memory_replace(name, old_content, new_content) when the user "
    "changes a constraint or you discover a key fact you must remember."
)

ADAPLANNER_SYSTEM = (
    "You are an agent. Before acting, write an explicit numbered plan "
    "(3-7 steps) for the task. Then carry out the plan step by step. "
    "If observations diverge from your plan, refine it before continuing."
)

AMEM_SYSTEM = (
    "You operate with an A-MEM (agentic memory) capability: as you take "
    "actions, internally maintain notes about what you've learned, and "
    "actively retrieve relevant past notes when planning your next step. "
    "If the user changes the goal mid-task, recall what's still applicable "
    "from your prior work before proceeding."
)


class InterventionStrategy(ABC):
    name: str = "abstract"

    def make_intervention_message(
        self,
        client: "ModelClient",
        prior_messages: list["ChatMessage"],
        intervention_text: str,
        modified_spec: Optional[str] = None,
    ) -> str:
        """Default = Phase 1 text. Subclasses override only if they NEED to (e.g.,
        injection-time summarization). For framework comparison, keep this default
        so the only difference is the agent's capabilities."""
        return _phase1_inject(intervention_text)

    def metadata(self) -> dict:
        return {"strategy": self.name}


class VanillaStrategy(InterventionStrategy):
    """Pure baseline — equivalent to Phase 1 vanilla agent."""
    name = "vanilla"


class PlanAndSolveStrategy(InterventionStrategy):
    """Plan-and-Solve (Wang 2023) — agent has a system prompt instructing it to
    plan-then-execute. Intervention text identical to Phase 1."""
    name = "plan_and_solve"


class MemGPTStrategy(InterventionStrategy):
    """MemGPT (Packer 2023) — agent has core_memory tools. Intervention identical."""
    name = "memgpt"


class AdaPlannerStrategy(InterventionStrategy):
    """AdaPlanner (Sun 2023) — agent re-plans on observations via step_callback.
    Intervention text identical to Phase 1."""
    name = "adaplanner"


# --- MemGPT tools (passed via Runner) ---

MEMGPT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "core_memory_append",
            "description": (
                "Append content to a section of core memory. Core memory is the "
                "always-visible scratchpad used to remember key facts across the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Section name."},
                    "content": {"type": "string", "description": "Text to append."},
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_replace",
            "description": "Replace exact content within a section of core memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "old_content": {"type": "string"},
                    "new_content": {"type": "string"},
                },
                "required": ["name", "old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_insert",
            "description": (
                "Insert content into archival memory (long-term vector store). "
                "Use for facts not needed every step but might be retrieved later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Text to archive."},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_search",
            "description": (
                "Search archival memory by semantic query. Returns top-k matching notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "k": {"type": "integer", "description": "Top-k (default 3)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conversation_search",
            "description": (
                "Search past conversation turns for a substring. Returns matching messages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for."},
                    "k": {"type": "integer", "description": "Max matches (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
]
