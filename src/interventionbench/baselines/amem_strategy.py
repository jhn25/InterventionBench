"""A-MEM (Xu 2025) — Zettelkasten-style autonomous agentic memory.

Per paper:
  - After each agent step, generate a structured note (content + category + tags + links)
  - Notes stored in vector DB (via amem_server, port 9100)
  - Before each step, retrieve top-k semantically related notes
  - Inject retrieved context into prompt

Implementation:
  - step_callback is called after each agent step → autonomous note-taking + retrieval
  - prefix_messages includes system prompt explaining memory + retrieved notes
  - amem_server provides Zettelkasten store with semantic+link retrieval
"""
from __future__ import annotations
from .strategies import InterventionStrategy
from .memory_client import MemoryClient


AMEM_SYSTEM_PROMPT = (
    "You operate with A-MEM agentic memory: structured notes are auto-stored after each step, "
    "and relevant notes are retrieved semantically before each step. "
    "Pay attention to any [retrieved_notes] block in observations — it contains key facts "
    "from earlier in the task that may inform your next action. "
    "Build a coherent picture of the task across steps using the memory."
)


class AMemStrategy(InterventionStrategy):
    """A-MEM (Xu 2025) — autonomous note-taking + retrieval via amem_server."""
    name = "amem"


def make_amem_step_callback(session_id: str, k: int = 3):
    """Create a step_callback that does autonomous note-taking + retrieval per step.

    Called after each agent step:
    1. Add note: latest action + observation as Zettelkasten note.
    2. Retrieve: top-k semantically relevant notes for next step's context.
    """
    client = MemoryClient(session_id)

    def callback(step_idx: int, observation: str = "", action: str = "",
                 thought: str = "", **kwargs) -> str:
        """Add note + return retrieved context for next step."""
        # 1. Note generation
        note_parts = []
        if thought: note_parts.append(f"Thought: {thought[:300]}")
        if action: note_parts.append(f"Action: {action[:300]}")
        if observation: note_parts.append(f"Observation: {observation[:400]}")
        note_content = " | ".join(note_parts)
        if note_content:
            client.add(content=note_content, category=f"step_{step_idx}",
                       tags=["agent_step"])

        # 2. Retrieval (top-k semantically + link-related)
        query = f"{action[:150]} {observation[:200]}".strip() or "task progress"
        hits = client.search(query=query, k=k)
        if not hits:
            return ""
        ctx_lines = ["[retrieved_notes]"]
        for h in hits[:k]:
            ctx_lines.append(f"  - ({h.get('score', 0):.2f}) {h.get('content', '')[:200]}")
        return "\n".join(ctx_lines)

    return callback
