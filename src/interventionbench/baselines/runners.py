"""Baseline orchestrators that drive the domain adapter for one A->B episode.

Strategy alone is sufficient for single-call baselines (Vanilla, Plan-and-Solve).
Multi-call baselines (Reflexion N=3 trials, MemGPT tool-driven, AdaPlanner with
step-wise refine) need a Runner to orchestrate adapter calls and additional LLM
calls.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from interventionbench.baselines.strategies import (
    VanillaStrategy,
    PlanAndSolveStrategy,
    PS_PREFIX,
    MemGPTStrategy,
    MEMGPT_TOOLS,
    AdaPlannerStrategy,
)

if TYPE_CHECKING:
    from interventionbench.data.models import BaseTask
    from interventionbench.domains.base import DomainAdapter, RunResult
    from interventionbench.models.base import ModelClient


class BaselineRunner(ABC):
    name: str = "abstract"

    @abstractmethod
    def run_episode(
        self,
        adapter: "DomainAdapter",
        task: "BaseTask",
        intervention: dict,
        client: "ModelClient",
    ) -> "RunResult":
        raise NotImplementedError


class VanillaRunner(BaselineRunner):
    """Reference implementation: single adapter call with VanillaStrategy."""

    name = "vanilla"

    def run_episode(self, adapter, task, intervention, client):
        return adapter.run_a_to_b(
            task,
            intervention,
            client, timing="late",
            strategy=VanillaStrategy(),
        )


class PlanAndSolveRunner(BaselineRunner):
    """Plan-and-Solve (Wang 2023): prepend PS prefix to system message; replan
    on intervention via PlanAndSolveStrategy."""

    name = "plan_and_solve"

    def run_episode(self, adapter, task, intervention, client):
        from interventionbench.models.base import ChatMessage
        prefix = [ChatMessage(role="system", content=PS_PREFIX)]
        return adapter.run_a_to_b(
            task,
            intervention,
            client, timing="late",
            strategy=PlanAndSolveStrategy(),
            prefix_messages=prefix,
        )


class ReflexionRunner(BaselineRunner):
    """Reflexion (Shinn 2023). N trials with verbal self-reflection between.

    Faithful to paper: trial 1 attempts, evaluator gives binary signal, on fail
    a self_reflection LLM call summarizes failure, next trial starts with
    reflection prepended. Final score = trial-N (paper convention).
    """

    name = "reflexion"

    REFLEXION_SYSTEM = (
        "You are an agent that just attempted a task and either failed or "
        "produced a suboptimal result. Reflect verbally on what went wrong, "
        "what specifically you should have done differently in light of the "
        "user's mid-task instruction, and what concrete approach you will take "
        "in the next attempt. Keep the reflection to 3-5 sentences."
    )

    def __init__(self, n_trials: int = 3):
        self.n_trials = n_trials

    def _reflect(self, client, last_result, intervention) -> str:
        from interventionbench.models.base import ChatMessage
        # Render last trajectory compactly
        steps = getattr(last_result.trajectory, "steps", []) or []
        traj_text = "\n".join(
            f"[{getattr(s, 'role', '?')}] {str(getattr(s, 'content', ''))[:600]}"
            for s in steps[-12:]
        )
        intervention_text = intervention.get("intervention_text", "")
        prompt = (
            f"Trajectory of last attempt:\n{traj_text}\n\n"
            f"User mid-task instruction: {intervention_text}\n\n"
            f"Outcome: success={last_result.success} score={last_result.score:.3f}\n\n"
            "Self-reflection:"
        )
        try:
            resp = client.chat(
                [
                    ChatMessage(role="system", content=self.REFLEXION_SYSTEM),
                    ChatMessage(role="user", content=prompt),
                ],
                max_tokens=400,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:
            return f"[reflection_error: {exc}]"

    def run_episode(self, adapter, task, intervention, client):
        from interventionbench.models.base import ChatMessage
        reflections: list[str] = []
        last = None
        for trial in range(self.n_trials):
            prefix = None
            if reflections:
                prefix_text = "[Past reflections from earlier trials]:\n" + "\n".join(
                    f"- Trial {i+1}: {r}" for i, r in enumerate(reflections)
                )
                prefix = [ChatMessage(role="system", content=prefix_text)]
            last = adapter.run_a_to_b(
                task,
                intervention,
                client, timing="late",
                strategy=VanillaStrategy(),
                prefix_messages=prefix,
            )
            if last.success:
                return last
            if trial < self.n_trials - 1:
                reflections.append(self._reflect(client, last, intervention))
        return last  # trial-N per paper convention


class MemGPTRunner(BaselineRunner):
    """MemGPT (Packer 2023) -- core_memory subset.

    Initializes core memory with the original task spec, exposes
    core_memory_append/replace functions via tools=, and uses MemGPTStrategy
    at intervention time to compress prior context into archival snapshot.
    """

    name = "memgpt"

    CORE_MEMORY_INIT_SYSTEM = (
        "You operate with a 3-tier MemGPT memory hierarchy:\n"
        "  1. Core memory — always-visible scratchpad. Tools: core_memory_append(name, content), "
        "core_memory_replace(name, old, new), core_memory_show().\n"
        "  2. Archival memory — long-term vector store. Tools: archival_memory_insert(content), "
        "archival_memory_search(query, k).\n"
        "  3. Conversation history search. Tool: conversation_search(query, k).\n"
        "Use core_memory for facts you need every step. Use archival_memory for facts that "
        "may be retrieved later. Use conversation_search to recall past dialog. "
        "Begin by calling core_memory_append('task', <task spec>)."
    )

    # Python helpers injected via setup_code so AppWorld's code-exec sandbox has them.
    # archival_memory and conversation_search call out to amem_server (HTTP, port 9100).
    SETUP_CODE = (
        "import os, requests, json as _j\n"
        "_CORE_MEMORY = {}\n"
        "_CONV_HISTORY = []  # list of (role, content) tuples\n"
        "_AMEM_URL = os.environ.get('AMEM_SERVER_URL', 'http://localhost:9100')\n"
        "_SESSION_ID = os.environ.get('MEMGPT_SESSION_ID', 'memgpt_default')\n"
        "try:\n"
        "    requests.post(f'{_AMEM_URL}/init', json={'session_id': _SESSION_ID}, timeout=3)\n"
        "except Exception:\n"
        "    pass\n"
        "def core_memory_append(name, content):\n"
        "    _CORE_MEMORY.setdefault(name, []).append(content)\n"
        "    return f'[core_memory] appended to {name}'\n"
        "def core_memory_replace(name, old_content, new_content):\n"
        "    items = _CORE_MEMORY.get(name, [])\n"
        "    for i, v in enumerate(items):\n"
        "        if old_content in v:\n"
        "            items[i] = v.replace(old_content, new_content)\n"
        "            return f'[core_memory] replaced in {name}'\n"
        "    items.append(new_content)\n"
        "    _CORE_MEMORY[name] = items\n"
        "    return f'[core_memory] not found in {name}; appended'\n"
        "def core_memory_show():\n"
        "    print('[core_memory]', _j.dumps(_CORE_MEMORY, ensure_ascii=False))\n"
        "def archival_memory_insert(content):\n"
        "    try:\n"
        "        r = requests.post(f'{_AMEM_URL}/add', json={'session_id': _SESSION_ID, 'content': content, 'category': 'archival', 'tags': []}, timeout=5)\n"
        "        return f\"[archival] inserted: {r.json().get('note_id', 'err')}\"\n"
        "    except Exception as e:\n"
        "        return f'[archival] error: {e}'\n"
        "def archival_memory_search(query, k=3):\n"
        "    try:\n"
        "        r = requests.post(f'{_AMEM_URL}/search', json={'session_id': _SESSION_ID, 'query': query, 'k': k}, timeout=5)\n"
        "        hits = r.json().get('hits', [])\n"
        "        return '\\n'.join(f\"[{h.get('score', 0):.2f}] {h.get('content', '')[:150]}\" for h in hits[:k]) or '[archival] no results'\n"
        "    except Exception as e:\n"
        "        return f'[archival] error: {e}'\n"
        "def conversation_search(query, k=5):\n"
        "    matches = [(r, c) for r, c in _CONV_HISTORY if query.lower() in c.lower()][:k]\n"
        "    return '\\n'.join(f'[{r}] {c[:200]}' for r, c in matches) or '[conv_search] no matches'\n"
    )

    def run_episode(self, adapter, task, intervention, client):
        from interventionbench.models.base import ChatMessage
        import os, uuid
        # Set per-task session id for amem_server isolation
        task_id = getattr(task, 'task_id', None) or str(uuid.uuid4())[:8]
        os.environ['MEMGPT_SESSION_ID'] = f'memgpt_{task_id}'
        prefix = [ChatMessage(role="system", content=self.CORE_MEMORY_INIT_SYSTEM)]
        return adapter.run_a_to_b(
            task,
            intervention,
            client, timing="late",
            strategy=MemGPTStrategy(),
            prefix_messages=prefix,
            tools=MEMGPT_TOOLS,
            setup_code=self.SETUP_CODE,
        )


class AdaPlannerRunner(BaselineRunner):
    """AdaPlanner (Sun 2023). Initial plan + intervention-time refine + optional
    step-wise refine for ReAct domains.

    Adaptation: skill library omitted; step-wise refine via step_callback (only
    applied by ReAct adapters). 1-call adapters ignore step_callback.
    """

    name = "adaplanner"

    INITIAL_PLAN_SYSTEM = (
        "You are an agent. Before acting, write an explicit numbered plan "
        "(3-7 steps) for the task. Then carry out the plan step by step."
    )

    def __init__(self, refine_every: int = 3):
        self.refine_every = refine_every

    def _make_step_callback(self, client):
        """Return a callback the adapter invokes after each step.
        Signature: cb(step_index, recent_messages) -> {"action": "in_plan"|"refine", ...}
        """
        last_refine_idx = {"i": -1}

        def cb(step_index: int, recent_messages: list) -> dict:
            if step_index - last_refine_idx["i"] < self.refine_every:
                return {"action": "in_plan"}
            # Cheap on-plan check (single short LLM call)
            from interventionbench.models.base import ChatMessage
            prior = "\n".join(
                f"[{getattr(m, 'role', '?')}] {str(getattr(m, 'content', ''))[:300]}"
                for m in recent_messages[-6:]
            )
            try:
                resp = client.chat(
                    [
                        ChatMessage(role="system", content=(
                            "You check if an agent is still on its plan. "
                            "Answer ON_PLAN or NEEDS_REFINE in one word, no rationale."
                        )),
                        ChatMessage(role="user", content=f"Recent steps:\n{prior}\nVerdict:"),
                    ],
                    max_tokens=8,
                    temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                verdict = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()
            except Exception:
                verdict = "ON_PLAN"
            last_refine_idx["i"] = step_index
            if "REFINE" in verdict:
                return {"action": "refine"}
            return {"action": "in_plan"}

        return cb

    def run_episode(self, adapter, task, intervention, client):
        from interventionbench.models.base import ChatMessage
        prefix = [ChatMessage(role="system", content=self.INITIAL_PLAN_SYSTEM)]
        return adapter.run_a_to_b(
            task,
            intervention,
            client, timing="late",
            strategy=AdaPlannerStrategy(),
            prefix_messages=prefix,
            step_callback=self._make_step_callback(client),
        )


class AMemRunner(BaselineRunner):
    """A-MEM (Xu 2025) — Agentic Memory with Zettelkasten-inspired note-taking.

    Adaptation: A-MEM's full per-step note construction is too slow for our setup;
    we construct notes only at intervention time from prior_messages, then retrieve
    top-k semantically relevant memories for the new instruction.
    """

    name = "amem"

    INIT_SYSTEM = (
        "You operate with an A-MEM agentic memory system. Two Python functions are "
        "available in your code execution context (HTTP-backed; safe to ignore on "
        "errors):\n"
        "  notebook_add(content)  # save a note (idempotent)\n"
        "  notebook_search(query, k=3)  # returns top-k matching note contents\n"
        "Save key facts as you discover them; search before re-deriving anything. "
        "When the user provides an intervention, you'll also receive a "
        "[A-MEM RETRIEVED CONTEXT] block — use it to maintain continuity."
    )

    AMEM_HTTP_PORT = 9100

    SETUP_CODE_TEMPLATE = (
        "import urllib.request as _ur, json as _j\n"
        "_AMEM_URL = 'http://127.0.0.1:{port}'\n"
        "_AMEM_SID = 'sess_' + str(id(_ur))[-8:]\n"
        "def _amem_post(path, payload):\n"
        "    try:\n"
        "        req = _ur.Request(_AMEM_URL + path, data=_j.dumps(payload).encode(), "
        "headers={{'Content-Type': 'application/json'}})\n"
        "        with _ur.urlopen(req, timeout=10) as r:\n"
        "            return _j.loads(r.read().decode())\n"
        "    except Exception as e:\n"
        "        return {{'error': str(e)}}\n"
        "def notebook_add(content):\n"
        "    res = _amem_post('/add', {{'session_id': _AMEM_SID, 'content': str(content)}})\n"
        "    return f'[A-MEM] added: {{res.get(\"id\", res)}}'\n"
        "def notebook_search(query, k=3):\n"
        "    res = _amem_post('/search', {{'session_id': _AMEM_SID, 'query': str(query), 'k': k}})\n"
        "    if isinstance(res, dict) and res.get('results'):\n"
        "        return [m.get('content', '') for m in res['results']]\n"
        "    return []\n"
        "_amem_post('/init', {{'session_id': _AMEM_SID}})\n"
    )

    def run_episode(self, adapter, task, intervention, client):
        from interventionbench.models.base import ChatMessage
        from .amem_strategy import AMemStrategy, AMEM_SYSTEM_PROMPT, make_amem_step_callback
        import uuid
        # Per-task session id for amem_server isolation
        task_id = getattr(task, 'task_id', None) or str(uuid.uuid4())[:8]
        session_id = f"amem_{task_id}"
        # Use the new full A-MEM impl (autonomous note-taking via step_callback)
        prefix = [ChatMessage(role="system", content=AMEM_SYSTEM_PROMPT)]
        callback = make_amem_step_callback(session_id, k=3)
        setup_code = self.SETUP_CODE_TEMPLATE.format(port=self.AMEM_HTTP_PORT)
        return adapter.run_a_to_b(
            task,
            intervention,
            client, timing="late",
            strategy=AMemStrategy(),
            prefix_messages=prefix,
            setup_code=setup_code,
            step_callback=callback,
        )


# Registry filled in by later tasks (B4-B5).
RUNNERS: dict[str, type[BaselineRunner]] = {
    "vanilla": VanillaRunner,
    "plan_and_solve": PlanAndSolveRunner,
    "reflexion": ReflexionRunner,
    "memgpt": MemGPTRunner,
    "adaplanner": AdaPlannerRunner,
    "amem": AMemRunner,
}


def get_runner(name: str) -> BaselineRunner:
    if name not in RUNNERS:
        raise ValueError(f"Unknown runner: {name}. Available: {list(RUNNERS.keys())}")
    return RUNNERS[name]()
