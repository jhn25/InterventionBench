"""Phase 2 framework hooks — uniform interface for all 5 frameworks.

Phase 2 reuses Phase 1's replay-based workers (workers/{appworld,webarena}.py).
Only the post-inject ReAct loop differs per framework. This module provides a
small per-framework configuration that the Phase 1 workers consult.

A worker calls `framework_config(framework_name, condition, task_id, iv_idx)` to get:
  - prefix_messages: list[dict] system messages prepended to ReAct prompt
  - setup_code: Python str executed at trajectory start (AW only — code-exec sandbox)
  - n_trials: int (Reflexion=3, others=1)
  - per_step_hook: callable(step_idx, recent_steps, world) -> None  (AdaPlanner refine, AMem note-take)
  - reflect: callable(prior_steps, intervention_text) -> str  (Reflexion only)

Frameworks: vanilla, plan_and_solve, reflexion, memgpt, adaplanner, amem, mem0, hipporag.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional


# ============================================================================
# System prefixes (faithful to source baselines)
# ============================================================================

PS_PREFIX = (
    "Let's first understand the task, extract the key constraints and "
    "deliverables, and devise a complete plan. Then, let's carry out the "
    "plan step by step, paying attention to each constraint, and complete "
    "the task."
)

REFLEXION_SYSTEM_TEMPLATE = (
    "[Past reflections from earlier trials]:\n{reflections}\n\n"
    "Use these insights to do better this attempt."
)

REFLEXION_REFLECT_PROMPT_SYSTEM = (
    "You are an agent that just attempted a task and either failed or "
    "produced a suboptimal result. Reflect verbally on what went wrong, "
    "what specifically you should have done differently in light of the "
    "user's mid-task instruction, and what concrete approach you will take "
    "in the next attempt. Keep the reflection to 3-5 sentences."
)

# Goal-Reflexion: Add-type specialized retry with maximum information injection.
# The reflection includes A spec, B spec, per-goal scores, type definition, and
# concrete guidance — agent gets everything it could possibly use to succeed.
GOAL_REFLEXION_REFLECT_SYSTEM = (
    "You are an agent that just attempted a multi-goal task and failed. "
    "Your goal is to identify exactly which sub-goal you missed and how to "
    "complete BOTH in the next trial. Be concrete and actionable. Output 4-6 "
    "sentences ending with a 'PLAN:' line listing the explicit steps for next trial."
)

MEMGPT_SYSTEM = (
    "You operate with a 3-tier MemGPT memory hierarchy:\n"
    "  1. Core memory — always-visible scratchpad. Use core_memory_append(name, content), "
    "core_memory_replace(name, old, new), core_memory_show().\n"
    "  2. Archival memory — long-term vector store. Use archival_memory_insert(content), "
    "archival_memory_search(query, k).\n"
    "  3. Conversation history search. Use conversation_search(query, k).\n"
    "Use core_memory for facts you need every step. Use archival_memory for facts that "
    "may be retrieved later. Use conversation_search to recall past dialog. "
    "Begin by calling core_memory_append('task', <task spec>)."
)

ADAPLANNER_SYSTEM = (
    "You are an agent. Before acting, write an explicit numbered plan "
    "(3-7 steps) for the task. Then carry out the plan step by step. "
    "If observations diverge from your plan, **revise the plan inline** "
    "before continuing — explicitly note what changed and why."
)

AMEM_SYSTEM = (
    "You operate with A-MEM agentic memory: structured notes are auto-stored after each step, "
    "and relevant notes are retrieved semantically before each step. "
    "Pay attention to any [retrieved_notes] block in observations — it contains key facts "
    "from earlier in the task that may inform your next action. "
    "Build a coherent picture of the task across steps using the memory. "
    "Two helpers are available in your code execution context (HTTP-backed; safe to ignore on errors):\n"
    "  notebook_add(content)  # save a note\n"
    "  notebook_search(query, k=3)  # returns top-k matching notes\n"
    "Save key facts as you discover them; search before re-deriving anything."
)

HIPPORAG_SYSTEM = (
    "You operate with HippoRAG 2 graph-based memory (Jimenez Gutierrez et al., "
    "arXiv:2502.14802). After each step the system extracts entities and "
    "(subject, predicate, object) triples from your latest action+observation "
    "and indexes them in a per-task knowledge graph. Before each subsequent "
    "step, the system runs a Personalized PageRank query over the graph "
    "(seeded by entities mentioned in your most recent action) and prepends "
    "a [hipporag_retrieved] block to the next observation containing the "
    "top-k relevant past passages. "
    "Pay attention to that block: it surfaces multi-hop connections that "
    "simple keyword search would miss (e.g., earlier facts that share an "
    "entity with your current action through an intermediate node). Use it "
    "to maintain a coherent picture of the task, recall constraints from "
    "user instructions (including any mid-task interventions), and avoid "
    "re-deriving facts you have already discovered."
)

ARIGRAPH_SYSTEM = (
    "You operate with AriGraph (Anokhin et al., IJCAI 2025; arXiv:2407.04363) "
    "agentic knowledge-graph memory. After each step the system extracts "
    "(subject, relation, object) triples from your latest action+observation "
    "via an LLM, refines the new triples against the existing graph (replacing "
    "outdated facts when entities change state), and adds them to a per-task "
    "knowledge graph. Before each subsequent step, the system runs a 2-hop BFS "
    "from entities mentioned in your most recent action and prepends a "
    "[arigraph_retrieved] block listing the associated triples. "
    "Pay attention to that block: it surfaces what you have learned about "
    "entities relevant to your current focus, including constraints from "
    "user instructions (any mid-task interventions) and any prior facts you "
    "should not re-derive. The graph keeps semantic+episodic facts separate "
    "from your current attention, so each retrieval is targeted to your goal."
)

MEM0_SYSTEM = (
    "You operate with Mem0 long-term memory (Chhikara et al., arXiv:2504.19413): "
    "after each step the system runs an LLM-driven fact-extraction pass over your "
    "latest action+observation and stores the extracted entities in a vector + "
    "entity-graph store. Before each subsequent step, Mem0 performs hybrid "
    "retrieval (semantic + entity match) and prepends a [mem0_retrieved] block "
    "to the next observation containing the top relevant memories. "
    "Trust the [mem0_retrieved] block as a high-signal summary of what you have "
    "already learned earlier in this task — use it to avoid re-deriving facts, "
    "to recall constraints from the user's instructions (including any mid-task "
    "interventions), and to keep multi-step plans coherent."
)

MEMBRIDGE_SYSTEM = (
    "You operate with MemBridge memory: a Mem0-style fact store augmented with a "
    "composite-planning hook. When the user intervenes mid-task with a new instruction, "
    "the system reads your progress on the original task, classifies how the new "
    "instruction relates to the old one (REPLACE / MODIFY / ADD_PARALLEL / "
    "ADD_SEQUENTIAL / ADD_REUSE / DROP), and writes a [COMPOSITE PLAN] entry into memory. "
    "From that point on, the [mem0_retrieved] block will always begin with the "
    "[COMPOSITE PLAN] line. Treat that line as authoritative — it tells you whether to "
    "abandon the original task (REPLACE), continue it alongside the new one (ADD_PARALLEL "
    "or ADD_SEQUENTIAL), reuse a captured artifact (ADD_REUSE), substitute a parameter "
    "(MODIFY), or drop one sub-step (DROP). Do not abandon the original task unless "
    "the plan explicitly says REPLACE; do not skip the new instruction unless it says DROP."
)

# Setup code: defines memory helpers in the AppWorld code-exec sandbox.
# Used by MemGPT and AMem (both rely on amem_server on port 9100).
def _memgpt_setup_code(session_id: str, amem_port: int = 9100) -> str:
    return f"""
import os, urllib.request as _ur, json as _j
_AMEM_URL = 'http://127.0.0.1:{amem_port}'
_SESSION_ID = '{session_id}'
_CORE_MEMORY = {{}}
_CONV_HISTORY = []
def _amem_post(path, payload):
    try:
        req = _ur.Request(_AMEM_URL + path, data=_j.dumps(payload).encode(),
                          headers={{'Content-Type': 'application/json'}})
        with _ur.urlopen(req, timeout=10) as r:
            return _j.loads(r.read().decode())
    except Exception as e:
        return {{'error': str(e)}}
_amem_post('/init', {{'session_id': _SESSION_ID}})
def core_memory_append(name, content):
    _CORE_MEMORY.setdefault(name, []).append(content)
    return f'[core_memory] appended to {{name}}'
def core_memory_replace(name, old, new):
    items = _CORE_MEMORY.get(name, [])
    for i, v in enumerate(items):
        if old in v:
            items[i] = v.replace(old, new)
            return f'[core_memory] replaced in {{name}}'
    items.append(new); _CORE_MEMORY[name] = items
    return f'[core_memory] not found; appended'
def core_memory_show():
    print('[core_memory]', _j.dumps(_CORE_MEMORY, ensure_ascii=False))
def archival_memory_insert(content):
    res = _amem_post('/add', {{'session_id': _SESSION_ID, 'content': str(content), 'category': 'archival', 'tags': []}})
    return f"[archival] inserted: {{res.get('note_id', res)}}"
def archival_memory_search(query, k=3):
    res = _amem_post('/search', {{'session_id': _SESSION_ID, 'query': str(query), 'k': k}})
    hits = res.get('results') or res.get('hits') or []
    return '\\n'.join(f"[{{h.get('score', 0):.2f}}] {{h.get('content', '')[:150]}}" for h in hits[:k]) or '[archival] no results'
def conversation_search(query, k=5):
    matches = [(r, c) for r, c in _CONV_HISTORY if query.lower() in c.lower()][:k]
    return '\\n'.join(f'[{{r}}] {{c[:200]}}' for r, c in matches) or '[conv_search] no matches'
""".strip()


def _amem_setup_code(session_id: str, amem_port: int = 9100) -> str:
    return f"""
import urllib.request as _ur, json as _j
_AMEM_URL = 'http://127.0.0.1:{amem_port}'
_AMEM_SID = '{session_id}'
def _amem_post(path, payload):
    try:
        req = _ur.Request(_AMEM_URL + path, data=_j.dumps(payload).encode(),
                          headers={{'Content-Type': 'application/json'}})
        with _ur.urlopen(req, timeout=10) as r:
            return _j.loads(r.read().decode())
    except Exception as e:
        return {{'error': str(e)}}
def notebook_add(content):
    res = _amem_post('/add', {{'session_id': _AMEM_SID, 'content': str(content)}})
    return f'[A-MEM] added: {{res.get("note_id", res)}}'
def notebook_search(query, k=3):
    res = _amem_post('/search', {{'session_id': _AMEM_SID, 'query': str(query), 'k': k}})
    hits = res.get('results') or res.get('hits') or []
    return [m.get('content', '') for m in hits[:k]]
_amem_post('/init', {{'session_id': _AMEM_SID}})
""".strip()


# ============================================================================
# AMem step callback — autonomous note-taking + retrieval
# ============================================================================

def _make_adaplanner_step_callback() -> Callable:
    """AdaPlanner (Sun 2023) — plan tracking + in-plan refinement.

    Per paper: agent writes a numbered plan first; during execution, an
    out-of-plan refiner detects when an action diverges from plan and
    triggers revision. We approximate this with a stateful callback:

    1. **Plan extraction (first opportunity)**: parse the agent's first
       assistant content for an explicit numbered plan (^N. step text).
       If absent, return a "[ADAPLANNER] write a numbered plan first"
       prompt to force one on the next turn.
    2. **Plan tracking** (after plan exists): increment plan-step pointer
       per agent step. Periodically (every 3 steps) inject a
       "[PLAN CHECK] currently on step K: <text>" reminder so the agent
       must explicitly acknowledge plan position.
    3. **Refinement signal on intervention/divergence**: when the agent's
       latest action mentions plan-keywords ("update", "change") OR after
       the intervention is injected, prepend a "[PLAN REVISION REQUIRED]
       observation diverged — revise plan inline before continuing"
       message so the agent must update the plan.
    """
    import re as _re
    state = {"plan": None, "plan_step": 0, "saw_intervention": False, "revised": False}

    PLAN_RE = _re.compile(r'^\s*(\d+)\.\s*(.+?)$', _re.MULTILINE)

    def cb(step_idx, recent_steps, world=None):
        # Find latest assistant + observation.
        latest_asst = None
        latest_obs = None
        saw_inject = False
        for s in reversed(recent_steps[-8:]):
            if not isinstance(s, dict):
                # Coerce ChatMessage-like objects.
                role = getattr(s, "role", None)
                content = getattr(s, "content", "")
                s = {"role": role, "content": content,
                     "metadata": getattr(s, "metadata", None) or {}}
            md = s.get("metadata") or {}
            if md.get("is_intervention"):
                saw_inject = True
            r = s.get("role")
            content = str(s.get("content", ""))
            # WA uses role="action" instead of "assistant"; treat both as agent output.
            if r in ("assistant", "action") and latest_asst is None:
                latest_asst = content
            elif r == "observation" and latest_obs is None:
                latest_obs = content
            if latest_asst is not None and latest_obs is not None:
                break

        # 1. Plan extraction (first turn that has assistant content).
        if state["plan"] is None and latest_asst:
            matches = PLAN_RE.findall(latest_asst)
            if len(matches) >= 3:
                state["plan"] = [(int(n), t.strip()) for n, t in matches]
                return f"[ADAPLANNER] Plan parsed ({len(matches)} steps). Now execute step-by-step. After each observation, briefly confirm whether the step succeeded; if not, revise the plan inline."
            else:
                # No plan yet — request one.
                return "[ADAPLANNER] No numbered plan detected. Before your NEXT action, write an explicit numbered plan (3-7 steps) for the task. Then execute it."

        # 2. Intervention detected → force plan revision.
        if saw_inject and not state["revised"]:
            state["revised"] = True
            state["saw_intervention"] = True
            return "[PLAN REVISION REQUIRED] User just injected a new instruction mid-task. REVISE your numbered plan now: list the updated steps explicitly before taking any further action."

        # 3. Periodic plan progress check (every 3 steps).
        if state["plan"] and (step_idx % 3 == 0):
            cur = state["plan_step"]
            if cur < len(state["plan"]):
                _, plan_text = state["plan"][cur]
                state["plan_step"] += 1
                return f"[PLAN CHECK] Currently on plan step {cur+1}: {plan_text[:120]}. If observation diverged, revise the plan."

        return None

    return cb


def _make_amem_step_callback(session_id: str, k: int = 3) -> Callable:
    """After each agent step: post recent action+observation as note, retrieve top-k.

    Returns a callable cb(step_idx, recent_messages, world) -> str | None.
    The returned string (if non-None) is prepended to the next observation as
    a [retrieved_notes] block. The callback is best-effort — exceptions silenced.
    """
    import urllib.request as _ur
    import json as _j

    AMEM_URL = "http://127.0.0.1:9100"

    def _post(path: str, payload: dict, timeout: float = 5.0) -> dict:
        try:
            req = _ur.Request(
                AMEM_URL + path,
                data=_j.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with _ur.urlopen(req, timeout=timeout) as r:
                return _j.loads(r.read().decode())
        except Exception:
            return {}

    # Init session once.
    _post("/init", {"session_id": session_id})

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        # Find most recent assistant + observation.
        # Coerce ChatMessage / non-dict items to dict format (WA / τ-bench
        # may pass typed message objects rather than the dict format that
        # AppWorld worker uses).
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None)
                content = getattr(s, "content", "")
                s = {"role": role, "content": content}
            r = s.get("role")
            content = str(s.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:400]
            elif r == "observation" and not last_obs:
                last_obs = content[:400]
            if last_action and last_obs:
                break
        if not (last_action or last_obs):
            return None
        note = " | ".join(p for p in [
            f"Action: {last_action}" if last_action else "",
            f"Obs: {last_obs}" if last_obs else "",
        ] if p)
        _post("/add", {
            "session_id": session_id,
            "content": note,
            "category": f"step_{step_idx}",
            "tags": ["agent_step"],
        })
        # Retrieve top-k for next step's context.
        query = (last_action[:200] or "task progress")
        res = _post("/search", {"session_id": session_id, "query": query, "k": k})
        hits = res.get("results") or res.get("hits") or []
        if not hits:
            return None
        lines = ["[retrieved_notes]"]
        for h in hits[:k]:
            lines.append(f"  - ({h.get('score', 0):.2f}) {h.get('content', '')[:180]}")
        return "\n".join(lines)

    return cb


# ============================================================================
# Mem0 step callback — paper-faithful (arXiv:2504.19413)
# ============================================================================

# Module-level cache: Memory() init is heavy (loads embedder client, opens
# qdrant collection). Reuse one instance per session_id across steps.
_MEM0_INSTANCES: dict = {}
# Sentinel for sessions where init failed — avoids retrying repeatedly.
_MEM0_FAILED: set = set()


def _build_mem0_instance(session_id: str):
    """Lazily construct a Mem0 Memory() for `session_id`.

    Strategy (paper-faithful + cost-conscious):
      * LLM (extraction): local vLLM via OpenAI-compatible API (port 8220 by
        default, model from STEER_MODEL_NS). vLLM-backed extraction has zero
        marginal cost and uses the same model the agent uses.
      * Embeddings: OpenAI text-embedding-3-small (~$0.02 / 1M tokens). Local
        vLLM does not expose /v1/embeddings, so we fall back to the cheap
        OpenAI embedder.
      * Vector store: per-session Qdrant local path under /tmp/interventionbench_mem0
        to prevent cross-process file-lock contention with the default
        /tmp/qdrant path.

    Returns the Memory instance, or None on init failure (so the callback can
    silently no-op).
    """
    if session_id in _MEM0_INSTANCES:
        return _MEM0_INSTANCES[session_id]
    # Disable telemetry: it spins up a singleton vector store at
    # ~/.mem0/migrations_qdrant which serializes across all forked workers.
    # Also pin per-process MEM0_DIR so history.db doesn't lock.
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    if not os.environ.get("MEM0_DIR"):
        os.environ["MEM0_DIR"] = f"/tmp/interventionbench_mem0_dir/pid_{os.getpid()}"
        os.makedirs(os.environ["MEM0_DIR"], exist_ok=True)
    if session_id in _MEM0_FAILED:
        return None
    try:
        from mem0 import Memory  # type: ignore
    except Exception:
        _MEM0_FAILED.add(session_id)
        return None

    # Mem0 extraction LLM choice — paper-faithful and reliable under load:
    # vLLM (Qwen3-27B) frequently returns malformed JSON for the entity-extraction
    # prompt under multiprocessing fork contention (~ all extractions fail, 0
    # notes stored). OpenAI gpt-4o-mini gives reliable JSON and the official
    # Mem0 paper uses gpt-4o-class models. Cost ~$5-10 across 3000 cells.
    use_openai_for_extract = os.environ.get("MEM0_USE_OPENAI", "1") == "1"
    if use_openai_for_extract:
        # OpenAI gpt-4o-mini for extraction (default).
        pass  # config set below
    base_port = int(os.environ.get("BASE_PORT", "8220"))
    vllm_url = os.environ.get("MEM0_VLLM_BASE_URL") or f"http://localhost:{base_port}/v1"
    vllm_model = (
        os.environ.get("MEM0_LLM_MODEL")
        or os.environ.get("STEER_MODEL_NS")
        or os.environ.get("MODEL_ID")
        or "Qwen3.6-27B-FP8"
    )

    # Per-session qdrant path. Use a hash to keep the path short and
    # filesystem-safe; collection_name is unique too so the local Qdrant
    # client doesn't cross-contaminate sessions.
    import hashlib as _hashlib
    sid_hash = _hashlib.md5(session_id.encode()).hexdigest()[:12]
    qdrant_path = f"/tmp/interventionbench_mem0/{sid_hash}"
    collection = f"mem0_{sid_hash}"

    if use_openai_for_extract:
        llm_cfg = {
            "provider": "openai",
            "config": {
                "model": os.environ.get("MEM0_OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        }
    else:
        llm_cfg = {
            "provider": "vllm",
            "config": {
                "model": vllm_model,
                "vllm_base_url": vllm_url,
                "api_key": os.environ.get("VLLM_API_KEY", "vllm-api-key"),
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        }
    cfg = {
        "llm": llm_cfg,
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": os.environ.get("OPENAI_API_KEY"),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "embedding_model_dims": 1536,
                # Per-cell local-path Qdrant. Telemetry singleton at
                # ~/.mem0/migrations_qdrant is the source of cross-worker
                # contention; we disable it via MEM0_TELEMETRY=false above.
                "path": qdrant_path,
                "on_disk": False,
            },
        },
    }
    try:
        mem = Memory.from_config(cfg)
    except Exception as exc:
        # Surface init failure so we can debug fork issues. Stderr only the
        # first time per worker process to avoid log spam.
        if not _MEM0_FAILED:
            import sys as _sys
            print(f"[MEM0_INIT_FAIL] session={session_id} err={type(exc).__name__}: {exc}",
                  file=_sys.stderr, flush=True)
        _MEM0_FAILED.add(session_id)
        return None
    _MEM0_INSTANCES[session_id] = mem
    return mem


def _make_mem0_step_callback(session_id: str, k: int = 3) -> Callable:
    """Mem0 (Chhikara et al., 2025) — autonomous fact-extract + hybrid retrieve.

    After each agent step:
      1. Pass latest action+observation to `Memory.add(messages, user_id=session_id)`.
         Mem0's internal LLM extracts entity-grounded facts and stores them in
         the vector store + entity graph.
      2. Issue `Memory.search(query, filters={'user_id': session_id}, top_k=k)`
         and format the hits as a `[mem0_retrieved]` block, which the worker
         prepends to the NEXT observation.

    The whole callback is best-effort: any exception (init failure, vLLM down,
    OpenAI quota) silently degrades to None so the cell still completes.
    """
    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        # Find most recent assistant + observation. Coerce ChatMessage-like
        # objects to dict (WA / τ-bench may pass typed messages).
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None)
                content = getattr(s, "content", "")
                s = {"role": role, "content": content}
            r = s.get("role")
            content = str(s.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if not (last_action or last_obs):
            return None

        # Mem0.add accepts either a string or a list of {role,content} dicts.
        # Use the list form so the extraction LLM can attribute each fact to
        # the assistant's action vs. the environment's observation.
        msgs = []
        if last_action:
            msgs.append({"role": "assistant", "content": last_action})
        if last_obs:
            msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
        # Paper-faithful add: try LLM-extracted facts (infer=True) first.
        # If extraction yields nothing (short msg, weird format) fall back to
        # raw-text store (infer=False) so search always has something to hit.
        added = False
        try:
            res = mem.add(
                msgs,
                user_id=session_id,
                metadata={"step": step_idx, "framework": "mem0"},
                infer=True,
            )
            # mem0 returns {'results': [...]} or list; non-empty = facts stored.
            if isinstance(res, dict):
                added = bool(res.get("results"))
            elif isinstance(res, list):
                added = bool(res)
        except Exception as exc:
            # Surface first add failure per worker for debug; subsequent silent.
            global _MEM0_ADD_FAIL_LOGGED
            if not globals().get("_MEM0_ADD_FAIL_LOGGED"):
                import sys as _sys
                print(f"[MEM0_ADD_FAIL] session={session_id} err={type(exc).__name__}: {exc}",
                      file=_sys.stderr, flush=True)
                globals()["_MEM0_ADD_FAIL_LOGGED"] = True

        # Fallback: if LLM extraction returned no facts (typical for short
        # action/obs pairs), store the raw text so search always has hits.
        if not added:
            try:
                mem.add(
                    msgs,
                    user_id=session_id,
                    metadata={"step": step_idx, "framework": "mem0", "fallback": "raw"},
                    infer=False,
                )
            except Exception:
                pass

        # Hybrid retrieval for next-step context.
        query = (last_action[:300] or last_obs[:300] or "task progress")
        try:
            res = mem.search(
                query,
                filters={"user_id": session_id},
                top_k=k,
                threshold=0.0,
            )
        except Exception:
            return None
        hits = (res or {}).get("results") or []
        if not hits:
            return None
        lines = ["[mem0_retrieved]"]
        for h in hits[:k]:
            text = str(h.get("memory") or h.get("content") or "")[:200]
            score = h.get("score")
            if score is not None:
                lines.append(f"  - ({float(score):.2f}) {text}")
            else:
                lines.append(f"  - {text}")
        return "\n".join(lines)

    return cb


# ============================================================================
# MemBridge step callback — Mem0 + composite-plan hook
# ============================================================================
#
# Wraps mem0 with one extra LLM call at the intervention boundary. The hook:
#  1. Pulls A-progress facts from mem0
#  2. Asks the planner LLM to classify the A↔B relationship and write a 3-5
#     sentence composite plan
#  3. Stores the composite plan back into mem0 with metadata composite_plan=True
#  4. On every subsequent retrieval, the composite plan is always included as
#     the first result (semantic search fills the remaining k-1 slots)
#
# MemBridge: memory-mediated task composition for handling multi-task interventions.

_MEMBRIDGE_PLANNED: set = set()  # session_ids where planner has already run

MEMBRIDGE_PLANNER_PROMPT = """You are the task-composition planner for an autonomous agent.

Original task A: {a_spec}
Agent's progress on A so far: {a_progress}
User just intervened with: "{iv_text}"
Hint: intervention type is {iv_type}

Classify the A↔B relationship (pick exactly ONE):
  REPLACE — A is obsolete; restart with B
  MODIFY — same task, only a parameter changes (the action or target)
  ADD_PARALLEL — keep both goals, interleave freely
  ADD_SEQUENTIAL — finish A first, then do B as a follow-up
  ADD_REUSE — B consumes A's intermediate output (e.g., A produced a value, B saves it)
  DROP — drop one sub-step of A; keep the rest

Then write a concrete 3-5 sentence plan covering:
  - What A-state must be preserved (if any)
  - What B-actions to take
  - Order of execution

Output STRICT JSON (no markdown):
{{"relationship": "<one of six>", "rationale": "<1 sentence>", "plan": "<3-5 sentence plan>"}}"""


def _membridge_make_planner_client():
    """Build a minimal OpenAI-compatible client for the planning LLM call.
    Reuses BASE_PORT / MODEL_ID / API_BASE_URL env from the worker."""
    import httpx
    base = os.environ.get("API_BASE_URL")
    if not base:
        base_port = int(os.environ.get("BASE_PORT", "8224"))
        base = f"http://localhost:{base_port}/v1"
    api_key = os.environ.get("OPENAI_API_KEY", "dummy-key")
    return httpx.Client(timeout=120.0, headers={"Authorization": f"Bearer {api_key}"}), base.rstrip("/")


def _membridge_call_planner(prompt: str, model: str | None = None) -> str:
    """One-shot chat call to the planner LLM. Returns raw text.

    Handles Qwen3 thinking mode: when enable_thinking is on, content is null
    and the answer is in the `reasoning` field. We disable thinking to get the
    answer in content; if content is still null, fall back to reasoning.
    """
    client, base = _membridge_make_planner_client()
    model = model or os.environ.get("STEER_MODEL_NS") or os.environ.get("MODEL_ID") or "Qwen3.6-27B-FP8"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    # Disable Qwen3 thinking mode so content is populated (not reasoning).
    if "Qwen3" in (model or ""):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        r = client.post(f"{base}/chat/completions", json=payload)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        text = msg.get("content")
        if not text:
            # Qwen3 fallback: reasoning field carries the answer when thinking
            # mode wasn't disabled or the model still emitted via reasoning.
            text = msg.get("reasoning") or ""
        return text or ""
    except Exception as exc:
        return f'{{"relationship": "ADD_PARALLEL", "rationale": "planner failed: {type(exc).__name__}", "plan": "Continue the original task and incorporate the new instruction in parallel."}}'
    finally:
        try: client.close()
        except: pass


def _membridge_extract_intervention(recent_steps: list) -> tuple[str | None, str | None]:
    """Scan all provided steps for the inject message; return (iv_text, iv_type) or (None, None).
    Looks for the worker's inject markers: 'User says:' (AW) or '[Mid-task user intervention]' (WA).

    Searches the entire recent_steps window (not just the tail) because by the time
    the callback fires, the inject step may be many steps back. Worker hooks pass
    a slice (e.g. steps[-8:] in AW) but we also pattern-match by prefix as a fallback."""
    # First pass: look for metadata flag (anywhere in window)
    for s in reversed(recent_steps):
        if not isinstance(s, dict):
            role = getattr(s, "role", None)
            content = getattr(s, "content", "")
            md = None
        else:
            role = s.get("role")
            content = str(s.get("content", ""))
            md = s.get("metadata")
        if md and (md.get("is_intervention") or md.get("intervention")):
            text = content
            for prefix in ("User says: ", "[Mid-task user intervention]: "):
                if text.startswith(prefix):
                    text = text[len(prefix):]
            text = text.replace(". Please continue with this update.", "")
            if "[INTERVENTION HINT" in text:
                text = text.split("[INTERVENTION HINT")[0].strip()
            iv_type = (md.get("type") or md.get("intervention_type")) if md else None
            return text, iv_type
    # Second pass: fallback by content prefix (in case metadata didn't carry over)
    for s in reversed(recent_steps):
        if not isinstance(s, dict): continue
        if s.get("role") != "user": continue
        content = str(s.get("content", ""))
        for prefix in ("User says: ", "[Mid-task user intervention]: "):
            if content.startswith(prefix):
                text = content[len(prefix):].replace(". Please continue with this update.", "")
                if "[INTERVENTION HINT" in text:
                    text = text.split("[INTERVENTION HINT")[0].strip()
                return text, None
    return None, None


def _make_membridge_step_callback(session_id: str, k: int = 3) -> Callable:
    """MemBridge — mem0 + composite-plan hook fired once per cell at intervention.

    Plain mem0 behavior pre-intervention; on first detection of an intervention
    step, runs the planner LLM and writes a [COMPOSITE PLAN] memory with
    composite_plan=True metadata. Retrieval includes that plan as the first
    result thereafter.
    """
    import json as _json
    import re as _re

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        # ---- 1. Pass through normal mem0 add (latest action+observation) ----
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role")
            content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                res = mem.add(msgs, user_id=session_id,
                              metadata={"step": step_idx, "framework": "membridge"},
                              infer=True)
                added = bool((res or {}).get("results")) if isinstance(res, dict) else bool(res)
                if not added:
                    mem.add(msgs, user_id=session_id,
                            metadata={"step": step_idx, "framework": "membridge", "fallback": "raw"},
                            infer=False)
            except Exception: pass

        # ---- 2. Composite-planning hook (fires once per cell at intervention) ----
        if session_id not in _MEMBRIDGE_PLANNED:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            import sys as _sys
            print(f"[MEMBRIDGE] step={step_idx} session={session_id[:30]} iv_found={iv_text is not None}",
                  file=_sys.stderr, flush=True)
            if iv_text:
                # Pull A-spec from very first system message in recent_steps
                a_spec = ""
                for s in recent_steps:
                    if not isinstance(s, dict):
                        role = getattr(s, "role", None); content = getattr(s, "content", "")
                    else:
                        role = s.get("role"); content = str(s.get("content", ""))
                    if role == "system":
                        a_spec = content[:800]; break
                # A-progress summary: top-5 facts from mem0
                a_progress = ""
                try:
                    res = mem.search("primary task progress completed actions",
                                     filters={"user_id": session_id}, top_k=5, threshold=0.0)
                    hits = (res or {}).get("results") or []
                    a_progress = "\n".join(f"- {h.get('memory') or h.get('content', '')[:120]}" for h in hits[:5])
                except Exception: pass

                prompt = MEMBRIDGE_PLANNER_PROMPT.format(
                    a_spec=a_spec[:500],
                    a_progress=a_progress[:600] or "(no progress recorded yet)",
                    iv_text=iv_text[:300],
                    iv_type=iv_type or "(unknown)",
                )
                raw = _membridge_call_planner(prompt)
                # Robust JSON parse
                try:
                    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
                    plan_json = _json.loads(m.group(0)) if m else {}
                except Exception:
                    plan_json = {}
                relationship = plan_json.get("relationship") or "ADD_PARALLEL"
                plan_text = plan_json.get("plan") or "Continue the original task; incorporate the new instruction in parallel."

                composite = f"[COMPOSITE PLAN — relationship={relationship}]\n{plan_text}"
                # Always use infer=False — mem0's LLM extraction strips our
                # composite_plan metadata when infer=True, so we can't filter
                # the plan back out by metadata. infer=False stores the raw text
                # verbatim with the metadata intact.
                try:
                    mem.add(
                        [{"role": "user", "content": composite}],
                        user_id=session_id,
                        metadata={"composite_plan": True, "priority": "always",
                                  "step": step_idx, "iv_type": iv_type,
                                  "relationship": relationship},
                        infer=False,
                    )
                except Exception: pass
                _MEMBRIDGE_PLANNED.add(session_id)

        # ---- 3. Retrieval — composite plan (if any) + semantic top (k-1) ----
        query = (last_action[:300] or last_obs[:300] or "task progress")
        composite_hits = []
        try:
            all_mems = mem.get_all(filters={"user_id": session_id})
            n_all = len((all_mems or {}).get("results", []))
            for m in (all_mems or {}).get("results", []):
                md = m.get("metadata") or {}
                if md.get("composite_plan"):
                    composite_hits.append(m)
            composite_hits = composite_hits[-1:]  # latest only
            if session_id in _MEMBRIDGE_PLANNED and not composite_hits:
                import sys as _sys
                print(f"[MEMBRIDGE] WARN: planner fired but get_all returned {n_all} mems, 0 composite",
                      file=_sys.stderr, flush=True)
        except Exception as _exc:
            import sys as _sys
            print(f"[MEMBRIDGE] get_all FAIL: {type(_exc).__name__}: {_exc}",
                  file=_sys.stderr, flush=True)

        semantic_hits = []
        try:
            res = mem.search(query, filters={"user_id": session_id},
                             top_k=max(1, k - len(composite_hits)), threshold=0.0)
            semantic_hits = (res or {}).get("results", [])
        except Exception: pass

        all_hits = composite_hits + semantic_hits
        if not all_hits: return None
        lines = ["[mem0_retrieved]"]
        for h in all_hits[:k]:
            text = str(h.get("memory") or h.get("content") or "")[:300]
            score = h.get("score")
            md = h.get("metadata") or {}
            tag = "PLAN" if md.get("composite_plan") else f"{float(score):.2f}" if score is not None else ""
            lines.append(f"  - ({tag}) {text}")
        return "\n".join(lines)

    return cb


# ============================================================================
# TypeBridge — mem0 + type-aware intervention bridging
# ============================================================================

# Cache: domain → dict[intervention_idx → type]. Loaded lazily.
_IV_TYPE_CACHE: dict[str, dict[int, str]] = {}


def _load_iv_type_map(domain: str) -> dict[int, str]:
    """Return {intervention_idx: type} for the domain, cached after first load."""
    if domain in _IV_TYPE_CACHE:
        return _IV_TYPE_CACHE[domain]
    from pathlib import Path
    # domain alias
    alias = {"aw": "appworld", "wa": "webarena", "os": "osworld"}
    d = alias.get(domain, domain)
    p = Path(f"steer-bench/{d}/interventions.jsonl")
    m: dict[int, str] = {}
    import json as _json
    if p.exists():
        # AW dataset doesn't have intervention_idx field — fallback to line index
        for i, ln in enumerate(p.open()):
            try:
                r = _json.loads(ln)
                idx = r.get("intervention_idx", i)  # fallback to line index
                t = r.get("type")
                if t:
                    m[int(idx)] = t
            except Exception:
                continue
    _IV_TYPE_CACHE[domain] = m
    return m


def _get_iv_type_for_cell(domain: str | None, iv_idx: int | None) -> str | None:
    if iv_idx is None or domain is None:
        return None
    return _load_iv_type_map(domain).get(int(iv_idx))

TYPEBRIDGE_SYSTEM = (
    "You have access to a mem0 retrieval store. Each step you will see a "
    "[mem0_retrieved] block summarizing past actions and observations. Use it "
    "to remember the original task across turns.\n\n"
    "CRITICAL — If the retrieved block contains a [TypeBridge ...] line, treat "
    "it as a TOP-PRIORITY directive from the user about how to reconcile the "
    "original task with the latest intervention. Follow the bridging "
    "instruction verbatim:\n"
    "  • [TypeBridge ADD] → keep doing the original task AND add the new part. "
    "Do not drop the original goal.\n"
    "  • [TypeBridge SWAP] → abandon the original task and switch to the new one.\n"
    "  • [TypeBridge DROP] → continue the original task but skip the specified part.\n"
    "  • [TypeBridge CANCEL] → stop all further actions.\n"
)

_TYPEBRIDGE_INJECTED: set[str] = set()


def _classify_iv_type(iv_type: str | None) -> str:
    """Map fine-grained type (e.g. VerbAdd) to coarse category (ADD/SWAP/DROP/CANCEL)."""
    if not iv_type:
        return "UNKNOWN"
    t = iv_type.lower()
    if "cancel" in t:
        return "CANCEL"
    if "add" in t:
        return "ADD"
    if "swap" in t:
        return "SWAP"
    if "drop" in t:
        return "DROP"
    return "UNKNOWN"


def _typebridge_template(iv_type: str | None, a_spec: str, iv_text: str) -> str:
    """Build the bridging directive based on intervention type."""
    cat = _classify_iv_type(iv_type)
    a_snip = a_spec[:300].strip()
    iv_snip = iv_text[:300].strip()
    if cat == "ADD":
        return (
            f"[TypeBridge ADD] The user is ADDING to your task. KEEP the original "
            f"goal: \"{a_snip}\". ALSO incorporate: \"{iv_snip}\". BOTH must succeed — "
            f"do not abandon the original."
        )
    if cat == "SWAP":
        return (
            f"[TypeBridge SWAP] The user is SWAPPING the task. STOP the original "
            f"goal (\"{a_snip}\"). FOCUS ONLY on: \"{iv_snip}\"."
        )
    if cat == "DROP":
        return (
            f"[TypeBridge DROP] The user wants to DROP a part of your task. "
            f"CONTINUE the original goal \"{a_snip}\" but EXCLUDE: \"{iv_snip}\"."
        )
    if cat == "CANCEL":
        return f"[TypeBridge CANCEL] The user has CANCELLED the task. Stop all further actions."
    return (
        f"[TypeBridge] Reconcile original task \"{a_snip}\" with user instruction \"{iv_snip}\"."
    )


def _make_typebridge_step_callback(session_id: str, k: int = 3,
                                    iv_type_hint: str | None = None) -> Callable:
    """TypeBridge — same as mem0 callback but on first intervention detection
    prepends a [TypeBridge <CAT>] bridging directive to the retrieval block.

    The directive is injected ONCE per cell (tracked in _TYPEBRIDGE_INJECTED).
    All subsequent retrievals carry the directive via the mem0 store itself
    (we add it as a high-priority memory).

    iv_type_hint is bound at framework_config time (looked up from dataset by
    intervention_idx). Used as fallback when the recent_steps scan can't recover
    the type from inject-message metadata.
    """
    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        # 1. Standard mem0 add (latest action+observation)
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role")
            content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                res = mem.add(msgs, user_id=session_id,
                              metadata={"step": step_idx, "framework": "typebridge"},
                              infer=True)
                added = bool((res or {}).get("results")) if isinstance(res, dict) else bool(res)
                if not added:
                    mem.add(msgs, user_id=session_id,
                            metadata={"step": step_idx, "framework": "typebridge", "fallback": "raw"},
                            infer=False)
            except Exception:
                pass

        # 2. Intervention detection (fires once per cell)
        if session_id not in _TYPEBRIDGE_INJECTED:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            # Fallback to hint from framework_config (dataset lookup by iv_idx)
            if iv_type is None and iv_type_hint:
                iv_type = iv_type_hint
            if iv_text:
                # Pull A-spec from first system message
                a_spec = ""
                for s in recent_steps:
                    if not isinstance(s, dict):
                        role = getattr(s, "role", None); content = getattr(s, "content", "")
                    else:
                        role = s.get("role"); content = str(s.get("content", ""))
                    if role == "system":
                        a_spec = content[:500]; break
                directive = _typebridge_template(iv_type, a_spec, iv_text)
                cat = _classify_iv_type(iv_type)
                # Store directive as high-priority memory (verbatim, infer=False)
                try:
                    mem.add(
                        [{"role": "user", "content": directive}],
                        user_id=session_id,
                        metadata={"typebridge_directive": True, "priority": "always",
                                  "step": step_idx, "iv_type": iv_type, "iv_cat": cat},
                        infer=False,
                    )
                except Exception:
                    pass
                _TYPEBRIDGE_INJECTED.add(session_id)
                # Log to stderr for live verification
                import sys as _sys
                print(f"[TYPEBRIDGE] session={session_id[:30]} step={step_idx} "
                      f"iv_type={iv_type} cat={cat} injected ✓",
                      file=_sys.stderr, flush=True)

        # 3. Retrieval — directive (if any) + semantic top-(k-1)
        directive_hits = []
        try:
            all_mems = mem.get_all(filters={"user_id": session_id})
            for m in (all_mems or {}).get("results", []):
                md = m.get("metadata") or {}
                if md.get("typebridge_directive"):
                    directive_hits.append(m)
            directive_hits = directive_hits[-1:]
        except Exception:
            pass

        query = (last_action[:300] or last_obs[:300] or "task progress")
        semantic_hits = []
        try:
            res = mem.search(query, filters={"user_id": session_id},
                             top_k=max(1, k - len(directive_hits)), threshold=0.0)
            semantic_hits = (res or {}).get("results", [])
        except Exception:
            pass

        all_hits = directive_hits + semantic_hits
        if not all_hits:
            return None
        lines = ["[mem0_retrieved]"]
        for h in all_hits[:k]:
            text = str(h.get("memory") or h.get("content") or "")[:400]
            md = h.get("metadata") or {}
            tag = "BRIDGE" if md.get("typebridge_directive") else (
                f"{float(h.get('score')):.2f}" if h.get("score") is not None else "")
            lines.append(f"  - ({tag}) {text}")
        return "\n".join(lines)

    return cb


# ============================================================================
# GoalForce — V2c+D: Force dual-goal state verbalization every step + A-interleave
# Built on top of mem0. Tests Finding 3: "agents lose goal, not context".
# ============================================================================

GOALFORCE_SYSTEM = (
    "You operate with mem0 retrieval. Each step shows a [mem0_retrieved] block.\n\n"
    "GOAL-STATE PROTOCOL — Before every action after an intervention:\n"
    "  1. STATE the status of BOTH goals on a single line:\n"
    "       GOAL_STATE: A=[in_progress|done] | B=[in_progress|done]\n"
    "  2. Choose your next action to advance WHICHEVER is still in_progress.\n"
    "  3. NEVER call stop() while either A or B is still in_progress.\n"
    "  4. If [TypeBridge ADD] directive is present, BOTH must be done before stopping.\n"
    "  5. If [TypeBridge SWAP/DROP/CANCEL], follow that directive's rule instead.\n\n"
    "The mem0 block reminds you of past actions; the GOAL_STATE forces you to "
    "treat both A and B as active goals, not as context."
)

_GOALFORCE_INJECTED: set[str] = set()


def _make_goalforce_step_callback(session_id: str, k: int = 3,
                                   iv_type_hint: str | None = None) -> Callable:
    """GoalForce — V2c (verbalize) + D (A-interleave).

    Builds on mem0 retrieval. On every step post-intervention, prepends a
    GOAL_STATE_REQUIRED reminder to the next observation, in addition to the
    standard mem0 retrieval block. For ADD-type cells, additionally injects an
    'A_REMAINS_OPEN' nudge every 2 steps to force interleaved A-progress.
    """
    step_counter = {"count": 0, "intervened": False, "iv_type": None}

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        # 1. Standard mem0 add (mirror typebridge logic)
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role")
            content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                res = mem.add(msgs, user_id=session_id,
                              metadata={"step": step_idx, "framework": "goalforce"},
                              infer=True)
                added = bool((res or {}).get("results")) if isinstance(res, dict) else bool(res)
                if not added:
                    mem.add(msgs, user_id=session_id,
                            metadata={"step": step_idx, "framework": "goalforce", "fallback": "raw"},
                            infer=False)
            except Exception:
                pass

        # 2. First-intervention detection — bind iv_type + flag
        if not step_counter["intervened"]:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            if iv_type is None and iv_type_hint:
                iv_type = iv_type_hint
            if iv_text:
                step_counter["intervened"] = True
                step_counter["iv_type"] = iv_type or "UNKNOWN"
                step_counter["count"] = 0
                # Pull A-spec from first system message for the directive
                a_spec = ""
                for s in recent_steps:
                    if not isinstance(s, dict):
                        role = getattr(s, "role", None); content = getattr(s, "content", "")
                    else:
                        role = s.get("role"); content = str(s.get("content", ""))
                    if role == "system":
                        a_spec = content[:500]; break
                # Reuse typebridge directive (gives ADD/SWAP/DROP/CANCEL classification)
                directive = _typebridge_template(iv_type, a_spec, iv_text)
                try:
                    mem.add([{"role": "user", "content": directive}],
                            user_id=session_id,
                            metadata={"goalforce_directive": True, "priority": "always"},
                            infer=False)
                except Exception:
                    pass

        # 3. Standard mem0 retrieval
        retrieved_text = ""
        try:
            query = last_action[:200] or last_obs[:200] or "task progress"
            res = mem.search(query=query, user_id=session_id, limit=k)
            hits = res.get("results", []) if isinstance(res, dict) else (res or [])
            if hits:
                lines = ["[mem0_retrieved]"]
                for h in hits[:k]:
                    txt = h.get("memory", "") if isinstance(h, dict) else str(h)
                    if txt:
                        lines.append(f"- {txt[:200]}")
                retrieved_text = "\n".join(lines)
        except Exception:
            pass

        # 4. GOAL_STATE_REQUIRED prepend (force verbalization)
        if step_counter["intervened"]:
            step_counter["count"] += 1
            goal_force = "[GOAL_STATE_REQUIRED] Before your next action, output one line:\n"
            goal_force += "  GOAL_STATE: A=[in_progress|done] | B=[in_progress|done]\n"
            goal_force += "Then act to advance whichever is in_progress."
            # 5. ADD-type extra nudge every 2 steps
            if step_counter["iv_type"] and "add" in step_counter["iv_type"].lower():
                if step_counter["count"] % 2 == 0:
                    goal_force += ("\n[A_REMAINS_OPEN] You added a goal but the ORIGINAL "
                                   "task A is still incomplete. Take one step toward A "
                                   "before continuing B.")
            if retrieved_text:
                return goal_force + "\n\n" + retrieved_text
            return goal_force
        return retrieved_text or None

    return cb


# ============================================================================
# GoalForce-Recap (V2a only — pure reminder, no GOAL_STATE force, no interleave)
# Ablation: tests if A-recap alone (without forcing) helps.
# ============================================================================

GOALFORCE_RECAP_SYSTEM = (
    "You operate with mem0 retrieval. Each step shows a [mem0_retrieved] block.\n\n"
    "REMINDER PROTOCOL — After every intervention, the [A_RECAP] block "
    "summarizes what you have done for the original task A. Use it to:\n"
    "  1. Track whether A is still incomplete.\n"
    "  2. Avoid abandoning A when handling the new intervention B.\n"
    "  3. Resume A once you are done with B (if A was incomplete).\n\n"
    "Follow any [TypeBridge ADD/SWAP/DROP/CANCEL] directive in the block."
)


def _make_goalforce_recap_step_callback(session_id: str, k: int = 3,
                                          iv_type_hint: str | None = None) -> Callable:
    """V2a: pure A-recap reminder + mem0. No GOAL_STATE forcing."""
    state = {"intervened": False}

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None
        # Standard mem0 add
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role")
            content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                mem.add(msgs, user_id=session_id, metadata={"step": step_idx, "framework": "goalforce_recap"}, infer=True)
            except Exception:
                pass

        # Intervention detection — fires once + typebridge directive
        if not state["intervened"]:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            if iv_type is None and iv_type_hint:
                iv_type = iv_type_hint
            if iv_text:
                state["intervened"] = True
                a_spec = ""
                for s in recent_steps:
                    if not isinstance(s, dict):
                        role = getattr(s, "role", None); content = getattr(s, "content", "")
                    else:
                        role = s.get("role"); content = str(s.get("content", ""))
                    if role == "system":
                        a_spec = content[:500]; break
                directive = _typebridge_template(iv_type, a_spec, iv_text)
                try:
                    mem.add([{"role": "user", "content": directive}],
                            user_id=session_id,
                            metadata={"goalforce_recap_directive": True, "priority": "always"},
                            infer=False)
                except Exception:
                    pass

        # Standard mem0 retrieval
        retrieved_text = ""
        try:
            query = last_action[:200] or last_obs[:200] or "task progress"
            res = mem.search(query=query, user_id=session_id, limit=k)
            hits = res.get("results", []) if isinstance(res, dict) else (res or [])
            if hits:
                lines = ["[mem0_retrieved]"]
                for h in hits[:k]:
                    txt = h.get("memory", "") if isinstance(h, dict) else str(h)
                    if txt: lines.append(f"- {txt[:200]}")
                retrieved_text = "\n".join(lines)
        except Exception:
            pass

        # A-recap reminder (post-intervention): pull A-related actions from recent_steps
        if state["intervened"]:
            a_actions = []
            for s in recent_steps[:12]:  # first 12 steps = mostly pre-injection A actions
                if not isinstance(s, dict): continue
                if s.get("role") in ("assistant", "action"):
                    c = (s.get("content") or "")[:120]
                    if c: a_actions.append(c)
            if a_actions:
                recap = "[A_RECAP] What you did for the original task A:\n"
                for a in a_actions[:3]:
                    recap += f"- {a}\n"
                recap += "Keep A in mind."
                if retrieved_text:
                    return recap + "\n\n" + retrieved_text
                return recap
        return retrieved_text or None

    return cb


# ============================================================================
# GoalForce V3 — Checklist: explicit per-subgoal TODO list (granular > binary)
# Hypothesis: agents forget the LAST step of A after handling B.
# A binary state ("A=in_progress") is too coarse; an explicit per-step
# checklist exposes WHICH step of A is still missing.
# ============================================================================

GOALFORCE_CHECKLIST_SYSTEM = (
    "You operate with mem0 retrieval. Each step shows a [mem0_retrieved] block.\n\n"
    "CHECKLIST PROTOCOL — After an intervention, treat A and B as two parallel "
    "task chains. Before EVERY action, output a checklist block:\n\n"
    "  CHECKLIST:\n"
    "    [A] 1. <step desc> [done]\n"
    "        2. <step desc> [todo]\n"
    "    [B] 1. <step desc> [todo]\n\n"
    "Rules:\n"
    "  1. List concrete sub-steps (NOT 'do A'). Decompose into 2-5 items per goal.\n"
    "  2. Mark [done] only after concrete evidence in observation.\n"
    "  3. NEVER call stop() while ANY [A] item or ANY [B] item is [todo] (unless "
    "the directive says one goal is dropped/cancelled).\n"
    "  4. Choose your next action to advance the FIRST [todo] item.\n"
)


def _make_goalforce_checklist_step_callback(session_id: str, k: int = 3,
                                              iv_type_hint: str | None = None) -> Callable:
    """V3: Granular per-step TODO checklist for both A and B."""
    state = {"intervened": False, "iv_type": None, "count": 0}

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        # mem0 add
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role"); content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                mem.add(msgs, user_id=session_id,
                        metadata={"step": step_idx, "framework": "goalforce_checklist"},
                        infer=True)
            except Exception:
                pass

        # intervention detection
        if not state["intervened"]:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            if iv_type is None and iv_type_hint:
                iv_type = iv_type_hint
            if iv_text:
                state["intervened"] = True
                state["iv_type"] = iv_type or "UNKNOWN"
                state["count"] = 0
                a_spec = ""
                for s in recent_steps:
                    if not isinstance(s, dict):
                        role = getattr(s, "role", None); content = getattr(s, "content", "")
                    else:
                        role = s.get("role"); content = str(s.get("content", ""))
                    if role == "system":
                        a_spec = content[:500]; break
                directive = _typebridge_template(iv_type, a_spec, iv_text)
                try:
                    mem.add([{"role": "user", "content": directive}],
                            user_id=session_id,
                            metadata={"checklist_directive": True, "priority": "always"},
                            infer=False)
                except Exception:
                    pass

        # retrieval
        retrieved_text = ""
        try:
            query = last_action[:200] or last_obs[:200] or "task progress"
            res = mem.search(query=query, user_id=session_id, limit=k)
            hits = res.get("results", []) if isinstance(res, dict) else (res or [])
            if hits:
                lines = ["[mem0_retrieved]"]
                for h in hits[:k]:
                    txt = h.get("memory", "") if isinstance(h, dict) else str(h)
                    if txt: lines.append(f"- {txt[:200]}")
                retrieved_text = "\n".join(lines)
        except Exception:
            pass

        # checklist force
        if state["intervened"]:
            state["count"] += 1
            iv_t = (state["iv_type"] or "").lower()
            force_block = ("[CHECKLIST_REQUIRED] Before your next action, output:\n"
                           "  CHECKLIST:\n"
                           "    [A] <numbered sub-steps with [done] or [todo]>\n"
                           "    [B] <numbered sub-steps with [done] or [todo]>\n")
            if "add" in iv_t:
                force_block += ("Reminder (ADD): BOTH A and B must reach all-[done] "
                                "before stop. Verify the LAST step of A explicitly.\n")
            elif "swap" in iv_t or "drop" in iv_t:
                force_block += ("Reminder (SWAP/DROP): A is replaced; only B needs to "
                                "be all-[done]. Mark A items [dropped].\n")
            elif "cancel" in iv_t:
                force_block += ("Reminder (CANCEL): stop NOW; do not start B.\n")
            if retrieved_text:
                return force_block + "\n" + retrieved_text
            return force_block
        return retrieved_text or None

    return cb


# ============================================================================
# GoalForce V4 — Verify: pre-stop verification guard (targets premature stop)
# Hypothesis: in ADD type, agent finishes B then immediately stops without
# revisiting A. A late-task "Have you done A? Have you done B?" prompt forces
# explicit verification before terminating.
# ============================================================================

GOALFORCE_VERIFY_SYSTEM = (
    "You operate with mem0 retrieval. Each step shows a [mem0_retrieved] block.\n\n"
    "VERIFY PROTOCOL — When you are about to terminate (e.g., apis...complete_task, "
    "stop(), final answer), you MUST first output a VERIFY block:\n\n"
    "  VERIFY:\n"
    "    Original A: <restate>\n"
    "    Evidence A is done: <concrete>\n"
    "    Intervention B: <restate>\n"
    "    Evidence B is done: <concrete>\n\n"
    "If you cannot point to concrete evidence for either, DO NOT stop. Take an action "
    "to gather the missing evidence first. Premature stops will be rejected."
)


def _make_goalforce_verify_step_callback(session_id: str, k: int = 3,
                                          iv_type_hint: str | None = None) -> Callable:
    """V4: pre-stop verification reminder, escalating after step 12."""
    state = {"intervened": False, "iv_type": None, "count": 0}

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        # mem0 add
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role"); content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                mem.add(msgs, user_id=session_id,
                        metadata={"step": step_idx, "framework": "goalforce_verify"},
                        infer=True)
            except Exception:
                pass

        # intervention
        if not state["intervened"]:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            if iv_type is None and iv_type_hint:
                iv_type = iv_type_hint
            if iv_text:
                state["intervened"] = True
                state["iv_type"] = iv_type or "UNKNOWN"
                state["count"] = 0
                a_spec = ""
                for s in recent_steps:
                    if not isinstance(s, dict):
                        role = getattr(s, "role", None); content = getattr(s, "content", "")
                    else:
                        role = s.get("role"); content = str(s.get("content", ""))
                    if role == "system":
                        a_spec = content[:500]; break
                directive = _typebridge_template(iv_type, a_spec, iv_text)
                try:
                    mem.add([{"role": "user", "content": directive}],
                            user_id=session_id,
                            metadata={"verify_directive": True, "priority": "always"},
                            infer=False)
                except Exception:
                    pass

        # retrieval
        retrieved_text = ""
        try:
            query = last_action[:200] or last_obs[:200] or "task progress"
            res = mem.search(query=query, user_id=session_id, limit=k)
            hits = res.get("results", []) if isinstance(res, dict) else (res or [])
            if hits:
                lines = ["[mem0_retrieved]"]
                for h in hits[:k]:
                    txt = h.get("memory", "") if isinstance(h, dict) else str(h)
                    if txt: lines.append(f"- {txt[:200]}")
                retrieved_text = "\n".join(lines)
        except Exception:
            pass

        # Detect stop-attempt in last action
        about_to_stop = False
        if last_action:
            la = last_action.lower()
            for sig in ("complete_task", "stop(", "stop[", "stop ", "final_answer",
                        "final answer", "task_complete", "i am done", "the task is complete"):
                if sig in la:
                    about_to_stop = True; break

        # verify force
        if state["intervened"]:
            state["count"] += 1
            iv_t = (state["iv_type"] or "").lower()
            force_block = ""
            # ALWAYS remind from step 5+ post-intervention, ESCALATE if stop detected
            if about_to_stop:
                force_block = (
                    "[VERIFY_REQUIRED — STOP DETECTED] You attempted to terminate. "
                    "Before stopping is accepted, output:\n"
                    "  VERIFY:\n"
                    "    Evidence A is done: <concrete observation>\n"
                    "    Evidence B is done: <concrete observation>\n"
                    "If you cannot, take one more action to gather evidence. "
                    "Do NOT call stop again until VERIFY is complete.\n"
                )
            elif state["count"] >= 5:
                force_block = (
                    "[VERIFY_REMINDER] Before any future stop call, you must output a "
                    "VERIFY block with concrete evidence that BOTH A and B are done.\n"
                )
            if "add" in iv_t and force_block:
                force_block += ("(ADD type: BOTH A and B must be done to pass.)\n")
            if force_block:
                if retrieved_text:
                    return force_block + "\n" + retrieved_text
                return force_block
        return retrieved_text or None

    return cb


# ============================================================================
# GoalForce V5 — Replan: mandatory unified plan immediately after intervention
# Hypothesis: agents don't re-organize when intervention hits; they just react
# locally to B. Forcing one explicit "UNIFIED_PLAN" output right after the
# intervention message — listing remaining A steps + B steps in execution order
# — creates a re-orientation moment.
# ============================================================================

GOALFORCE_REPLAN_SYSTEM = (
    "You operate with mem0 retrieval. Each step shows a [mem0_retrieved] block.\n\n"
    "REPLAN PROTOCOL — Immediately after receiving any user intervention, you MUST "
    "output ONCE a UNIFIED_PLAN block before any further action:\n\n"
    "  UNIFIED_PLAN:\n"
    "    Original A remaining: <ordered list of A steps still needed>\n"
    "    New B steps: <ordered list of B steps>\n"
    "    Execution order: <interleave or sequence A and B steps; explain why>\n\n"
    "Rules:\n"
    "  1. For ADD: both A_remaining and B steps appear; choose interleave order.\n"
    "  2. For SWAP/DROP: A_remaining = [DROPPED]; only B steps appear.\n"
    "  3. For CANCEL: both lists = []; stop immediately.\n"
    "  4. Do NOT skip UNIFIED_PLAN — it is required even if obvious.\n"
)


def _make_goalforce_replan_step_callback(session_id: str, k: int = 3,
                                          iv_type_hint: str | None = None) -> Callable:
    """V5: one-shot UNIFIED_PLAN forced exactly once post-intervention."""
    state = {"intervened": False, "iv_type": None, "plan_demanded": False, "count": 0}

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        # mem0 add
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role"); content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                mem.add(msgs, user_id=session_id,
                        metadata={"step": step_idx, "framework": "goalforce_replan"},
                        infer=True)
            except Exception:
                pass

        # intervention
        if not state["intervened"]:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            if iv_type is None and iv_type_hint:
                iv_type = iv_type_hint
            if iv_text:
                state["intervened"] = True
                state["iv_type"] = iv_type or "UNKNOWN"
                state["plan_demanded"] = True  # demand once
                state["count"] = 0
                a_spec = ""
                for s in recent_steps:
                    if not isinstance(s, dict):
                        role = getattr(s, "role", None); content = getattr(s, "content", "")
                    else:
                        role = s.get("role"); content = str(s.get("content", ""))
                    if role == "system":
                        a_spec = content[:500]; break
                directive = _typebridge_template(iv_type, a_spec, iv_text)
                try:
                    mem.add([{"role": "user", "content": directive}],
                            user_id=session_id,
                            metadata={"replan_directive": True, "priority": "always"},
                            infer=False)
                except Exception:
                    pass

        # retrieval
        retrieved_text = ""
        try:
            query = last_action[:200] or last_obs[:200] or "task progress"
            res = mem.search(query=query, user_id=session_id, limit=k)
            hits = res.get("results", []) if isinstance(res, dict) else (res or [])
            if hits:
                lines = ["[mem0_retrieved]"]
                for h in hits[:k]:
                    txt = h.get("memory", "") if isinstance(h, dict) else str(h)
                    if txt: lines.append(f"- {txt[:200]}")
                retrieved_text = "\n".join(lines)
        except Exception:
            pass

        # Check if last action contains a UNIFIED_PLAN block — clear demand
        if state["plan_demanded"] and last_action:
            if "UNIFIED_PLAN" in last_action.upper() or "unified_plan" in last_action.lower():
                state["plan_demanded"] = False  # satisfied

        if state["intervened"]:
            state["count"] += 1
            if state["plan_demanded"]:
                # Hammer the demand until satisfied
                iv_t = (state["iv_type"] or "").lower()
                force_block = (
                    "[UNIFIED_PLAN_REQUIRED] You just received an intervention. "
                    "Before any action, output:\n"
                    "  UNIFIED_PLAN:\n"
                    "    Original A remaining: <list>\n"
                    "    New B steps: <list>\n"
                    "    Execution order: <ordered, explained>\n"
                )
                if "add" in iv_t:
                    force_block += "(ADD: interleave A_remaining + B.)\n"
                elif "swap" in iv_t or "drop" in iv_t:
                    force_block += "(SWAP/DROP: A_remaining = DROPPED.)\n"
                elif "cancel" in iv_t:
                    force_block += "(CANCEL: both empty; stop.)\n"
                if retrieved_text:
                    return force_block + "\n" + retrieved_text
                return force_block
        return retrieved_text or None

    return cb


# ============================================================================
# GoalForce V6 — Combo: Checklist (V3) + Verify (V4) merged
# Hypothesis: V3 maintains granular goal state; V4 catches premature stop.
# Combining them should hit both "missed last step of A" + "stopped too early".
# ============================================================================

GOALFORCE_COMBO_SYSTEM = (
    "You operate with mem0 retrieval. Each step shows a [mem0_retrieved] block.\n\n"
    "DUAL-FORCE PROTOCOL:\n\n"
    "1. CHECKLIST: Before EVERY action, output:\n"
    "     CHECKLIST:\n"
    "       [A] 1. <step> [done|todo]   2. ...\n"
    "       [B] 1. <step> [done|todo]   2. ...\n"
    "   Decompose into concrete sub-steps. Mark [done] only with observation evidence.\n\n"
    "2. VERIFY (on stop): When attempting to terminate (complete_task / stop / "
    "final answer), output BEFORE termination:\n"
    "     VERIFY:\n"
    "       Evidence A is done: <concrete>\n"
    "       Evidence B is done: <concrete>\n"
    "   If you cannot, do NOT stop — gather missing evidence first.\n\n"
    "Type rules — ADD: both A and B must be all-[done]. SWAP/DROP: A is dropped, "
    "only B needs all-[done]. CANCEL: stop immediately."
)


def _make_goalforce_combo_step_callback(session_id: str, k: int = 3,
                                          iv_type_hint: str | None = None) -> Callable:
    """V6: V3 checklist + V4 verify dual-force."""
    state = {"intervened": False, "iv_type": None, "count": 0}

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role"); content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                mem.add(msgs, user_id=session_id,
                        metadata={"step": step_idx, "framework": "goalforce_combo"},
                        infer=True)
            except Exception:
                pass

        if not state["intervened"]:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            if iv_type is None and iv_type_hint:
                iv_type = iv_type_hint
            if iv_text:
                state["intervened"] = True
                state["iv_type"] = iv_type or "UNKNOWN"
                state["count"] = 0
                a_spec = ""
                for s in recent_steps:
                    if not isinstance(s, dict):
                        role = getattr(s, "role", None); content = getattr(s, "content", "")
                    else:
                        role = s.get("role"); content = str(s.get("content", ""))
                    if role == "system":
                        a_spec = content[:500]; break
                directive = _typebridge_template(iv_type, a_spec, iv_text)
                try:
                    mem.add([{"role": "user", "content": directive}],
                            user_id=session_id,
                            metadata={"combo_directive": True, "priority": "always"},
                            infer=False)
                except Exception:
                    pass

        retrieved_text = ""
        try:
            query = last_action[:200] or last_obs[:200] or "task progress"
            res = mem.search(query=query, user_id=session_id, limit=k)
            hits = res.get("results", []) if isinstance(res, dict) else (res or [])
            if hits:
                lines = ["[mem0_retrieved]"]
                for h in hits[:k]:
                    txt = h.get("memory", "") if isinstance(h, dict) else str(h)
                    if txt: lines.append(f"- {txt[:200]}")
                retrieved_text = "\n".join(lines)
        except Exception:
            pass

        # Stop detection
        about_to_stop = False
        if last_action:
            la = last_action.lower()
            for sig in ("complete_task", "stop(", "stop[", "stop ", "final_answer",
                        "final answer", "task_complete", "the task is complete"):
                if sig in la:
                    about_to_stop = True; break

        if state["intervened"]:
            state["count"] += 1
            iv_t = (state["iv_type"] or "").lower()
            force_block = "[CHECKLIST_REQUIRED] Output CHECKLIST before next action.\n"
            if about_to_stop:
                force_block += (
                    "[VERIFY_REQUIRED — STOP DETECTED] You attempted to terminate. "
                    "Output VERIFY with concrete evidence for BOTH A and B. "
                    "If incomplete, do NOT stop — gather evidence.\n"
                )
            elif state["count"] >= 5:
                force_block += ("[VERIFY_REMINDER] When you next attempt stop, you must "
                                "output VERIFY block.\n")
            if "add" in iv_t:
                force_block += "(ADD: BOTH all-[done] required to pass.)\n"
            elif "swap" in iv_t or "drop" in iv_t:
                force_block += "(SWAP/DROP: A dropped; only B needs all-[done].)\n"
            elif "cancel" in iv_t:
                force_block += "(CANCEL: stop now, no B.)\n"
            if retrieved_text:
                return force_block + "\n" + retrieved_text
            return force_block
        return retrieved_text or None

    return cb


# ============================================================================
# GoalForce V7 — Quote: every step requires direct quotation of A's spec
# Hypothesis: agents lose A because it falls out of effective working memory
# under context pressure. Forcing them to QUOTE A's exact text every step
# keeps A persistent in the action prefix where attention is strongest.
# Distinct from recap (paraphrase) — exact quote forces literal re-attention.
# ============================================================================

GOALFORCE_QUOTE_SYSTEM = (
    "You operate with mem0 retrieval. Each step shows a [mem0_retrieved] block.\n\n"
    "QUOTE PROTOCOL — Before EVERY action after an intervention, you MUST quote "
    "the original task A verbatim on its own line:\n\n"
    "  A_QUOTE: \"<exact text of the original task as given>\"\n"
    "  STATUS: A=[in_progress|done|dropped] B=[in_progress|done]\n\n"
    "Rules:\n"
    "  1. Quote must be character-exact, not paraphrased.\n"
    "  2. NEVER stop while A=in_progress (unless directive says A is dropped).\n"
    "  3. The quote keeps A literally in your prompt context to prevent drift."
)


def _make_goalforce_quote_step_callback(session_id: str, k: int = 3,
                                          iv_type_hint: str | None = None) -> Callable:
    """V7: force exact quotation of A every step (persistence over paraphrase)."""
    state = {"intervened": False, "iv_type": None, "count": 0, "a_text": ""}

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        mem = _build_mem0_instance(session_id)
        if mem is None:
            return None

        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None); content = getattr(s, "content", "")
                s_dict = {"role": role, "content": content}
            else:
                s_dict = s
            r = s_dict.get("role"); content = str(s_dict.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break

        # Capture A's text from first user/system message (one-time)
        if not state["a_text"]:
            for s in recent_steps:
                if not isinstance(s, dict):
                    role = getattr(s, "role", None); content = getattr(s, "content", "")
                else:
                    role = s.get("role"); content = str(s.get("content", ""))
                if role == "user":
                    # AppWorld puts task as user message
                    if "task" in content.lower() or len(content) < 800:
                        state["a_text"] = content[:300]; break
                if role == "system" and len(content) < 800:
                    state["a_text"] = content[:300]; break

        if last_action or last_obs:
            msgs = []
            if last_action: msgs.append({"role": "assistant", "content": last_action})
            if last_obs: msgs.append({"role": "user", "content": f"[observation] {last_obs}"})
            try:
                mem.add(msgs, user_id=session_id,
                        metadata={"step": step_idx, "framework": "goalforce_quote"},
                        infer=True)
            except Exception:
                pass

        if not state["intervened"]:
            iv_text, iv_type = _membridge_extract_intervention(recent_steps)
            if iv_type is None and iv_type_hint:
                iv_type = iv_type_hint
            if iv_text:
                state["intervened"] = True
                state["iv_type"] = iv_type or "UNKNOWN"
                state["count"] = 0
                a_spec = state["a_text"]
                directive = _typebridge_template(iv_type, a_spec, iv_text)
                try:
                    mem.add([{"role": "user", "content": directive}],
                            user_id=session_id,
                            metadata={"quote_directive": True, "priority": "always"},
                            infer=False)
                except Exception:
                    pass

        retrieved_text = ""
        try:
            query = last_action[:200] or last_obs[:200] or "task progress"
            res = mem.search(query=query, user_id=session_id, limit=k)
            hits = res.get("results", []) if isinstance(res, dict) else (res or [])
            if hits:
                lines = ["[mem0_retrieved]"]
                for h in hits[:k]:
                    txt = h.get("memory", "") if isinstance(h, dict) else str(h)
                    if txt: lines.append(f"- {txt[:200]}")
                retrieved_text = "\n".join(lines)
        except Exception:
            pass

        if state["intervened"]:
            state["count"] += 1
            a_snippet = (state["a_text"] or "<A spec not captured>")[:250]
            iv_t = (state["iv_type"] or "").lower()
            force_block = (
                "[QUOTE_REQUIRED] Before your next action, output:\n"
                f"  A_QUOTE: \"{a_snippet}\"\n"
                "  STATUS: A=[in_progress|done|dropped] B=[in_progress|done]\n"
            )
            if "add" in iv_t:
                force_block += "(ADD: A must reach [done]; do not skip A's last step.)\n"
            elif "swap" in iv_t or "drop" in iv_t:
                force_block += "(SWAP/DROP: A=[dropped] is correct; finish B.)\n"
            elif "cancel" in iv_t:
                force_block += "(CANCEL: stop now.)\n"
            if retrieved_text:
                return force_block + "\n" + retrieved_text
            return force_block
        return retrieved_text or None

    return cb


# ============================================================================
# HippoRAG step callback — paper-faithful (arXiv:2502.14802)
# ============================================================================

def _make_hipporag_step_callback(
    session_id: str,
    k: int = 3,
    index_every: int = 2,
) -> Callable:
    """HippoRAG 2 — graph-based RAG via OpenIE + Personalized PageRank.

    After each agent step:
      1. POST the latest action+observation to /add. The hipporag_server
         buffers passages and runs OpenIE (NER + triple extraction via the
         local vLLM) once the buffer reaches BATCH_SIZE, building a per-cell
         knowledge graph.
      2. POST a search query to /search. The server flushes any pending
         passages, then runs HippoRAG 2 retrieval (fact rerank + PPR over the
         graph) and returns the top-k passages.
      3. Format the hits as a `[hipporag_retrieved]` block, which the worker
         prepends to the NEXT observation.

    Performance: indexing is the bottleneck (2 LLM calls per passage). We
    only call /add every `index_every` steps to keep per-cell wall time
    reasonable. Search is done every step.

    All exceptions silenced — best-effort; if the server is down or slow,
    the cell still completes (hooks return None).
    """
    import urllib.request as _ur
    import json as _j

    HR_URL = os.environ.get("HIPPORAG_URL", "http://127.0.0.1:9101")
    # Search includes (a) flush of any buffered docs (each takes 2 LLM calls
    # for OpenIE, ~30s) and (b) DSPy rerank LLM call (~30s) + PPR (fast).
    # With BATCH_SIZE=4 and index_every=2, a typical 8-step task indexes ~16
    # docs across 4 batches => ~3-4min total indexing wall time. Per-step
    # search incurs ~30s rerank only (graph already built).
    SEARCH_TIMEOUT = float(os.environ.get("HIPPORAG_SEARCH_TIMEOUT", "240"))
    ADD_TIMEOUT = float(os.environ.get("HIPPORAG_ADD_TIMEOUT", "240"))

    def _post(path: str, payload: dict, timeout: float) -> dict:
        try:
            req = _ur.Request(
                HR_URL + path,
                data=_j.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with _ur.urlopen(req, timeout=timeout) as r:
                return _j.loads(r.read().decode())
        except Exception:
            return {}

    # Init session once. (Best-effort; server creates lazily on first /add too.)
    _post("/init", {"session_id": session_id}, timeout=5.0)

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        # Find most recent assistant + observation. Coerce ChatMessage-like
        # objects to dict (WA / τ-bench may pass typed messages).
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None)
                content = getattr(s, "content", "")
                s = {"role": role, "content": content}
            r = s.get("role")
            content = str(s.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if not (last_action or last_obs):
            return None

        # Index passage every `index_every` steps to amortize OpenIE cost.
        if (step_idx % max(1, index_every)) == 0:
            passage = " ".join(p for p in [
                f"Action: {last_action}" if last_action else "",
                f"Observation: {last_obs}" if last_obs else "",
            ] if p)
            _post(
                "/add",
                {"session_id": session_id, "content": passage},
                timeout=ADD_TIMEOUT,
            )

        # Retrieve top-k via PPR for next step's context. Use the latest
        # action as the query (entities in the agent's intent drive PPR).
        query = (last_action[:300] or last_obs[:300] or "task progress")
        res = _post(
            "/search",
            {"session_id": session_id, "query": query, "k": k},
            timeout=SEARCH_TIMEOUT,
        )
        hits = res.get("hits") or []
        if not hits:
            return None
        lines = ["[hipporag_retrieved]"]
        for h in hits[:k]:
            text = str(h.get("content", ""))[:200]
            score = h.get("score")
            if score is not None:
                lines.append(f"  - ({float(score):.2f}) {text}")
            else:
                lines.append(f"  - {text}")
        return "\n".join(lines)

    return cb


# ============================================================================
# AriGraph step callback (Anokhin 2025, IJCAI; arXiv:2407.04363)
# ============================================================================
def _make_arigraph_step_callback(session_id: str, k: int = 5) -> Callable:
    """AriGraph — paper-faithful agentic knowledge graph.

    Per agent step:
      1. POST /add with current action+observation. The arigraph_server LLM
         extracts (subj, rel, obj) triples, refines them vs the existing
         per-session graph (replacing outdated facts when entities move/change),
         and adds them to the in-memory triplet set.
      2. POST /search with the latest action as the query. The server extracts
         entities from the query and runs a 2-hop BFS over the graph (paper
         Section 3.2: "associative retrieval"), returning top-k associated
         triple strings.
      3. Format the hits as a [arigraph_retrieved] block prepended to the next
         observation.

    All exceptions silenced — best-effort; if the server is down or LLM extract
    fails, the cell still completes with no marker (degrades to vanilla).
    """
    import urllib.request as _ur
    import json as _j

    AG_URL = os.environ.get("ARIGRAPH_URL", "http://127.0.0.1:9102")
    SEARCH_TIMEOUT = float(os.environ.get("ARIGRAPH_SEARCH_TIMEOUT", "30"))
    ADD_TIMEOUT = float(os.environ.get("ARIGRAPH_ADD_TIMEOUT", "60"))

    def _post(path: str, payload: dict, timeout: float) -> dict:
        try:
            req = _ur.Request(
                AG_URL + path,
                data=_j.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with _ur.urlopen(req, timeout=timeout) as r:
                return _j.loads(r.read().decode())
        except Exception:
            return {}

    _post("/init", {"session_id": session_id}, timeout=5.0)

    def cb(step_idx: int, recent_steps: list, world=None) -> Optional[str]:
        last_action, last_obs = "", ""
        for s in reversed(recent_steps[-6:]):
            if not isinstance(s, dict):
                role = getattr(s, "role", None)
                content = getattr(s, "content", "")
                s = {"role": role, "content": content}
            r = s.get("role")
            content = str(s.get("content", ""))
            if r in ("assistant", "action") and not last_action:
                last_action = content[:600]
            elif r == "observation" and not last_obs:
                last_obs = content[:600]
            if last_action and last_obs:
                break
        if not (last_action or last_obs):
            return None

        passage = " ".join(p for p in [
            f"Action: {last_action}" if last_action else "",
            f"Observation: {last_obs}" if last_obs else "",
        ] if p)
        _post("/add", {"session_id": session_id, "content": passage},
              timeout=ADD_TIMEOUT)

        query = (last_action[:300] or last_obs[:300] or "task progress")
        res = _post("/search",
                    {"session_id": session_id, "query": query, "k": k},
                    timeout=SEARCH_TIMEOUT)
        hits = res.get("hits") or []
        if not hits:
            return None
        lines = ["[arigraph_retrieved]"]
        for h in hits[:k]:
            text = str(h.get("content", ""))[:200]
            lines.append(f"  - {text}")
        return "\n".join(lines)

    return cb


# ============================================================================
# Public API
# ============================================================================

@dataclass
class FrameworkConfig:
    name: str
    prefix_messages: list[dict] = field(default_factory=list)
    setup_code: Optional[str] = None
    n_trials: int = 1
    per_step_hook: Optional[Callable] = None  # cb(step_idx, recent_steps, world) -> str|None
    reflect_system_prompt: Optional[str] = None  # for reflexion only


def framework_config(framework: str, condition: str, task_id: str,
                     iv_idx: Optional[int] = None,
                     domain: Optional[str] = None) -> FrameworkConfig:
    """Build per-cell framework configuration. Worker uses this to wire hooks."""
    fw = (framework or "vanilla").lower()
    cell_id = f"{task_id}_{iv_idx if iv_idx is not None else 'A'}_{condition}"

    if fw == "vanilla":
        return FrameworkConfig(name="vanilla")

    if fw == "plan_and_solve":
        return FrameworkConfig(
            name="plan_and_solve",
            prefix_messages=[{"role": "system", "content": PS_PREFIX}],
        )

    if fw == "reflexion":
        return FrameworkConfig(
            name="reflexion",
            n_trials=int(os.environ.get("REFLEXION_N_TRIALS", "3")),
            reflect_system_prompt=REFLEXION_REFLECT_PROMPT_SYSTEM,
        )

    if fw == "goal_reflexion":
        # Add-type specialized: 2 trials, reflection prompt enriched with
        # A spec, B spec, type info, and per-goal score breakdown.
        # Worker reads `goal_reflexion=True` flag and switches the reflection
        # prompt rendering to include the goal-decomposed info.
        return FrameworkConfig(
            name="goal_reflexion",
            n_trials=int(os.environ.get("GOAL_REFLEXION_N_TRIALS", "2")),
            reflect_system_prompt=GOAL_REFLEXION_REFLECT_SYSTEM,
        )

    if fw == "memgpt":
        session_id = f"memgpt_{cell_id}"
        return FrameworkConfig(
            name="memgpt",
            prefix_messages=[{"role": "system", "content": MEMGPT_SYSTEM}],
            setup_code=_memgpt_setup_code(session_id),
        )

    if fw == "adaplanner":
        return FrameworkConfig(
            name="adaplanner",
            prefix_messages=[{"role": "system", "content": ADAPLANNER_SYSTEM}],
            per_step_hook=_make_adaplanner_step_callback(),
        )

    if fw == "amem":
        session_id = f"amem_{cell_id}"
        cb = _make_amem_step_callback(session_id, k=3)
        return FrameworkConfig(
            name="amem",
            prefix_messages=[{"role": "system", "content": AMEM_SYSTEM}],
            setup_code=_amem_setup_code(session_id),
            per_step_hook=cb,
        )

    if fw == "mem0":
        session_id = f"mem0_{cell_id}"
        cb = _make_mem0_step_callback(session_id, k=int(os.environ.get("MEM0_K", "3")))
        return FrameworkConfig(
            name="mem0",
            prefix_messages=[{"role": "system", "content": MEM0_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "hipporag":
        session_id = f"hr_{cell_id}"
        cb = _make_hipporag_step_callback(
            session_id,
            k=int(os.environ.get("HIPPORAG_K", "3")),
            index_every=int(os.environ.get("HIPPORAG_INDEX_EVERY", "1")),
        )
        return FrameworkConfig(
            name="hipporag",
            prefix_messages=[{"role": "system", "content": HIPPORAG_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "arigraph":
        session_id = f"ag_{cell_id}"
        cb = _make_arigraph_step_callback(
            session_id,
            k=int(os.environ.get("ARIGRAPH_K", "5")),
        )
        return FrameworkConfig(
            name="arigraph",
            prefix_messages=[{"role": "system", "content": ARIGRAPH_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "membridge":
        session_id = f"mb_{cell_id}"
        cb = _make_membridge_step_callback(session_id, k=int(os.environ.get("MEM0_K", "3")))
        return FrameworkConfig(
            name="membridge",
            prefix_messages=[{"role": "system", "content": MEMBRIDGE_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "typebridge":
        session_id = f"tb_{cell_id}"
        iv_type_hint = _get_iv_type_for_cell(domain, iv_idx) if domain else None
        cb = _make_typebridge_step_callback(session_id, k=int(os.environ.get("MEM0_K", "3")),
                                             iv_type_hint=iv_type_hint)
        return FrameworkConfig(
            name="typebridge",
            prefix_messages=[{"role": "system", "content": TYPEBRIDGE_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "goalforce":
        session_id = f"gf_{cell_id}"
        iv_type_hint = _get_iv_type_for_cell(domain, iv_idx) if domain else None
        cb = _make_goalforce_step_callback(session_id, k=int(os.environ.get("MEM0_K", "3")),
                                            iv_type_hint=iv_type_hint)
        return FrameworkConfig(
            name="goalforce",
            prefix_messages=[{"role": "system", "content": GOALFORCE_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "goalforce_recap":
        # V2a only — A-recap reminder, no GOAL_STATE force, no interleave.
        # Light-touch variant. Tests pure reminder effect.
        session_id = f"gfr_{cell_id}"
        iv_type_hint = _get_iv_type_for_cell(domain, iv_idx) if domain else None
        cb = _make_goalforce_recap_step_callback(
            session_id, k=int(os.environ.get("MEM0_K", "3")),
            iv_type_hint=iv_type_hint
        )
        return FrameworkConfig(
            name="goalforce_recap",
            prefix_messages=[{"role": "system", "content": GOALFORCE_RECAP_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "goalforce_checklist":
        # V3 — per-subgoal TODO checklist (granular over binary GOAL_STATE).
        session_id = f"gfc_{cell_id}"
        iv_type_hint = _get_iv_type_for_cell(domain, iv_idx) if domain else None
        cb = _make_goalforce_checklist_step_callback(
            session_id, k=int(os.environ.get("MEM0_K", "3")),
            iv_type_hint=iv_type_hint
        )
        return FrameworkConfig(
            name="goalforce_checklist",
            prefix_messages=[{"role": "system", "content": GOALFORCE_CHECKLIST_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "goalforce_verify":
        # V4 — pre-stop verification guard (targets Add premature stop).
        session_id = f"gfv_{cell_id}"
        iv_type_hint = _get_iv_type_for_cell(domain, iv_idx) if domain else None
        cb = _make_goalforce_verify_step_callback(
            session_id, k=int(os.environ.get("MEM0_K", "3")),
            iv_type_hint=iv_type_hint
        )
        return FrameworkConfig(
            name="goalforce_verify",
            prefix_messages=[{"role": "system", "content": GOALFORCE_VERIFY_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "goalforce_replan":
        # V5 — one-shot UNIFIED_PLAN forced post-intervention.
        session_id = f"gfp_{cell_id}"
        iv_type_hint = _get_iv_type_for_cell(domain, iv_idx) if domain else None
        cb = _make_goalforce_replan_step_callback(
            session_id, k=int(os.environ.get("MEM0_K", "3")),
            iv_type_hint=iv_type_hint
        )
        return FrameworkConfig(
            name="goalforce_replan",
            prefix_messages=[{"role": "system", "content": GOALFORCE_REPLAN_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "goalforce_combo":
        # V6 — V3 checklist + V4 verify merged.
        session_id = f"gfco_{cell_id}"
        iv_type_hint = _get_iv_type_for_cell(domain, iv_idx) if domain else None
        cb = _make_goalforce_combo_step_callback(
            session_id, k=int(os.environ.get("MEM0_K", "3")),
            iv_type_hint=iv_type_hint
        )
        return FrameworkConfig(
            name="goalforce_combo",
            prefix_messages=[{"role": "system", "content": GOALFORCE_COMBO_SYSTEM}],
            per_step_hook=cb,
        )

    if fw == "iv_restate":
        # Ablation: rewrite IV msg as A+B combined (Add only). Worker-level via env.
        # No prompt-side hooks — pure vanilla agent + modified IV message.
        return FrameworkConfig(name="iv_restate", prefix_messages=None, per_step_hook=None)

    if fw == "goalforce_quote":
        # V7 — force exact verbatim quote of A every step (persistence).
        session_id = f"gfq_{cell_id}"
        iv_type_hint = _get_iv_type_for_cell(domain, iv_idx) if domain else None
        cb = _make_goalforce_quote_step_callback(
            session_id, k=int(os.environ.get("MEM0_K", "3")),
            iv_type_hint=iv_type_hint
        )
        return FrameworkConfig(
            name="goalforce_quote",
            prefix_messages=[{"role": "system", "content": GOALFORCE_QUOTE_SYSTEM}],
            per_step_hook=cb,
        )

    raise ValueError(f"Unknown framework: {framework}")
