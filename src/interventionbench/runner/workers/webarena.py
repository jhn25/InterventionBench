"""WebArena worker — A/B/A→B with strict replay.

A→B replay: re-execute A's recorded actions in fresh A→B env (with patched
eval config) until inject_step, then inject intervention message and continue
with model-driven loop. Guarantees 100% intervention firing.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from playwright.sync_api import sync_playwright as _sync_playwright_init  # noqa: F401

# Module-level import so _run_wa_replay's `except BudgetExceeded` works.
# Without this, every model API failure (Connection refused, timeout, etc.) in
# the agent loop raises NameError, which surfaces as [AGENT LOOP CRASH] instead
# of the intended [MODEL ERROR] / [BUDGET EXCEEDED] cleanup.
try:
    from interventionbench.runner.budget import BudgetExceeded  # noqa: F401
except ImportError:
    class BudgetExceeded(Exception):  # type: ignore[no-redef]
        pass

INTERVENTIONS = Path(os.environ.get(
    "INTERVENTIONS_FILE",
    str(REPO_ROOT / "steer-bench/webarena/interventions.jsonl"),
))
STRICT_EVAL = REPO_ROOT / "steer-bench/webarena/strict_eval.json"
DOMAIN = "webarena"
MODEL_ID = os.environ.get("MODEL_ID") or os.environ.get("WEBARENA_MODEL", "Qwen3.6-27B-FP8")
NUM_ENDPOINTS = int(os.environ.get("NUM_ENDPOINTS", "4"))
BASE_PORT = int(os.environ.get("BASE_PORT", "8000"))
MAX_STEPS = int(os.environ.get("WEBARENA_MAX_STEPS", "30"))
INJECT_FRAC = float(os.environ.get("INJECT_FRAC", "0.5"))
USE_ACK = os.environ.get("USE_ACK", "0") == "1"
ACK_PROMPT_SUFFIX = " Before continuing, restate the updated task in one sentence"
USE_SYSPROMPT_MITIGATION = os.environ.get("USE_SYSPROMPT_MITIGATION", "0") == "1"
# TypeHint mitigation — append type-specific cue right after intervention text.
USE_TYPEHINT = os.environ.get("USE_TYPEHINT", "0") == "1"
TYPE_HINTS = {
    "VerbSwap":   "User changed the action. Stop the previous action; perform the new action on the same target.",
    "TargetSwap": "User changed the target. Same action, on the new target.",
    "BothSwap":   "User replaced the goal. Abandon the previous task; do the new one only.",
    "VerbAdd":    "User added an extra action. Complete BOTH the original task AND this new action.",
    "TargetAdd":  "User added an extra target. Apply the same action to BOTH targets.",
    "BothAdd":    "User added a wholly new sub-task. Complete BOTH the original task AND this new sub-task.",
    "VerbDrop":   "User wants to skip one sub-action of the original task. Do the rest unchanged.",
    "TargetDrop": "User wants to skip one target. Do the original task without that target.",
    "Cancel":     "User wants to stop. Acknowledge briefly and halt within 1-2 steps.",
}
SYSPROMPT_WARNING = (
    "IMPORTANT: The user may interject mid-task with updated instructions. "
    "Any user message after the task starts is authoritative — acknowledge it, "
    "apply the change, and continue from there.\n\n"
)


def _maybe_prepend_sysprompt(messages):
    if not USE_SYSPROMPT_MITIGATION:
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = SYSPROMPT_WARNING + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": SYSPROMPT_WARNING.strip()})
    return messages


def item_key(item):
    if isinstance(item, (str, int)):
        return (str(item), None)
    return (str(item["base_task_id"]), int(item["_idx"]))


def _load_patched() -> dict[int, dict]:
    out = {}
    if STRICT_EVAL.exists():
        for d in json.loads(STRICT_EVAL.read_text()):
            pe = d.get("patched_eval") or d.get("patched_evaluator")
            if d.get("_status") == "ok" and pe:
                out[int(d["intervention_idx"])] = pe
    return out


def build_workitems(*, condition, phase, framework, limit=None):
    rows = []
    for i, line in enumerate(INTERVENTIONS.open()):
        if not line.strip(): continue
        d = json.loads(line)
        d["_idx"] = i
        rows.append(d)
    patched = _load_patched()
    if condition == "A":
        items = sorted({r["base_task_id"] for r in rows})
    else:
        items = []
        for r in rows:
            # Skip cells marked as skipped (no valid B for this type+task)
            if r.get("skip") is True or r.get("_skipped"):
                continue
            # Cancel cells: no B-task / no eval — trajectory-based scoring
            # via steps_to_cancel. Only meaningful in A_to_B.
            if r.get("type") == "Cancel":
                if condition != "A_to_B":
                    continue
                r = dict(r)
                r["_condition"] = condition
                r["_phase"] = phase
                r["_framework"] = framework
                r["_patched_evaluator"] = None  # no B-eval
                items.append(r)
                continue
            if not r.get("modified_spec_b"):
                continue
            # Prefer embedded patched_eval (new cells); fall back to strict_eval.json
            # keyed first by intervention_idx (the proper cell ID), then by line index
            # for backward compat with old aligned data.
            embedded = r.get("patched_eval") or r.get("patched_evaluator")
            iv_idx_for_lookup = r.get("intervention_idx", r["_idx"])
            pe = embedded or patched.get(int(iv_idx_for_lookup)) or patched.get(r["_idx"])
            if not pe:
                continue
            r = dict(r)
            r["_condition"] = condition
            r["_phase"] = phase
            r["_framework"] = framework
            r["_patched_evaluator"] = pe
            items.append(r)
    return items[:limit] if limit else items


def _endpoint(task_id: str) -> str:
    # API_BASE_URL override → use external API (OpenAI / OpenRouter / Together / etc.)
    override = os.environ.get("API_BASE_URL")
    if override:
        return override.rstrip("/")
    # Auto-route API models to their cloud endpoint (never fall back to localhost vLLM).
    # Previously a missing API_BASE_URL silently routed gpt/gemini/claude to Qwen vLLM,
    # which returned RST → [MODEL ERROR] Connection refused (or 400 invalid model).
    mid = (MODEL_ID or "").lower()
    if mid.startswith("gpt-") or mid.startswith("o1-") or mid.startswith("o3-") or mid.startswith("o4-"):
        return "https://api.openai.com/v1"
    if mid.startswith("gemini-"):
        return "https://generativelanguage.googleapis.com/v1beta/openai"
    if mid.startswith("claude-"):
        return "https://api.anthropic.com/v1"
    # Local vLLM fallback (Qwen-27B / Llama-70B / Qwen-72B etc.)
    # VLLM_HOST env allows routing to remote vLLM (e.g., b200-2: 100.87.32.89)
    import hashlib
    h = int(hashlib.sha1(str(task_id).encode()).hexdigest(), 16) % NUM_ENDPOINTS
    host = os.environ.get("VLLM_HOST", "localhost")
    return f"http://{host}:{BASE_PORT + h}/v1"


def _api_key_for_model() -> str:
    """Pick the right API key based on MODEL_ID. vLLM endpoints accept any key."""
    mid = (MODEL_ID or "").lower()
    if mid.startswith("gemini-"):
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "dummy"
    if mid.startswith("claude-"):
        return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or "dummy"
    return (os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy")


def _make_client(task_id: str):
    from interventionbench.models.api import OpenAICompatibleClient
    api_key = _api_key_for_model()
    return OpenAICompatibleClient(
        base_url=_endpoint(task_id), api_key=api_key, model=MODEL_ID,
        timeout=600.0, max_retries=15,  # absorb vLLM TCP-backlog refusals at high concurrency
    )


def _make_adapter():
    from interventionbench.domains.webarena import WebArenaAdapter
    return WebArenaAdapter(headless=True, max_steps=MAX_STEPS)


def _trajectory_to_dict(traj):
    return traj.model_dump() if hasattr(traj, "model_dump") else traj


def _load_a_actions(task_id: str) -> list[str]:
    """Return list of A trajectory action_str values (stable-id form), skipping parse failures."""
    from interventionbench.runner.replay import load_a_trajectory
    t = load_a_trajectory(DOMAIN, task_id)
    if not t:
        return []
    out = []
    for s in (t.get("steps") or []):
        if s.get("role") != "action":
            continue
        md = s.get("metadata") or {}
        if md.get("parse_failure") or md.get("early_stop"):
            continue
        c = s.get("content") or ""
        if not c or c.startswith("[PARSE FAILURE]") or c.startswith("[INVALID ACTION]"):
            continue
        out.append(c)
    return out


def _load_a_num_steps(task_id: str) -> int | None:
    """Return num_steps metadata from A trajectory."""
    from interventionbench.runner.replay import load_a_trajectory
    t = load_a_trajectory(DOMAIN, task_id)
    if not t:
        return None
    return (t.get("metadata") or {}).get("num_steps")


def _run_wa_replay(task, intervention, client, inject_step, patched_eval,
                   *, prefix_messages=None, per_step_hook=None,
                   prior_reflections=None, framework_name=None,
                   trial_idx=0, budget=None):
    """Strict-replay A→B for WA. Replays A's actions until inject_step-1 in
    fresh env with patched config, then injects intervention and continues
    model-driven. Returns (steps_list, success, b_score, a_score, num_steps,
    inject_actual, b_eval_error, a_eval_error).

    Phase 2 hooks (all optional, no-op when None — Phase 1 path is unchanged):
      prefix_messages: list[dict] system messages to merge into the system
        prompt of every post-injection model call. Logged as ``framework_prefix``
        steps at trajectory start.
      per_step_hook: callable(step_idx, recent_steps, world_or_None) -> str|None.
        If returned string is non-None it is prepended to the next observation.
      prior_reflections: list[str] prepended as a [Past reflections] system
        message (Reflexion only).
      framework_name: name string for metadata logging.
      trial_idx: Reflexion trial index (0-based) for metadata logging.
    """
    from interventionbench.domains.webarena import (
        _CONFIG_DIR, _StableIDMap, _build_messages, _extract_action_from_response,
        _generate_config_file, _load_official_prompt, _require_webarena_env,
        _WEBARENA_REPO,
    )
    from interventionbench.models.base import ChatMessage
    # Function-local re-import so `except BudgetExceeded` at the inner agent
    # loop always resolves the name, even when the module-level import scope
    # got perturbed by a forked worker or by an importer that loaded this
    # module via a non-standard path. Mem0/HippoRAG/etc. fork mid-run and
    # previously hit `NameError: name 'BudgetExceeded' is not defined`.
    from interventionbench.runner.budget import BudgetExceeded  # noqa: F401

    _require_webarena_env()
    if str(_WEBARENA_REPO) not in sys.path:
        sys.path.insert(0, str(_WEBARENA_REPO))
    from browser_env import ActionTypes, ScriptBrowserEnv, create_stop_action
    from browser_env.actions import (
        ActionParsingError, create_id_based_action, create_none_action,
    )
    from browser_env.helper_functions import get_action_description
    from evaluation_harness import evaluator_router

    task_id = task.task_id
    task_config = task.metadata.get("config", {})
    iv_text = intervention.get("intervention_text", "")
    modified_spec = intervention.get("modified_spec_b", task.spec)

    # Two configs: patched (for B-eval, this is the live env config) and original (for A-eval).
    config_path = _generate_config_file(task_config, _CONFIG_DIR, patched_eval=patched_eval)
    original_config_path = (
        _generate_config_file(task_config, _CONFIG_DIR, patched_eval=None)
        if patched_eval is not None else None
    )

    a_actions = _load_a_actions(task_id)
    replay_until = max(0, int(inject_step) - 1)
    replay_until = min(replay_until, len(a_actions))

    steps = []
    steps.append({
        "role": "system",
        "content": f"Task {task_id}: {task.spec}",
        "metadata": {"task_id": task_id, "condition": "A_to_B",
                     "inject_step": inject_step, "replay_until": replay_until,
                     "a_total_actions": len(a_actions),
                     "framework": framework_name, "trial": trial_idx},
    })

    # Phase 2: log framework prefix + setup-skip + prior-reflections as
    # synthetic system steps (does not affect Phase 1 — all None by default).
    if prefix_messages:
        for pm in prefix_messages:
            steps.append({"role": "system", "content": pm.get("content", ""),
                          "metadata": {"framework_prefix": framework_name,
                                       "trial": trial_idx}})
    # WA has no code-exec sandbox; setup_code is unsupported. Log a marker.
    setup_code = intervention.get("_setup_code")
    if setup_code:
        steps.append({"role": "observation",
                      "content": "[setup_code skipped: WA has no code exec]",
                      "metadata": {"framework_setup_skipped": framework_name}})
    if prior_reflections:
        from interventionbench.runner.workers._phase2_hooks import REFLEXION_SYSTEM_TEMPLATE
        ref_text = REFLEXION_SYSTEM_TEMPLATE.format(
            reflections="\n".join(f"- Trial {i+1}: {r}" for i, r in enumerate(prior_reflections))
        )
        steps.append({"role": "system", "content": ref_text,
                      "metadata": {"reflexion_trial": trial_idx,
                                   "framework_prefix": framework_name}})

    env = ScriptBrowserEnv(
        headless=True, observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 720},
        save_trace_enabled=False, sleep_after_execution=0.0,
    )
    success, b_score, a_score, num_steps = False, None, None, 0
    b_eval_error, a_eval_error, inject_actual = None, None, None

    try:
        env.reset(options={"config_file": str(config_path)})
        obs, info = env.reset(options={"config_file": str(config_path)})
        # Bump Playwright default timeout (30s -> 90s) — many concurrent workers
        # overload single-process docker apps (Magento, Postmill, GitLab) and the
        # 30s default surfaces as 'Timeout 30000ms exceeded while waiting for event
        # "load"', killing the cell.
        try:
            _to = int(os.environ.get("PLAYWRIGHT_TIMEOUT_MS", "90000"))
            env.page.set_default_timeout(_to)
            env.page.set_default_navigation_timeout(_to)
        except Exception:
            pass
        from browser_env.utils import StateInfo
        state_info: StateInfo = {"observation": obs, "info": info}
        trajectory: list = [state_info]

        meta_data: dict = {"action_history": ["None"]}
        prompt = _load_official_prompt()
        id_map = _StableIDMap()
        current_objective = task.spec
        injected = False
        replayed_count = 0

        # ---- REPLAY phase: re-execute A's actions ----
        for replay_idx in range(replay_until):
            step_num = replay_idx + 1
            obs_dict = state_info["observation"]
            current_obs_raw = obs_dict.get("text", "") or obs_dict.get("accessibility_tree", "")
            current_obs = id_map.stabilize_obs(current_obs_raw)
            current_url = state_info["info"]["page"].url
            steps.append({"role": "observation",
                          "content": f"URL: {current_url}\n{current_obs[:500]}...",
                          "metadata": {"step": step_num, "url": current_url, "replay": True}})

            action_str = a_actions[replay_idx]
            action_str_chrome = id_map.chromify_action(action_str)
            try:
                action = create_id_based_action(action_str_chrome)
            except ActionParsingError as exc:
                steps.append({"role": "action",
                              "content": f"[REPLAY INVALID] {action_str}",
                              "metadata": {"step": step_num, "replay": True, "error": str(exc)}})
                trajectory.append(create_none_action())
                replayed_count = step_num
                continue

            trajectory.append(action)
            try:
                action_desc = get_action_description(
                    action, state_info["info"]["observation_metadata"],
                    action_set_tag="id_accessibility_tree", prompt_constructor=None,
                )
            except Exception:
                action_desc = action_str
            meta_data["action_history"].append(action_desc)
            steps.append({"role": "action", "content": action_str,
                          "metadata": {"step": step_num, "action_desc": action_desc, "replay": True}})

            if action["action_type"] == ActionTypes.STOP:
                replayed_count = step_num
                break

            try:
                obs, _, terminated, _, info = env.step(action)
            except Exception as exc:
                steps.append({"role": "observation",
                              "content": f"[REPLAY ENV ERROR] {exc}",
                              "metadata": {"step": step_num, "replay": True, "error": True}})
                trajectory.append(create_stop_action(f"Replay env error: {exc}"))
                replayed_count = step_num
                break

            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)
            replayed_count = step_num
            if terminated:
                trajectory.append(create_stop_action(""))
                replayed_count = step_num
                break

        # ---- INJECT + model-driven continuation ----
        # Multi-step support: pull lists from intervention dict (set by run_one).
        # Single-step path: lists have one entry each (= legacy single inject).
        inject_steps_list = intervention.get("_inject_steps_list") or [inject_step]
        inject_texts_list = intervention.get("_inject_texts_list") or [iv_text]
        multi_step = bool(intervention.get("_multi_step"))
        ivs_meta_inner = intervention.get("interventions") if multi_step else None
        # Parallel list of intervention types (one per inject step). Used by TypeHint
        # mitigation to append a type-specific cue after the user message.
        if multi_step and ivs_meta_inner:
            inject_types_list = [iv.get("type") for iv in ivs_meta_inner]
        else:
            inject_types_list = [intervention.get("type")]

        def _format_iv(text: str, iv_type: str | None = None) -> str:
            out = (text + ACK_PROMPT_SUFFIX) if (USE_ACK and text) else text
            if USE_TYPEHINT and iv_type and TYPE_HINTS.get(iv_type):
                out = f"{out}\n\n[INTERVENTION HINT — type={iv_type}] {TYPE_HINTS[iv_type]}"
            return out

        # First inject (immediately after replay).
        first_iv_text = _format_iv(inject_texts_list[0], inject_types_list[0] if inject_types_list else None)
        inject_step_actual = replayed_count + 1
        inject_actual = inject_step_actual
        steps.append({"role": "user",
                      "content": f"[Mid-task user intervention]: {first_iv_text}",
                      "metadata": {"step": inject_step_actual, "intervention": True,
                                   "is_intervention": True, "inject_seq": 0,
                                   "ack_used": USE_ACK}})
        current_objective = modified_spec
        injected = True
        injected_texts_so_far = [first_iv_text]
        next_inject_idx = 1

        consecutive_parse_failures = 0
        consecutive_same_actions = 0
        last_action_str = None
        pending_hook_ctx: str | None = None  # Phase 2: prepend to next obs

        # Build the Phase 2 system-prefix blob once: prefix_messages +
        # optional reflections. Empty for Phase 1 (vanilla).
        _phase2_sys_extra = ""
        if prefix_messages:
            _phase2_sys_extra += "\n\n".join(
                pm.get("content", "") for pm in prefix_messages
                if pm.get("role", "system") == "system"
            )
        if prior_reflections:
            from interventionbench.runner.workers._phase2_hooks import REFLEXION_SYSTEM_TEMPLATE
            ref_text = REFLEXION_SYSTEM_TEMPLATE.format(
                reflections="\n".join(f"- Trial {i+1}: {r}"
                                      for i, r in enumerate(prior_reflections))
            )
            _phase2_sys_extra = (_phase2_sys_extra + "\n\n" + ref_text).strip()

        agent_loop_error_msg: str | None = None
        try:
            for step_num in range(inject_step_actual, MAX_STEPS + 1):
                if consecutive_parse_failures >= 5 or consecutive_same_actions >= 5:
                    trajectory.append(create_stop_action("Early stop"))
                    break

                # Multi-step: check if step_num matches the next pending inject point.
                # When triggered, log a synthetic user message and (optionally) update
                # current_objective from the next iv's modified_spec_b.
                while (next_inject_idx < len(inject_steps_list)
                       and step_num >= inject_steps_list[next_inject_idx]):
                    _nxt_type = inject_types_list[next_inject_idx] if next_inject_idx < len(inject_types_list) else None
                    nxt_iv_text = _format_iv(inject_texts_list[next_inject_idx], _nxt_type)
                    steps.append({"role": "user",
                                  "content": f"[Mid-task user intervention]: {nxt_iv_text}",
                                  "metadata": {"step": step_num, "intervention": True,
                                               "is_intervention": True,
                                               "inject_seq": next_inject_idx,
                                               "ack_used": USE_ACK}})
                    injected_texts_so_far.append(nxt_iv_text)
                    if multi_step and ivs_meta_inner and next_inject_idx < len(ivs_meta_inner):
                        nxt_spec = ivs_meta_inner[next_inject_idx].get("modified_spec_b")
                        if nxt_spec:
                            current_objective = nxt_spec
                    next_inject_idx += 1

                obs_dict = state_info["observation"]
                current_obs_raw = obs_dict.get("text", "") or obs_dict.get("accessibility_tree", "")
                current_obs = id_map.stabilize_obs(current_obs_raw)
                # Phase 2: prepend retrieved-notes / refine signal to obs.
                if pending_hook_ctx:
                    current_obs = f"{pending_hook_ctx}\n\n{current_obs}"
                    pending_hook_ctx = None
                current_url = state_info["info"]["page"].url
                previous_action_str = meta_data["action_history"][-1]

                steps.append({"role": "observation",
                              "content": f"URL: {current_url}\n{current_obs[:500]}...",
                              "metadata": {"step": step_num, "url": current_url, "post_injection": True}})

                messages_raw = _build_messages(
                    prompt=prompt, observation=current_obs, url=current_url,
                    objective=current_objective, previous_action=previous_action_str,
                    max_obs_length=3840,
                )
                chat_messages = []
                _system_seen = False
                for m in messages_raw:
                    name = m.get("name", "")
                    if name == "example_user":
                        chat_messages.append(ChatMessage(role="user", content=m["content"]))
                    elif name == "example_assistant":
                        chat_messages.append(ChatMessage(role="assistant", content=m["content"]))
                    elif m["role"] in ("system", "user", "assistant"):
                        # Phase 2: merge prefix into the first system message only.
                        if m["role"] == "system" and not _system_seen and _phase2_sys_extra:
                            chat_messages.append(ChatMessage(
                                role="system",
                                content=m["content"] + "\n\n" + _phase2_sys_extra,
                            ))
                        else:
                            chat_messages.append(ChatMessage(role=m["role"], content=m["content"]))
                        if m["role"] == "system":
                            _system_seen = True
                # If no system was emitted but we have a Phase 2 prefix, prepend it.
                if _phase2_sys_extra and not _system_seen:
                    chat_messages.insert(0, ChatMessage(role="system", content=_phase2_sys_extra))
                if injected_texts_so_far:
                    if len(injected_texts_so_far) == 1:
                        chat_messages.append(ChatMessage(
                            role="user",
                            content=f"[Earlier mid-task user intervention]: {injected_texts_so_far[0]}",
                        ))
                    else:
                        combined_ivs = "\n".join(
                            f"({i+1}) {t}" for i, t in enumerate(injected_texts_so_far)
                        )
                        chat_messages.append(ChatMessage(
                            role="user",
                            content=f"[Earlier mid-task user interventions]:\n{combined_ivs}",
                        ))

                try:
                    # Mistral/Mixtral templates reject system role — merge system into first user.
                    if "Mixtral" in MODEL_ID or "Mistral" in MODEL_ID:
                        sys_chunks = [m.content for m in chat_messages if m.role == "system"]
                        non_sys = [m for m in chat_messages if m.role != "system"]
                        if sys_chunks:
                            sys_blob = "\n\n".join(sys_chunks)
                            if non_sys and non_sys[0].role == "user":
                                non_sys[0] = ChatMessage(role="user", content=f"{sys_blob}\n\n{non_sys[0].content}")
                            else:
                                non_sys.insert(0, ChatMessage(role="user", content=sys_blob))
                            chat_messages = non_sys
                    # Qwen3-only chat_template_kwargs; skip for other models to avoid errors
                    _eb = {"chat_template_kwargs": {"enable_thinking": False}} if "Qwen3" in MODEL_ID else None
                    # Sysprompt mitigation: prepend warning to first system message (idempotent).
                    if USE_SYSPROMPT_MITIGATION:
                        _injected = False
                        for _i, _m in enumerate(chat_messages):
                            if _m.role == "system" and not _m.content.startswith("IMPORTANT: The user may interject"):
                                chat_messages[_i] = ChatMessage(role="system", content=SYSPROMPT_WARNING + _m.content)
                                _injected = True
                                break
                        if not _injected:
                            chat_messages.insert(0, ChatMessage(role="system", content=SYSPROMPT_WARNING.strip()))
                    response = client.chat(
                        chat_messages, max_tokens=1024, temperature=0.0,
                        extra_body=_eb,
                    )
                    response_text = response.content or ""
                    if budget is not None:
                        budget.add(getattr(response, "usage", None))
                except BudgetExceeded as exc:
                    steps.append({"role": "observation",
                                  "content": f"[BUDGET EXCEEDED] {exc}",
                                  "metadata": {"step": step_num, "budget_exceeded": True,
                                               "budget": budget.snapshot() if budget else {}}})
                    trajectory.append(create_stop_action(f"Budget exceeded: {exc}"))
                    break
                except Exception as exc:
                    steps.append({"role": "observation",
                                  "content": f"[MODEL ERROR] {exc}",
                                  "metadata": {"step": step_num, "error": True}})
                    trajectory.append(create_stop_action(f"Model error: {exc}"))
                    break

                steps.append({"role": "thought", "content": response_text,
                              "metadata": {"step": step_num, "post_injection": True}})

                try:
                    action_str = _extract_action_from_response(response_text, prompt)
                    consecutive_parse_failures = 0
                except (ValueError, ActionParsingError):
                    consecutive_parse_failures += 1
                    steps.append({"role": "action", "content": "[PARSE FAILURE]",
                                  "metadata": {"step": step_num, "parse_failure": True}})
                    trajectory.append(create_none_action())
                    continue

                if action_str == last_action_str:
                    consecutive_same_actions += 1
                else:
                    consecutive_same_actions = 0
                last_action_str = action_str

                action_str_chrome = id_map.chromify_action(action_str)
                try:
                    action = create_id_based_action(action_str_chrome)
                except ActionParsingError:
                    consecutive_parse_failures += 1
                    trajectory.append(create_none_action())
                    steps.append({"role": "action", "content": f"[INVALID ACTION] {action_str}",
                                  "metadata": {"step": step_num}})
                    continue

                trajectory.append(action)
                try:
                    action_desc = get_action_description(
                        action, state_info["info"]["observation_metadata"],
                        action_set_tag="id_accessibility_tree", prompt_constructor=None,
                    )
                except Exception:
                    action_desc = action_str
                meta_data["action_history"].append(action_desc)
                steps.append({"role": "action", "content": action_str,
                              "metadata": {"step": step_num, "action_desc": action_desc,
                                           "post_injection": True}})

                if action["action_type"] == ActionTypes.STOP:
                    break

                try:
                    obs, _, terminated, _, info = env.step(action)
                    # Transient env retry (1x, 5s): browser-level network failures
                    # (Chromium error pages, gateway 5xx, ERR_ABORTED). These otherwise
                    # propagate into the obs and look like the agent took an empty action.
                    _obs_text_check = ""
                    if isinstance(obs, dict):
                        _obs_text_check = (obs.get("text") or obs.get("accessibility_tree") or "")[:2000]
                    _ENV_RETRY_PATS = ("ERR_ABORTED", "ERR_CONNECTION", "ERR_NAME_NOT_RESOLVED",
                                       "ERR_EMPTY_RESPONSE", "502 Bad Gateway", "503 Service",
                                       "504 Gateway", "500 Internal Server", "HTTP ERROR 5",
                                       "HTTP ERROR 403", "site can't be reached",
                                       "took too long to respond")
                    if any(p in _obs_text_check for p in _ENV_RETRY_PATS):
                        steps.append({"role": "observation",
                                      "content": "[ENV TRANSIENT] retrying same action after 5s",
                                      "metadata": {"step": step_num, "env_retry": True}})
                        import time as _time
                        _time.sleep(5)
                        try:
                            obs, _, terminated, _, info = env.step(action)
                        except Exception as _exc2:
                            steps.append({"role": "observation",
                                          "content": f"[ENV ERROR after retry] {_exc2}",
                                          "metadata": {"step": step_num, "error": True,
                                                       "env_retry_failed": True}})
                            trajectory.append(create_stop_action(f"Env error: {_exc2}"))
                            break
                except Exception as exc:
                    steps.append({"role": "observation",
                                  "content": f"[ENV ERROR] {exc}",
                                  "metadata": {"step": step_num, "error": True}})
                    trajectory.append(create_stop_action(f"Env error: {exc}"))
                    break

                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                # Phase 2: per-step framework hook (AMem note-take + retrieval, etc).
                # Best-effort; exceptions silenced. Returned string (if non-None) is
                # prepended to the next observation as a [retrieved_notes] block.
                if per_step_hook is not None:
                    try:
                        pending_hook_ctx = per_step_hook(step_num, steps[-8:], None)
                    except Exception:
                        pending_hook_ctx = None

                if terminated:
                    trajectory.append(create_stop_action(""))
                    break

        except Exception as exc:
            import traceback as _tb
            agent_loop_error_msg = f"{type(exc).__name__}: {exc}"[:300]
            steps.append({"role": "observation",
                          "content": f"[AGENT LOOP CRASH] {agent_loop_error_msg}",
                          "metadata": {"crash": True,
                                       "trace": _tb.format_exc()[-500:]}})
            try:
                trajectory.append(create_stop_action(f"crash: {agent_loop_error_msg[:80]}"))
            except Exception: pass

        # ---- Dual eval (multi-step: combine across each iv's patched_eval) ----
        patched_evals_list = intervention.get("_patched_evals_list") or [patched_eval]
        b_scores_per_iv = []
        b_errors_per_iv = []
        for _pi, _pe in enumerate(patched_evals_list):
            if _pe is None:
                b_scores_per_iv.append(None)
                b_errors_per_iv.append("no_patched_eval")
                continue
            try:
                # First iv reuses config_path (already generated above).
                _cfg = config_path if _pi == 0 else _generate_config_file(
                    task_config, _CONFIG_DIR, patched_eval=_pe
                )
                _ev = evaluator_router(str(_cfg))
                _s = float(_ev(
                    trajectory=trajectory, config_file=str(_cfg),
                    page=env.page, client=env.get_page_client(env.page),
                ))
                b_scores_per_iv.append(_s)
            except Exception as exc:
                # WA StringEvaluator does `last_action["answer"]` — when the
                # agent didn't issue stop(answer="...") this is a KeyError
                # 'answer'. That's a definitive agent failure (no deliverable),
                # so score = 0.0, NOT None (would otherwise be excluded as
                # "unevaluable" and inflate fail rates).
                if isinstance(exc, KeyError) and str(exc) == "'answer'":
                    b_scores_per_iv.append(0.0)
                    b_errors_per_iv.append("no_stop_answer")
                else:
                    b_scores_per_iv.append(None)
                    b_errors_per_iv.append(str(exc))
        valid_bs = [s for s in b_scores_per_iv if s is not None]
        b_score = min(valid_bs) if valid_bs else None
        if [e for e in b_errors_per_iv if e]:
            b_eval_error = "; ".join(e for e in b_errors_per_iv if e)
            steps.append({"role": "observation",
                          "content": f"[B-EVAL ERROR] {b_eval_error}",
                          "metadata": {"eval_error": b_eval_error, "which": "b"}})
        if original_config_path is not None:
            try:
                evaluator_a = evaluator_router(str(original_config_path))
                a_score = float(evaluator_a(
                    trajectory=trajectory, config_file=str(original_config_path),
                    page=env.page, client=env.get_page_client(env.page),
                ))
            except Exception as exc:
                # Same KeyError 'answer' handling for A: agent didn't stop.
                if isinstance(exc, KeyError) and str(exc) == "'answer'":
                    a_score = 0.0
                    a_eval_error = "no_stop_answer"
                else:
                    a_eval_error = str(exc)
                steps.append({"role": "observation",
                              "content": f"[A-EVAL ERROR] {a_eval_error}",
                              "metadata": {"eval_error": a_eval_error, "which": "a"}})

        success = bool(b_score == 1.0)
        num_steps = len([s for s in steps if s.get("role") == "action"])

    finally:
        try: env.close()
        except Exception: pass

    return steps, success, b_score, a_score, num_steps, inject_actual, b_eval_error, a_eval_error


def run_one(item):
    from interventionbench.runner.storage import make_trajectory
    from interventionbench.runner.workers._phase2_hooks import (
        framework_config, REFLEXION_REFLECT_PROMPT_SYSTEM,
    )
    from interventionbench.models.base import ChatMessage

    if isinstance(item, (str, int)):
        task_id = str(item)
        condition = "A"
        iv_idx, iv_type, iv_text, spec_b = None, None, None, None
        patched = None
        framework = "vanilla"
    else:
        task_id = str(item["base_task_id"])
        iv_idx = int(item["_idx"])
        iv_type = item["type"]
        iv_text = item.get("intervention_text")
        spec_b = item["modified_spec_b"]
        condition = item["_condition"]
        patched = item.get("_patched_evaluator")
        framework = item.get("_framework") or "vanilla"

    fw_cfg = framework_config(framework, condition, task_id, iv_idx, domain="webarena")
    # Convert prefix_messages (list[dict]) -> list[ChatMessage] for adapter
    # signature compat. Keep raw dict form for the replay worker (which
    # consumes dicts directly).
    prefix_msgs_chat = [
        ChatMessage(role=pm.get("role", "system"), content=pm.get("content", ""))
        for pm in (fw_cfg.prefix_messages or [])
    ]
    prefix_msgs_dict = list(fw_cfg.prefix_messages or [])

    t0 = time.time()
    adapter = _make_adapter()
    client = _make_client(task_id)
    steps_out = []
    success, a_score, b_score, num_steps, inject_actual = False, None, None, 0, None
    extra = {"model_id": MODEL_ID, "framework": framework}
    # Per-trajectory token + cost budget guard.
    try:
        from interventionbench.runner.budget import BudgetExceeded, TrajectoryBudget
        budget = TrajectoryBudget(
            max_total_tokens=int(os.environ.get("MAX_TOKENS_PER_CELL", "600000")),
            max_cost_usd=float(os.environ.get("MAX_COST_PER_CELL_USD", "1.5")),
            max_calls=int(os.environ.get("MAX_CALLS_PER_CELL", "200")),
            model_id=MODEL_ID,
        )
    except ImportError:
        class BudgetExceeded(Exception): pass
        budget = None

    def _adapter_step_callback(*, step_index, recent_messages):
        """Adapt fw_cfg.per_step_hook(step_idx, recent_steps, world) signature
        to the adapter's step_callback(step_index=, recent_messages=) form.
        Best-effort; exceptions silenced. Return value ignored by adapter
        (WA's run_baseline doesn't have a per-step text-prepend channel —
        AMem hook side-effects (note add) still fire correctly)."""
        if fw_cfg.per_step_hook is None:
            return None
        try:
            return fw_cfg.per_step_hook(step_index, list(recent_messages), None)
        except Exception:
            return None
    _step_cb = _adapter_step_callback if fw_cfg.per_step_hook is not None else None

    try:
        tasks = adapter.list_candidate_tasks(limit=None)
        task = next((t for t in tasks if t.task_id == task_id), None)
        if not task:
            return None

        if condition == "A":
            # Phase 1: vanilla A. Phase 2 frameworks are only meaningful for
            # B / A_to_B (A is the unmodified baseline). Pass framework hooks
            # through anyway so a Phase-2 A run still gets the prefix.
            result = adapter.run_baseline(
                task, client,
                prefix_messages=prefix_msgs_chat or None,
                step_callback=_step_cb,
            )
            traj_dict = _trajectory_to_dict(result.trajectory)
            steps_out = traj_dict.get("steps") or []
            md = traj_dict.get("metadata") or {}
            success = result.success
            a_score = md.get("b_score")
            num_steps = md.get("num_steps") or 0
            extra["sites"] = md.get("sites")
            if fw_cfg.setup_code:
                steps_out.insert(1, {"role": "observation",
                                     "content": "[setup_code skipped: WA has no code exec]",
                                     "metadata": {"framework_setup_skipped": fw_cfg.name}})
            for pm in reversed(prefix_msgs_dict):
                steps_out.insert(1, {"role": "system",
                                     "content": pm.get("content", ""),
                                     "metadata": {"framework_prefix": fw_cfg.name}})

        elif condition == "B":
            modified_task = task.model_copy()
            modified_task.spec = spec_b
            result = adapter.run_baseline(
                modified_task, client, patched_eval=patched,
                prefix_messages=prefix_msgs_chat or None,
                step_callback=_step_cb,
            )
            traj_dict = _trajectory_to_dict(result.trajectory)
            steps_out = traj_dict.get("steps") or []
            md = traj_dict.get("metadata") or {}
            success = result.success
            b_score = md.get("b_score")
            num_steps = md.get("num_steps") or 0
            extra["patched_eval_used"] = True
            # Insert framework_prefix steps + setup-skip marker right after
            # the first system step so they're visible in the trajectory.
            if fw_cfg.setup_code:
                steps_out.insert(1, {"role": "observation",
                                     "content": "[setup_code skipped: WA has no code exec]",
                                     "metadata": {"framework_setup_skipped": fw_cfg.name}})
            for pm in reversed(prefix_msgs_dict):
                steps_out.insert(1, {"role": "system",
                                     "content": pm.get("content", ""),
                                     "metadata": {"framework_prefix": fw_cfg.name}})

        else:  # A_to_B with strict replay
            a_n = _load_a_num_steps(task_id)

            # Multi-step detection: cell has "interventions" list + "inject_ratios".
            multi_step = isinstance(item.get("interventions"), list) and len(item["interventions"]) >= 2
            if multi_step:
                ivs_meta = item["interventions"]
                ratios = item.get("inject_ratios") or [0.5, 0.75]
                inject_steps_planned = [max(2, round((a_n or 10) * r)) for r in ratios]
                for i in range(1, len(inject_steps_planned)):
                    inject_steps_planned[i] = max(inject_steps_planned[i],
                                                   inject_steps_planned[i-1] + 2)
                inject_texts_list = [iv.get("text") or iv.get("intervention_text") for iv in ivs_meta]
                patched_evals_list = [patched.get(int(iv["idx"])) for iv in ivs_meta]
                iv_text = inject_texts_list[0]
            else:
                inject_steps_planned = [max(2, round((a_n or 10) * INJECT_FRAC))]
                inject_texts_list = [iv_text]
                # `patched` IS this cell's eval dict (eval_types/reference_url/etc),
                # not a dict-of-evals keyed by intervention_idx. The earlier code
                # `patched.get(iv_idx)` always returned None for single-step cells,
                # causing the dual-eval loop to short-circuit with "no_patched_eval"
                # and produce b_score=None for the majority of WA A→B cells.
                # Cancel cells pass patched=None so propagate that explicitly.
                patched_evals_list = [patched]
                ivs_meta = None

            inject_step_planned = inject_steps_planned[0]
            iv_dict = dict(item)
            iv_dict["intervention_idx"] = iv_idx
            iv_dict["_setup_code"] = fw_cfg.setup_code
            iv_dict["_inject_steps_list"] = inject_steps_planned
            iv_dict["_inject_texts_list"] = inject_texts_list
            iv_dict["_patched_evals_list"] = patched_evals_list
            iv_dict["_multi_step"] = multi_step

            extra["inject_step_planned"] = inject_step_planned
            extra["inject_steps_planned"] = inject_steps_planned
            extra["a_total_steps"] = a_n
            extra["inject_frac"] = INJECT_FRAC
            extra["use_ack"] = USE_ACK
            extra["multi_step"] = multi_step
            extra["intervention_text"] = iv_text
            extra["patched_eval_used"] = True
            if multi_step:
                extra["intervention_texts"] = inject_texts_list
                extra["interventions_metadata"] = [
                    {"text": t, "type": iv.get("type"), "idx": iv.get("idx")}
                    for t, iv in zip(inject_texts_list, ivs_meta)
                ]

            # ----- Reflexion N-trial loop wrapper -----
            # n_trials==1 for vanilla / non-Reflexion. Reflexion: re-init env
            # per trial, replay A, inject, run framework. If fail, generate a
            # reflection and prepend to next trial's system prefix.
            n_trials = max(1, fw_cfg.n_trials)
            reflections: list[str] = []
            best_steps: list = []
            best_b_score: float | None = None
            best_a_score: float | None = None
            best_num_steps: int = 0
            best_inject: int | None = None
            best_b_err: str | None = None
            best_a_err: str | None = None
            trial_idx = 0
            for trial_idx in range(n_trials):
                t_steps, t_success, t_b_score, t_a_score, t_num, t_inject, t_b_err, t_a_err = (
                    _run_wa_replay(
                        task, iv_dict, client, inject_step_planned, patched,
                        prefix_messages=prefix_msgs_dict or None,
                        per_step_hook=fw_cfg.per_step_hook,
                        prior_reflections=list(reflections) if reflections else None,
                        framework_name=fw_cfg.name,
                        trial_idx=trial_idx,
                        budget=budget,
                    )
                )
                trial_b_pass = (t_b_score is not None and t_b_score >= 0.999)
                # Always overwrite — paper convention is final-trial trajectory.
                best_steps = t_steps
                best_b_score = t_b_score
                best_a_score = t_a_score
                best_num_steps = t_num
                best_inject = t_inject
                best_b_err = t_b_err
                best_a_err = t_a_err

                # Reflect for next trial (Reflexion only, only if not last + not success).
                if n_trials > 1 and trial_idx < n_trials - 1 and not trial_b_pass:
                    recent = "\n".join(
                        f"[{s.get('role', '?')}] {str(s.get('content', ''))[:600]}"
                        for s in t_steps[-12:]
                    )
                    reflect_prompt = (
                        f"Trajectory of last attempt:\n{recent}\n\n"
                        f"User mid-task instruction: {iv_text}\n\n"
                        f"Outcome: b_pass={trial_b_pass} b_score={t_b_score}\n\n"
                        "Self-reflection:"
                    )
                    try:
                        _eb = ({"chat_template_kwargs": {"enable_thinking": False}}
                               if "Qwen3" in MODEL_ID else None)
                        r = client.chat(
                            [ChatMessage(role="system", content=REFLEXION_REFLECT_PROMPT_SYSTEM),
                             ChatMessage(role="user", content=reflect_prompt)],
                            max_tokens=400, temperature=0.0,
                            extra_body=_eb,
                        )
                        ref = (r.content or "")[:800]
                    except Exception as exc:
                        ref = f"[reflection_error: {exc}]"
                    reflections.append(ref)
                if trial_b_pass:
                    break

            steps_out = best_steps
            success = bool(best_b_score is not None and best_b_score >= 0.999)
            b_score = best_b_score
            a_score = best_a_score
            num_steps = best_num_steps
            inject_actual = best_inject
            if best_b_err: extra["b_eval_error"] = best_b_err
            if best_a_err: extra["a_eval_error"] = best_a_err
            extra["n_trials_run"] = trial_idx + 1
            extra["all_reflections"] = reflections

    except Exception as exc:
        import traceback
        steps_out.append({"role": "error", "content": str(exc),
                          "metadata": {"trace": traceback.format_exc()[-500:]}})

    extra["elapsed_s"] = round(time.time() - t0, 1)
    extra["num_steps"] = num_steps
    if budget is not None:
        extra["budget"] = budget.snapshot()

    # Cancel (BothDrop): trajectory-based score = post-inject actionable steps.
    # Pass = ≤ 2 (one acknowledgment + stop). Only computed for A_to_B.
    if iv_type == "Cancel" and condition == "A_to_B" and inject_actual is not None:
        post = 0
        for s in (steps_out or []):
            md = s.get("metadata") or {}
            st = md.get("step")
            if st is None or st <= inject_actual:
                continue
            if s.get("role") not in ("assistant", "thought", "action"):
                continue
            if md.get("replay"):
                continue
            content = (s.get("content") or "").strip()
            if not content:
                continue
            post += 1
        extra["steps_to_cancel"] = post
        extra["cancel_pass"] = post <= 2

    traj = make_trajectory(
        task_ref=task_id, condition=condition,
        intervention_idx=iv_idx, intervention_type=iv_type,
        modified_spec_b=spec_b, inject_step=inject_actual,
        a_score=a_score, b_score=b_score,
        success=success, steps=steps_out, extra_metadata=extra,
    )
    fired = " ✓" if (condition == "A_to_B" and inject_actual) else (" ✗" if condition == "A_to_B" else "")
    print(f"  [{condition}] {task_id}/{iv_idx if iv_idx is not None else '-'} {iv_type or ''}: "
          f"a={a_score} b={b_score} steps={num_steps}{fired} dt={extra['elapsed_s']:.0f}s", flush=True)
    return traj
