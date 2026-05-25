"""AppWorld worker — A/B/A→B with replay.

Worker functions are at module-top so they're picklable for fork pool.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("APPWORLD_ROOT", str(REPO_ROOT / "data/external/appworld"))
sys.path.insert(0, str(REPO_ROOT / "src"))

INTERVENTIONS = Path(os.environ.get(
    "INTERVENTIONS_FILE",
    str(REPO_ROOT / "steer-bench/appworld/interventions.jsonl"),
))
TASK_IDS = REPO_ROOT / "steer-bench/appworld/task_ids.txt"
DOMAIN = "appworld"
MAX_STEPS = int(os.environ.get("APPWORLD_MAX_STEPS", "40"))
INJECT_FRAC = float(os.environ.get("INJECT_FRAC", "0.5"))
# Ack mitigation: append a restate-the-task prompt to inject_text when set.
USE_ACK = os.environ.get("USE_ACK", "0") == "1"
ACK_PROMPT_SUFFIX = " Before continuing, restate the updated task in one sentence"
# IV-Restate-A: rewrite the IV message to include A verbatim + "Also, [B]".
# Tests whether prompt-level re-injection of A at IV time prevents A loss.
USE_IV_RESTATE_A = os.environ.get("USE_IV_RESTATE_A", "0") == "1"
# Sysprompt mitigation: prepend warning to system prompt about mid-task user intervention.
USE_SYSPROMPT_MITIGATION = os.environ.get("USE_SYSPROMPT_MITIGATION", "0") == "1"
# TypeHint mitigation: append type-specific guidance after the intervention message.
# Tells the agent exactly what kind of change the user is requesting so it doesn't
# have to infer from prose. Read in _react_loop; cells must carry intervention_type.
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


def _maybe_prepend_sysprompt(messages: list[dict]) -> list[dict]:
    if not USE_SYSPROMPT_MITIGATION:
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = SYSPROMPT_WARNING + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": SYSPROMPT_WARNING.strip()})
    return messages
MODEL_ID = os.environ.get("MODEL_ID", "Qwen3.6-27B-FP8")
NUM_ENDPOINTS = int(os.environ.get("NUM_ENDPOINTS", "4"))
BASE_PORT = int(os.environ.get("BASE_PORT", "8000"))


def _eval_y_inline(world, eval_code: str) -> tuple[float | None, int, int, int, str | None]:
    """Run intervention evaluation_code against the LIVE world (no replay).

    Mirrors scripts/eval/aw_post_eval.py._replay_eval but skips the replay step
    since the world is already in its final post-agent state. Uses the same
    AppWorld TestTracker + ModelCollectionPair plumbing and the same strict
    AST patches (extend_ignore_lists, strip_answer_match_if_no_gt_answer).

    Returns (b_score, n_pass, n_fail, n_total, error).
    """
    if not eval_code:
        return None, 0, 0, 0, "no eval_code"
    try:
        import importlib.util
        import tempfile
        import uuid as _uuid
        from copy import deepcopy
        from appworld.evaluator import TestTracker
        from appworld.collections.models import ModelCollectionPair
        from munch import Munch
        # Reuse AST patches from the post-hoc evaluator.
        # 2026-05-22: eval_appworld_v2.py was renamed to eval_appworld.py during
        # cleanup; new runs (typehint, membridge) all silently got b_score=None
        # because of the stale import. Vanilla cells captured earlier still have
        # correct scores. Fixed by importing the renamed module.
        sys.path.insert(0, str(REPO_ROOT / "scripts/eval"))
        from eval_appworld import (  # type: ignore
            _extend_ignore_lists,
            _strip_answer_match_if_no_gt_answer,
        )
        # Monkey-patch appworld.common.errors.get_traceback_stack_data: its
        # context_lines indexing throws IndexError when our cell eval runs as a
        # tempfile (linecache returns mismatched lines). This swallows the real
        # eval error and produces the misleading "IndexError: list index out of
        # range" that affected 256/745 mem0 cells. Defensive bound-check returns
        # an empty stack info instead of crashing.
        from appworld.common import errors as _aw_errors
        if not getattr(_aw_errors, "_steer_traceback_patched", False):
            _orig_tb = _aw_errors.get_traceback_stack_data
            def _safe_tb(*args, **kw):
                try:
                    return _orig_tb(*args, **kw)
                except Exception:
                    return []
            _aw_errors.get_traceback_stack_data = _safe_tb
            _aw_errors._steer_traceback_patched = True
    except Exception as e:
        return None, 0, 0, 0, f"import failed: {type(e).__name__}: {e}"

    try:
        task = world.task
        gt = task.ground_truth
        tracker = TestTracker(test_data=None,
                              difficulty=gt.metadata.get("difficulty", 1),
                              suppress_errors=True)
        pub = Munch(deepcopy(dict(gt.public_data)))
        priv = Munch(deepcopy(dict(gt.private_data)))

        # Apply the strict patches (match aw_post_eval.py: only the two
        # normalization patches; do NOT apply _strip_picky_blocks or
        # _relax_set_equality which would over-soften vs. A's official eval).
        code = _extend_ignore_lists(eval_code)
        if gt.answer is None:
            code = _strip_answer_match_if_no_gt_answer(code)

        # end_db_home_path must be the live in-memory write target — see
        # comment in aw_post_eval.py and appworld.evaluator.evaluate_task.
        pair = ModelCollectionPair(
            start_db_home_path=task.model_collection.from_db_home_path,
            start_model_collection=task.model_collection,
            end_db_home_path=world.models.to_db_home_path,
            end_model_collection=world.models,
        )
        main_user = pair.start.admin.MainUser.find_one(**task.supervisor)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
            tf.write(code.encode())
            tmp = tf.name
        try:
            spec = importlib.util.spec_from_file_location(
                f"__bz_inline_{_uuid.uuid4().hex[:6]}__", tmp)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.evaluate(test=tracker, public_data=pub, private_data=priv,
                            main_user=main_user, models=pair,
                            ground_truth_answer=gt.answer)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

        n_pass = len(tracker.passes)
        n_fail = len(tracker.failures)
        n_total = n_pass + n_fail
        if n_total == 0:
            return None, 0, 0, 0, "no assertions"
        return n_pass / n_total, n_pass, n_fail, n_total, None
    except Exception as e:
        import traceback
        return None, 0, 0, 0, f"{type(e).__name__}: {str(e)[:200]} | {traceback.format_exc()[-200:]}"


def item_key(item):
    """Extract (task_ref, intervention_idx) for skip-done."""
    if isinstance(item, str):
        return (item, None)
    return (item["base_task_id"], int(item["intervention_index"]))


def build_workitems(*, condition, phase, framework, limit=None):
    """Return list of work items for the given condition."""
    if condition == "A":
        items = [t.strip() for t in TASK_IDS.read_text().split() if t.strip()]
    else:
        items = []
        for i, line in enumerate(INTERVENTIONS.open()):
            if not line.strip():
                continue
            d = json.loads(line)
            # Skip cells marked as skipped (no valid B for this type+task)
            if d.get("skip") is True or d.get("_skipped"):
                continue
            # Cancel cells have no B-task / no eval — trajectory-based scoring
            # via steps_to_cancel. Only meaningful in A_to_B.
            if d.get("type") == "Cancel":
                if condition != "A_to_B":
                    continue
            # Multistep cells: have `interventions` list + `inject_ratios`, no
            # single modified_spec_b / evaluation_code (each iv has its own).
            # Only meaningful in A_to_B; run_one handles the multi-inject loop.
            elif d.get("_multistep") or d.get("interventions"):
                if condition != "A_to_B":
                    continue
            elif not d.get("modified_spec_b") or not d.get("evaluation_code"):
                continue
            d.setdefault("intervention_index", i)
            d["_condition"] = condition
            d["_phase"] = phase
            d["_framework"] = framework
            items.append(d)
    return items[:limit] if limit else items


def _endpoint(task_id: str) -> str:
    # API_BASE_URL override → use external API (OpenAI / OpenRouter / Together / etc.)
    override = os.environ.get("API_BASE_URL")
    if override:
        return override.rstrip("/")
    # Auto-route API models to their cloud endpoint (never fall back to vLLM).
    mid = (MODEL_ID or "").lower()
    if mid.startswith("gpt-") or mid.startswith("o1-") or mid.startswith("o3-") or mid.startswith("o4-"):
        return "https://api.openai.com/v1"
    if mid.startswith("gemini-"):
        return "https://generativelanguage.googleapis.com/v1beta/openai"
    if mid.startswith("claude-"):
        return "https://api.anthropic.com/v1"
    import hashlib
    h = int(hashlib.sha1(str(task_id).encode()).hexdigest(), 16) % NUM_ENDPOINTS
    host = os.environ.get("VLLM_HOST", "localhost")
    return f"http://{host}:{BASE_PORT + h}/v1"


def _api_key_for_model() -> str:
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
        timeout=300.0, max_retries=15,  # absorb vLLM TCP-backlog refusals at high concurrency
    )


def _call_model(client, messages: list[dict], budget=None) -> str:
    from interventionbench.models.base import ChatMessage
    typed = [ChatMessage(role=m["role"], content=m["content"]) for m in messages
             if m["role"] in ("user", "assistant", "system")]
    # Some vLLM chat templates (Qwen3.6) reject system messages that appear after
    # non-system messages ("System message must be at the beginning"). Hoist all
    # system messages to the start as a single concatenated system message.
    if any(m.role == "system" for m in typed) and any(m.role != "system" for m in typed):
        sys_chunks = [m.content for m in typed if m.role == "system"]
        non_sys = [m for m in typed if m.role != "system"]
        if len(sys_chunks) > 1 or (sys_chunks and typed[0].role != "system"):
            typed = [ChatMessage(role="system", content="\n\n".join(sys_chunks))] + non_sys
    # Mistral/Mixtral chat templates reject system role and require strict user/assistant alternation.
    if "Mixtral" in MODEL_ID or "Mistral" in MODEL_ID:
        # 1) Merge system into first user (or promote to user if no non-system).
        sys_chunks = [m.content for m in typed if m.role == "system"]
        non_sys = [m for m in typed if m.role != "system"]
        if sys_chunks and non_sys:
            sys_blob = "\n\n".join(sys_chunks)
            first = non_sys[0]
            if first.role == "user":
                non_sys[0] = ChatMessage(role="user", content=f"{sys_blob}\n\n{first.content}")
            else:
                non_sys.insert(0, ChatMessage(role="user", content=sys_blob))
            typed = non_sys
        elif sys_chunks and not non_sys:
            typed = [ChatMessage(role="user", content="\n\n".join(sys_chunks))]
        # 2) Coalesce consecutive same-role messages (Mixtral requires strict alternation).
        coalesced: list[ChatMessage] = []
        for m in typed:
            if coalesced and coalesced[-1].role == m.role:
                coalesced[-1] = ChatMessage(role=m.role, content=f"{coalesced[-1].content}\n\n{m.content}")
            else:
                coalesced.append(m)
        typed = coalesced
    # Qwen3-specific reasoning toggle; harmless on other models if vLLM ignores unknown kwargs,
    # but some models error on unknown chat_template_kwargs — only send for Qwen3.
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if "Qwen3" in MODEL_ID else None
    response = client.chat(
        typed, max_tokens=2048, temperature=0.0,
        extra_body=extra_body,
    )
    if budget is not None:
        budget.add(getattr(response, "usage", None))
    return response.content or ""


try:
    from interventionbench.runner.budget import BudgetExceeded, TrajectoryBudget
except ImportError:
    class BudgetExceeded(Exception): pass
    class TrajectoryBudget: pass


def _react_loop(world, messages, steps, client, max_steps,
                inject_after=None, inject_text=None,
                inject_after_list=None, inject_text_list=None,
                inject_type_list=None,
                step_offset=0, initial_obs=None,
                per_step_hook=None, budget=None,
                a_spec=None):
    """ReAct loop. Returns (success, score, inject_actual_step, n_assistant).

    Single-inject (legacy): pass inject_after + inject_text.
    Multi-step inject: pass inject_after_list + inject_text_list (parallel lists).
    inject_after / inject_after_list are assistant_count thresholds (relative
    to this _react_loop call — step_offset is added when logging abs_step).
    inject_actual returns the FIRST inject's abs_step.

    Ack mitigation: if USE_ACK env var is set, each injected message gets
    ACK_PROMPT_SUFFIX appended.

    per_step_hook(step_idx, recent_steps, world) -> str | None: if non-None, the
    string is prepended to the next observation. Used by AMem (post-step note +
    retrieval) and AdaPlanner (refine signal).
    """
    from interventionbench.domains.appworld import _extract_code

    # Normalize: single-inject -> wrap into lists for unified handling.
    if inject_after_list is None:
        inject_after_list = [inject_after] if inject_after is not None else []
        inject_text_list = [inject_text] if inject_text is not None else []
    inject_after_list = list(inject_after_list)
    inject_text_list = list(inject_text_list)
    assert len(inject_after_list) == len(inject_text_list), \
        "inject_after_list and inject_text_list must have same length"

    execution_outputs = [initial_obs] if initial_obs is not None else []
    pending_hook_ctx: str | None = None  # text to prepend to next observation
    assistant_count = 0
    next_inject_idx = 0  # next entry in inject_*_list to fire
    inject_actual = None  # records FIRST inject's abs_step
    last_code = None
    repeat_code_count = 0
    no_code_count = 0
    STUCK_THRESHOLD = 5

    for local_step in range(1, max_steps + 1):
        abs_step = local_step + step_offset

        if (next_inject_idx < len(inject_after_list)
                and assistant_count >= inject_after_list[next_inject_idx]):
            this_text = inject_text_list[next_inject_idx]
            if USE_ACK:
                this_text = this_text + ACK_PROMPT_SUFFIX
            # TypeHint: append a 1-sentence type-specific cue right after the user
            # intervention text. The agent gets ground-truth signaling about
            # whether to swap / add / drop / cancel, so failures cannot be
            # attributed to ambiguity about the intervention's intent.
            if USE_TYPEHINT and inject_type_list is not None:
                _this_type = inject_type_list[next_inject_idx] if next_inject_idx < len(inject_type_list) else None
                _hint = TYPE_HINTS.get(_this_type or "", "")
                if _hint:
                    this_text = f"{this_text}\n\n[INTERVENTION HINT — type={_this_type}] {_hint}"
            _this_type_iv = (inject_type_list[next_inject_idx]
                             if inject_type_list is not None and next_inject_idx < len(inject_type_list)
                             else None)
            _is_add = _this_type_iv and "Add" in _this_type_iv
            if USE_IV_RESTATE_A and _is_add and a_spec:
                inject_msg = f"User says: {a_spec} {this_text}"
            else:
                inject_msg = f"User says: {this_text}. Please continue with this update."
            messages.append({"role": "user", "content": inject_msg})
            steps.append({"role": "user", "content": inject_msg,
                          "metadata": {"step": abs_step, "is_intervention": True,
                                       "inject_seq": next_inject_idx,
                                       "ack_used": USE_ACK}})
            if inject_actual is None:
                inject_actual = abs_step
            next_inject_idx += 1
        elif execution_outputs:
            obs = execution_outputs[-1] or ""
            if pending_hook_ctx:
                obs = f"{pending_hook_ctx}\n\n{obs}"
                pending_hook_ctx = None
            obs_msg = "Output:\n```\n" + obs + ("" if obs.endswith("\n") else "\n") + "```\n\n"
            messages.append({"role": "user", "content": obs_msg})
            steps.append({"role": "observation", "content": obs, "metadata": {"step": abs_step}})

        try:
            assistant_text = _call_model(client, messages, budget=budget)
        except BudgetExceeded as exc:
            steps.append({"role": "observation",
                          "content": f"[BUDGET EXCEEDED] {exc}",
                          "metadata": {"step": abs_step, "budget_exceeded": True,
                                       "budget": budget.snapshot() if budget else {}}})
            break
        except Exception as exc:
            steps.append({"role": "observation", "content": f"[MODEL ERROR] {exc}",
                          "metadata": {"step": abs_step, "error": True}})
            break

        code, fixed_text = _extract_code(assistant_text)
        messages.append({"role": "assistant", "content": fixed_text + "\n\n"})
        steps.append({"role": "assistant", "content": assistant_text,
                      "metadata": {"step": abs_step, "has_code": bool(code), "post_injection": next_inject_idx > 0}})
        assistant_count += 1

        if not code:
            no_code_count += 1
            if world.task_completed():
                break
            if no_code_count >= STUCK_THRESHOLD:
                steps.append({"role": "observation", "content": "[STUCK: no-code repeat ≥5]",
                              "metadata": {"step": abs_step, "stuck": "no_code"}})
                break
            execution_outputs = ["(no code executed)"]
            continue
        no_code_count = 0

        if code == last_code:
            repeat_code_count += 1
            if repeat_code_count >= STUCK_THRESHOLD:
                steps.append({"role": "observation", "content": "[STUCK: same-code repeat ≥5]",
                              "metadata": {"step": abs_step, "stuck": "same_code"}})
                break
        else:
            repeat_code_count = 0
        last_code = code

        res = world.batch_execute([code])
        execution_outputs = [res[0] if res else ""]

        # Per-step framework hook (AMem, AdaPlanner). Best-effort; exceptions silenced.
        if per_step_hook is not None:
            try:
                pending_hook_ctx = per_step_hook(abs_step, steps[-8:], world)
            except Exception:
                pending_hook_ctx = None

        if world.task_completed():
            break

    try:
        tt = world.evaluate(suppress_errors=True)
        return bool(tt.success), float(tt.pass_percentage) / 100.0, inject_actual, assistant_count
    except Exception:
        return False, 0.0, inject_actual, assistant_count


def _load_a_steps(task_id: str) -> list[dict]:
    """Load A trajectory's assistant steps for this task (for replay)."""
    from interventionbench.runner.storage import trajectories_path
    p = trajectories_path(DOMAIN, "phase1", "vanilla", "A")
    if not p.exists():
        return []
    for line in p.open():
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue
        if t.get("task_ref") == task_id:
            return [s for s in (t.get("steps") or []) if s.get("role") == "assistant"]
    return []


def run_one(item):
    """Worker entry. Item is task_id string (A) or intervention dict (B/AB)."""
    from appworld import AppWorld
    from interventionbench.domains.appworld import _render_prompt, _prompt_to_messages, _extract_code
    from interventionbench.runner.storage import make_trajectory
    from interventionbench.runner.workers._phase2_hooks import framework_config

    if isinstance(item, str):
        # A condition
        task_id = item
        condition = "A"
        iv_idx, iv_type, iv_text, spec_b = None, None, None, None
        framework = "vanilla"
    else:
        task_id = item["base_task_id"]
        iv_idx = int(item["intervention_index"])
        # Multi-step cell: top-level "type"/"intervention_text"/"modified_spec_b"
        # are absent; derive from interventions[] list.
        if isinstance(item.get("interventions"), list) and len(item["interventions"]) >= 2:
            ivs0 = item["interventions"]
            iv_type = "MultiStep[" + "+".join(iv.get("type", "?") for iv in ivs0) + "]"
            iv_text = ivs0[0].get("text") or ivs0[0].get("intervention_text")
            spec_b = ivs0[-1].get("modified_spec_b") or item.get("modified_spec_b")
        else:
            iv_type = item["type"]
            iv_text = item.get("intervention_text")
            spec_b = item["modified_spec_b"]
        condition = item["_condition"]
        framework = item.get("_framework") or "vanilla"

    fw_cfg = framework_config(framework, condition, task_id, iv_idx, domain="appworld")
    t0 = time.time()
    client = _make_client(task_id)
    exp = f"{condition}_{task_id}_{iv_idx if iv_idx is not None else 'A'}_{framework}"
    steps = []
    success, score, n_steps, inject_actual = False, 0.0, 0, None
    inline_b_score = None
    # Per-trajectory token + cost budget (raises BudgetExceeded inside _call_model).
    budget = TrajectoryBudget(
        max_total_tokens=int(os.environ.get("MAX_TOKENS_PER_CELL", "300000")),
        max_cost_usd=float(os.environ.get("MAX_COST_PER_CELL_USD", "1.0")),
        max_calls=int(os.environ.get("MAX_CALLS_PER_CELL", "200")),
        model_id=MODEL_ID,
    )
    extra = {"model_id": MODEL_ID, "framework": framework}

    def _apply_framework_setup(world, messages):
        """Insert prefix_messages and execute setup_code. Mutates messages in-place."""
        if fw_cfg.prefix_messages:
            for pm in reversed(fw_cfg.prefix_messages):
                # Insert after the first system message (the task prompt).
                insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                messages.insert(insert_at, dict(pm))
                steps.append({"role": "system", "content": pm["content"],
                              "metadata": {"framework_prefix": fw_cfg.name}})
        if fw_cfg.setup_code:
            try:
                world.batch_execute([fw_cfg.setup_code])
                steps.append({"role": "observation",
                              "content": f"[setup_code executed: {len(fw_cfg.setup_code)} chars, framework={fw_cfg.name}]",
                              "metadata": {"framework_setup": fw_cfg.name}})
            except Exception as exc:
                steps.append({"role": "observation",
                              "content": f"[setup_code error: {exc}]",
                              "metadata": {"framework_setup": fw_cfg.name, "error": True}})

    try:
        if condition == "A":
            with AppWorld(task_id=task_id, experiment_name=exp) as world:
                instr = world.task.instruction
                messages = _prompt_to_messages(_render_prompt(instr, world.task.supervisor))
                steps.append({"role": "system", "content": f"Task: {instr}", "metadata": {"task_id": task_id}})
                _apply_framework_setup(world, messages)
                success, score, _, n_steps = _react_loop(
                    world, messages, steps, client, MAX_STEPS,
                    per_step_hook=fw_cfg.per_step_hook, budget=budget)

        elif condition == "B":
            with AppWorld(task_id=task_id, experiment_name=exp) as world:
                world.task.instruction = spec_b
                messages = _prompt_to_messages(_render_prompt(spec_b, world.task.supervisor))
                steps.append({"role": "system", "content": f"Task: {spec_b}",
                              "metadata": {"task_id": task_id, "intervention_idx": iv_idx, "type": iv_type}})
                _apply_framework_setup(world, messages)
                success, score, _, n_steps = _react_loop(
                    world, messages, steps, client, MAX_STEPS,
                    per_step_hook=fw_cfg.per_step_hook, budget=budget)
                # Inline B-eval against the live world (replaces post-hoc replay).
                eval_code = item.get("evaluation_code")
                inline_b_score, _np, _nf, _nt, _err = _eval_y_inline(world, eval_code)
                extra["b_eval_inline_n_pass"] = _np
                extra["b_eval_inline_n_fail"] = _nf
                extra["b_eval_inline_n_total"] = _nt
                if _err:
                    extra["b_eval_inline_error"] = _err

        else:  # A_to_B
            a_assistants = _load_a_steps(task_id)
            if not a_assistants:
                return None
            n_a = len(a_assistants)

            # Multi-step detection: cell has "interventions" list (>=2) + "inject_ratios".
            # Otherwise: single-inject path (legacy + timing/ack experiments).
            multi_step = isinstance(item.get("interventions"), list) and len(item["interventions"]) >= 2
            if multi_step:
                ivs = item["interventions"]
                ratios = item.get("inject_ratios") or [0.5, 0.75]
                inject_steps_abs = [max(1, round(n_a * r)) for r in ratios]
                for i in range(1, len(inject_steps_abs)):
                    inject_steps_abs[i] = max(inject_steps_abs[i], inject_steps_abs[i-1] + 2)
                inject_steps_abs = [min(s, n_a) for s in inject_steps_abs]
                inject_texts = [iv.get("text") or iv.get("intervention_text") for iv in ivs]
                inject_types_for_loop = [iv.get("type") for iv in ivs]
                iv_text = inject_texts[0]  # legacy field = first iv's text
            else:
                inject_after = max(1, round(n_a * INJECT_FRAC))
                inject_steps_abs = [inject_after]
                inject_texts = [iv_text]
                inject_types_for_loop = [iv_type]
                ivs = None

            replay_until = inject_steps_abs[0] - 1
            extra["a_total_steps"] = n_a
            extra["inject_step_planned"] = inject_steps_abs[0]
            extra["inject_steps_planned"] = inject_steps_abs
            extra["inject_frac"] = INJECT_FRAC
            extra["use_ack"] = USE_ACK
            extra["multi_step"] = multi_step
            extra["intervention_text"] = iv_text
            if multi_step:
                extra["intervention_texts"] = inject_texts
                extra["interventions_metadata"] = [
                    {"text": t, "type": iv.get("type"), "idx": iv.get("idx")}
                    for t, iv in zip(inject_texts, ivs)
                ]

            # ----- Reflexion N-trial loop wrapper -----
            # Vanilla / non-Reflexion frameworks: n_trials=1 (single attempt).
            # Reflexion: re-init the world per trial, replay A actions, inject B,
            # run framework. If b_eval fails, generate a reflection and prepend
            # it to the next trial's system prefix.
            n_trials = max(1, fw_cfg.n_trials)
            reflections: list[str] = []
            best_b_score: float | None = None
            best_b_extras: dict = {}
            best_a_score: float = 0.0
            best_inject: int | None = None
            best_n_steps: int = 0
            best_steps_snapshot: list = []

            for trial_idx in range(n_trials):
                trial_steps: list[dict] = []
                trial_exp = f"{exp}_t{trial_idx}" if n_trials > 1 else exp
                with AppWorld(task_id=task_id, experiment_name=trial_exp) as world:
                    instr = world.task.instruction
                    messages = _prompt_to_messages(_render_prompt(instr, world.task.supervisor))
                    messages = _maybe_prepend_sysprompt(messages)
                    trial_steps.append({"role": "system", "content": f"Task: {instr}",
                                        "metadata": {"task_id": task_id, "intervention_idx": iv_idx,
                                                     "type": iv_type, "inject_after": inject_steps_abs[0],
                                                     "inject_steps": inject_steps_abs,
                                                     "a_total_steps": n_a, "trial": trial_idx}})

                    # Reflexion: prepend prior reflections as a system message.
                    if reflections:
                        from interventionbench.runner.workers._phase2_hooks import REFLEXION_SYSTEM_TEMPLATE
                        ref_text = REFLEXION_SYSTEM_TEMPLATE.format(
                            reflections="\n".join(f"- Trial {i+1}: {r}" for i, r in enumerate(reflections))
                        )
                        messages.insert(1, {"role": "system", "content": ref_text})
                        trial_steps.append({"role": "system", "content": ref_text,
                                            "metadata": {"reflexion_trial": trial_idx}})

                    # Apply framework setup (prefix + setup_code) BEFORE replay.
                    def _apply_setup_inner(world, messages):
                        if fw_cfg.prefix_messages:
                            for pm in reversed(fw_cfg.prefix_messages):
                                insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                                messages.insert(insert_at, dict(pm))
                                trial_steps.append({"role": "system", "content": pm["content"],
                                                    "metadata": {"framework_prefix": fw_cfg.name}})
                        if fw_cfg.setup_code:
                            try:
                                world.batch_execute([fw_cfg.setup_code])
                                trial_steps.append({"role": "observation",
                                                    "content": f"[setup_code executed: {len(fw_cfg.setup_code)} chars, framework={fw_cfg.name}]",
                                                    "metadata": {"framework_setup": fw_cfg.name}})
                            except Exception as exc:
                                trial_steps.append({"role": "observation",
                                                    "content": f"[setup_code error: {exc}]",
                                                    "metadata": {"framework_setup": fw_cfg.name, "error": True}})
                    _apply_setup_inner(world, messages)

                    # ----- REPLAY phase (always: ensures intervention timing parity with Phase 1) -----
                    execution_outputs = []
                    n_replayed = 0
                    for i in range(replay_until):
                        a_text = a_assistants[i].get("content") or ""
                        code, fixed_text = _extract_code(a_text)
                        if execution_outputs:
                            obs = execution_outputs[-1] or ""
                            obs_msg = "Output:\n```\n" + obs + ("" if obs.endswith("\n") else "\n") + "```\n\n"
                            messages.append({"role": "user", "content": obs_msg})
                            trial_steps.append({"role": "observation", "content": obs,
                                                "metadata": {"step": i + 1, "replay": True}})
                        messages.append({"role": "assistant", "content": fixed_text + "\n\n"})
                        trial_steps.append({"role": "assistant", "content": a_text,
                                            "metadata": {"step": i + 1, "has_code": bool(code), "replay": True}})
                        if code:
                            res = world.batch_execute([code])
                            execution_outputs = [res[0] if res else ""]
                        if world.task_completed():
                            break
                        n_replayed = i + 1

                    # ----- inject + continue -----
                    # Build per-call inject_after_list (relative to post-replay).
                    # First inject at count 0 (immediately after replay), subsequent
                    # at deltas from inject_steps_abs[0].
                    rel_after = [s - inject_steps_abs[0] for s in inject_steps_abs]
                    remaining = max(1, MAX_STEPS - n_replayed)
                    t_success, t_score, t_inject, t_n_post = _react_loop(
                        world, messages, trial_steps, client, max_steps=remaining,
                        inject_after_list=rel_after,
                        inject_text_list=inject_texts,
                        inject_type_list=inject_types_for_loop,
                        step_offset=n_replayed,
                        initial_obs=execution_outputs[-1] if execution_outputs else None,
                        per_step_hook=fw_cfg.per_step_hook, budget=budget,
                        a_spec=instr)
                    t_n_steps = n_replayed + t_n_post

                    # Eval THIS trial's b_score.
                    if multi_step:
                        # Run each iv's evaluation_code separately; min score = combined.
                        sub_evals = []
                        for iv in ivs:
                            sub_code = iv.get("evaluation_code")
                            sub_score, sub_np, sub_nf, sub_nt, sub_err = _eval_y_inline(world, sub_code)
                            sub_evals.append({"score": sub_score, "n_pass": sub_np,
                                              "n_fail": sub_nf, "n_total": sub_nt,
                                              "error": sub_err, "iv_idx": iv.get("idx")})
                        valid = [e["score"] for e in sub_evals if e["score"] is not None]
                        t_b_score = min(valid) if valid else None
                        t_np = sum((e["n_pass"] or 0) for e in sub_evals)
                        t_nf = sum((e["n_fail"] or 0) for e in sub_evals)
                        t_nt = sum((e["n_total"] or 0) for e in sub_evals)
                        t_err = "; ".join(e["error"] for e in sub_evals if e["error"]) or None
                        extra["sub_evals"] = sub_evals
                    else:
                        eval_code = item.get("evaluation_code")
                        t_b_score, t_np, t_nf, t_nt, t_err = _eval_y_inline(world, eval_code)
                    trial_b_pass = (t_b_score is not None and t_b_score >= 0.999)

                    # Track best trial. For goal_reflexion on Add, we use a joint
                    # pass-aware metric: prefer trial with (a≥0.5 AND b=1.0).
                    # For others, follow Reflexion paper convention (final trial).
                    is_add_goal = (fw_cfg.name == "goal_reflexion"
                                   and iv_type and "Add" in iv_type)
                    trial_relaxed_pass = (
                        (t_b_score or 0) >= 0.999 and (t_score or 0) >= 0.5)
                    if is_add_goal:
                        # Best-of-trials for goal_reflexion: keep trial that
                        # satisfies Add criterion, falling back to higher b_score.
                        prev_pass = (best_b_score is not None
                                      and best_b_score >= 0.999
                                      and best_a_score >= 0.5)
                        if trial_relaxed_pass and not prev_pass:
                            best_b_score = t_b_score; best_a_score = t_score
                            best_inject = t_inject; best_n_steps = t_n_steps
                            best_steps_snapshot = list(trial_steps)
                            best_b_extras = {"b_eval_inline_n_pass": t_np,
                                "b_eval_inline_n_fail": t_nf,
                                "b_eval_inline_n_total": t_nt,
                                "n_replayed": n_replayed}
                            if t_err: best_b_extras["b_eval_inline_error"] = t_err
                        elif not prev_pass and (
                              best_b_score is None
                              or (t_b_score or 0) > (best_b_score or -1)):
                            best_b_score = t_b_score; best_a_score = t_score
                            best_inject = t_inject; best_n_steps = t_n_steps
                            best_steps_snapshot = list(trial_steps)
                            best_b_extras = {"b_eval_inline_n_pass": t_np,
                                "b_eval_inline_n_fail": t_nf,
                                "b_eval_inline_n_total": t_nt,
                                "n_replayed": n_replayed}
                            if t_err: best_b_extras["b_eval_inline_error"] = t_err
                    else:
                        # Standard Reflexion paper convention: report final trial.
                        if best_b_score is None or (t_b_score or 0) > (best_b_score or -1):
                            best_b_score = t_b_score
                        best_a_score = t_score
                        best_inject = t_inject
                        best_n_steps = t_n_steps
                        best_steps_snapshot = list(trial_steps)
                        best_b_extras = {"b_eval_inline_n_pass": t_np, "b_eval_inline_n_fail": t_nf,
                                         "b_eval_inline_n_total": t_nt, "n_replayed": n_replayed}
                        if t_err:
                            best_b_extras["b_eval_inline_error"] = t_err

                    # Reflect for next trial (Reflexion only, and only if not last trial + not success).
                    if n_trials > 1 and trial_idx < n_trials - 1 and not trial_b_pass:
                        from interventionbench.runner.workers._phase2_hooks import REFLEXION_REFLECT_PROMPT_SYSTEM
                        from interventionbench.models.base import ChatMessage
                        # Render last trajectory compactly.
                        recent = "\n".join(
                            f"[{s.get('role', '?')}] {str(s.get('content', ''))[:600]}"
                            for s in trial_steps[-12:]
                        )
                        if fw_cfg.name == "goal_reflexion":
                            # Goal-decomposed reflection with full info dump.
                            # For BothAdd specifically, A and B are unrelated sub-tasks
                            # and the agent typically over-attends to B (the easier one).
                            # Add explicit A-first ordering hint.
                            is_both_add = (iv_type == "BothAdd")
                            ordering_hint = ""
                            if is_both_add:
                                ordering_hint = (
                                    "\n[CRITICAL — BothAdd type]: Task A and Task B are "
                                    "two unrelated sub-tasks. In trial 1 you likely completed "
                                    "B (often the simpler one) and skipped A. In trial 2, "
                                    "COMPLETE A FIRST — produce A's answer or final action "
                                    "BEFORE you begin any action for B.\n"
                                )
                            reflect_prompt = (
                                f"=== TRIAL {trial_idx+1} OUTCOME ===\n"
                                f"Original task A: {instr}\n"
                                f"  → A completion score: {t_score} (target: 0.5+ for relaxed pass)\n"
                                f"Intervention B (Add type, sub-type={iv_type}): {iv_text}\n"
                                f"  → B completion score: {t_b_score} (target: 1.0 for pass)\n\n"
                                f"This is an ADD-type intervention: BOTH A and B must be completed.\n"
                                f"{ordering_hint}"
                                f"\n=== LAST 12 STEPS OF TRIAL {trial_idx+1} ===\n{recent}\n\n"
                                f"=== REFLECT FOR TRIAL {trial_idx+2} ===\n"
                                f"Identify exactly: (i) which specific action(s) of A you missed; "
                                f"(ii) which specific action(s) of B you missed. "
                                f"Then write a PLAN: line with the explicit ordered steps "
                                f"that completes BOTH A and B in the next trial."
                                + (" Put A's steps FIRST in the PLAN." if is_both_add else "")
                            )
                            sys_prompt = fw_cfg.reflect_system_prompt
                        else:
                            reflect_prompt = (
                                f"Trajectory of last attempt:\n{recent}\n\n"
                                f"User mid-task instruction: {iv_text}\n\n"
                                f"Outcome: b_pass={trial_b_pass} b_score={t_b_score}\n\n"
                                "Self-reflection:"
                            )
                            sys_prompt = REFLEXION_REFLECT_PROMPT_SYSTEM
                        try:
                            r = client.chat(
                                [ChatMessage(role="system", content=sys_prompt),
                                 ChatMessage(role="user", content=reflect_prompt)],
                                max_tokens=500, temperature=0.0,
                                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                            )
                            ref = (r.content or "")[:1200]
                        except Exception as exc:
                            ref = f"[reflection_error: {exc}]"
                        reflections.append(ref)
                    # Early-exit on trial success. For goal_reflexion on Add types,
                    # we use the Add criterion (a≥0.5 ∧ b=1.0). Otherwise standard b_pass.
                    if fw_cfg.name == "goal_reflexion" and iv_type and "Add" in iv_type:
                        if trial_b_pass and (t_score or 0) >= 0.5:
                            break
                    elif trial_b_pass:
                        break

            # Commit best/final trial's results.
            steps.extend(best_steps_snapshot)
            success = best_b_score is not None and best_b_score >= 0.999
            score = best_a_score
            inject_actual = best_inject
            n_steps = best_n_steps
            inline_b_score = best_b_score
            extra.update(best_b_extras)
            extra["n_trials_run"] = trial_idx + 1
            extra["all_reflections"] = reflections

    except Exception as exc:
        import traceback
        steps.append({"role": "error", "content": str(exc),
                      "metadata": {"trace": traceback.format_exc()[-500:]}})

    extra["elapsed_s"] = round(time.time() - t0, 1)
    extra["assistant_steps"] = n_steps

    # Cancel (BothDrop): trajectory-based score = how many actionable steps
    # the agent took after the inject point. Pass = ≤ 2 (one acknowledgment +
    # stop). Only computed for A_to_B (Cancel is filtered out of B).
    if iv_type == "Cancel" and condition == "A_to_B" and inject_actual is not None:
        post = 0
        for s in steps:
            md = s.get("metadata") or {}
            st = md.get("step")
            if st is None or st <= inject_actual:
                continue
            if s.get("role") != "assistant":
                continue
            if md.get("replay"):
                continue
            content = (s.get("content") or "").strip()
            if not content:
                continue
            post += 1
        extra["steps_to_cancel"] = post
        extra["cancel_pass"] = post <= 2

    a_score = score if condition in ("A", "A_to_B") else None
    # b_score is now set INLINE for B and A_to_B (was: None / post-hoc).
    # If inline eval failed (None) the post-hoc b_eval.jsonl side-file still
    # takes precedence in the inspector/compute_results layer.
    b_score = inline_b_score if condition in ("B", "A_to_B") else None
    if condition == "B":
        extra["a_oracle_score_on_b_run"] = score

    # Record token + cost usage snapshot for this trajectory.
    extra["budget"] = budget.snapshot()

    traj = make_trajectory(
        task_ref=task_id, condition=condition,
        intervention_idx=iv_idx, intervention_type=iv_type,
        modified_spec_b=spec_b, inject_step=inject_actual,
        a_score=a_score, b_score=b_score,
        success=success, steps=steps, extra_metadata=extra,
    )
    fired_marker = " ✓" if (condition == "A_to_B" and inject_actual) else (" ✗" if condition == "A_to_B" else "")
    b_str = f" b={b_score:.2f}" if b_score is not None else (" b=?" if condition in ("B", "A_to_B") else "")
    print(f"  [{condition}] {task_id}/{iv_idx if iv_idx is not None else '-'} {iv_type or ''}: "
          f"score={score:.2f}{b_str} steps={n_steps}{fired_marker} dt={extra['elapsed_s']:.0f}s", flush=True)
    return traj
