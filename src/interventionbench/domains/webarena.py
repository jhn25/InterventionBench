"""WebArena domain adapter — OFFICIAL CoT-then-Act implementation.

Uses:
- WebArena's official p_cot_id_actree_2s prompt template (verbatim)
- WebArena's official ScriptBrowserEnv (Playwright-backed browser)
- WebArena's official evaluator_router for programmatic state-based scoring
- Qwen via OpenAI-compatible vLLM endpoint

Environment requirements (docker containers):
  shopping     localhost:7770   (OneStopShop Magento)
  shopping_admin localhost:7780  (Magento admin CMS)
  forum        localhost:9999   (Postmill Reddit-clone)
  gitlab       localhost:8023   (GitLab)
  map          localhost:3000   (OpenStreetMap)
  wikipedia    localhost:8888   (Kiwix mirror, optional)
  homepage     localhost:4399   (webarena-homepage flask app)

Set env vars before running:
  export REDDIT="localhost:9999"
  export SHOPPING="localhost:7770"
  export SHOPPING_ADMIN="localhost:7780/admin"
  export GITLAB="localhost:8023"
  export WIKIPEDIA="localhost:8888"
  export MAP="localhost:3000"
  export HOMEPAGE="localhost:4399"

Image tars: http://metis.lti.cs.cmu.edu/webarena-images/
  shopping_final_0712.tar         (~63 GB)
  shopping_admin_final_0719.tar   (~9 GB)
  postmill-populated-exposed-withimg.tar (~50 GB)
  gitlab-populated-final-port8023.tar    (~72 GB)

BLOCKED status as of 2026-04-28: CMU mirror unreachable (7 bytes/sec),
archive.org returns 403. Adapter code is complete and correct; will work
once containers are available.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from interventionbench.data.models import BaseTask, Trajectory, TrajectoryStep
from interventionbench.domains.base import DomainAdapter, RunResult
from interventionbench.models.base import ChatMessage, ModelClient

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
_WEBARENA_REPO = Path(__file__).resolve().parents[3] / "data" / "raw" / "webarena"
_CONFIG_DIR = _WEBARENA_REPO / "config_files"
_PROMPT_RAW = _WEBARENA_REPO / "agent" / "prompts" / "raw" / "p_cot_id_actree_2s.py"

# ---------------------------------------------------------------------------
# Official p_cot_id_actree_2s prompt — read VERBATIM from the repo
# ---------------------------------------------------------------------------
def _load_official_prompt() -> dict:
    """Load the official prompt dict from the WebArena repo."""
    if not _PROMPT_RAW.exists():
        raise FileNotFoundError(
            f"Official prompt not found at {_PROMPT_RAW}. "
            "Clone the WebArena repo: git clone --depth=1 "
            "https://github.com/web-arena-x/webarena data/raw/webarena"
        )
    ns: dict = {}
    exec(_PROMPT_RAW.read_text(), ns)  # noqa: S102
    return ns["prompt"]


# ---------------------------------------------------------------------------
# Task config helpers
# ---------------------------------------------------------------------------
def _load_task_configs() -> list[dict]:
    """Load all 812 task configs from test.raw.json."""
    raw_json = _CONFIG_DIR / "test.raw.json"
    if not raw_json.exists():
        return []
    with raw_json.open() as f:
        return json.load(f)


def _generate_config_file(
    task_config: dict,
    config_dir: Path,
    patched_eval: dict | None = None,
) -> Path:
    """Write a per-task config JSON for the env (substituting real URLs).

    If `patched_eval` is provided, the config's `eval` dict is replaced with
    the patched version (URL placeholders still get substituted) so that the
    end-of-episode evaluator scores against the modified spec B.
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    cfg = dict(task_config)
    if patched_eval is not None:
        cfg["eval"] = dict(patched_eval)

    def _norm_url(env_val: str, default: str) -> str:
        """Return env_val with http:// scheme, stripping any existing scheme."""
        raw = env_val if env_val else default
        # Strip existing http:// or https:// so we never double-prefix
        for scheme in ("https://", "http://"):
            if raw.startswith(scheme):
                raw = raw[len(scheme):]
        return f"http://{raw}"

    # Substitute URL placeholders with real env-var values
    url_subs = {
        "__SHOPPING__": _norm_url(os.environ.get("SHOPPING", ""), "localhost:7770"),
        "__SHOPPING_ADMIN__": _norm_url(os.environ.get("SHOPPING_ADMIN", ""), "localhost:7780/admin"),
        "__REDDIT__": _norm_url(os.environ.get("REDDIT", ""), "localhost:9999"),
        "__GITLAB__": _norm_url(os.environ.get("GITLAB", ""), "localhost:8023"),
        "__MAP__": _norm_url(os.environ.get("MAP", ""), "localhost:3000"),
        "__WIKIPEDIA__": _norm_url(os.environ.get("WIKIPEDIA", ""), "localhost:8888"),
        "__HOMEPAGE__": _norm_url(os.environ.get("HOMEPAGE", ""), "localhost:4399"),
    }
    def _sub_placeholders(s: str) -> str:
        for placeholder, val in url_subs.items():
            s = s.replace(placeholder, val)
        return s

    # Substitute in start_url
    cfg["start_url"] = _sub_placeholders(cfg.get("start_url", ""))

    # Substitute in eval.program_html URLs (reference_url, program_html[*].url)
    if "eval" in cfg:
        eval_cfg = dict(cfg["eval"])
        if eval_cfg.get("reference_url"):
            eval_cfg["reference_url"] = _sub_placeholders(eval_cfg["reference_url"])
        if eval_cfg.get("program_html"):
            ph_list = []
            for ph in eval_cfg["program_html"]:
                ph = dict(ph)
                if ph.get("url"):
                    ph["url"] = _sub_placeholders(ph["url"])
                ph_list.append(ph)
            eval_cfg["program_html"] = ph_list
        cfg["eval"] = eval_cfg

    # Fix storage_state path. We prefer the UNIVERSAL state (all sites pre-
    # authenticated in one context) over the per-task single-site state when
    # available, so that the agent can navigate freely across sites for
    # cross-site interventions (e.g. BothSwap shopping->reddit). See
    # scripts/wa/build_universal_storage_state.py and
    # docs/audit/wa_multi_site_setup.md for details.
    universal_state = _WEBARENA_REPO / ".auth" / "all_sites.json"
    use_universal = universal_state.exists() and os.environ.get(
        "WA_USE_UNIVERSAL_STATE", "1"
    ) != "0"
    if use_universal:
        cfg["storage_state"] = str(universal_state)
    elif cfg.get("storage_state"):
        # Fall back to original per-task storage_state.
        # cfg["storage_state"] is like "./.auth/reddit_state.json"
        # We need to strip only the leading "./" prefix, not all leading dots/slashes.
        raw_state = cfg["storage_state"]
        if raw_state.startswith("./"):
            raw_state = raw_state[2:]  # strip leading "./" exactly
        elif raw_state.startswith("/"):
            pass  # already absolute
        state_abs = _WEBARENA_REPO / raw_state
        cfg["storage_state"] = str(state_abs) if state_abs.exists() else None

    out_path = tmp / f"{cfg['task_id']}.json"
    out_path.write_text(json.dumps(cfg))
    return out_path


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
def _site_directory_hint() -> str:
    """Return a system-prompt addendum listing every WebArena site's real URL.

    The official prompt tells the agent to visit ``http://homepage.com`` to
    discover other sites, but our hosted homepage uses the
    ``<your-server-hostname>`` placeholder in its anchor hrefs (see
    data/raw/webarena/environment_docker/webarena-homepage). With the universal
    storage_state the agent has cookies for every site; this hint provides the
    actual base URLs so it can ``goto [url]`` directly.
    """
    if os.environ.get("WA_DISABLE_SITE_HINT", "0") == "1":
        return ""
    sites = [
        ("Shopping (OneStopShop)",       os.environ.get("SHOPPING", "http://localhost:7770")),
        ("Shopping admin (Magento CMS)", os.environ.get("SHOPPING_ADMIN", "http://localhost:7780/admin")),
        ("Reddit-like forum",            os.environ.get("REDDIT", "http://localhost:9999")),
        ("GitLab",                       os.environ.get("GITLAB", "http://localhost:8023")),
        ("OpenStreetMap",                os.environ.get("MAP", "http://localhost:3000")),
        ("Wikipedia (Kiwix)",            os.environ.get("WIKIPEDIA", "http://localhost:8888")),
        ("Homepage hub",                 os.environ.get("HOMEPAGE", "http://localhost:4399")),
    ]
    lines = [
        "",
        "Available WebArena sites (you are pre-authenticated on all of them; "
        "use `goto [url]` to switch sites freely):",
    ]
    for name, url in sites:
        lines.append(f"  - {name}: {url}")
    return "\n".join(lines)


def _build_messages(
    prompt: dict,
    observation: str,
    url: str,
    objective: str,
    previous_action: str,
    max_obs_length: int = 1920,
) -> list[dict[str, str]]:
    """Build the chat messages list using the official CoT prompt structure.

    The official OpenAI chat format from CoTPromptConstructor:
      - system: intro
      - system (example_user): example[0][0]
      - system (example_assistant): example[0][1]
      - ... (for each example)
      - user: current observation formatted via template
    """
    intro = prompt["intro"] + _site_directory_hint()
    examples: list[tuple[str, str]] = [tuple(e) for e in prompt["examples"]]  # type: ignore[misc]
    template: str = prompt["template"]

    # Truncate observation (replicate tokenizer truncation with char-level fallback)
    if max_obs_length and len(observation) > max_obs_length * 4:
        observation = observation[: max_obs_length * 4]

    current = template.format(
        observation=observation,
        url=url,
        objective=objective,
        previous_action=previous_action,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": intro}]
    for x, y in examples:
        messages.append({"role": "system", "name": "example_user", "content": x})
        messages.append({"role": "system", "name": "example_assistant", "content": y})
    messages.append({"role": "user", "content": current})
    return messages


def _extract_action_from_response(response: str, prompt: dict) -> str:
    """Extract the action string from model response using official CoT logic."""
    action_splitter = prompt["meta_data"]["action_splitter"]  # "```"
    pattern = rf"{re.escape(action_splitter)}((.|\n)*?){re.escape(action_splitter)}"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Cannot parse action from response: {response[:200]}")


# ---------------------------------------------------------------------------
# Stable element-ID renumbering
# ---------------------------------------------------------------------------
# WebArena's a11y tree uses Chrome DevTools' backendDOMNodeId which is assigned
# anew on every browser session. Same page → different IDs. To make A and A→B
# trajectories deterministic for the same task, we rewrite observation IDs to
# stable per-step numbers (DFS order) and reverse-map the model's action IDs
# before executing. The env still operates on Chrome IDs internally.
class _StableIDMap:
    """Per-observation Chrome ID ⇄ stable [1, 2, 3, …] mapping."""

    def __init__(self) -> None:
        self.chrome_to_stable: dict[str, str] = {}
        self.stable_to_chrome: dict[str, str] = {}

    def stabilize_obs(self, obs_text: str) -> str:
        """Replace [chrome_id] with [stable_id] in obs text, in first-seen DFS order."""
        self.chrome_to_stable.clear()
        self.stable_to_chrome.clear()
        next_stable = [1]

        def _replace(m: re.Match) -> str:
            cid = m.group(1)
            if cid not in self.chrome_to_stable:
                sid = str(next_stable[0])
                next_stable[0] += 1
                self.chrome_to_stable[cid] = sid
                self.stable_to_chrome[sid] = cid
            return f"[{self.chrome_to_stable[cid]}]"

        return re.sub(r"\[(\d+)\]", _replace, obs_text)

    def chromify_action(self, action_str: str) -> str:
        """Replace ONLY the first [N] (the element ID) with the matching [chrome_id].
        Subsequent [N] tokens (e.g., press_enter_after flag in `type [id] [content] [1]`,
        or numeric content) are left alone.
        """
        mapped = [False]

        def _replace(m: re.Match) -> str:
            if mapped[0]:
                return m.group(0)
            mapped[0] = True
            sid = m.group(1)
            cid = self.stable_to_chrome.get(sid)
            return f"[{cid}]" if cid else m.group(0)

        return re.sub(r"\[(\d+)\]", _replace, action_str)


# ---------------------------------------------------------------------------
# Official eval helpers
# ---------------------------------------------------------------------------
def _patch_openai_compat() -> None:
    """Patch openai.error for webarena compatibility with openai>=2.0.

    The WebArena repo was written against openai==0.27.x which exposed
    openai.error.RateLimitError etc. In openai>=2.0 those moved to
    openai.RateLimitError. We inject a shim so the webarena llms module
    can be imported without error.
    """
    import types
    import openai as _openai

    if not hasattr(_openai, "error"):
        error_mod = types.ModuleType("openai.error")
        # Map old-style error names to new-style equivalents (or Exception)
        error_mod.RateLimitError = getattr(_openai, "RateLimitError", Exception)  # type: ignore[attr-defined]
        error_mod.APIError = getattr(_openai, "APIError", Exception)  # type: ignore[attr-defined]
        error_mod.APIConnectionError = getattr(_openai, "APIConnectionError", Exception)  # type: ignore[attr-defined]
        error_mod.Timeout = getattr(_openai, "APITimeoutError", Exception)  # type: ignore[attr-defined]
        error_mod.InvalidRequestError = getattr(_openai, "BadRequestError", Exception)  # type: ignore[attr-defined]
        error_mod.AuthenticationError = getattr(_openai, "AuthenticationError", Exception)  # type: ignore[attr-defined]
        error_mod.ServiceUnavailableError = getattr(_openai, "InternalServerError", Exception)  # type: ignore[attr-defined]
        _openai.error = error_mod  # type: ignore[attr-defined]
        sys.modules["openai.error"] = error_mod


def _require_webarena_env() -> None:
    """Verify that the WebArena Python package and env vars are set up."""
    if str(_WEBARENA_REPO) not in sys.path:
        sys.path.insert(0, str(_WEBARENA_REPO))

    # Patch openai.error compat before any webarena imports
    _patch_openai_compat()

    # Check env vars
    required = ["REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing WebArena env vars: {missing}\n"
            "Set them to the docker container ports, e.g.:\n"
            "  export SHOPPING=http://localhost:7770\n"
            "  export SHOPPING_ADMIN=http://localhost:7780/admin\n"
            "  export REDDIT=http://localhost:9999\n"
            "  export GITLAB=http://localhost:8023\n"
            "  export MAP=http://localhost:3000\n"
            "  export WIKIPEDIA=http://localhost:8888\n"
            "  export HOMEPAGE=http://localhost:4399\n"
        )

    try:
        import browser_env  # noqa: F401
        import evaluation_harness  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "WebArena Python packages not importable.\n"
            f"Add to PYTHONPATH: {_WEBARENA_REPO}\n"
            f"Original error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------
class WebArenaAdapter(DomainAdapter):
    """Official CoT-then-Act adapter for WebArena benchmark.

    Implements the OFFICIAL evaluation loop from run.py:
    1. ScriptBrowserEnv (headless Playwright) serves a11y tree observations
    2. p_cot_id_actree_2s prompt (verbatim) drives CoT reasoning
    3. evaluator_router scores via programmatic state checks
    4. 30-step cap with early-stop on 3x parse-failure or 3x repeat-action

    Docker containers required on localhost (see module docstring).
    """

    domain_name = "webarena"

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        max_obs_length: int = 1920,
        parsing_failure_th: int = 3,
        repeating_action_failure_th: int = 3,
        sleep_after_execution: float = 2.5,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        sites_filter: list[str] | None = None,
        min_task_id: int = 0,
    ):
        if max_steps is None:
            max_steps = int(os.environ.get("WEBARENA_MAX_STEPS", "30"))
        self.max_steps = max_steps
        self.max_obs_length = max_obs_length
        self.parsing_failure_th = parsing_failure_th
        self.repeating_action_failure_th = repeating_action_failure_th
        self.sleep_after_execution = sleep_after_execution
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.sites_filter = sites_filter  # e.g. ["shopping"] to limit domains
        self.min_task_id = min_task_id
        self._prompt: dict | None = None

    @property
    def _official_prompt(self) -> dict:
        if self._prompt is None:
            self._prompt = _load_official_prompt()
        return self._prompt

    def list_candidate_tasks(self, *, limit: int | None = None) -> list[BaseTask]:
        """Return BaseTask objects from the 812 WebArena test tasks."""
        all_configs = _load_task_configs()
        tasks: list[BaseTask] = []
        for cfg in all_configs:
            task_id = cfg["task_id"]
            if task_id < self.min_task_id:
                continue
            sites = cfg.get("sites", [])
            if self.sites_filter and not any(s in sites for s in self.sites_filter):
                continue
            tasks.append(
                BaseTask(
                    domain="webarena",
                    task_id=str(task_id),
                    spec=cfg["intent"],
                    metadata={
                        "sites": sites,
                        "start_url": cfg.get("start_url", ""),
                        "eval_types": cfg.get("eval", {}).get("eval_types", []),
                        "require_login": cfg.get("require_login", False),
                        "config": cfg,
                    },
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
        return tasks

    def run_baseline(
        self,
        task: BaseTask,
        model_client: ModelClient,
        *,
        patched_eval: dict | None = None,
        prefix_messages: list[ChatMessage] | None = None,
        step_callback=None,
        tools: list[dict] | None = None,        # accepted but unused (no tool-calling)
        setup_code: str | None = None,          # accepted but unused (no exec sandbox)
    ) -> RunResult:
        """Run one WebArena task using the official CoT-then-Act loop.

        If `patched_eval` is provided, the per-task config's `eval` block is
        replaced with the patched version before running, so end-of-episode
        scoring matches the modified spec B.

        Phase 2 hook parameters (``prefix_messages``, ``step_callback``,
        ``tools``, ``setup_code``) all default to ``None`` and are no-ops when
        not supplied. ``prefix_messages`` are merged into the first system
        message on every model call. ``step_callback`` is invoked after each
        completed step as ``step_callback(step_index=step_num,
        recent_messages=steps[-6:])`` (return value ignored). ``tools`` /
        ``setup_code`` are accepted for API parity with other adapters but
        ignored (WA has no tool-calling pipe and no code-exec sandbox).
        """
        _require_webarena_env()

        # Import webarena packages (they need PYTHONPATH set)
        if str(_WEBARENA_REPO) not in sys.path:
            sys.path.insert(0, str(_WEBARENA_REPO))

        from browser_env import (
            ActionTypes,
            ScriptBrowserEnv,
            create_stop_action,
        )
        from browser_env.actions import ActionParsingError, create_id_based_action, is_equivalent
        from browser_env.helper_functions import get_action_description
        from evaluation_harness import evaluator_router

        task_id = task.task_id
        task_config = task.metadata.get("config", {})

        steps: list[TrajectoryStep] = []
        success = False
        score = 0.0

        # Write task config to temp file (with optional patched eval rules)
        config_path = _generate_config_file(task_config, _CONFIG_DIR, patched_eval=patched_eval)

        steps.append(
            TrajectoryStep(
                role="system",
                content=f"Task {task_id}: {task.spec}",
                metadata={"task_id": task_id, "sites": task.metadata.get("sites", [])},
            )
        )

        env = ScriptBrowserEnv(
            headless=self.headless,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": self.viewport_width, "height": self.viewport_height},
            save_trace_enabled=False,
            sleep_after_execution=self.sleep_after_execution,
        )

        try:
            env.reset(options={"config_file": str(config_path)})
            obs, info = env.reset(options={"config_file": str(config_path)})
            # Bump Playwright default timeout from 30s -> 90s.  Many concurrent workers
            # hammering single-process docker apps (Magento, Postmill, GitLab) push page
            # loads well past 30s and surface as 'Timeout 30000ms exceeded while waiting
            # for event "load"', killing the cell.
            try:
                _to = int(os.environ.get("PLAYWRIGHT_TIMEOUT_MS", "90000"))
                env.page.set_default_timeout(_to)
                env.page.set_default_navigation_timeout(_to)
            except Exception:
                pass

            # Official trajectory format: alternating StateInfo / Action
            from browser_env.utils import StateInfo
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory: list[Any] = [state_info]

            meta_data: dict[str, Any] = {"action_history": ["None"]}
            prompt = self._official_prompt

            consecutive_parse_failures = 0
            consecutive_same_actions = 0
            last_action_str: str | None = None
            id_map = _StableIDMap()

            for step_num in range(1, self.max_steps + 1):
                # ---- Early stop checks (official logic) ----
                early_stop = False
                stop_reason = ""
                if consecutive_parse_failures >= self.parsing_failure_th:
                    early_stop = True
                    stop_reason = f"Parse failure threshold ({self.parsing_failure_th}) exceeded"
                if consecutive_same_actions >= self.repeating_action_failure_th:
                    early_stop = True
                    stop_reason = f"Repeating action threshold ({self.repeating_action_failure_th}) exceeded"

                if early_stop:
                    action = create_stop_action(f"Early stop: {stop_reason}")
                    trajectory.append(action)
                    steps.append(TrajectoryStep(
                        role="action",
                        content=f"stop [Early stop: {stop_reason}]",
                        metadata={"step": step_num, "early_stop": True},
                    ))
                    break

                # ---- Build observation ----
                # ScriptBrowserEnv with observation_type="accessibility_tree"
                # stores the a11y tree under the "text" key (not "accessibility_tree").
                obs_dict = state_info["observation"]
                current_obs_raw = obs_dict.get("text", "") or obs_dict.get("accessibility_tree", "")
                # Stabilize Chrome's per-session DOM IDs to deterministic [1, 2, 3, …]
                current_obs = id_map.stabilize_obs(current_obs_raw)
                current_url = state_info["info"]["page"].url
                previous_action_str = meta_data["action_history"][-1]

                steps.append(TrajectoryStep(
                    role="observation",
                    content=f"URL: {current_url}\n{current_obs[:500]}...",
                    metadata={"step": step_num, "url": current_url},
                ))

                # ---- Build prompt and call model ----
                messages_raw = _build_messages(
                    prompt=prompt,
                    observation=current_obs,
                    url=current_url,
                    objective=task.spec,
                    previous_action=previous_action_str,
                    max_obs_length=self.max_obs_length,
                )

                # Convert to ChatMessage.
                # The official prompt uses system messages with name="example_user"
                # and name="example_assistant" for few-shot examples. Many LLM APIs
                # (including Qwen via vLLM) require the system message to be first
                # only — they reject system messages after user/assistant turns.
                # We remap: name=example_user -> role=user, name=example_assistant
                # -> role=assistant, keeping the first (unnamed) system as system.
                chat_messages: list[ChatMessage] = []
                first_system_added = False
                for m in messages_raw:
                    name = m.get("name", "")
                    if name == "example_user":
                        chat_messages.append(ChatMessage(role="user", content=m["content"]))
                    elif name == "example_assistant":
                        chat_messages.append(ChatMessage(role="assistant", content=m["content"]))
                    elif m["role"] in ("system", "user", "assistant"):
                        chat_messages.append(ChatMessage(role=m["role"], content=m["content"]))  # type: ignore[arg-type]
                        # Phase 2: merge prefix_messages into the first system message
                        # (vLLM requires only one system at position 0).
                        if (
                            not first_system_added
                            and m["role"] == "system"
                            and prefix_messages
                        ):
                            sys_extra = "\n\n".join(
                                pm.content for pm in prefix_messages if pm.role == "system"
                            )
                            if sys_extra:
                                chat_messages[-1] = ChatMessage(
                                    role="system",
                                    content=chat_messages[-1].content + "\n\n" + sys_extra,
                                )
                            non_sys = [pm for pm in prefix_messages if pm.role != "system"]
                            for pm in non_sys:
                                chat_messages.append(pm)
                            first_system_added = True
                        elif m["role"] == "system":
                            first_system_added = True

                # If no system was emitted (shouldn't happen) but we want
                # prefix_messages, prepend them.
                if prefix_messages and not first_system_added:
                    chat_messages = list(prefix_messages) + chat_messages

                try:
                    _MID = os.environ.get("MODEL_ID", "")
                    if "Mixtral" in _MID or "Mistral" in _MID:
                        sys_chunks = [m.content for m in chat_messages if m.role == "system"]
                        non_sys = [m for m in chat_messages if m.role != "system"]
                        if sys_chunks:
                            sys_blob = "\n\n".join(sys_chunks)
                            if non_sys and non_sys[0].role == "user":
                                non_sys[0] = ChatMessage(role="user", content=f"{sys_blob}\n\n{non_sys[0].content}")
                            else:
                                non_sys.insert(0, ChatMessage(role="user", content=sys_blob))
                            chat_messages = non_sys
                    response = model_client.chat(
                        chat_messages,
                        max_tokens=1024,
                        temperature=0.0,
                        extra_body=({"chat_template_kwargs": {"enable_thinking": False}} if "Qwen3" in _MID else None),
                    )
                    response_text = response.content or ""
                except Exception as exc:
                    steps.append(TrajectoryStep(
                        role="observation",
                        content=f"[MODEL ERROR] {exc}",
                        metadata={"step": step_num, "error": True},
                    ))
                    action = create_stop_action(f"Model error: {exc}")
                    trajectory.append(action)
                    break

                steps.append(TrajectoryStep(
                    role="thought",
                    content=response_text,
                    metadata={"step": step_num},
                ))

                # ---- Extract action ----
                try:
                    action_str = _extract_action_from_response(response_text, prompt)
                    consecutive_parse_failures = 0
                except (ValueError, ActionParsingError):
                    consecutive_parse_failures += 1
                    steps.append(TrajectoryStep(
                        role="action",
                        content="[PARSE FAILURE]",
                        metadata={"step": step_num, "parse_failure": True},
                    ))
                    # Use NONE action (official approach)
                    from browser_env.actions import create_none_action
                    trajectory.append(create_none_action())
                    # No env step needed for NONE
                    continue

                # ---- Repeating action check ----
                if action_str == last_action_str:
                    consecutive_same_actions += 1
                else:
                    consecutive_same_actions = 0
                last_action_str = action_str

                # ---- Create Action object ----
                # Map stable IDs in model output → Chrome IDs the env expects
                action_str_chrome = id_map.chromify_action(action_str)
                try:
                    action = create_id_based_action(action_str_chrome)
                except ActionParsingError:
                    consecutive_parse_failures += 1
                    from browser_env.actions import create_none_action
                    trajectory.append(create_none_action())
                    steps.append(TrajectoryStep(
                        role="action",
                        content=f"[INVALID ACTION] {action_str}",
                        metadata={"step": step_num},
                    ))
                    continue

                # Record action in official trajectory format
                trajectory.append(action)

                action_desc = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag="id_accessibility_tree",
                    prompt_constructor=None,
                )
                meta_data["action_history"].append(action_desc)

                steps.append(TrajectoryStep(
                    role="action",
                    content=action_str,  # stable-ID form for trajectory log
                    metadata={"step": step_num, "action_desc": action_desc, "chrome_action": action_str_chrome},
                ))

                # ---- Check for STOP ----
                if action["action_type"] == ActionTypes.STOP:
                    break

                # ---- Step environment ----
                try:
                    obs, _, terminated, _, info = env.step(action)
                except Exception as exc:
                    steps.append(TrajectoryStep(
                        role="observation",
                        content=f"[ENV ERROR] {exc}",
                        metadata={"step": step_num, "error": True},
                    ))
                    trajectory.append(create_stop_action(f"Env error: {exc}"))
                    break

                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                # Phase 2 framework hook: step_callback can return a string to
                # be prepended as a synthetic observation step. This enables
                # AdaPlanner [PLAN CHECK] / AMem [retrieved_notes] to actually
                # reach the agent + persist in trajectory for audit.
                if step_callback is not None:
                    try:
                        hook_text = step_callback(
                            step_index=step_num, recent_messages=steps[-6:])
                    except Exception:
                        hook_text = None
                    if hook_text:
                        steps.append(TrajectoryStep(
                            role="observation",
                            content=str(hook_text),
                            metadata={"step": step_num,
                                      "framework_hook": True},
                        ))

                if terminated:
                    trajectory.append(create_stop_action(""))
                    break

            # ---- Official evaluation ----
            b_score = None
            b_eval_error = None
            try:
                evaluator = evaluator_router(str(config_path))
                b_score = float(evaluator(
                    trajectory=trajectory,
                    config_file=str(config_path),
                    page=env.page,
                    client=env.get_page_client(env.page),
                ))
                score = b_score
                success = score == 1.0
            except Exception as exc:
                b_eval_error = str(exc)
                steps.append(TrajectoryStep(
                    role="observation",
                    content=f"[EVAL ERROR] {exc}",
                    metadata={"eval_error": str(exc)},
                ))
                success = False
                score = 0.0

        finally:
            try:
                env.close()
            except Exception:
                pass

        traj = Trajectory(
            episode_id=str(uuid.uuid4()),
            task_ref=task.task_id,
            condition="A",
            model=getattr(model_client, "model", "unknown"),
            steps=steps,
            success=success,
            score=score,
            metadata={
                "timestamp": time.time(),
                "sites": task.metadata.get("sites", []),
                "num_steps": len([s for s in steps if s.role == "action"]),
                "eval_types": task.metadata.get("eval_types", []),
                "b_score": b_score,
                "b_eval_error": b_eval_error,
                "patched_eval_used": patched_eval is not None,
            },
        )
        return RunResult(trajectory=traj, success=success, score=score)

    def run_a_to_b(
        self,
        task: BaseTask,
        intervention: dict[str, Any],
        model_client: ModelClient,
        *,
        timing: str = "mid",
        strategy: "InterventionStrategy | None" = None,  # noqa: F821
        prefix_messages: list[ChatMessage] | None = None,
        tools: list[dict] | None = None,
        step_callback=None,
        patched_eval: dict | None = None,
        setup_code: str | None = None,            # accepted but unused (no exec sandbox)
        inject_step: int | None = None,           # if set, overrides timing-based computation
    ) -> RunResult:
        """A->B mode: run the same loop as run_baseline but inject a user
        intervention mid-trajectory. After injection, the agent's objective
        switches from the original task.spec to ``intervention["modified_spec_b"]``.

        Phase 2 hook parameters (``strategy``, ``prefix_messages``, ``tools``,
        ``step_callback``) all default to ``None`` and are wired through to be
        no-ops when not supplied.

        Args:
            task: The base WebArena task (provides start config + original spec).
            intervention: Dict with keys ``intervention_text``, ``modified_spec_b``,
                optionally ``intervention_idx``, ``type``, ``difficulty``.
            model_client: The model client (e.g. OpenAICompatibleClient).
            timing: One of "early" / "mid" / "late" — controls inject_step.
            strategy: If provided, used to generate the user-message content at
                the inject step instead of the plain "[Mid-task ...]" template.
            prefix_messages: If provided, prepended to the chat after the system
                prompt (and before few-shot examples) on every model call.
            tools: If provided, forwarded to ``model_client.chat`` (via extra_body
                fallback when the client does not accept the kwarg directly).
            step_callback: If provided, called after each completed step as
                ``step_callback(step_index=step_num, recent_messages=steps[-6:])``.
                Return value is currently ignored.
        """
        _require_webarena_env()

        if str(_WEBARENA_REPO) not in sys.path:
            sys.path.insert(0, str(_WEBARENA_REPO))

        from browser_env import (
            ActionTypes,
            ScriptBrowserEnv,
            create_stop_action,
        )
        from browser_env.actions import ActionParsingError, create_id_based_action
        from browser_env.helper_functions import get_action_description
        from evaluation_harness import evaluator_router

        task_id = task.task_id
        task_config = task.metadata.get("config", {})

        intervention_text = intervention.get("intervention_text", "")
        modified_spec = intervention.get("modified_spec_b", task.spec)
        intervention_idx = intervention.get("intervention_idx") or intervention.get("_idx")
        intervention_type = intervention.get("type", "")
        intervention_difficulty = intervention.get("difficulty", "")

        # Compute inject step. If caller passed an explicit inject_step (e.g.,
        # adaptive from A's actual trajectory length), use it. Otherwise fall
        # back to timing fraction × max_steps.
        if inject_step is None:
            if timing == "early":
                inject_step = max(2, self.max_steps // 4)
            elif timing == "late":
                inject_step = max(2, 3 * self.max_steps // 4)
            else:  # default "mid"
                inject_step = max(2, self.max_steps // 2)
        else:
            inject_step = max(2, int(inject_step))

        steps: list[TrajectoryStep] = []
        success = False
        score = 0.0

        # Generate two config files: one with patched eval (for B-eval) and one
        # with the original eval (for A-eval). Trajectory is run once against
        # the patched config so the agent's environment matches the spec the
        # patched evaluator scores.
        config_path = _generate_config_file(task_config, _CONFIG_DIR, patched_eval=patched_eval)
        original_config_path = (
            _generate_config_file(task_config, _CONFIG_DIR, patched_eval=None)
            if patched_eval is not None
            else None
        )

        steps.append(
            TrajectoryStep(
                role="system",
                content=f"Task {task_id}: {task.spec}",
                metadata={
                    "task_id": task_id,
                    "sites": task.metadata.get("sites", []),
                    "condition": "A_to_B",
                    "inject_step": inject_step,
                    "timing": timing,
                },
            )
        )

        env = ScriptBrowserEnv(
            headless=self.headless,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": self.viewport_width, "height": self.viewport_height},
            save_trace_enabled=False,
            sleep_after_execution=self.sleep_after_execution,
        )

        # Track current objective: switches to modified_spec at inject step.
        current_objective = task.spec
        injected = False

        def _client_chat(chat_messages: list[ChatMessage]):
            """Wrapper for model_client.chat. tools= forwarding disabled because
            our vLLM endpoint isn't started with --enable-auto-tool-choice
            (Phase 2 D2 smoke confirmed 400). MemGPT degrades to prompt-based
            tool simulation via MemGPTStrategy inject text."""
            _MID = os.environ.get("MODEL_ID", "")
            if "Mixtral" in _MID or "Mistral" in _MID:
                sys_chunks = [m.content for m in chat_messages if m.role == "system"]
                non_sys = [m for m in chat_messages if m.role != "system"]
                if sys_chunks:
                    sys_blob = "\n\n".join(sys_chunks)
                    if non_sys and non_sys[0].role == "user":
                        non_sys[0] = ChatMessage(role="user", content=f"{sys_blob}\n\n{non_sys[0].content}")
                    else:
                        non_sys.insert(0, ChatMessage(role="user", content=sys_blob))
                    chat_messages = non_sys
            base_extra = {"chat_template_kwargs": {"enable_thinking": False}} if "Qwen3" in _MID else None
            return model_client.chat(
                chat_messages,
                max_tokens=1024,
                temperature=0.0,
                extra_body=base_extra,
            )

        try:
            env.reset(options={"config_file": str(config_path)})
            obs, info = env.reset(options={"config_file": str(config_path)})

            from browser_env.utils import StateInfo
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory: list[Any] = [state_info]

            meta_data: dict[str, Any] = {"action_history": ["None"]}
            prompt = self._official_prompt

            consecutive_parse_failures = 0
            consecutive_same_actions = 0
            last_action_str: str | None = None
            id_map = _StableIDMap()

            for step_num in range(1, self.max_steps + 1):
                # ---- Mid-trajectory injection (before this step's model call) ----
                if not injected and step_num == inject_step:
                    if strategy is not None:
                        prior_for_strategy = [
                            ChatMessage(
                                role="user" if s.role in ("system", "user", "observation") else "assistant",
                                content=str(s.content),
                            )
                            for s in steps
                        ]
                        inject_text = strategy.make_intervention_message(
                            client=model_client,
                            prior_messages=prior_for_strategy,
                            intervention_text=intervention_text,
                            modified_spec=modified_spec,
                        )
                        steps.append(
                            TrajectoryStep(
                                role="user",
                                content=inject_text,
                                metadata={
                                    "step": step_num,
                                    "intervention": True,
                                    "strategy": strategy.name,
                                },
                            )
                        )
                    else:
                        steps.append(
                            TrajectoryStep(
                                role="user",
                                content=f"[Mid-task user intervention]: {intervention_text}",
                                metadata={"step": step_num, "intervention": True},
                            )
                        )
                    current_objective = modified_spec
                    injected = True

                # ---- Early stop checks (official logic) ----
                early_stop = False
                stop_reason = ""
                if consecutive_parse_failures >= self.parsing_failure_th:
                    early_stop = True
                    stop_reason = f"Parse failure threshold ({self.parsing_failure_th}) exceeded"
                if consecutive_same_actions >= self.repeating_action_failure_th:
                    early_stop = True
                    stop_reason = f"Repeating action threshold ({self.repeating_action_failure_th}) exceeded"

                if early_stop:
                    action = create_stop_action(f"Early stop: {stop_reason}")
                    trajectory.append(action)
                    steps.append(TrajectoryStep(
                        role="action",
                        content=f"stop [Early stop: {stop_reason}]",
                        metadata={"step": step_num, "early_stop": True},
                    ))
                    break

                # ---- Build observation ----
                obs_dict = state_info["observation"]
                current_obs_raw = obs_dict.get("text", "") or obs_dict.get("accessibility_tree", "")
                # Stabilize Chrome's per-session DOM IDs to deterministic [1, 2, 3, …]
                current_obs = id_map.stabilize_obs(current_obs_raw)
                current_url = state_info["info"]["page"].url
                previous_action_str = meta_data["action_history"][-1]

                steps.append(TrajectoryStep(
                    role="observation",
                    content=f"URL: {current_url}\n{current_obs[:500]}...",
                    metadata={"step": step_num, "url": current_url},
                ))

                # ---- Build prompt and call model ----
                messages_raw = _build_messages(
                    prompt=prompt,
                    observation=current_obs,
                    url=current_url,
                    objective=current_objective,
                    previous_action=previous_action_str,
                    max_obs_length=self.max_obs_length,
                )

                chat_messages: list[ChatMessage] = []
                first_system_added = False
                for m in messages_raw:
                    name = m.get("name", "")
                    if name == "example_user":
                        chat_messages.append(ChatMessage(role="user", content=m["content"]))
                    elif name == "example_assistant":
                        chat_messages.append(ChatMessage(role="assistant", content=m["content"]))
                    elif m["role"] in ("system", "user", "assistant"):
                        chat_messages.append(ChatMessage(role=m["role"], content=m["content"]))  # type: ignore[arg-type]
                        # Merge prefix_messages into the first system message
                        # (vLLM requires only one system at position 0).
                        if (
                            not first_system_added
                            and m["role"] == "system"
                            and prefix_messages
                        ):
                            sys_extra = "\n\n".join(
                                pm.content for pm in prefix_messages if pm.role == "system"
                            )
                            if sys_extra:
                                chat_messages[-1] = ChatMessage(
                                    role="system",
                                    content=chat_messages[-1].content + "\n\n" + sys_extra,
                                )
                            non_sys = [pm for pm in prefix_messages if pm.role != "system"]
                            for pm in non_sys:
                                chat_messages.append(pm)
                            first_system_added = True
                        elif m["role"] == "system":
                            first_system_added = True

                # If no system was emitted (shouldn't happen) but we still want
                # prefix_messages, prepend them.
                if prefix_messages and not first_system_added:
                    chat_messages = list(prefix_messages) + chat_messages

                # If we have already injected, append the intervention text as
                # an additional user message so the model sees it in context
                # too. (The current_objective swap covers the prompt template,
                # but a visible turn-style message is more reliable.)
                if injected and intervention_text:
                    chat_messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                f"[Earlier mid-task user intervention]: {intervention_text}"
                            ),
                        )
                    )

                try:
                    response = _client_chat(chat_messages)
                    response_text = response.content or ""
                except Exception as exc:
                    steps.append(TrajectoryStep(
                        role="observation",
                        content=f"[MODEL ERROR] {exc}",
                        metadata={"step": step_num, "error": True},
                    ))
                    action = create_stop_action(f"Model error: {exc}")
                    trajectory.append(action)
                    break

                steps.append(TrajectoryStep(
                    role="thought",
                    content=response_text,
                    metadata={"step": step_num},
                ))

                # ---- Extract action ----
                try:
                    action_str = _extract_action_from_response(response_text, prompt)
                    consecutive_parse_failures = 0
                except (ValueError, ActionParsingError):
                    consecutive_parse_failures += 1
                    steps.append(TrajectoryStep(
                        role="action",
                        content="[PARSE FAILURE]",
                        metadata={"step": step_num, "parse_failure": True},
                    ))
                    from browser_env.actions import create_none_action
                    trajectory.append(create_none_action())
                    if step_callback is not None:
                        try:
                            step_callback(step_index=step_num, recent_messages=steps[-6:])
                        except Exception:
                            pass
                    continue

                # ---- Repeating action check ----
                if action_str == last_action_str:
                    consecutive_same_actions += 1
                else:
                    consecutive_same_actions = 0
                last_action_str = action_str

                # ---- Create Action object ----
                # Map stable IDs in model output → Chrome IDs the env expects
                action_str_chrome = id_map.chromify_action(action_str)
                try:
                    action = create_id_based_action(action_str_chrome)
                except ActionParsingError:
                    consecutive_parse_failures += 1
                    from browser_env.actions import create_none_action
                    trajectory.append(create_none_action())
                    steps.append(TrajectoryStep(
                        role="action",
                        content=f"[INVALID ACTION] {action_str}",
                        metadata={"step": step_num},
                    ))
                    if step_callback is not None:
                        try:
                            step_callback(step_index=step_num, recent_messages=steps[-6:])
                        except Exception:
                            pass
                    continue

                trajectory.append(action)

                action_desc = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag="id_accessibility_tree",
                    prompt_constructor=None,
                )
                meta_data["action_history"].append(action_desc)

                steps.append(TrajectoryStep(
                    role="action",
                    content=action_str,
                    metadata={"step": step_num, "action_desc": action_desc},
                ))

                if action["action_type"] == ActionTypes.STOP:
                    if step_callback is not None:
                        try:
                            step_callback(step_index=step_num, recent_messages=steps[-6:])
                        except Exception:
                            pass
                    break

                # ---- Step environment ----
                try:
                    obs, _, terminated, _, info = env.step(action)
                except Exception as exc:
                    steps.append(TrajectoryStep(
                        role="observation",
                        content=f"[ENV ERROR] {exc}",
                        metadata={"step": step_num, "error": True},
                    ))
                    trajectory.append(create_stop_action(f"Env error: {exc}"))
                    break

                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                # Phase 2 framework hook: step_callback can return a string to
                # be prepended as a synthetic observation step. This enables
                # AdaPlanner [PLAN CHECK] / AMem [retrieved_notes] to actually
                # reach the agent + persist in trajectory for audit.
                if step_callback is not None:
                    try:
                        hook_text = step_callback(
                            step_index=step_num, recent_messages=steps[-6:])
                    except Exception:
                        hook_text = None
                    if hook_text:
                        steps.append(TrajectoryStep(
                            role="observation",
                            content=str(hook_text),
                            metadata={"step": step_num,
                                      "framework_hook": True},
                        ))

                if terminated:
                    trajectory.append(create_stop_action(""))
                    break

            # ---- Dual evaluation ----
            # B-eval: against patched config (matches modified spec B)
            b_score = None
            a_score = None
            b_eval_error = None
            a_eval_error = None
            try:
                evaluator = evaluator_router(str(config_path))
                b_score = float(evaluator(
                    trajectory=trajectory,
                    config_file=str(config_path),
                    page=env.page,
                    client=env.get_page_client(env.page),
                ))
            except Exception as exc:
                b_eval_error = str(exc)
                steps.append(TrajectoryStep(
                    role="observation",
                    content=f"[B-EVAL ERROR] {exc}",
                    metadata={"eval_error": str(exc), "which": "b"},
                ))
            # A-eval: against ORIGINAL config (did the model also satisfy A's spec?)
            if original_config_path is not None:
                try:
                    evaluator_a = evaluator_router(str(original_config_path))
                    a_score = float(evaluator_a(
                        trajectory=trajectory,
                        config_file=str(original_config_path),
                        page=env.page,
                        client=env.get_page_client(env.page),
                    ))
                except Exception as exc:
                    a_eval_error = str(exc)
                    steps.append(TrajectoryStep(
                        role="observation",
                        content=f"[A-EVAL ERROR] {exc}",
                        metadata={"eval_error": str(exc), "which": "a"},
                    ))
            # success/score reflect B (the only metric that matters for SR)
            if b_score is not None:
                score = b_score
                success = b_score == 1.0
            else:
                success = False
                score = 0.0

        finally:
            try:
                env.close()
            except Exception:
                pass

        traj = Trajectory(
            episode_id=str(uuid.uuid4()),
            task_ref=task.task_id,
            condition="A->B",  # type: ignore[arg-type]
            model=getattr(model_client, "model", "unknown"),
            steps=steps,
            success=success,
            score=score,
            metadata={
                "timestamp": time.time(),
                "sites": task.metadata.get("sites", []),
                "num_steps": len([s for s in steps if s.role == "action"]),
                "eval_types": task.metadata.get("eval_types", []),
                "condition": "A_to_B",
                "intervention_idx": intervention_idx,
                "intervention_type": intervention_type,
                "intervention_difficulty": intervention_difficulty,
                "intervention_text": intervention_text,
                "modified_spec_b": modified_spec,
                "inject_step": inject_step,
                "timing": timing,
                "strategy": getattr(strategy, "name", None) if strategy is not None else None,
                "injected": injected,
                "b_score": b_score,
                "a_score": a_score,
                "b_eval_error": b_eval_error,
                "a_eval_error": a_eval_error,
                "patched_eval_used": patched_eval is not None,
            },
        )
        return RunResult(trajectory=traj, success=success, score=score)
