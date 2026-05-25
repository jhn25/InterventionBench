"""Phase 1/2 trajectory storage — single jsonl per (domain, phase, framework, condition).

Layout:
    outputs/<domain>/<phase>/<framework>/<condition>/trajectories.jsonl
        phase     = "phase1" | "phase2"
        framework = "vanilla" (phase 1) or {plan_and_solve, memgpt, ...} (phase 2)
        condition = "A" | "B" | "A_to_B"

Writer pattern: SINGLE writer in the main orchestrator. Workers return
trajectory dicts via Pool.imap_unordered; main appends each to the jsonl.
No file locks, no race conditions, no shard files, no dedup.
"""
from __future__ import annotations

import json
from pathlib import Path

import os
REPO_ROOT = Path(__file__).resolve().parents[3]
# Model namespace: outputs/{model}/<domain>/... when STEER_MODEL_NS is set.
# Default (unset): outputs/<domain>/... (legacy single-model layout for backward compat).
_MODEL_NS = os.environ.get("STEER_MODEL_NS", "").strip()
OUTPUTS_ROOT = REPO_ROOT / "outputs" / _MODEL_NS if _MODEL_NS else REPO_ROOT / "outputs"


def trajectories_path(domain: str, phase: str, framework: str, condition: str,
                      experiment_name: str = "") -> Path:
    """Single jsonl path for this run.

    experiment_name suffixes the condition dir (e.g. "A_to_B" + "timing_0.25"
    -> "A_to_B__timing_0.25"). Used for new experiments (timing ablation,
    multi-step intervention, ack mitigation) so results land in separate
    directories from the main runs.
    """
    cond_dir = condition if not experiment_name else f"{condition}__{experiment_name}"
    p = OUTPUTS_ROOT / domain / phase / framework / cond_dir / "trajectories.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_existing_keys(path: Path) -> set[tuple[str, int | None]]:
    """Read existing jsonl, return set of (task_ref, intervention_idx) keys for skip-done."""
    out = set()
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            md = t.get("metadata") or {}
            iv_idx = md.get("intervention_idx")
            out.add((str(t.get("task_ref")), iv_idx))
    return out


def append_trajectory(path: Path, traj: dict) -> None:
    """Append one trajectory dict as a JSONL line. Single-writer safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(traj, ensure_ascii=False) + "\n")


def make_trajectory(
    *,
    task_ref: str,
    condition: str,                # "A" | "B" | "A_to_B"
    intervention_idx: int | None,
    intervention_type: str | None,
    modified_spec_b: str | None,
    inject_step: int | None,        # absolute step where intervention fired (None if no inject or didn't fire)
    a_score: float | None,
    b_score: float | None,
    success: bool,
    steps: list[dict],
    extra_metadata: dict | None = None,
) -> dict:
    """Construct a unified Phase 1/2 trajectory dict."""
    metadata = {
        "intervention_idx": intervention_idx,
        "intervention_type": intervention_type,
        "condition": condition,
        "modified_spec_b": modified_spec_b,
        "inject_step": inject_step,
        "intervention_fired": (inject_step is not None) if condition == "A_to_B" else None,
        "a_score": a_score,
        "b_score": b_score,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    score = b_score if b_score is not None else (a_score if a_score is not None else 0.0)
    return {
        "task_ref": str(task_ref),
        "condition": condition,
        "intervention_idx": intervention_idx,
        "intervention_type": intervention_type,
        "modified_spec_b": modified_spec_b,
        "success": bool(success),
        "score": float(score),
        "steps": steps,
        "metadata": metadata,
    }


def load_trajectories(path: Path) -> list[dict]:
    """Load all trajectories from a jsonl file."""
    out = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out
