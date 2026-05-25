"""Replay helper — load A trajectory's actions for A→B replay.

A→B replay: re-execute A's recorded actions in fresh A→B environment until
inject_step, then inject intervention message and continue with model-driven loop.
Guarantees 100% intervention firing (no env-nondeterminism early-termination).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_a_trajectory(domain: str, task_id: str) -> dict | None:
    """Load A trajectory dict for the given task. Returns None if not found.

    Honors STEER_MODEL_NS so that A→B replay reads the per-model A cache.
    For tau-bench (task_id="<sub_domain>/<base_task_id>"), the sub_domain is
    spliced into the path (matching the storage layout).
    """
    ns = os.environ.get("STEER_MODEL_NS", "").strip()
    base = REPO_ROOT / "outputs" / ns if ns else REPO_ROOT / "outputs"

    # tau-bench layout: outputs/<ns>/taubench/<sub_domain>/phase1/...
    sub_domain = None
    if domain == "taubench" and isinstance(task_id, str) and "/" in task_id:
        sub_domain, _ = task_id.split("/", 1)

    if sub_domain:
        p = base / domain / sub_domain / "phase1" / "vanilla" / "A" / "trajectories.jsonl"
    else:
        p = base / domain / "phase1" / "vanilla" / "A" / "trajectories.jsonl"

    if not p.exists():
        return None
    for line in p.open():
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue
        if str(t.get("task_ref")) == str(task_id):
            return t
    return None


def extract_steps(traj: dict, role: str) -> list[dict]:
    """Return list of step dicts with the given role from a trajectory."""
    return [s for s in (traj.get("steps") or []) if s.get("role") == role]
