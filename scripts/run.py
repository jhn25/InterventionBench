#!/usr/bin/env python3
"""Unified Phase 1/2 inference runner for InterventionBench.

Usage:
    python scripts/run.py --domain {aw|wa} --condition {A|B|A_to_B} --workers N
    python scripts/run.py --domain aw --phase 2 --framework plan_and_solve --condition A_to_B

Phase 1: vanilla agent. Phase 2: framework-wrapped agent (plan_and_solve, memgpt, etc).
A→B uses REPLAY mode (replay A's recorded actions in fresh env until inject_step,
then inject intervention message and continue with model-driven loop).

Architecture:
- Single jsonl output per (domain, phase, framework, condition)
- multiprocessing.Pool fork; workers return dicts; main process appends to jsonl
- Skip already-done by (task_ref, intervention_idx) key
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Set OPENAI_BUDGET_USD env var to enforce per-job API spend cap (optional).

from interventionbench.runner.storage import (  # noqa: E402
    trajectories_path, load_existing_keys, append_trajectory,
)

DOMAIN_ALIASES = {
    "aw": "appworld", "wa": "webarena",
}

# Per-domain config (workers default, executor type)
DOMAIN_CONFIG = {
    "appworld":  {"workers_default": 16, "executor": "fork"},
    "webarena":  {"workers_default": 4,  "executor": "fork"},
}


def _import_worker(domain: str):
    """Lazy import per-domain worker module."""
    mod = importlib.import_module(f"interventionbench.runner.workers.{domain}")
    return mod


def _build_workitems(worker_mod, condition: str, phase: int, framework: str, limit: int | None):
    """Build list of work items (filtered by skip-done check happens in caller)."""
    return worker_mod.build_workitems(condition=condition, phase=phase, framework=framework, limit=limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=[
        "aw", "wa",
        "appworld", "webarena",
    ], required=True)
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--condition", choices=["A", "B", "A_to_B"], required=True)
    ap.add_argument("--framework", default="vanilla",
                    help="vanilla (phase 1) or {plan_and_solve, memgpt, adaplanner, amem, reflexion} (phase 2)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: only first N items")
    ap.add_argument("--experiment-name", default="",
                    help="Suffix output path: A_to_B__<name>. Used for timing/multistep/ack experiments.")
    args = ap.parse_args()

    domain = DOMAIN_ALIASES.get(args.domain, args.domain)
    cfg = DOMAIN_CONFIG[domain]
    workers = args.workers or cfg["workers_default"]
    phase = f"phase{args.phase}"
    framework = args.framework
    condition = args.condition

    # Setup output path (with experiment_name suffix if provided)
    out_path = trajectories_path(domain, phase, framework, condition, args.experiment_name)
    exp_tag = f" exp={args.experiment_name}" if args.experiment_name else ""
    print(f"[{domain}/{phase}/{framework}/{condition}{exp_tag}] output: {out_path}", flush=True)

    # Path-level master lock — prevents two masters from writing the same
    # trajectories.jsonl simultaneously (which produces duplicate cells).
    # Lock dies with the process, so a crashed master frees its slot automatically.
    import fcntl, hashlib
    lock_key = hashlib.sha1(str(out_path).encode()).hexdigest()[:16]
    lock_path = Path(f"/tmp/interventionbench-master-{lock_key}.lock")
    lock_fp = lock_path.open("w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"  ABORT: another master already running for {out_path} (lock {lock_path})", flush=True)
        sys.exit(0)
    lock_fp.write(f"{os.getpid()}\n{out_path}\n")
    lock_fp.flush()

    # Load worker module
    worker_mod = _import_worker(domain)

    # Build work items (each item has a key (task_ref, intervention_idx))
    items = worker_mod.build_workitems(condition=condition, phase=args.phase, framework=framework, limit=args.limit)
    if not items:
        print("No work items.", flush=True)
        return

    # Skip already-done
    done_keys = load_existing_keys(out_path)
    todo = [it for it in items if worker_mod.item_key(it) not in done_keys]
    print(f"  total={len(items)} done={len(done_keys)} todo={len(todo)} workers={workers}", flush=True)
    if not todo:
        print("All done.", flush=True)
        return

    # Run
    t0 = time.time()
    if cfg["executor"] == "thread":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker_mod.run_one, item) for item in todo]
            for fut in as_completed(futures):
                try:
                    traj = fut.result()
                except Exception as exc:
                    import traceback
                    print(f"  worker exc: {exc}", flush=True)
                    traceback.print_exc()
                    continue
                if traj is not None:
                    append_trajectory(out_path, traj)
    else:
        # 2026-05-21: Switched fork → spawn for WebArena to dodge Python 3.14
        # asyncio-loop-leak that causes Playwright sync API errors. Forked
        # workers inherit any asyncio event loop created in the parent (or in
        # earlier cells via httpx/openai/etc.), and WebArena's ScriptBrowserEnv
        # uses sync_playwright which errors when an asyncio loop is live.
        # Spawn = fresh interpreter per worker = no inherited loop. Slightly
        # higher startup cost (~1s/worker) but eliminates the 3,553-cell crash.
        start_method = "spawn" if domain in ("wa", "webarena") else "fork"
        ctx = mp.get_context(start_method)
        with ctx.Pool(workers, maxtasksperchild=(1 if domain in ("wa","webarena") else None)) as pool:
            for traj in pool.imap_unordered(worker_mod.run_one, todo, chunksize=1):
                if traj is not None:
                    append_trajectory(out_path, traj)

    elapsed = time.time() - t0
    n = sum(1 for _ in out_path.open())
    print(f"\nDONE [{domain}/{phase}/{framework}/{condition}] cells={n} elapsed={elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
