#!/usr/bin/env python3
"""Replay AppWorld Add trajectories and score both tasks assertion by assertion.

At each trajectory's final state, this scorer applies both the original-task and
added-task evaluators. A trajectory passes only when every substantive assertion
on both sides passes. An assertion is excluded only when it clearly forbids a
state change required by the other task; mixed and ambiguous assertions remain
substantive. Evaluator outcomes use the 0.999 pass threshold.

AppWorld must be installed separately. Set ``APPWORLD_ROOT`` to the directory
containing its ``data`` directory before running this script.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import multiprocessing as mp
import os
import re
import signal
import sys
import tempfile
import textwrap
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
PASS = 0.999


class ReplayTimeout(TimeoutError):
    """Raised when one recorded code block exceeds the replay time limit."""


def _timeout_handler(signum, frame):
    raise ReplayTimeout("recorded code block exceeded replay timeout")


ANTICHEAT_PATTERN = re.compile(
    r"anti[- ]?cheat|assert\s+(?:there (?:are|were)\s+)?no model changes|"
    r"no other|only .{0,60}(?:changed|modified)|did not change|unchanged|"
    r"no new\s|none .{0,60}(?:changed|updated|deleted|removed|added)|"
    r"len\(\s*changed|==\s*set\(\)|-\s*\{[^}]*\}\s*==",
    re.I | re.S,
)
MIXED_PATTERN = re.compile(
    r"changed_model_names|changed_field_names|model changes match|"
    r"(?:\b0\b|\bno\b|\bnone\b)\s+(?:\w+\s+){0,5}"
    r"(?:added|updated|deleted|removed|changed|records)",
    re.I | re.S,
)

_AUXILIARY_MODEL_TYPES = [
    "venmo.Notification", "venmo.User", "venmo.AccountAccessToken",
    "phone.AccountAccessToken", "phone.User",
    "gmail.AccountAccessToken", "gmail.User",
    "spotify.Notification", "spotify.AccountAccessToken", "spotify.User",
    "splitwise.Notification", "splitwise.User", "splitwise.AccountAccessToken",
    "splitwise.GroupMembership", "splitwise.Group",
    "todoist.AccountAccessToken", "todoist.User", "todoist.Project", "todoist.Section",
    "amazon.User", "amazon.AccountAccessToken", "amazon.BrowsingHistoryEntry",
    "simple_note.AccountAccessToken", "simple_note.User",
    "file_system.AccountAccessToken", "file_system.User",
    "supervisor.Task",
]


def normalized_requirement(value: str | None) -> str:
    return textwrap.dedent(value or "").strip()


def extract_code(text: str) -> str:
    """Return the first complete or partial Python code block in a response."""
    match = re.search(r"```python\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.match(r".*```python\n(.*)", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def guard_recorded_loops(code: str, max_iterations: int = 20) -> tuple[str, int]:
    """Cap agent-authored while loops so malformed pagination cannot hang replay."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0

    class Guard(ast.NodeTransformer):
        def __init__(self) -> None:
            self.count = 0

        def visit_While(self, node: ast.While):
            self.generic_visit(node)
            self.count += 1
            replacement = ast.For(
                target=ast.Name(id=f"__replay_loop_guard_{self.count}", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(max_iterations)], keywords=[],
                ),
                body=[
                    ast.If(
                        test=ast.UnaryOp(op=ast.Not(), operand=node.test),
                        body=[ast.Break()], orelse=[],
                    ),
                    *node.body,
                ],
                orelse=node.orelse,
                type_comment=None,
            )
            return ast.copy_location(replacement, node)

    guard = Guard()
    rewritten = guard.visit(tree)
    if not guard.count:
        return code, 0
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten), guard.count


def extract_requirement_sources(eval_code: str) -> dict[str, str]:
    """Map evaluator requirement text to its ``with test`` source block."""
    try:
        tree = ast.parse(eval_code)
    except SyntaxError:
        return {}
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "test"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                continue
            requirement = normalized_requirement(call.args[0].value)
            result[requirement] = ast.get_source_segment(eval_code, node) or ""
            break
    return result


def _set_failure_direction(trace: str) -> tuple[bool, bool]:
    lower = trace.lower()
    return "in left but not right" in lower, "in right but not left" in lower


def classify_assertion(
    requirement: str,
    source: str,
    evaluator_label: str | None,
    status: str,
    failure_trace: str = "",
) -> dict[str, Any]:
    """Conservatively classify an evaluator assertion."""
    text = f"{requirement}\n{source}"
    lower = text.lower()
    if evaluator_label == "no_op_pass":
        return {
            "class": "anticheat", "ambiguous": False,
            "classification_reason": "no_op_invariant",
        }
    if re.search(r"anti[- ]?cheat", lower):
        return {
            "class": "anticheat", "ambiguous": False,
            "classification_reason": "explicit_anticheat_language",
        }

    is_model_set = "changed_model_names" in lower or "model changes match" in lower
    is_field_set = "changed_field_names" in lower
    if status == "fail" and (is_model_set or is_field_set):
        extras, missing = _set_failure_direction(failure_trace)
        noun = "field" if is_field_set else "model"
        if extras and not missing:
            return {
                "class": "anticheat", "ambiguous": False,
                "classification_reason": f"exact_{noun}_change_set_extra_only",
            }
        if missing and not extras:
            return {
                "class": "substantive", "ambiguous": False,
                "classification_reason": f"exact_{noun}_change_set_missing_required_only",
            }
        return {
            "class": "substantive", "ambiguous": True,
            "classification_reason": f"mixed_{noun}_change_set_failure",
        }

    if ANTICHEAT_PATTERN.search(text) and not MIXED_PATTERN.search(text):
        return {
            "class": "anticheat", "ambiguous": False,
            "classification_reason": "pure_textual_invariant",
        }
    if MIXED_PATTERN.search(text) or ANTICHEAT_PATTERN.search(text):
        return {
            "class": "substantive", "ambiguous": True,
            "classification_reason": "positive_and_invariant_checks_mixed",
        }
    return {
        "class": "substantive", "ambiguous": False,
        "classification_reason": "positive_task_requirement",
    }


def tracker_assertions(tracker: Any, eval_code: str, test_data: Any = None) -> list[dict[str, Any]]:
    """Convert an AppWorld TestTracker into ordered assertion records."""
    sources = extract_requirement_sources(eval_code)
    passed = {normalized_requirement(x.get("requirement")): x for x in tracker.passes}
    failed = {normalized_requirement(x.get("requirement")): x for x in tracker.failures}
    ordered = test_data or [
        {"requirement": x.get("requirement"), "label": x.get("label")}
        for x in tracker.passes + tracker.failures
    ]
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ordered:
        requirement = normalized_requirement(item.get("requirement"))
        seen.add(requirement)
        if requirement in passed:
            entry, status, trace = passed[requirement], "pass", ""
        elif requirement in failed:
            entry, status = failed[requirement], "fail"
            trace = entry.get("trace") or ""
        else:
            entry, status, trace = item, "not_run", ""
        label = entry.get("label") or item.get("label")
        classification = classify_assertion(
            requirement, sources.get(requirement, ""), label, status, trace
        )
        blocks.append({
            "order": len(blocks), "requirement": requirement, "status": status,
            "evaluator_label": label, **classification,
            "failure_trace_excerpt": trace[-1200:] if trace else None,
        })

    for entry, status in [
        *((x, "pass") for x in tracker.passes),
        *((x, "fail") for x in tracker.failures),
    ]:
        requirement = normalized_requirement(entry.get("requirement"))
        if requirement in seen:
            continue
        trace = entry.get("trace") or ""
        classification = classify_assertion(
            requirement, sources.get(requirement, ""), entry.get("label"), status, trace
        )
        blocks.append({
            "order": len(blocks), "requirement": requirement, "status": status,
            "evaluator_label": entry.get("label"), **classification,
            "failure_trace_excerpt": trace[-1200:] if trace else None,
            "metadata_mismatch": True,
        })
    return blocks


def _extend_ignore_lists(eval_code: str) -> str:
    """Add incidental AppWorld model types to changed-model ignore lists."""
    try:
        tree = ast.parse(eval_code)
    except SyntaxError:
        return eval_code

    class Extender(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            self.generic_visit(node)
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "changed_model_names"):
                return node
            ignore = next((kw for kw in node.keywords if kw.arg == "ignore"), None)
            auxiliary = ast.List(
                elts=[ast.Constant(value=x) for x in _AUXILIARY_MODEL_TYPES],
                ctx=ast.Load(),
            )
            if ignore is None:
                node.keywords.append(ast.keyword(arg="ignore", value=auxiliary))
            elif isinstance(ignore.value, ast.List):
                present = {
                    x.value for x in ignore.value.elts
                    if isinstance(x, ast.Constant) and isinstance(x.value, str)
                }
                ignore.value.elts.extend(
                    ast.Constant(value=x) for x in _AUXILIARY_MODEL_TYPES if x not in present
                )
            else:
                ignore.value = ast.BinOp(left=ignore.value, op=ast.Add(), right=auxiliary)
            return node

    rewritten = Extender().visit(tree)
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten)


def _strip_answer_assertions(eval_code: str) -> str:
    """Remove answer assertions when the base task has no ground-truth answer."""
    try:
        tree = ast.parse(eval_code)
    except SyntaxError:
        return eval_code

    class Stripper(ast.NodeTransformer):
        def visit_With(self, node: ast.With):
            is_test = any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "test"
                for item in node.items
            )
            if not is_test:
                self.generic_visit(node)
                return node
            body = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
            if "test.answer(" in body:
                return ast.copy_location(ast.Pass(), node)
            self.generic_visit(node)
            return node

    rewritten = Stripper().visit(tree)
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten)


def run_inline_evaluator(world: Any, eval_code: str) -> tuple[Any, str]:
    """Run an intervention evaluator against a live AppWorld final state."""
    from appworld.collections.models import ModelCollectionPair
    from appworld.common import errors as appworld_errors
    from appworld.evaluator import TestTracker
    from munch import Munch

    if not getattr(appworld_errors, "_assertion_replay_traceback_patch", False):
        original = appworld_errors.get_traceback_stack_data

        def safe_traceback(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except Exception:
                return []

        appworld_errors.get_traceback_stack_data = safe_traceback
        appworld_errors._assertion_replay_traceback_patch = True

    task = world.task
    ground_truth = task.ground_truth
    tracker = TestTracker(
        test_data=None,
        difficulty=ground_truth.metadata.get("difficulty", 1),
        suppress_errors=True,
    )
    code = _extend_ignore_lists(eval_code)
    if ground_truth.answer is None:
        code = _strip_answer_assertions(code)
    models = ModelCollectionPair(
        start_db_home_path=task.model_collection.from_db_home_path,
        start_model_collection=task.model_collection,
        end_db_home_path=world.models.to_db_home_path,
        end_model_collection=world.models,
    )
    main_user = models.start.admin.MainUser.find_one(**task.supervisor)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as handle:
        handle.write(code.encode())
        temporary_path = handle.name
    try:
        spec = importlib.util.spec_from_file_location(
            f"__add_assertions_{uuid.uuid4().hex[:8]}__", temporary_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.evaluate(
            test=tracker,
            public_data=Munch(deepcopy(dict(ground_truth.public_data))),
            private_data=Munch(deepcopy(dict(ground_truth.private_data))),
            main_user=main_user,
            models=models,
            ground_truth_answer=ground_truth.answer,
        )
    finally:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
    return tracker, code


def summarize_side(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    substantive = [x for x in blocks if x["class"] == "substantive"]
    failures = [x for x in substantive if x["status"] != "pass"]
    return {
        "substantive_pass": bool(substantive) and not failures,
        "n_assertions": len(blocks),
        "n_substantive_assertions": len(substantive),
        "n_substantive_fail": len(failures),
        "n_anticheat_fail": sum(
            x["status"] == "fail" and x["class"] == "anticheat" for x in blocks
        ),
        "n_ambiguous_fail": sum(
            x["status"] == "fail" and x["ambiguous"] for x in blocks
        ),
        "assertions": blocks,
    }


def apply_known_mixed_assertion_case(row: dict[str, Any]) -> None:
    """Resolve one evaluator block that embeds a clear cross-task exclusion."""
    if row.get("error") or row.get("intervention_idx") != 1065:
        return
    if not row["original"]["substantive_pass"]:
        return
    for block in row["added"]["assertions"]:
        trace = block.get("failure_trace_excerpt") or ""
        if (
            block.get("status") == "fail"
            and "PaymentRequest added (NOT venmo.Transaction)" in block.get("requirement", "")
            and re.search(r"AssertionError:\s+[1-9]\d*\s*==\s*0\s*$", trace)
        ):
            block.update({
                "class": "anticheat",
                "ambiguous": False,
                "classification_reason": "embedded_cross_task_exclusion",
            })
    row["added"] = summarize_side(row["added"]["assertions"])


def replay_one(job: tuple[str, str, int, str, dict[str, Any], str]) -> dict[str, Any]:
    model, task_ref, intervention_idx, intervention_type, trajectory, eval_code = job
    result: dict[str, Any] = {
        "model": model,
        "task_ref": task_ref,
        "intervention_idx": intervention_idx,
        "intervention_type": intervention_type,
        "pass_threshold": PASS,
    }
    try:
        from appworld import AppWorld

        experiment = f"add_assertions_{intervention_idx}_{uuid.uuid4().hex[:10]}"
        code_blocks = execution_exceptions = execution_timeouts = guarded_loops = 0
        with AppWorld(task_id=task_ref, experiment_name=experiment) as world:
            for step in trajectory.get("steps") or []:
                if step.get("role") != "assistant":
                    continue
                code = extract_code(step.get("content") or "")
                if not code:
                    continue
                code, guarded = guard_recorded_loops(code)
                guarded_loops += guarded
                code_blocks += 1
                previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                try:
                    signal.alarm(20)
                    world.batch_execute([code])
                except ReplayTimeout:
                    execution_timeouts += 1
                except Exception:
                    execution_exceptions += 1
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous_handler)
                if world.task_completed():
                    break

            original_tracker = world.evaluate(suppress_errors=True)
            ground_truth = world.task.ground_truth
            added_tracker, patched_added_code = run_inline_evaluator(world, eval_code)

        original_blocks = tracker_assertions(
            original_tracker,
            ground_truth.evaluation_code,
            ground_truth.test_data,
        )
        added_blocks = tracker_assertions(added_tracker, patched_added_code)
        result.update({
            "original": summarize_side(original_blocks),
            "added": summarize_side(added_blocks),
            "n_code_blocks_replayed": code_blocks,
            "n_execution_exceptions": execution_exceptions,
            "n_execution_timeouts": execution_timeouts,
            "n_guarded_loops": guarded_loops,
        })
        apply_known_mixed_assertion_case(result)
        result["assertion_pass"] = bool(
            result["original"]["substantive_pass"]
            and result["added"]["substantive_pass"]
        )
    except Exception as exc:
        import traceback

        result["assertion_pass"] = False
        result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        result["traceback"] = traceback.format_exc()[-2000:]
    return result


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jobs(
    data_path: Path,
    results_path: Path,
    models: list[str] | None,
) -> tuple[list[tuple[str, str, int, str, dict[str, Any], str]], list[dict[str, Any]]]:
    interventions: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(data_path):
        if row.get("skip") or row.get("_skipped") or not row.get("type", "").endswith("Add"):
            continue
        interventions[(str(row["base_task_id"]), int(row["intervention_idx"]))] = row

    selected_models = models or sorted(
        path.name for path in results_path.iterdir() if path.is_dir()
    )
    jobs = []
    issues = []
    for model in selected_models:
        trajectory_path = results_path / model / "AtoB_trajectories.jsonl"
        if not trajectory_path.exists():
            issues.append({"model": model, "reason": "trajectory_file_missing"})
            continue
        seen: set[tuple[str, int]] = set()
        for line_number, trajectory in enumerate(read_jsonl(trajectory_path), 1):
            if not (trajectory.get("intervention_type") or "").endswith("Add"):
                continue
            metadata = trajectory.get("metadata") or {}
            key = (
                str(trajectory.get("task_ref")),
                int(trajectory.get("intervention_idx", metadata.get("intervention_idx"))),
            )
            if key in seen:
                issues.append({
                    "model": model, "line": line_number, "task_ref": key[0],
                    "intervention_idx": key[1], "reason": "duplicate_trajectory",
                })
                continue
            seen.add(key)
            intervention = interventions.get(key)
            if intervention is None:
                issues.append({
                    "model": model, "line": line_number, "task_ref": key[0],
                    "intervention_idx": key[1], "reason": "not_an_active_add_instance",
                })
                continue
            jobs.append((
                model, key[0], key[1], intervention["type"], trajectory,
                intervention.get("evaluation_code") or "",
            ))
    return jobs, issues


def existing_rows(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        (row["model"], str(row["task_ref"]), int(row["intervention_idx"])): row
        for row in read_jsonl(path)
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / "data/appworld_interventions.jsonl",
        help="AppWorld intervention JSONL (default: release data file).",
    )
    parser.add_argument(
        "--results", type=Path, default=REPO_ROOT / "results/appworld",
        help="Directory containing <model>/AtoB_trajectories.jsonl.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "results/_results/appworld_add_assertions.jsonl",
        help="Output JSONL consumed by scripts/eval.py.",
    )
    parser.add_argument("--models", nargs="+", help="Optional model directory names.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel replay workers.")
    parser.add_argument(
        "--limit", type=int,
        help="Replay at most this many rows in total; useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Discard matching rows already present in the output.",
    )
    args = parser.parse_args()

    appworld_root = os.environ.get("APPWORLD_ROOT")
    if not appworld_root:
        parser.error(
            "APPWORLD_ROOT is required; set it to the AppWorld installation directory"
        )
    if not (Path(appworld_root) / "data").is_dir():
        parser.error(f"APPWORLD_ROOT does not contain a data directory: {appworld_root}")
    if not args.data.exists():
        parser.error(f"data file not found: {args.data}")
    if not args.results.is_dir():
        parser.error(f"results directory not found: {args.results}")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    jobs, issues = load_jobs(args.data, args.results, args.models)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    selected_keys = {(job[0], job[1], job[2]) for job in jobs}
    completed = {} if args.overwrite else {
        key: row for key, row in existing_rows(args.output).items() if key in selected_keys
    }
    pending = [job for job in jobs if (job[0], job[1], job[2]) not in completed]
    print(
        f"scope={len(jobs)} done={len(completed)} todo={len(pending)} "
        f"workers={args.workers} issues={len(issues)}",
        flush=True,
    )

    if pending:
        if args.workers == 1:
            replayed = map(replay_one, pending)
            for position, row in enumerate(replayed, 1):
                completed[(row["model"], row["task_ref"], row["intervention_idx"])] = row
                print(f"replayed={position}/{len(pending)}", flush=True)
        else:
            with mp.get_context("fork").Pool(args.workers) as pool:
                for position, row in enumerate(
                    pool.imap_unordered(replay_one, pending, chunksize=1), 1
                ):
                    completed[(row["model"], row["task_ref"], row["intervention_idx"])] = row
                    print(f"replayed={position}/{len(pending)}", flush=True)

    ordered = [completed[(job[0], job[1], job[2])] for job in jobs]
    write_jsonl(args.output, ordered)
    errors = sum("error" in row for row in ordered)
    passes = sum(bool(row.get("assertion_pass")) for row in ordered)
    print(f"wrote={args.output} rows={len(ordered)} passes={passes} errors={errors}")


if __name__ == "__main__":
    main()
