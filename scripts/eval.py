"""Score saved trajectories on InterventionBench.

Computes, for each (model, domain) under results/:
  * SR_A, SR_B, SR_{A->B} — marginal success rates
  * Adaptability — P(A->B type-aware pass | vanilla B-alone pass)
      - Point estimate: Wilson estimator (k + z^2/2) / (n + z^2), z = 1.96
      - 95% CI: Wilson score interval
  * Per-type Adaptability (Swap / Drop / Add / Cancel)

Type-aware A->B success rule:
  AppWorld Add -> both evaluators pass at the substantive-assertion level
  WebArena Add -> a passes AND b passes
  Swap   -> b passes AND a does not pass
  Drop   -> b passes AND a does not pass
  Cancel -> metadata.cancel_pass (trajectory-based)

Usage:
  APPWORLD_ROOT=/path/to/appworld python scripts/score_add_assertions.py
  python scripts/eval.py --results results/
  python scripts/eval.py --results results/ --output adaptability.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# All scalar evaluator outcomes use the paper's threshold.
PASS = 0.999

# Wilson constants for 95% CI.
Z = 1.959963984540054
Z2 = Z * Z


def wilson_point(k: int, n: int) -> float:
    """Wilson point estimator (CI midpoint): (k + z^2/2) / (n + z^2)."""
    if n == 0:
        return 0.5
    return (k + Z2 / 2) / (n + Z2)


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% CI [low, high] for binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + Z2 / n
    center = (p_hat + Z2 / (2 * n)) / denom
    half = (Z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + Z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def type_aware_pass(
    iv_type: str | None,
    a_score,
    b_score,
    md: dict | None = None,
    *,
    domain: str,
    appworld_add_pass: bool | None = None,
) -> bool:
    """Type-aware A->B success rule. Returns True iff trajectory passes its type."""
    if iv_type is None:
        return False
    if iv_type == 'Cancel':
        if md is not None and 'cancel_pass' in md:
            return bool(md.get('cancel_pass'))
        a_p = (a_score or 0) >= PASS if a_score is not None else None
        b_p = (b_score or 0) >= PASS if b_score is not None else False
        return (a_p is False) and (not b_p)
    if a_score is None and b_score is None:
        return False
    a_p = (a_score or 0) >= PASS if a_score is not None else None
    b_p = (b_score or 0) >= PASS if b_score is not None else False

    if 'Add' in iv_type:
        if domain == 'appworld':
            if appworld_add_pass is None:
                raise ValueError('missing AppWorld Add assertion-level verdict')
            return appworld_add_pass
        return bool(a_p) and bool(b_p)
    if 'Swap' in iv_type or 'Drop' in iv_type:
        return bool(b_p) and (a_p is False if a_p is not None else True)
    return bool(b_p)


def coarse_type(t: str | None) -> str:
    if not t:
        return 'Unknown'
    if 'Add' in t: return 'Add'
    if 'Swap' in t: return 'Swap'
    if 'Drop' in t: return 'Drop'
    if t == 'Cancel': return 'Cancel'
    return 'Other'


def load_jsonl(path: Path, active_keys: set[tuple[str, int]] | None = None) -> dict:
    """Return dict keyed by (task_ref, intervention_idx)."""
    out = {}
    if not path.exists():
        return out
    for ln in path.open():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        md = r.get('metadata') or {}
        if md.get('placeholder'):
            continue
        key = (str(r.get('task_ref')), r.get('intervention_idx'))
        if active_keys is not None and key not in active_keys:
            continue
        out[key] = {
            'a_score': md.get('a_score'),
            'b_score': md.get('b_score'),
            'type': r.get('intervention_type'),
            'cancel_pass': md.get('cancel_pass'),
        }
    return out


def load_active_keys(data_dir: Path, domain: str) -> set[tuple[str, int]]:
    """Load active dataset cells so screened rows do not enter scoring."""
    path = data_dir / f'{domain}_interventions.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'dataset file not found: {path}')
    keys = set()
    for lineno, line in enumerate(path.open(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('skip') or row.get('_skipped'):
            continue
        keys.add((str(row['base_task_id']), int(row['intervention_idx'])))
    return keys


def load_appworld_add_assertions(path: Path) -> dict[tuple[str, str, int], bool]:
    """Load assertion-level AppWorld Add verdicts emitted by the replay scorer."""
    if not path.exists():
        raise FileNotFoundError(
            f'AppWorld Add assertion verdicts not found: {path}\n'
            'Run scripts/score_add_assertions.py before scripts/eval.py.'
        )
    verdicts = {}
    for lineno, line in enumerate(path.open(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('error'):
            continue
        key = (str(row['model']), str(row['task_ref']), int(row['intervention_idx']))
        if key in verdicts:
            raise ValueError(f'duplicate assertion verdict at {path}:{lineno}: {key}')
        verdicts[key] = bool(row.get('assertion_pass'))
    return verdicts


def load_a_keyed_by_task(path: Path) -> dict:
    """A-condition has no intervention_idx; key by task_ref only."""
    out = {}
    if not path.exists():
        return out
    for ln in path.open():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        md = r.get('metadata') or {}
        if md.get('placeholder'):
            continue
        out[str(r.get('task_ref'))] = {'a_score': md.get('a_score')}
    return out


def compute_metrics(
    model_dir: Path,
    domain: str,
    active_keys: set[tuple[str, int]],
    appworld_add_assertions: dict[tuple[str, str, int], bool],
) -> dict | None:
    """Compute SR + Adaptability for one (model, domain)."""
    A = load_a_keyed_by_task(model_dir / 'A_trajectories.jsonl')
    B = load_jsonl(model_dir / 'B_trajectories.jsonl', active_keys)
    AB = load_jsonl(model_dir / 'AtoB_trajectories.jsonl', active_keys)
    if not (B and AB):
        return None

    # Marginal SRs.
    sr_a = sum(1 for v in A.values() if (v.get('a_score') or 0) >= PASS) / max(len(A), 1) if A else None
    sr_b = sum(1 for v in B.values() if (v.get('b_score') or 0) >= PASS) / max(len(B), 1)
    def passes(key, value):
        assertion_verdict = appworld_add_assertions.get(
            (model_dir.name, key[0], int(key[1]))
        ) if domain == 'appworld' and coarse_type(value['type']) == 'Add' else None
        return type_aware_pass(
            value['type'], value['a_score'], value['b_score'], value,
            domain=domain, appworld_add_pass=assertion_verdict,
        )

    non_cancel_ab = {
        key: value for key, value in AB.items() if coarse_type(value['type']) != 'Cancel'
    }
    sr_ab = (
        sum(1 for key, value in non_cancel_ab.items() if passes(key, value))
        / max(len(non_cancel_ab), 1)
    )

    # Restricted Adaptability: condition on vanilla B-pass.
    paired_keys = set(B.keys()) & set(AB.keys())
    n_b_pass = 0
    n_both = 0
    by_type = defaultdict(lambda: {'n': 0, 'b_pass': 0, 'both': 0})

    for k in paired_keys:
        bd, abd = B[k], AB[k]
        cls = coarse_type(abd['type'])
        if cls == 'Cancel':
            continue
        by_type[cls]['n'] += 1
        if (bd.get('b_score') or 0) >= PASS:
            n_b_pass += 1
            by_type[cls]['b_pass'] += 1
            if passes(k, abd):
                n_both += 1
                by_type[cls]['both'] += 1

    for k, abd in AB.items():
        if coarse_type(abd['type']) != 'Cancel':
            continue
        by_type['Cancel']['n'] += 1
        by_type['Cancel']['b_pass'] += 1
        if abd.get('cancel_pass'):
            by_type['Cancel']['both'] += 1

    adapt_raw = n_both / n_b_pass if n_b_pass > 0 else None
    adapt_wilson = wilson_point(n_both, n_b_pass)
    ci_lo, ci_hi = wilson_ci(n_both, n_b_pass)

    by_type_out = {}
    for cls, d in by_type.items():
        cls_lo, cls_hi = wilson_ci(d['both'], d['b_pass'])
        by_type_out[cls] = {
            'n': d['n'], 'b_pass': d['b_pass'], 'both': d['both'],
            'adapt_raw': (d['both'] / d['b_pass']) if d['b_pass'] > 0 else None,
            'adapt_wilson': wilson_point(d['both'], d['b_pass']),
            'ci_low': cls_lo, 'ci_high': cls_hi,
        }

    return {
        'model': model_dir.name,
        'domain': domain,
        'n_A': len(A), 'SR_A': sr_a,
        'n_B': len(B), 'SR_B': sr_b,
        'n_AB': len(non_cancel_ab), 'n_AB_with_cancel': len(AB), 'SR_AB': sr_ab,
        'n_paired': len(paired_keys), 'n_b_pass': n_b_pass, 'n_both': n_both,
        'Adapt_raw': adapt_raw,
        'Adapt_wilson': adapt_wilson,
        'CI_low': ci_lo, 'CI_high': ci_hi,
        'by_type': by_type_out,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=REPO_ROOT / 'results',
                    help='Directory containing <domain>/<model>/*.jsonl. Default: results/')
    ap.add_argument('--output', type=Path, default=None,
                    help='Optional path to write JSON summary.')
    ap.add_argument('--data', type=Path, default=REPO_ROOT / 'data',
                    help='Directory containing released intervention JSONL files. Default: data/')
    ap.add_argument(
        '--appworld-add-assertions', type=Path,
        default=REPO_ROOT / 'results/_results/appworld_add_assertions.jsonl',
        help='Assertion-level AppWorld Add JSONL from score_add_assertions.py.',
    )
    args = ap.parse_args()

    if not args.results.exists():
        print(f"ERROR: results directory not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    domain_names = {
        path.name for path in args.results.iterdir()
        if path.is_dir() and path.name in ('appworld', 'webarena')
    }
    try:
        active_by_domain = {
            domain: load_active_keys(args.data, domain) for domain in domain_names
        }
        appworld_add_assertions = (
            load_appworld_add_assertions(args.appworld_add_assertions)
            if 'appworld' in domain_names else {}
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)

    rows = []
    for domain_dir in sorted(args.results.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith('_'):
            continue
        domain = domain_dir.name
        if domain not in ('appworld', 'webarena'):
            continue
        for model_dir in sorted(domain_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            try:
                r = compute_metrics(
                    model_dir, domain, active_by_domain[domain], appworld_add_assertions
                )
            except ValueError as exc:
                print(f'ERROR: {model_dir}: {exc}', file=sys.stderr)
                sys.exit(2)
            if r is not None:
                rows.append(r)

    # Header.
    print(f"{'Model':<28}{'Dom':>5}{'SR_A':>7}{'SR_B':>7}{'SR_AB':>7}"
          f"{'n_Bp':>6}{'k':>5}{'Adapt':>8}{'95% CI':>16}")
    print('-' * 90)
    for r in rows:
        sr_a_str = f"{(r['SR_A'] or 0)*100:5.1f}" if r['SR_A'] is not None else "  -- "
        sr_b_str = f"{r['SR_B']*100:5.1f}"
        sr_ab_str = f"{r['SR_AB']*100:5.1f}"
        ci_str = f"[{r['CI_low']*100:.1f}, {r['CI_high']*100:.1f}]"
        print(f"  {r['model'][:26]:<26} {r['domain'][:3]:>5}"
              f"{sr_a_str:>7}{sr_b_str:>7}{sr_ab_str:>7}"
              f"{r['n_b_pass']:>6}{r['n_both']:>5}"
              f"{r['Adapt_wilson']*100:>7.1f} {ci_str:>15}")

    print()
    print(f"{'Per-type Adaptability (%, Wilson point, 95% CI)':<32}")
    print(f"{'Model/Dom':<32}{'Add':>20}{'Swap':>20}{'Drop':>20}{'Cancel':>20}")
    print('-' * 112)
    for r in rows:
        tag = f"{r['model'][:22]}/{r['domain'][:3]}"
        parts = []
        for cls in ['Add', 'Swap', 'Drop', 'Cancel']:
            v = r['by_type'].get(cls)
            if v is None or v['b_pass'] == 0:
                parts.append('  --  ')
            else:
                parts.append(f"{v['adapt_wilson']*100:5.1f} [{v['ci_low']*100:4.1f},{v['ci_high']*100:5.1f}]")
        print(f"  {tag:<30}{parts[0]:>20}{parts[1]:>20}{parts[2]:>20}{parts[3]:>20}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nWrote JSON summary to {args.output} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
