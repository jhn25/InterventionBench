"""Score saved trajectories on InterventionBench.

Computes, for each (model, domain) under results/:
  * SR_A, SR_B, SR_{A->B} — marginal success rates
  * Adaptability — P(A->B type-aware pass | vanilla B-alone pass)
      - Point estimate: Wilson estimator (k + z^2/2) / (n + z^2), z = 1.96
      - 95% CI: Wilson score interval
  * Per-type Adaptability (Swap / Drop / Add / Cancel)

Type-aware A->B success rule:
  Add    -> a passes (relaxed >= 0.5) AND b passes
  Swap   -> b passes AND a does not pass
  Drop   -> b passes AND a does not pass
  Cancel -> metadata.cancel_pass (trajectory-based)

Usage:
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

# Pass thresholds.
PASS = 0.999          # strict: a == 1.0 and b == 1.0
ADD_RELAX = 0.5       # for Add: A passes when its answer assertion passes
                      # (set ADD_RELAX = PASS to recover strict a && b).

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


def type_aware_pass(iv_type: str | None, a_score, b_score, md: dict | None = None) -> bool:
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
        a_relax = (a_score or 0) >= ADD_RELAX if a_score is not None else False
        return a_relax and bool(b_p)
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


def load_jsonl(path: Path) -> dict:
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
        out[key] = {
            'a_score': md.get('a_score'),
            'b_score': md.get('b_score'),
            'type': r.get('intervention_type'),
            'cancel_pass': md.get('cancel_pass'),
        }
    return out


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


def compute_metrics(model_dir: Path, domain: str) -> dict | None:
    """Compute SR + Adaptability for one (model, domain)."""
    A = load_a_keyed_by_task(model_dir / 'A_trajectories.jsonl')
    B = load_jsonl(model_dir / 'B_trajectories.jsonl')
    AB = load_jsonl(model_dir / 'AtoB_trajectories.jsonl')
    if not (B and AB):
        return None

    # Marginal SRs.
    sr_a = sum(1 for v in A.values() if (v.get('a_score') or 0) >= PASS) / max(len(A), 1) if A else None
    sr_b = sum(1 for v in B.values() if (v.get('b_score') or 0) >= PASS) / max(len(B), 1)
    sr_ab = sum(1 for v in AB.values() if type_aware_pass(v['type'], v['a_score'], v['b_score'], v)) / max(len(AB), 1)

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
            if type_aware_pass(abd['type'], abd['a_score'], abd['b_score'], abd):
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
        'n_AB': len(AB), 'SR_AB': sr_ab,
        'n_paired': len(paired_keys), 'n_b_pass': n_b_pass, 'n_both': n_both,
        'Adapt_raw': adapt_raw,
        'Adapt_wilson': adapt_wilson,
        'CI_low': ci_lo, 'CI_high': ci_hi,
        'by_type': by_type_out,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=Path('results'),
                    help='Directory containing <domain>/<model>/*.jsonl. Default: results/')
    ap.add_argument('--output', type=Path, default=None,
                    help='Optional path to write JSON summary.')
    args = ap.parse_args()

    if not args.results.exists():
        print(f"ERROR: results directory not found: {args.results}", file=sys.stderr)
        sys.exit(1)

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
            r = compute_metrics(model_dir, domain)
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
