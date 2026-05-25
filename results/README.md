# Results

Pre-computed trajectories for 12 LLM agents × 2 domains × 3 scenarios (A, B, A→B).

```
results/<domain>/<model>/
  ├── A_trajectories.jsonl         # τ_A standalone
  ├── B_trajectories.jsonl         # τ_B standalone
  ├── AtoB_trajectories.jsonl      # mid-task intervention
  ├── phase2/<framework>/          # 9 frameworks (Qwen3.6-27B only)
  └── variants/                    # ablations (Qwen3.6-27B only)
```

Per-trajectory schema:

```json
{
  "task_ref": "383cbac_1",
  "condition": "A_to_B",
  "intervention_idx": 290,
  "intervention_type": "VerbSwap",
  "steps": [...],
  "metadata": {
    "a_score": 0.0 | 1.0 | null,    // τ_A's evaluator
    "b_score": 0.0 | 1.0 | null,    // τ_B's evaluator
    "cancel_pass": true | false,     // Cancel type only
    "inject_step": 7,
    ...
  }
}
```

Run `python ../scripts/eval.py --results .` to recompute SR + Adaptability.
