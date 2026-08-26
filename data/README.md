# Dataset

2,250 active intervention instances: 1,088 AppWorld + 1,162 WebArena, spanning 9 intervention types. Screened AppWorld rows remain in the JSONL with explicit skip flags.

One JSON object per line:

```json
{
  "base_task_id": "383cbac_1",
  "type": "VerbSwap",
  "intervention_idx": 290,
  "modified_spec_b": "...",                // standalone-executable τ_B
  "intervention_text": "Wait — actually ...",
  "primary_action_a": "send",
  "primary_target_a": "email",
  "new_action_b": "save",
  "new_target_b": "note",
  "evaluation_code": "<python>",           // AppWorld: patched evaluator for τ_B
  "patched_eval": {...},                   // WebArena: patched evaluator JSON
  "skip": false
}
```

Each `modified_spec_b` is used both as a mid-task intervention in scenario A→B and as a standalone task in scenario B.

`strict_eval.json` contains the released WebArena strict-evaluator metadata. See `CHANGELOG.md` for dataset v1.1 repairs and `known_limitations.json` for the residual BothAdd phrasing-conflict indices.
