# InterventionBench

Code and data for the camera-ready InterventionBench paper.

## Reproduce scores

```bash
pip install -r requirements.txt
APPWORLD_ROOT=/path/to/appworld python scripts/score_add_assertions.py
python scripts/eval.py --results results/
```

Prints SR_A, SR_B, SR_{A→B}, and per-type Adaptability (Wilson 95% CI) for each (model, domain). The bundled `results/_results/adaptability.json` is the earlier submission snapshot and should be regenerated with `--output` after assertion replay.

Swap and non-Cancel Drop require the revised-task evaluator to pass and the original-task evaluator to fail; WebArena Add requires both evaluators to pass, and Cancel uses the trajectory-level stopping rule. For AppWorld Add, the released replay scorer applies both evaluators at the final state and requires every substantive assertion on both sides to pass, excluding only clear cross-task no-other-change assertions while retaining mixed or ambiguous assertions.

## Run new inference

```bash
python scripts/run.py --domain {aw|wa} --condition {A|B|A_to_B} --workers N
```

AppWorld and WebArena environments must be installed separately per their upstream repos. Set `MODEL_ID` and `API_BASE_URL` env vars to point at any OpenAI-compatible endpoint.

## Layout

- `data/`: 2,250 active intervention instances (1,088 AppWorld + 1,162 WebArena)
- `src/interventionbench/`: runner + 9 baseline frameworks
- `scripts/`: `run.py` (inference), `score_add_assertions.py` (AppWorld Add replay), `eval.py` (aggregate scoring)
- `results/_results/adaptability.json`: per-(model, domain) metrics

Full trajectories (~3.3 GB) are hosted separately; the link will be added after deanonymization.

## Dataset changelog

Dataset v1.1 repairs AppWorld and WebArena evaluators, strengthens covered WebArena URL-only checks, and excludes four AppWorld instances whose source state or evaluator cannot support a reliable score. Screened AppWorld instances remain in the distributed JSONL with explicit skip flags, so the 1,197 rows on disk contain 1,088 active instances; see `data/CHANGELOG.md` for the complete index-level record.

## Known limitations

Forty-four BothAdd intervention texts contain a residual phrasing conflict with the original task and are being revised. Their task references and intervention indices are listed in `data/known_limitations.json`.
