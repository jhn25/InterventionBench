# InterventionBench

Code and data for the InterventionBench paper. **For ARR May 2026 submission.**

## Reproduce scores

```bash
pip install -r requirements.txt
python scripts/eval.py --results results/
```

Prints SR_A, SR_B, SR_{A→B}, and per-type Adaptability (Wilson 95% CI) for each (model, domain). The pre-computed summary is at `results/_results/adaptability.json`.

## Run new inference

```bash
python scripts/run.py --domain {aw|wa} --condition {A|B|A_to_B} --workers N
```

AppWorld and WebArena environments must be installed separately per their upstream repos. Set `MODEL_ID` and `API_BASE_URL` env vars to point at any OpenAI-compatible endpoint.

## Layout

- `data/` — 2,254 intervention instances (1,092 AppWorld + 1,162 WebArena)
- `src/interventionbench/` — runner + 9 baseline frameworks
- `scripts/` — `run.py` (inference), `eval.py` (scoring)
- `results/_results/adaptability.json` — per-(model, domain) metrics

Full trajectories (~3.3 GB) are hosted separately; the link will be added after deanonymization.
