# InterventionBench dataset v1.1 changelog

This is a versioned, non-destructive repair. The v1.0 files under `data/` are unchanged. Exact
before/after values for every top-level field below are in `diff_v10_v11.json`.

The final v1.1 release contains 1,088 active AppWorld rows and 1,162 active WebArena rows, for
2,250 active instances. All 109 screened AppWorld rows remain on disk with explicit skip flags.

## AppWorld interventions

- **437** — Replaced `evaluation_code` and added `_v11_fix`. The new evaluator follows the live
  `modified_spec_b`: export the meal-plan note, attach it to Vickie's draft, and **send** it.
  Evidence: `revision/analysis/dataset_census/REPORT.md:123-133` and the v1.0 row's
  `modified_spec_b`.
- **750** — Replaced `evaluation_code` and added `_v11_fix`. It now requires the target draft to
  remain unsent, the resume to be detached, and the headshot to be attached to that draft.
  Evidence: `revision/analysis/dataset_census/REPORT.md:123-133` and the v1.0 row's
  `modified_spec_b`.
- **757** — Replaced `evaluation_code` and added `_v11_fix`. It now requires export and attachment
  of the workout plan to the retained draft, with no sent email. Evidence:
  `revision/analysis/dataset_census/REPORT.md:123-133` and the v1.0 row's `modified_spec_b`.
- **542, 622, 1132** — Added `skip: true` and `_skipped: "corrupt_source_db_v11"`; rows were not
  deleted. Evidence: the v1.1 lane repair specification and the three v1.0 rows for corrupt base
  task `3d9a636_2`.

## WebArena interventions

- **464, 466, 467, 492, 493, 553, 664, 666, 667, 692, 693, 764, 766, 767, 1120** — Added
  `locator: ""` only to `program_html` targets that lacked the key. Evidence:
  `revision/analysis/dataset_census/REPORT.md:151-157`; the harness reads the key at
  `data/raw/webarena/evaluation_harness/evaluators.py:300` and sends an empty locator through the
  full-page fallback at lines 307-309.
- **2, 4, 17, 22, 28, 29, 56, 72, 73, 79, 86, 101, 103, 105, 107, 108, 111, 112, 114, 119,
  121, 124, 127, 130, 133, 140, 142, 143, 145, 147, 153, 157, 166, 168, 173, 179, 186, 209,
  216, 218, 220, 228, 239** — Replaced the weak URL-only `patched_eval` with the corresponding
  covered strict evaluator. Evidence: `revision/analysis/dataset_census/wa_rescore_manifest.json`.
- **13, 41, 85, 139, 141, 234** — Kept the original evaluator and added
  `_v11_todo: "strict_eval_vacuous"`. These are exactly the six definitions whose strict
  `program_html` check was already true before the first replay in
  `revision/analysis/wa_strict_rescore_v2/impact_tables.json` → `definition_entry_state`.
- **1020** — Replaced `patched_eval` with the WI-C300 cart-state draft and added
  `_v11_fix: "draft_needs_author_review"`. Evidence:
  `revision/analysis/wa_strict_rescore_v2/run_rescore.py:69-85`.
- **56** — Changed `modified_spec_b` from sending to “Upvote the newest post in the gaming
  subreddit,” matching its intervention text. Evidence:
  `revision/analysis/dataset_census/REPORT.md:182-186`.
- **166, 168** — Within the adopted strict evaluators, replaced the throwing
  `.page__sidebar.outerText` expression with
  `(document.querySelector('.subscribe-button') || document.body).className` and a lowercase
  `unsubscribe` requirement. A Playwright check against the live authenticated Postmill pages
  found zero `.page__sidebar` nodes and one `.subscribe-button` on each page; the new expression
  evaluated without exception. Runtime observations are recorded in `validation.json`.

## WebArena strict-evaluator metadata

Evaluator payloads and statuses are byte-for-byte equivalent as parsed JSON; only metadata was
realigned from the v1.0 intervention rows.

- **base_task_id filled (89):** 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308,
  309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326,
  327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344,
  345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,
  363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380,
  381, 382, 383, 384, 385.
- **type corrected (133):** 0, 1, 2, 3, 5, 12, 13, 14, 23, 25, 26, 27, 28, 30, 31, 34,
  37, 38, 39, 41, 44, 48, 49, 55, 56, 152, 157, 170, 171, 177, 180, 183, 205, 207, 209,
  252, 253, 254, 256, 257, 265, 286, 290, 291, 297, 298, 299, 300, 301, 302, 303, 304, 305,
  306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
  324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341,
  342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
  360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
  378, 379, 380, 381, 382, 383, 384, 385.

Evidence for both metadata groups: `revision/analysis/dataset_census/REPORT.md:188-211` and the
v1.0 WebArena rows joined by `intervention_idx`.

## Amendment (2026-08-18)
- idx 437: repaired evaluator withdrawn; cell excluded (`skip: true`) — B spec references a meal-plan note that does not exist in the source environment state; author decided to exclude rather than adjudicate.

## Known limitation

Forty-four BothAdd intervention texts retain a phrasing conflict with the original task and are
being revised. Their task references and intervention indices are listed in
`known_limitations.json`.
