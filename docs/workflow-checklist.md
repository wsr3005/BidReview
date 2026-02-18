# Workflow Checklist

Use this checklist for each task lane.

## Before Coding

1. State task goal in one sentence.
2. Define Definition of Done.
3. List expected files to touch.
4. Note major risk if this change fails.
5. Define at least one business acceptance scenario (realistic input + expected outcome).

## During Coding

1. Keep the diff minimal and focused.
2. Validate assumptions early.
3. Avoid unrelated cleanup.
4. Write down decisions that future sessions must know.
5. Add or update tests for every bugfix and every changed business rule.

## Before Handoff

Run `.\scripts\verify.ps1` first, then fill gaps manually if needed.
Optional discipline check: `.\scripts\check-task-cards.ps1` (use `-FailOnUntracked` to enforce).
Use `docs/l3-test-matrix.md` as the canonical L3 test dimensions and thresholds.
Optional matrix runner: `.\scripts\test-matrix.ps1 -Level t0|t1|t2|rc`.

1. Run lint or static checks.
2. Run relevant unit tests.
3. Run integration/business-path tests with representative inputs.
4. Run build or type checks.
5. Perform smoke test for critical path.
6. If gold data exists, run evaluation metrics and verify thresholds.
   Recommended when gold fixtures are changed: `uv run python scripts/validate-goldset.py`
7. If any gate fails: fix -> rerun failed gate -> rerun full gate.
8. Verify release artifacts for release lanes: `cross-audit.jsonl`, `ingest/entity-pool.json`, `release/run-metadata.json`, `release/canary-result.json`, `release/release-trace.json`.
9. Prepare handoff summary.

## Handoff Template

- Task:
- Outcome:
- Files:
- Checks run:
- Risks:
- Next best action:

## Active Parallel Board (L3 Phase 1)

Date: 2026-02-17
Target: move current `bidagent` to L3-ready Phase 1 baseline.

Lane A
- Objective: requirement -> review tasks decomposition (`plan-tasks` core logic).
- Owner: Codex-Lane-A
- Task card: `docs/l3-lane-cards.md` (Phase 1 / Lane A)
- File ownership: `bidagent/task_planner.py`, `tests/test_task_planner.py`

Lane B
- Objective: LLM-first task verdict engine and verdict protocol writer.
- Owner: Codex-Lane-B
- Task card: `docs/l3-lane-cards.md` (Phase 1 / Lane B)
- File ownership: `bidagent/llm_judge.py`, `tests/test_llm_judge.py`

Lane C
- Objective: CLI/pipeline integration and gate decision (`assist_only` vs `auto_final`).
- Owner: Codex-Lane-C
- Task card: `docs/l3-lane-cards.md` (Phase 1 / Lane C)
- File ownership: `bidagent/cli.py`, `bidagent/pipeline.py`, `bidagent/eval.py`, `tests/test_pipeline_review.py`

Merge order
1. Lane A
2. Lane B
3. Lane C (integrates A/B outputs)

Required checks per lane
1. `.\scripts\verify.ps1`
2. lane-specific unit tests listed in task card
3. if eval fixtures changed: `uv run bidagent eval --out <run_dir>`

## Forward Plan (Phase 2 + Phase 3)

Source of truth: `docs/l3-task-board.md`
Lane briefs: `docs/l3-lane-cards.md`
Architecture blueprint: `docs/bid-review-architecture-blueprint.md`

Prepared phase bundles:
1. Phase 2: evidence indexing, active evidence harvesting, counter-evidence auditing
2. Phase 3: gold set expansion, gate threshold tuning, release hardening and canary

Recommended execution policy:
1. Do not open Phase 2 lanes before Phase 1 merge is green.
2. Do not open Phase 3 lanes before Phase 2 eval targets are met.
3. Keep one integration lane owner for `bidagent/pipeline.py` and `bidagent/cli.py`.
