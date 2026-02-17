# L3 Lane Cards

This file is tracked by git and can be used as the stable source for multi-thread execution.

## Phase 1

### Lane A: Task Planner

- Lane ID: `L3P1-A`
- Goal: requirement -> review task decomposition.
- Expected outputs: `review-tasks.jsonl`.
- File ownership:
1. `bidagent/task_planner.py`
2. `tests/test_task_planner.py`
- DoD:
1. planner emits deterministic tasks with required schema fields.
2. tasks include at least `task_id`, `requirement_id`, `task_type`, `query`, `expected_logic`, `priority`.
3. lane unit tests pass.
- Required checks:
1. `uv run python -m unittest tests.test_task_planner -v`
2. `.\scripts\verify.ps1`

### Lane B: LLM Verdict Engine

- Lane ID: `L3P1-B`
- Goal: task-level LLM judging with structured verdict protocol.
- Expected outputs: `verdicts.jsonl`.
- File ownership:
1. `bidagent/llm_judge.py`
2. `tests/test_llm_judge.py`
- DoD:
1. verdict status enum is strictly `pass|risk|fail|needs_ocr|insufficient_evidence`.
2. each verdict contains confidence + evidence refs + trace.
3. error/low-confidence fallback behavior is tested.
- Required checks:
1. `uv run python -m unittest tests.test_llm_judge -v`
2. `.\scripts\verify.ps1`

### Lane C: Gate Integration

- Lane ID: `L3P1-C`
- Goal: pipeline/cli integration for planner + verdict + gate.
- Expected outputs:
1. `review-tasks.jsonl`
2. `verdicts.jsonl`
3. `gate-result.json`
- File ownership:
1. `bidagent/cli.py`
2. `bidagent/pipeline.py`
3. `bidagent/eval.py`
4. `tests/test_pipeline_review.py`
- DoD:
1. `plan-tasks` and `gate` are integrated.
2. gate emits `release_mode` in `assist_only|auto_final`.
3. below-threshold runs must be blocked from `auto_final`.
- Required checks:
1. `uv run python -m unittest tests.test_pipeline_review -v`
2. `.\scripts\verify.ps1`

## Integration Policy

1. Merge order: A -> B -> C.
2. Lane C integrates A/B artifacts only after their checks are green.
3. Any lane blocked by missing context should reference:
1. `docs/l3-task-board.md`
2. `docs/l3-lane-cards.md`
