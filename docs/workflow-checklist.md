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

1. Run lint or static checks.
2. Run relevant unit tests.
3. Run integration/business-path tests with representative inputs.
4. Run build or type checks.
5. Perform smoke test for critical path.
6. If gold data exists, run evaluation metrics and verify thresholds.
7. If any gate fails: fix -> rerun failed gate -> rerun full gate.
8. Prepare handoff summary.

## Handoff Template

- Task:
- Outcome:
- Files:
- Checks run:
- Risks:
- Next best action:
