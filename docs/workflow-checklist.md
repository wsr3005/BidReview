# Workflow Checklist

Use this checklist for each task lane.

## Before Coding

1. State task goal in one sentence.
2. Define Definition of Done.
3. List expected files to touch.
4. Note major risk if this change fails.

## During Coding

1. Keep the diff minimal and focused.
2. Validate assumptions early.
3. Avoid unrelated cleanup.
4. Write down decisions that future sessions must know.

## Before Handoff

Run `.\scripts\verify.ps1` first, then fill gaps manually if needed.
Optional discipline check: `.\scripts\check-task-cards.ps1` (use `-FailOnUntracked` to enforce).

1. Run lint or static checks.
2. Run relevant tests.
3. Run build or type checks.
4. Perform smoke test for critical path.
5. Prepare handoff summary.

## Handoff Template

- Task:
- Outcome:
- Files:
- Checks run:
- Risks:
- Next best action:
