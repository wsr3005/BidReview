# AI Collaboration Contract

This file defines how the human and AI agent work together in this repo.

## Goal

Ship useful changes fast with tight quality control.

## Core Principles

- CLI first. Prefer reproducible commands over manual UI steps.
- Small diffs. Keep each change focused and reversible.
- Fast loop. Plan -> implement -> verify -> report.
- Business correctness first. A green unit test run is not enough if business e2e fails.
- Control blast radius. Touch the minimum files needed.
- Document decisions. Keep context in `docs/` so future sessions stay aligned.

## Roles

### Human (Owner)

- Sets priorities, deadlines, and risk tolerance.
- Approves architecture-level decisions.
- Makes final ship and rollback calls.

### Agent (Executor)

- Breaks work into small, testable tasks.
- Implements one focused change at a time.
- Runs verification commands before handoff.
- Reports assumptions, risks, and exact file changes.

## Default Working Loop

1. Define one task with a clear Definition of Done.
2. Confirm constraints: scope, risk level, deadline.
3. Implement minimal change set.
4. Run verification checklist.
5. Summarize results and next actions.

If verification fails, continue in an auto-repair loop:
fix -> rerun failing gate -> rerun full gate -> report.
Do not hand off while any required gate is red unless explicitly blocked.

## Task Sizing Rules

- Prefer tasks that can complete within 20-60 minutes.
- Prefer one concern per commit.
- Defer non-essential refactors unless requested.
- Stop and escalate if requirements are ambiguous or conflicting.

## Parallel Session Rules

- Start with at most 2-3 concurrent lanes.
- One lane = one objective + one owner.
- Do not let two lanes edit the same file set at the same time.
- Merge only after each lane passes its own verification.

## Verification Gate

Before handoff, run the strongest available checks for this project:

Preferred command: `.\scripts\verify.ps1`

1. Lint or static checks
2. Unit tests for touched areas
3. Integration/business-path tests for touched areas (must include representative real input shape)
4. Build or type check
5. Quick manual smoke test for critical path
6. If a gold dataset exists, run evaluation metrics and enforce thresholds (example: hard-fail recall floor)

If any gate fails:
- Fix and rerun until all required gates pass, or
- Escalate with explicit blocker, impact, and rollback-safe state.

If any check is unavailable, state it explicitly in the handoff note with reason and replacement check.

## Commit and Change Discipline

- Keep commits atomic and reversible.
- Use imperative commit messages.
- Include only related files in each commit.
- Note migration or rollback steps when relevant.

## Handoff Format

Every completed task should include:

1. What changed
2. Why this approach
3. Files touched
4. Verification run and results
5. Risks and follow-up options

## Documentation Rules

- Keep workflow notes in `docs/workflow-checklist.md`.
- Track per-task details using `docs/task-template.md`.
- Update docs whenever behavior or process changes.

## Python Environment

- Use `uv` as the default tool to manage Python environments and dependencies.
- Create/update local environment with `uv sync`.
- Run Python tooling and scripts via `uv run <command>`.
