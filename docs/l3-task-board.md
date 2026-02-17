# L3 Task Board

This document preserves multi-phase execution context for L3 automation.

## Program Targets

1. Auto-review coverage >=95%
2. hard_fail_recall >=0.98
3. false_positive_fail <=1%
4. Evidence traceability >=99%
5. LLM coverage in AI mode =100%

## Phase Snapshot

| Phase | Goal | Status | Open Lanes | Exit Criteria |
| --- | --- | --- | --- | --- |
| Phase 1 | LLM-first baseline + gate wiring | In Progress | 3 | planner+verdict+gate outputs are stable |
| Phase 2 | Active evidence retrieval quality | Planned | 3 | evidence recall and conflict handling targets met |
| Phase 3 | Release readiness for auto_final | Planned | 3 | eval thresholds passed on expanded gold set |

## Phase 1 Lane Cards

1. `docs/tasks/TASK-2026-02-17-l3p1-task-planner.md`
2. `docs/tasks/TASK-2026-02-17-l3p1-llm-verdict-engine.md`
3. `docs/tasks/TASK-2026-02-17-l3p1-gate-integration.md`

## Phase 2 Lane Cards

1. `docs/tasks/TASK-2026-02-17-l3p2-evidence-index.md`
2. `docs/tasks/TASK-2026-02-17-l3p2-evidence-harvester.md`
3. `docs/tasks/TASK-2026-02-17-l3p2-counter-auditor.md`

Phase 2 focus:
1. Build unified evidence index for text/table/OCR blocks.
2. Add active retrieval per review task with top-k evidence pack output.
3. Add counter-evidence checks to reduce false pass and unstable verdicts.

Phase 2 target metrics:
1. Evidence retrieval recall >=0.95 on labeled benchmark set.
2. Reference-only evidence incorrectly judged as pass = 0.
3. Conflict detection coverage >=90% on synthetic contradiction fixtures.

## Phase 3 Lane Cards

1. `docs/tasks/TASK-2026-02-17-l3p3-goldset-expansion.md`
2. `docs/tasks/TASK-2026-02-17-l3p3-gate-tuning.md`
3. `docs/tasks/TASK-2026-02-17-l3p3-release-hardening.md`

Phase 3 focus:
1. Expand and stratify gold set for realistic business clauses.
2. Tune release gate thresholds and add fail-fast checks in pipeline.
3. Add release hardening (metadata, prompt/model versioning, canary flow).

Phase 3 target metrics:
1. hard_fail_recall >=0.98
2. false_positive_fail <=1%
3. Auto-review coverage >=95%

## Multi-Thread Rules

1. Max 2-3 lanes in parallel.
2. One lane owns one primary objective and non-overlapping file set.
3. Integration lane merges outputs only after each lane's verify gates pass.
4. Use `.\scripts\verify.ps1` before handoff in every lane.

## Suggested Merge Order

1. Phase 1: A -> B -> C
2. Phase 2: A -> B -> C
3. Phase 3: A -> B -> C

## Blockers Policy

If a lane blocks:
1. record blocker + impacted files in its task card
2. stop cross-lane edits on overlapping files
3. re-route via integration lane after unblock
