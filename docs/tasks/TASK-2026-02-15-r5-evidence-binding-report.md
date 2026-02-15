# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r5-evidence-binding-report
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

在报告中强化“结论-证据”绑定展示：每条 finding 输出主要证据的定位与摘录片段，降低“有结论无片段/位置”的不可复核问题。

## Definition of Done

1. `review-report.md` 每条 finding 都包含 `evidence:` 段（无证据则 `evidence: none`）。
2. `evidence:` 至少包含 `evidence_id`、`block/page`、`score`、`excerpt`（截断）。
3. 单测 smoke 验证报告包含 `evidence:` 且含关键字（如“营业执照”）。

## Scope

- In scope:
  - `bidagent/pipeline.py` 报告输出格式
  - `tests/test_cli_smoke.py` 回归测试
- Out of scope:
  - 证据质量门槛与自动降级（R5 后续子任务）

## Plan

1. 在 `pipeline.report()` 增加 `evidence:` 摘要输出。
2. 更新 CLI smoke 测试确保回归门禁。

## Verification

- Tests: `python -m unittest discover -s tests -v`
- Build: `python -m compileall bidagent tests`
- Script: `.\scripts\verify.ps1`

## Change Log

- Files touched:
  - `docs/tasks/TASK-2026-02-15-r5-evidence-binding-report.md`
  - `bidagent/pipeline.py`
  - `tests/test_cli_smoke.py`
