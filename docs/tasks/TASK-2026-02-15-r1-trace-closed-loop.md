# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r1-trace-closed-loop
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: High

## Objective

实现审查结果的“条款-证据-结论”闭环追溯，使每条结论都可回链到条款来源与证据位置。

## Definition of Done

1. finding 输出包含 `clause_id`、证据 `evidence_id` 与 `decision_trace`。
2. `decision_trace` 至少包含：条款来源位置、规则引擎版本、证据引用列表、最终结论。
3. `review-report.md` 展示 trace 摘要，支持人工快速回溯。
4. 单测覆盖：有证据与无证据两类场景均可追溯。

## Scope

- In scope:
  - `bidagent/models.py`
  - `bidagent/review.py`
  - `bidagent/llm.py`
  - `bidagent/pipeline.py`
  - `tests/test_review.py`
  - `tests/test_cli_smoke.py`
- Out of scope:
  - 条款结构化深度升级（R3）
  - 跨文档一致性（R6）

## Plan

1. 扩展 Finding 数据结构并保持兼容性。
2. 在 review 规则层生成 trace 元数据。
3. 对 LLM 改判场景同步 trace 的最终结论来源。
4. 在 report 中输出 trace 摘要并补测试。

## Verification

- Lint/static: `.\scripts\verify.ps1`
- Tests: `python -m unittest discover -s tests -v`
- Build/typecheck: `python -m compileall bidagent tests`
- Smoke test: `python -m bidagent run --tender <file> --bid <file> --out <dir> --focus business --resume`
- Verification results (2026-02-15):
  - `python -m unittest discover -s tests -v` -> `28 passed`
  - `python -m compileall bidagent tests` -> pass
  - `.\scripts\verify.ps1` -> pass

## Change Log

- Files touched:
  - `docs/tasks/TASK-2026-02-15-r1-trace-closed-loop.md`
  - `bidagent/models.py`
  - `bidagent/review.py`
  - `bidagent/llm.py`
  - `bidagent/pipeline.py`
  - `tests/test_review.py`
  - `tests/test_cli_smoke.py`
  - `README.md`
- Key decisions:
  - 用 `decision_trace.rule.version` 固化判定版本，支持后续规则迭代回溯
  - `evidence_id` 采用稳定位置编码（doc/page/block/section）生成
- Tradeoffs:
  - 当前 trace 仍以规则层 Top-N 证据为主，未引入跨段聚合图（后续 R3/R6 扩展）

## Handoff

- Summary:
  - 交付完成后，每条 finding 都能从报告层回链到条款和证据。
- Remaining risks:
  - 如果源文档后续被替换，旧 `block_index` 回链仍可能失效（需版本锁定策略）。
- Recommended next step:
  - 进入 R5，提升“定位缺失”场景下的证据质量门槛。
