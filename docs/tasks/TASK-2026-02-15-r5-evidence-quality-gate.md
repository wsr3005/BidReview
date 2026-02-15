# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r5-evidence-quality-gate
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

为 `pass/risk` 结论增加“证据可用性门槛”，避免仅凭“见附件/扫描件”等引用性片段就给出通过或风险判断，确保报告与标注具备可追溯、可定位的证据落点。

## Definition of Done

1. 对 `pass/risk` findings 执行 evidence gate:
   - 若无可用的非 `reference_only` 证据片段，则降级为 `needs_ocr` 或 `insufficient_evidence`。
2. gate 结果落盘到 `decision_trace.evidence_gate`（含 downgraded_to/reason、最小摘录长度等）。
3. 单测覆盖：
   - 非强制项在“仅命中引用性证据”时会被 gate 降级。
   - 存在非引用性证据时不应被误降级。
4. `python -m unittest discover -s tests -v` 与 `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/review.py`
  - `bidagent/pipeline.py`
  - `tests/test_review.py`
- Out of scope:
  - 证据召回/相似度算法重构（R3/R6 后续）
  - OCR 准确率提升（R2 后续）

