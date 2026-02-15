# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r7-low-confidence-fallback
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

补齐 R7：在 LLM 二次审查阶段加入低置信兜底，避免低置信度结论被自动判为 `pass`，并确保低置信条目进入人工复核队列。

## Definition of Done

1. 增加 `--ai-min-confidence`（默认 0.65）。
2. LLM 返回 `pass` 但 `confidence < min_confidence` 时，自动降级为 `risk/high`，reason 固定为“LLM置信度低(...)，需人工复核”。
3. 在 `decision_trace.low_confidence_fallback` 记录阈值、置信度、降级动作与前值。
4. 单测覆盖该降级逻辑。
5. `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/llm.py`
  - `bidagent/pipeline.py`
  - `bidagent/cli.py`
  - `tests/test_llm_review.py`
- Out of scope:
  - 引入新 status（如 `manual_required`）的全链路改造

