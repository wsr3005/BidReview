# Task Card

## Metadata

- Task ID: TASK-2026-02-15-code-review-fixes-trace-sanitize
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Low

## Objective

落实 `code-review-0f2e26f / 2559be7 / 11b85b2` 的可执行建议：增强 trace 的稳定回链能力（增加证据指纹），并清洗报告输出避免 Markdown 被异常字符污染。

## Definition of Done

1. evidence 增加 `excerpt_hash` 字段，并同步写入 `decision_trace.evidence_refs[*].excerpt_hash`。
2. `review-report.md` 中的 excerpt/trace 输出做空白与引号清洗，并限制最大长度。
3. `_choose_alternate_evidence` 对非数字 `score` 不抛异常。
4. 单测覆盖 `excerpt_hash` 透传。
5. `python -m unittest discover -s tests -v` 与 `python -m compileall bidagent tests` 通过。

## Scope

- In scope:
  - `bidagent/review.py`
  - `bidagent/pipeline.py`
  - `tests/test_review.py`

