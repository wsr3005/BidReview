# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r6-consistency-check
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

补齐 R6：对投标文件做跨章节一致性校验（公司名、统一社会信用代码、投标总价、关键日期等），输出可复核的冲突清单并写入报告。

## Definition of Done

1. 生成 `consistency-findings.jsonl`，包含：
   - 字段类型、严重级别、原因
   - 各取值的出现次数与示例位置/摘录
2. `review-report.md` 增加 Consistency Findings 小节并展示摘要。
3. 单测覆盖：
   - 公司名出现 2 个不同取值会被检出
   - 单一取值不应误报
4. `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/consistency.py`
  - `bidagent/pipeline.py`
  - `tests/test_consistency.py`
- Out of scope:
  - 规则抽取/证据召回重构（R3/R5）
  - OCR 识别准确率提升（R2）

