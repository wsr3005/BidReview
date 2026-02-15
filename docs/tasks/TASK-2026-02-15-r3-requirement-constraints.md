# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r3-requirement-constraints
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

补齐 R3 的最小可判定建模：把招标要求句中的“数量/金额/期限”抽取为结构化约束字段，写入 `requirements.jsonl`，为后续确定性校验与评测打基础。

## Definition of Done

1. `requirements.jsonl` 的每条 requirement 增加 `constraints` 列表字段。
2. 至少支持三类约束抽取：
   - amount（金钱，输出 `value_fen`）
   - term（期限/工期，输出 `value+unit`）
   - quantity（数量，输出 `value+unit`）
3. 合并重复 requirements 时保留 constraints 的并集。
4. 单测覆盖三类约束抽取。
5. `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/constraints.py`
  - `bidagent/models.py`
  - `bidagent/review.py`
  - `bidagent/pipeline.py`
  - `tests/test_constraints.py`
- Out of scope:
  - 约束抽取准确率评测集（R8）
  - rule_tier 分层判定（R4）

