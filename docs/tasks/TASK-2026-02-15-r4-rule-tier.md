# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r4-rule-tier
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

补齐 R4：对 requirement 做分层建模（硬性否决项/评分项/一般项），并在报告与人工复核清单中体现优先级，避免风险优先级失真。

## Definition of Done

1. `requirements.jsonl` 增加 `rule_tier` 字段（`hard_fail|scored|general`）。
2. 抽取阶段对典型文本可正确识别 tier：
   - 含“否则/否决/废标/无效投标”等后果描述 -> `hard_fail`
   - 含“评分/得分/分值/加分/扣分”等 -> `scored`
3. `manual-review.jsonl`/`manual-review.md` 增加 tier 并按 tier 排序（hard_fail 优先）。
4. `review-report.md` summary 输出各 tier 数量与 hard_fail 的状态分布。
5. `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/models.py`
  - `bidagent/review.py`
  - `bidagent/pipeline.py`
  - `tests/test_rule_tier.py`
- Out of scope:
  - hard_fail 召回率评测集（R8）

