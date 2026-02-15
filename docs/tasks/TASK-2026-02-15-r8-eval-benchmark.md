# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r8-eval-benchmark
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

补齐 R8：建立可重复执行的评测命令与指标输出，支持持续量化改进（漏检/误报/hard_fail 召回等）。

## Definition of Done

1. 新增命令：`python -m bidagent eval --out <run_dir>`
2. 读取 `<run_dir>/eval/gold.jsonl`（人工标注的基准集），输出 `<run_dir>/eval/metrics.json`
3. 指标至少包含：
   - hard_fail_total / hard_fail_fail / hard_fail_missed / hard_fail_recall
   - false_positive_fail
4. 单测覆盖评测逻辑。
5. `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/eval.py`
  - `bidagent/cli.py`
  - `tests/test_eval.py`
- Out of scope:
  - 真实基准集内容（由你在 runs/<x>/eval/gold.jsonl 维护）

