# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r5-primary-evidence-selection
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

统一 report/annotate/checklist 的“主证据”选择策略，避免把引用/目录/弱定位内容当作落点，提升可复核性。

## Definition of Done

1. `annotate/checklist/report` 均不再直接使用 `evidence[0]` 作为主证据。
2. 对 `needs_ocr`：主证据优先选择 `reference_only=true` 的扫描件引用证据。
3. 对其他状态：主证据优先选择 `reference_only=false` 且可定位的证据。
4. 回归测试覆盖：
   - 非 `needs_ocr` 时会避开 `reference_only` 主证据
   - `needs_ocr` 时会选择 `reference_only` 主证据
5. `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/pipeline.py`
  - `tests/test_annotate_output.py`
- Out of scope:
  - 规则层的 evidence 生成策略重构（R3/R5 后续）

## Verification

- Tests: `python -m unittest discover -s tests -v`
- Script: `.\scripts\verify.ps1`

