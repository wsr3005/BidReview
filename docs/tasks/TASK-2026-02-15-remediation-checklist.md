# Task Card

## Metadata

- Task ID: TASK-2026-02-15-remediation-checklist
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: High

## Objective

将当前审查模块的核心设计问题落盘为可执行清单，后续按优先级逐项修复并可量化验收。

## Definition of Done

1. 形成统一问题清单（R1-R9），每项包含当前状态、目标产出、验收标准。
2. 清单可直接作为后续单任务开发输入，不需要二次拆解。
3. 明确执行顺序与依赖，避免并行冲突和重复返工。

## Scope

- In scope:
  - 需求/证据/结论闭环能力
  - 图片证据主流程能力
  - 条款结构化与判定分层
  - 证据定位与跨文档一致性
  - 低置信兜底、评测基准、执行纪律
- Out of scope:
  - 本任务不直接提交功能代码
  - 本任务不引入新模型供应商

## Plan

1. 固化问题列表并标注当前状态（Open / Partial / Done）。
2. 为每项定义单独 DoD 和验证命令。
3. 给出执行顺序与里程碑，后续按单项任务卡推进。

## Remediation Checklist

| ID | 问题 | 当前状态 | 目标产出 | DoD（验收标准） |
|---|---|---|---|---|
| R1 | 审查流程未形成“条款-证据-结论”闭环，结论不可追溯 | Done | 引入 `clause_id` + `evidence_id` + `decision_trace` 结构，报告可回链到原文位置 | 任意一条 finding 可追溯到 requirement 源位置、bid 证据位置、判定规则版本；`review-report.md` 输出 trace 字段 |
| R2 | 图片审查未纳入主流程，扫描件证据漏检 | Partial | OCR 默认纳入可配置主流程，缺 OCR 时强制 `needs_ocr` | `ingest` 输出 `bid_ocr_blocks`；扫描件类 requirement 在无文本证据时不再误判 `pass/fail`；真实样本回归通过；OCR 自检与统计可观测 |
| R3 | 条款抽取与建模不完整 | Open | requirement 增加结构化字段（章节路径、约束类型、数值条件、对象） | 至少支持“数量/金额/期限”3类约束抽取；抽取准确率在基准集达到目标阈值 |
| R4 | 缺少“硬性否决项/评分项”分层判定 | Open | requirement 增加 `rule_tier`（hard_fail/scored/general） | 审查输出分层统计；hard_fail 召回率单独统计；manual checklist 按层级排序 |
| R5 | 结论与证据绑定弱，定位缺失 | Partial | 强制证据定位质量门槛（页码/块号/摘录长度） | 非 `fail(no_evidence)` 场景下 evidence 至少含一条可定位证据；标注副本落点命中率达标 |
| R6 | 跨文档一致性校验不足 | Done | 新增 consistency checker（公司名、金额、日期、证照编号） | 输出 `consistency-findings.jsonl`；可检出样例冲突并进入报告 |
| R7 | 低置信度场景缺少可靠兜底 | Done | 增加低置信策略（降级到人工复核、禁止自动 pass） | LLM 低置信或证据弱时状态进入 `manual_required`/`risk`；无“低置信 pass” |
| R8 | 缺少稳定评测基准，质量不可量化 | Open | 建立固定评测集与指标板（漏检率/误报率/hard_fail召回） | 新增评测命令可重复执行；PR 必须附核心指标变化 |
| R9 | 任务卡与提交不同步，影响追踪 | Done | 形成“任务卡先落盘、实现后即提交”的纪律门禁 | 每次功能提交对应一个 task card；`git log` 可按 task id 回溯变更 |

## Execution Order

1. Milestone-A（先保真）: R1, R5, R9
2. Milestone-B（补漏检）: R2, R6, R7
3. Milestone-C（提准确）: R3, R4
4. Milestone-D（可持续）: R8

## Verification

- Lint/static: `.\scripts\verify.ps1`
- Tests: `python -m unittest discover -s tests -v`
- Build/typecheck: `python -m compileall bidagent tests`
- Smoke test: `python -m bidagent run --tender <file> --bid <file> --out <dir> --focus business --resume`

## Change Log

- Files touched:
  - `docs/tasks/TASK-2026-02-15-remediation-checklist.md`
- Key decisions:
  - 采用 R1-R9 编号作为后续任务拆分入口
  - 按“先可追溯/再补漏检/再提准确/最后评测”排序
- Tradeoffs:
  - 先不做一次性大重构，优先保证每项可独立交付与回归

## Handoff

- Summary:
  - 清单已落盘，可直接按 R1-R9 开工并逐项关闭。
- Remaining risks:
  - 若不先做 R1（trace）与 R9（纪律），后续质量改进仍难回溯。
- Recommended next step:
  - 下一任务先执行 R1，并创建独立任务卡与验收样例。
