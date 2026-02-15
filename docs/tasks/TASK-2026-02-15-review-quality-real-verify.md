# Task Card

## Metadata

- Task ID: TASK-2026-02-15-review-quality-real-verify
- Owner: Codex
- Date: 2026-02-15
- Priority: High
- Risk level: High

## Objective

基于真实产物 `runs/real-verify` 评估“审查结论准确性 + 标注落点有效性”，明确当前功能可用边界。

## Scope

- In scope:
  - `runs/real-verify/requirements.jsonl`
  - `runs/real-verify/findings.jsonl`
  - `runs/real-verify/annotations.jsonl`
  - `runs/real-verify/annotated/北京为是科技有限公司-安徽芜湖投标文件(1).annotated.docx`
- Out of scope:
  - 代码修复实现
  - 新模型/新算法引入

## Findings

1. High: 核心能力（规则抽取 + 审查判定 + 证据定位）当前不可用，误判与错位是系统性问题。
   - 典型案例：`R0068`
   - 判定理由为“未提供承诺函/保密条款响应”，但落点定位到 `bid block_index=1`（项目标题）。
   - 该落点文本是项目名称，不是承诺/保密条款正文，属于明显错位。

2. High: 标注副本功能正常，但仅证明“写回成功”，不能证明“审查正确”。
   - 批注副本具备 Word 原生结构：`word/comments.xml`、`commentRangeStart/commentReference`、`document.xml.rels` 的 comments 关系均存在。
   - 这属于次要能力（导出），不是核心能力（正确审查）。

3. High: 量化结果支持“错位严重、有效证据不足”。
   - `annotations.jsonl` 总数：`78`
   - 有定位（含 `block_index`）：`47`
   - 无定位：`31`
   - 已定位中，落在 `TOC1`（目录/标题样式）的有：`20`
   - 已定位中，目标文本长度 `<=30` 的有：`31`（大量标题/短句）
   - 结论：大量标注并未落在可支撑判定的正文证据上。

## Assessment

- 次要功能（批注导出/写回）：可用
- 核心功能（审查准确性）：不可用

当前版本可被描述为“能把结果写进文档”，但“结果本身不可信”。

## Why It Fails

1. 规则抽取噪声高，目录/模板句混入 requirements。
2. 证据召回偏关键词共现，容易命中标题块和无效短文本。
3. 批注采用 top1 落点，放大错配。
4. LLM 基于错误候选证据复判，无法实质纠偏。

## Recommended Next Steps

1. 先做规则和证据候选清洗（剔除目录/标题样式块、短文本块）。
2. 增加“证据可用性门槛”（正文长度、关键词覆盖、语义相似度阈值）。
3. 批注落点改为“主证据 + 备选证据”并输出可回溯依据，不再单点 top1。
4. 以真实样本建立误判回归集（含 `R0068`）作为发布门禁。

## Fixes Applied (2026-02-15)

1. 证据候选清洗已落地：
   - 在 `review` 阶段过滤目录样式块（`TOC*`）、标题短文本块、项目标题类文本（缺少动作词）。
   - 代码：`bidagent/review.py`
2. 批注落点已支持“主证据 + 备选证据”：
   - `annotations.jsonl` 增加 `alternate_targets`（最多2条）及 `target.score`。
   - `annotations.md` 增加 `score` 和备选计数。
   - 代码：`bidagent/pipeline.py`
3. 回归测试补充：
   - `tests/test_review.py` 增加目录/标题误命中测试。

## Re-Verification (runs/real-verify)

- 本轮重跑后：
  - `annotations`: 79
  - `with_block`: 46
  - `no_block`: 33
  - `toc_section`: 0（由 20 降到 0）
  - `short_excerpt<=30`: 5（由 31 降到 5）

说明：目录/标题错位显著下降，但“无定位”仍偏高（33），核心准确性仍需继续提升。
