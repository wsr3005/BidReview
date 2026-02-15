# Task Card

## Metadata

- Task ID: TASK-2026-02-15-review-design-issues
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: High

## Objective

系统性梳理审查模块当前设计的核心缺陷，为后续改进提供明确问题清单和优先级依据。

## Scope

- In scope:
  - `bidagent/review.py` — 需求提取 + 审查匹配
  - `bidagent/llm.py` — LLM 复核
  - `bidagent/document.py` — 文档解析
  - `bidagent/annotators.py` — 批注生成
  - `bidagent/pipeline.py` — 流水线编排
  - `bidagent/models.py` — 数据模型
- Out of scope:
  - 代码修复实现
  - 新模型/新算法引入

## 已识别问题清单

### P1: 需求提取缺乏结构感知

- 模块: `review.py` → `extract_requirements()`
- 现状: 按段落逐句扫描，用正则 (`is_requirement_sentence`) 判断是否为需求语句。
- 问题:
  - 丢失层级上下文：无法得知一条需求属于哪个章节/评分项（如 "第三章 > 3.1.2"）。
  - 丢失条件约束关系：定量要求（如 "≥ 3 个同类业绩"）被平铺为普通句子，约束值未结构化。
  - 表格型需求几乎无法处理：评分标准表、资质清单表格中的需求被拼合或遗漏。

### P2: 审查匹配策略过于粗糙

- 模块: `review.py` → `review_requirements()`
- 现状: 纯关键词命中计数 → Top-3 匹配，阈值硬切。
- 问题:
  - 语义相近但用词不同的内容无法匹配（如招标 "ISO 9001 认证" vs 投标 "质量管理体系认证证书"）。
  - 定量要求无法校验：只算关键词重叠，不解析数值做比对。
  - 多个需求可能匹配到同一段投标内容，无法分辨"是否真的在回应这条需求"。

### P3: LLM 审查粒度受限于初筛结果

- 模块: `llm.py` → `DeepSeekReviewer._build_messages()`
- 现状: LLM 看到的 evidence 只有 `excerpt`（240 字截断），加上规则层给出的 status/severity。
- 问题:
  - 240 字截断可能截断关键证据信息。
  - 缺乏章节上下文，LLM 无法判断证据段落在投标文件中的位置和意义。
  - LLM 基于错误的候选证据复判时，无法实质纠偏——"垃圾进垃圾出"。

### P4: 批注定位 block_index 对齐不可靠

- 模块: `document.py` → `iter_docx_blocks()` 与 `annotators.py` → `annotate_docx_copy()`
- 现状: ingest 阶段通过 `iter_docx_blocks()` 生成 block_index；annotate 阶段重新遍历 DOCX 段落，用 `split_text_blocks()` 重建 block → paragraph 映射。
- 问题:
  - 两次遍历独立执行，没有共享状态或校验机制。
  - 如果文件版本不一致（如 ingest 后投标文件被修改），block_index 整体偏移，所有批注定位错误。
  - `split_text_blocks()` 对同一段落可能产生多个 block，但 annotator 把它们全映射到同一个 `<w:p>`，导致多条批注堆积在同一段落、而相邻段落被遗漏。

### P5: 需求去重/合并有信息丢失和误判风险

- 模块: `review.py` → `_same_requirement()` + `_merge_requirement()`
- 现状: 用关键词重叠度 ≥ 0.75 或归一化文本互相包含来判断是否为"同一条需求"；合并时保留更长的文本。
- 问题:
  - 误合并：关键词相似但约束不同的需求会被合并（如 "保证金金额不低于 5 万" 和 "保证金有效期不少于 90 天"）。
  - 漏合并：同一需求在不同章节用了不同措辞，关键词重叠度不够。
  - 合并只保留更长文本，短文本中的独有约束信息会被丢弃。

### P6: DOCX 解析只读了 document.xml，遗漏内容

- 模块: `document.py` → `iter_docx_blocks()`
- 现状: 只遍历 `word/document.xml` 中的 `<w:p>` 节点。
- 问题:
  - 表格 (`<w:tbl>`) 的结构信息全部丢失：多个单元格的文本被拼合，行列关系消失。
  - 页眉/页脚 (`header*.xml` / `footer*.xml`) 中的内容被忽略。
  - 脚注/尾注 (`footnotes.xml` / `endnotes.xml`) 被忽略。
  - 文本框/浮动内容被忽略。

### P7: 缺少矛盾检测（反向审查）

- 模块: `review.py` → `review_requirements()`
- 现状: 只做单向检查——"招标每条需求 → 投标文件中是否有对应响应"。
- 问题:
  - 正面冲突无法发现：招标要求 30 天交付，投标承诺 60 天，但因关键词命中足够会判 pass。
  - 数值不达标无法发现：招标要求 ≥ 3 个业绩，投标写了 2 个，因为 "业绩" 关键词命中就通过。
  - 投标文件主动资格声明与招标条款的矛盾无法发现。
  - 总结：当前系统能发现"缺失"，但无法发现"有但不对"。

### P8: 投标文件交叉引用无法追踪

- 模块: `review.py` → `review_requirements()`
- 现状: 每个 block 作为独立单元评估。
- 问题:
  - 投标文件中大量出现 "详见附件X"、"参照第X章"、"具体方案见技术标" 等引用。
  - 如果关键词命中的是 "详见附件" 这句话本身，算作证据，但实质上没有具体内容支撑。
  - 一条需求的真正证据可能散落在多处，靠碰巧关键词命中只能捕获部分。
  - 与 P3 中的 `is_reference_only_evidence` 有部分重叠，但当前仅针对 OCR 引用做了处理，通用交叉引用未处理。

### P9: LLM 并发审查无断点续传、无重试

- 模块: `llm.py` → `apply_llm_review()`
- 现状: `ThreadPoolExecutor` 并发调用 LLM，全部完成后整体写入 `findings.jsonl`。
- 问题:
  - 无重试：单次 HTTP 失败直接记录 `error`，不重试。网络波动即产生漏审。
  - 无断点续传：100 条里处理到第 80 条时崩溃，前 80 条 LLM 结果全部丢失，下次全量重跑。
  - `--resume` 仅检查文件是否存在 + provider 是否覆盖完，不支持部分完成状态的恢复。

### P10: 评审状态分类体系存在灰色地带

- 模块: `models.py` → `Finding.status`
- 现状: 5 种 status: `pass / risk / fail / needs_ocr / insufficient_evidence`。
- 问题:
  - `risk` 和 `insufficient_evidence` 界限模糊，纯粹靠分数阈值硬切，缺乏语义区分。
  - 缺少 "部分满足"（partial）状态：需求有多个子条件，投标满足了部分。
  - 缺少 "偏离"（deviation）状态：投标做了偏离响应或让步说明，这在实际投标中很常见。
  - 缺少 "矛盾"（conflict）状态：与 P7 关联，投标有响应但内容与要求矛盾。

## 问题依赖关系

```
P1 (结构感知) ──┐
                 ├──→ P2 (匹配策略) ──→ P3 (LLM 输入质量) ──→ P7 (矛盾检测)
P6 (DOCX 解析) ─┘                                             P10 (状态体系)
                                                                    ↑
P5 (去重合并) ─── 依赖 P1 的结构信息才能正确去重                      │
P4 (批注定位) ─── 依赖 P6 的解析完整性                               │
P8 (交叉引用) ─── 依赖 P1/P6 的结构信息                              │
P9 (断点续传) ─── 独立可修                                          │
P7 (矛盾检测) ─── 依赖 P2 匹配 + P10 状态体系 ─────────────────────┘
```

## 待讨论

1. 结构化解析的深度：招标文件（PDF）的质量如何——文字型还是扫描件？表格需求占比多大？
2. 语义匹配的取舍：先做关键词 + 章节映射，还是直接引入 Embedding / LLM 做语义匹配？
3. LLM 成本控制：需求量大（100+）时，是否需要分层策略（只对 risk/fail 调 LLM）？
4. 改进策略：在现有代码上渐进增强，还是对核心模块做一次较大重构？

## Change Log

- 2026-02-15: 初始版本，识别 10 个核心问题，尚未开始修复。
