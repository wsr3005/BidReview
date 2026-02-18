# L3 测试矩阵（优化版：分层执行 + 并行回归 + 质量门槛）

本文把“上线前收口”固化为可执行测试矩阵，并对现有方案做四项优化：
1. 从“三维”升级为“三维 x 四层执行节奏（T0/T1/T2/RC）”。
2. 补齐当前仓库实际存在但未纳入矩阵的关键测试文件。
3. 增加并行执行方案，缩短回归总时长。
4. 把缺口从“建议项”改为可验收行动项（含 DoD）。

## 总目标（与 gate 严格对齐）

1. `auto_review_coverage >= 0.95`
2. `hard_fail_recall >= 0.98`
3. `false_positive_fail_rate <= 0.01`
4. `evidence_traceability >= 0.99`
5. `llm_coverage == 1.0`（AI 模式）

## 执行层级（新增）

| 层级 | 触发时机 | 目标时长 | 必跑范围 | 放行条件 |
| --- | --- | --- | --- | --- |
| T0 快速回归 | 本地改动后 | 5-10 分钟 | 改动相关测试子集（见“按改动选测”） | 0 失败 |
| T1 提交前门禁 | 每次提交前 | 10-20 分钟 | `.\scripts\verify.ps1` + 必要业务链路补测 | 0 失败 |
| T2 每日回归 | 每日/夜间 | 20-40 分钟 | 三维矩阵全量离线测试 + goldset 校验 | 0 失败 |
| RC 发布候选 | 发版前 | 40+ 分钟 | T2 + 真实样本 E2E + gate/canary 放行 | 指标全部达标 |

统一脚本入口：
1. `.\scripts\test-matrix.ps1 -Level t0 -Areas rules,pipeline`
2. `.\scripts\test-matrix.ps1 -Level t1`
3. `.\scripts\test-matrix.ps1 -Level t2 -Parallel on`
4. `.\scripts\test-matrix.ps1 -Level rc -Parallel on -Tender "<tender>" -Bid "<bid>" -OutDir "<run_dir>"`

## 按改动选测（T0 最小集合）

| 改动范围 | 必跑测试 |
| --- | --- |
| `bidagent/ocr.py`, `bidagent/document.py`, `bidagent/annotators.py` | `tests.test_ocr`, `tests.test_pipeline_pdf_ocr_smoke`, `tests.test_annotate_output` |
| `bidagent/review.py`, `bidagent/constraints.py`, `bidagent/task_planner.py` | `tests.test_review`, `tests.test_constraints`, `tests.test_rule_tier`, `tests.test_task_planner` |
| `bidagent/llm.py`, `bidagent/llm_judge.py` | `tests.test_llm_review`, `tests.test_llm_judge`, `tests.test_deepseek_reviewer` |
| `bidagent/evidence_index.py`, `bidagent/evidence_harvester.py` | `tests.test_evidence_index`, `tests.test_evidence_harvester` |
| `bidagent/pipeline.py`, `bidagent/cli.py`, `bidagent/eval.py` | `tests.test_pipeline_review`, `tests.test_cli_smoke`, `tests.test_eval`, `tests.test_goldset_validation` |
| `bidagent/consistency.py`, `report/checklist` 相关逻辑 | `tests.test_consistency`, `tests.test_report`, `tests.test_checklist` |

## 维度 1：代码级测试（稳定性与可回归）

### 1. OCR + 解析 + 标注链路

目标：
1. OCR 结果可落到 `ingest/bid_blocks.jsonl` 且可被后续阶段消费。
2. `manifest.json` 中 OCR 元数据完整。
3. 注释文件输出在 PDF/DOCX 场景稳定。

测试集合：
1. `tests/test_ocr.py`
2. `tests/test_pipeline_pdf_ocr_smoke.py`
3. `tests/test_annotate_output.py`

执行命令：
```powershell
uv run python -m unittest tests.test_ocr tests.test_pipeline_pdf_ocr_smoke tests.test_annotate_output -v
```

### 2. 规则提取与任务分解（召回优先）

目标：
1. 硬性条款不漏抽。
2. 金额/数量/层级约束抽取稳定。
3. requirement -> review-task 分解可重复、字段完整。

测试集合：
1. `tests/test_review.py`
2. `tests/test_constraints.py`
3. `tests/test_rule_tier.py`
4. `tests/test_task_planner.py`

执行命令：
```powershell
uv run python -m unittest tests.test_review tests.test_constraints tests.test_rule_tier tests.test_task_planner -v
```

### 3. 判决与证据（TP/TN/反幻觉）

目标：
1. 低置信 pass 可降级，异常有 fallback。
2. 冲突证据触发二次审计，避免“盲 pass”。
3. verdict 与 evidence trace 字段完整且结构可验。

测试集合：
1. `tests/test_llm_review.py`
2. `tests/test_llm_judge.py`
3. `tests/test_evidence_index.py`
4. `tests/test_evidence_harvester.py`
5. `tests/test_pipeline_review.py`
6. `tests/test_deepseek_reviewer.py`

执行命令：
```powershell
uv run python -m unittest tests.test_llm_review tests.test_llm_judge tests.test_evidence_index tests.test_evidence_harvester tests.test_pipeline_review tests.test_deepseek_reviewer -v
```

### 4. 结果输出一致性（新增）

目标：
1. 报告、清单、一致性检查输出与业务预期一致。

测试集合：
1. `tests/test_consistency.py`
2. `tests/test_report.py`
3. `tests/test_checklist.py`

执行命令：
```powershell
uv run python -m unittest tests.test_consistency tests.test_report tests.test_checklist -v
```

## 维度 2：模型评估（质量门槛）

### 1. Goldset 结构与分层校验

执行：
```powershell
uv run python scripts/validate-goldset.py --path docs/goldset/l3-gold.jsonl
uv run python -m unittest tests.test_goldset_validation -v
```

通过标准：
1. 样本量、tier 分层、status 分布达到脚本阈值。
2. 非法标签可被脚本和单测同时拦截。

### 2. 运行评估与 gate 放行

执行：
```powershell
uv run bidagent eval --out <run_dir>
uv run bidagent gate --out <run_dir> --release-mode auto_final --gate-fail-fast critical
```

指标映射：
1. `hard_fail_recall`：TN 拦截能力。
2. `false_positive_fail_rate`：TP 保护能力。
3. `evidence_traceability`：证据可回溯完整度。
4. `llm_coverage`：AI 模式覆盖率。

放行标准（默认阈值）：
1. `hard_fail_recall >= 0.98`
2. `false_positive_fail_rate <= 0.01`
3. `evidence_traceability >= 0.99`
4. `llm_coverage == 1.0`

说明：
1. `--gate-fail-fast critical` 作为默认 RC 参数，用于在关键指标失效时提前终止并缩短失败反馈时间。

## 维度 3：端到端测试（业务链路）

目标：
1. 从原始文件输入到 `release_mode` 输出全链路可复现。
2. 关键发布产物齐全且结构可读。

测试集合：
1. `tests/test_cli_smoke.py`
2. `tests/test_pipeline_pdf_ocr_smoke.py`
3. `tests/test_pipeline_review.py`
4. `tests/test_eval.py`

执行命令：
```powershell
uv run python -m unittest tests.test_cli_smoke tests.test_pipeline_pdf_ocr_smoke tests.test_pipeline_review tests.test_eval -v
```

真实样本验收命令：
```powershell
uv run bidagent run `
  --tender "<tender>" `
  --bid "<bid>" `
  --out "<run_dir>" `
  --ocr-mode auto `
  --ai-provider deepseek `
  --ai-model deepseek-chat `
  --release-mode auto_final `
  --gate-fail-fast critical
```

验收产物：
1. `requirements.jsonl`
2. `review-tasks.jsonl`
3. `evidence-packs.jsonl`
4. `findings.jsonl`
5. `verdicts.jsonl`
6. `gate-result.json`
7. `release/run-metadata.json`
8. `release/canary-result.json`
9. `release/release-trace.json`

## 并行执行方案（新增，适用于 T2/RC）

并行分 3 组，避免同类问题串行阻塞：
1. Lane A（解析与规则）：`test_ocr` + `test_review/constraints/rule_tier/task_planner`
2. Lane B（LLM与证据）：`test_llm_review/llm_judge/evidence_index/evidence_harvester/deepseek_reviewer`
3. Lane C（流水线与发布）：`test_pipeline_review/cli_smoke/pipeline_pdf_ocr_smoke/eval/goldset_validation/report/checklist/consistency`

PowerShell 参考：
```powershell
$env:BIDAGENT_ALLOW_NO_AI = "1"
$jobs = @(
  Start-Job { uv run python -m unittest tests.test_ocr tests.test_review tests.test_constraints tests.test_rule_tier tests.test_task_planner -v },
  Start-Job { uv run python -m unittest tests.test_llm_review tests.test_llm_judge tests.test_evidence_index tests.test_evidence_harvester tests.test_deepseek_reviewer -v },
  Start-Job { uv run python -m unittest tests.test_pipeline_review tests.test_cli_smoke tests.test_pipeline_pdf_ocr_smoke tests.test_eval tests.test_goldset_validation tests.test_report tests.test_checklist tests.test_consistency -v }
)
$jobs | Wait-Job | Receive-Job
if (($jobs | Where-Object { $_.State -ne "Completed" }).Count -gt 0) { exit 1 }
```

## 当前缺口与下一步（可验收）

1. P1：补 `extract-req` 召回评估脚本。DoD：输入 goldset，输出 `extract_recall` 与漏抽 TopN 样本。
2. P1：补“证据引用存在性反查”测试。DoD：`evidence_refs` 可在 `ingest/bid_blocks.jsonl` 命中，否则 gate 失败。
3. P2：补埋雷样本集（金额阈值、时间约束、证照有效期）。DoD：每类至少 20 条且纳入每日回归。
4. P2：补离线 LLM-as-a-Judge 评分脚本。DoD：输出 1-5 分分布，低于阈值时自动降级 `assist_only`。
