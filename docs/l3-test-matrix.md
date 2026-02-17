# L3 测试矩阵（代码级 + 模型评估 + 端到端）

本文把“上线前收口”具体化为三维测试体系，并映射到本仓库现有命令。

## 总目标（与 gate 对齐）

1. Auto-review coverage >= 95%
2. hard_fail_recall >= 0.98
3. false_positive_fail_rate <= 1%
4. Evidence traceability >= 99%
5. LLM coverage（AI 模式）= 100%

## 维度 1：代码级测试（稳定性与可回归）

### 1. OCR 与解析层测试

目标：
1. PDF/DOCX/TXT 解析后，关键信息不丢失。
2. OCR 产物可进入后续判决链路。

现有测试：
1. `tests/test_ocr.py`
2. `tests/test_pipeline_pdf_ocr_smoke.py`
3. `tests/test_annotate_output.py`

执行命令：
```powershell
uv run python -m unittest tests.test_ocr tests.test_pipeline_pdf_ocr_smoke tests.test_annotate_output -v
```

通过标准：
1. OCR 分支可产出 `bid_blocks.jsonl` 增量块。
2. `manifest.json` 中 `ocr` 字段完整。
3. 注释/定位阶段不因 OCR 块崩溃。

### 2. 规则提取测试（召回优先）

目标：
1. 防“漏判”为第一优先，Recall > Precision。
2. 约束抽取（金额、数量、条款层级）稳定。

现有测试：
1. `tests/test_review.py`
2. `tests/test_constraints.py`
3. `tests/test_rule_tier.py`
4. `tests/test_task_planner.py`

执行命令：
```powershell
uv run python -m unittest tests.test_review tests.test_constraints tests.test_rule_tier tests.test_task_planner -v
```

通过标准：
1. 硬性条款（hard_fail）不应被漏抽。
2. 金额/数量约束抽取结果与断言一致。
3. mixed sentence、噪声目录、模板文本场景均可回归。

说明：
1. 当前仓库尚无“规则提取 Recall 量化脚本”，建议下一步补一个 `extract-req` goldset 评测脚本（仅评召回也可先落地）。

### 3. 判决逻辑测试（TP/TN/反幻觉）

目标：
1. True Positive：应过尽过。
2. True Negative：埋雷可拦截（例如保证金 49 万）。
3. 幻觉防线：证据引用必须可追溯，不允许伪造页码/块号。

现有测试：
1. `tests/test_llm_judge.py`
2. `tests/test_pipeline_review.py`
3. `tests/test_eval.py`
4. `tests/test_evidence_index.py`
5. `tests/test_evidence_harvester.py`

执行命令：
```powershell
uv run python -m unittest tests.test_llm_judge tests.test_pipeline_review tests.test_eval tests.test_evidence_index tests.test_evidence_harvester -v
```

通过标准：
1. LLM 低置信 pass 能降级，异常有 fallback。
2. 冲突证据可触发二次审计并降级。
3. `decision_trace` 和 `evidence_refs` 字段完整。

## 维度 2：模型评估（质量门槛）

### 1. Goldset 结构校验

执行：
```powershell
uv run python scripts/validate-goldset.py
```

标准：
1. 样本量、分层、标签分布满足脚本阈值。

### 2. 运行评估（TP/TN 主指标）

执行：
```powershell
uv run bidagent eval --out <run_dir>
uv run bidagent gate --out <run_dir> --release-mode auto_final
```

指标映射：
1. `hard_fail_recall` 对应 TN 拦截能力（关键硬性项漏判风险）。
2. `false_positive_fail_rate` 对应 TP 保护能力（误杀率）。

放行标准（默认 gate）：
1. `hard_fail_recall >= 0.98`
2. `false_positive_fail_rate <= 0.01`
3. `evidence_traceability >= 0.99`

### 3. 幻觉检测（证据可回溯）

现有保障：
1. gate 中 `evidence_traceability` 检查。
2. canary 中 `verdict_trace_complete` 与 release artifact 完整性检查。

建议增强：
1. 增加“证据引用存在性”专项断言：`evidence.page/block` 必须能在 `ingest/bid_blocks.jsonl` 反查到。

### 4. LLM-as-a-Judge（推理质量）

现状：
1. 仓库尚未内置第二模型裁判评分流程。

建议落地目标：
1. 对 `verdicts.jsonl` 中 `reason + decision_trace` 打分（1-5）。
2. 平均分 < 4 时告警并强制 `assist_only`。

## 维度 3：端到端测试（业务链路）

目标：
1. 从原始文件输入到 `release_mode` 输出全链路可复现。

现有测试：
1. `tests/test_cli_smoke.py`
2. `tests/test_pipeline_pdf_ocr_smoke.py`
3. `tests/test_pipeline_review.py`（覆盖 gate/canary/release trace）

执行命令：
```powershell
uv run python -m unittest tests.test_cli_smoke tests.test_pipeline_pdf_ocr_smoke tests.test_pipeline_review -v
```

全链路验收命令（真实样本）：
```powershell
uv run bidagent run `
  --tender "<tender>" `
  --bid "<bid>" `
  --out "<run_dir>" `
  --ocr-mode auto `
  --ai-provider deepseek `
  --ai-model deepseek-chat `
  --release-mode auto_final
```

验收产物：
1. `requirements.jsonl`
2. `review-tasks.jsonl`
3. `findings.jsonl`
4. `verdicts.jsonl`
5. `gate-result.json`
6. `release/run-metadata.json`
7. `release/canary-result.json`
8. `release/release-trace.json`

## 执行节奏（建议）

1. 每次提交前：`.\scripts\verify.ps1`
2. 每日回归：三维矩阵中的关键子集（OCR + 判决 + E2E smoke）
3. 版本候选（RC）：全量矩阵 + 真实业务样本 + gate/canary 放行检查

## 当前缺口与下一步（按优先级）

1. P1：补“规则提取 Recall 评估脚本”（专注漏抽）。
2. P1：补“证据引用存在性反查测试”（反幻觉硬校验）。
3. P2：补“埋雷样本集”（金额阈值、时间约束、证照有效期等）。
4. P2：补“LLM-as-a-Judge”离线评分脚本与告警门槛。
