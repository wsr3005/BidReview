# Task Card

## Metadata

- Task ID: TASK-2026-02-15-code-review-89d26d7
- Owner: Codex
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

对提交 `89d26d7` 进行代码审查，识别行为缺陷、回归风险和测试缺口，并形成可执行修复清单。

## Scope

- In scope:
  - `bidagent/pipeline.py`
  - `bidagent/review.py`
  - `bidagent/llm.py`
  - `scripts/verify.ps1`
  - `tests/`
- Out of scope:
  - 新功能实现
  - 规则抽取算法重构

## Verification

- Commit inspected: `git show --stat --oneline 89d26d7`
- Tests run: `python -m pytest -q` (`9 passed`)
- Targeted repro:
  - `review(..., resume=True, ai_provider='deepseek')` 的半完成 llm 场景复现

## Findings

1. High: `--resume` + `--ai-provider deepseek` 会在“任意一条存在 llm.provider=deepseek”时提前返回，导致未复审条目被跳过。
   - Location: `bidagent/pipeline.py:128`, `bidagent/pipeline.py:129`
   - Impact: findings 可能处于“部分 LLM 化”不一致状态，下游 `annotate/checklist/report` 基于旧结果输出。
   - Repro: 构造 2 条 findings，仅 1 条带 `llm.provider=deepseek`，调用 `review(..., resume=True, ai_provider='deepseek')` 后第 2 条仍无 `llm`，但函数已返回成功统计。

2. Medium: 规则抽取会把目录/模板段落误识别为强制要求，导致审查基线噪声偏高。
   - Location: `bidagent/review.py:39`, `bidagent/review.py:106`, `bidagent/review.py:109`, `bidagent/review.py:171`, `bidagent/review.py:179`
   - Impact: requirement 集合包含非“可核查义务句”，提升误报和人工复核成本。
   - Evidence: 真实运行产物 `runs/real-verify/requirements.jsonl` 中出现目录段（如 `R0001`）。

3. Medium: 项目默认验证脚本对 Python 工程无效，可能造成“空跑成功”。
   - Location: `scripts/verify.ps1:54`, `scripts/verify.ps1:56`, `scripts/verify.ps1:57`
   - Impact: 若仅执行该脚本，`pytest`/类型检查不会触发，回归可能漏检。

## Test Coverage Gaps

- 缺少 `resume + ai_provider` 的半完成/增量复审用例。
- 缺少“规则抽取噪声过滤”基准样例（目录、模板、格式说明）。
- 缺少面向 Python 项目的统一 verify 脚本门禁。

## Open Questions

1. `resume` 语义是否应定义为“全部 findings 已有指定 provider 的 llm 结果才跳过”？
2. requirement 抽取目标是否仅保留“可核查义务句”并剔除目录/模板段？

## Recommended Next Steps

1. 修复 `pipeline.review` 的 resume 判定：从 `any(...)` 改为“全量覆盖检查”。
2. 为上述修复增加单测：覆盖“部分已有 llm”时仍继续补齐。
3. 收紧 requirement 抽取条件：去除目录/模板/格式性段落。
4. 为 Python 项目补充 `scripts/verify.ps1` 的 `pytest` 路径或新增 Python verify 脚本。

## Fixes Applied (2026-02-15)

1. `pipeline.review` 已改为“全量覆盖检查”：
   - 仅当所有 findings 均有 `llm.provider == ai_provider` 才在 `resume + ai_provider` 下直接复用。
   - 若为“部分 LLM 化”，将继续执行审查并补齐。
2. 已新增测试覆盖：
   - `tests/test_pipeline_review.py::test_resume_ai_partial_llm_triggers_refill`
   - `tests/test_pipeline_review.py::test_resume_ai_full_llm_skips_recompute`
3. requirement 抽取已收紧：
   - 增加句级切分、目录/标题识别、模板/格式说明过滤、可核查义务句判定。
   - 新增测试：
     - `tests/test_review.py::test_extract_requirements_skips_catalog_and_template_noise`
     - `tests/test_review.py::test_extract_requirements_splits_mixed_sentences`
4. `scripts/verify.ps1` 已支持 Python 项目门禁：
   - 无 `package.json` 时可检测 Python 工程并执行测试与编译检查，不再“空跑成功”。

## Verification Rerun

- `python -m unittest discover -s tests -v` -> `13 passed`
- `python -m compileall bidagent tests` -> pass
- `.\scripts\verify.ps1` -> pass（已执行 Python gates）
