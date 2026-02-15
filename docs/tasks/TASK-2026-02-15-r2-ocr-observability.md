# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r2-ocr-observability
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

补齐 R2 的“可观测性与环境自检”：当开启 `--ocr-mode auto/tesseract` 时，明确告诉用户 OCR 是否真的工作（引擎可用性、图片数量、成功/失败、OCR 字符量），并将统计落盘到 `ingest/manifest.json` 与 CLI 输出 summary。

## Definition of Done

1. `ingest` summary 增加 `ocr` 字段（engine_available、images_total、images_succeeded、images_failed、chars_total、blocks_emitted 等）。
2. `ingest/manifest.json` 落盘同样的 `ocr` 统计，便于 resume 后回溯。
3. 单测覆盖：
   - `ocr_mode=auto` 时 `manifest.json` 含 `ocr` 字段。
4. `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/ocr.py`
  - `bidagent/pipeline.py`
  - `tests/test_pipeline_review.py`
- Out of scope:
  - 引入 OCR 依赖的自动安装
  - OCR 识别准确率提升

