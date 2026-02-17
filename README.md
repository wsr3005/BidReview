# bidagent CLI

`bidagent` is a local CLI workflow for business-focused bid review.

## Setup (uv)

```bash
uv sync
```

## Quick Start

```bash
uv run bidagent run \
  --tender "电子招标文件.pdf" \
  --bid "bid.docx" \
  --out "runs/demo" \
  --focus business \
  --ocr-mode auto \
  --ai-provider deepseek \
  --ai-model deepseek-chat \
  --ai-workers 8 \
  --ai-api-key-file "deepseek api key.txt" \
  --resume
```

## Commands

- `ingest`: parse tender and bid files into JSONL blocks
- `extract-req`: extract business requirements from tender blocks
- `plan-tasks`: decompose each requirement into multiple review tasks (`2-6` tasks when constraints exist)
- `review`: check bid document against extracted requirements
- `verdict`: aggregate task-level verdict protocol into requirement-level final verdicts
- `gate`: apply release thresholds and output `gate-result.json`
- `annotate`: generate issue annotations sidecar files and annotated document copy (`.docx` / `.pdf`)
- `checklist`: export manual review list (`fail` + `high risk` + `needs_ocr`)
- `report`: generate markdown summary report
- `run`: execute the full pipeline

## Release Outputs

`run` also writes release hardening artifacts under `<out>/release/`:

- `run-metadata.json`: model/prompt/strategy version metadata for the run
- `canary-result.json`: canary checks used before `auto_final` promotion
- `release-trace.json`: release decision + key artifact checksums for audit traceability

When `runs/<x>/eval/gold.jsonl` exists, `run` auto-generates `eval/metrics.json` before gate checks.
`run` also writes `evidence-packs.jsonl` to preserve task-level support/counter evidence traces.
`verdict` applies a second-pass conflict audit before finalizing `pass` decisions.
With `--ai-provider deepseek`, `verdict` can run task-level LLM judging before requirement aggregation.
Task-level LLM judging uses `--ai-workers` with a built-in safety cap to reduce 429 burst risk.

## Notes

- Large files are handled in a staged, resumable pipeline.
- Findings include a trace chain (`clause_id`, `evidence_id`, `decision_trace`) for auditability.
- For image evidence (e.g. scanned licenses), use `--ocr-mode auto|tesseract`.
- If only "scan attachment reference" is detected but OCR text is missing, status becomes `needs_ocr`.
- PDF parsing depends on `pypdf` (optional). If unavailable, use TXT/DOCX or install `pypdf`.
- OCR currently supports `pytesseract + pillow` when installed.
- DOCX parser uses built-in XML streaming and does not validate stamp/seal by design.
- If `--ai-provider deepseek` is enabled, the tool reads API key from:
  1) `DEEPSEEK_API_KEY`, 2) `--ai-api-key-file`, 3) `deepseek api key.txt`.
- `gate` / `run` support threshold tuning via `--gate-threshold-*` and short-circuit strategy via `--gate-fail-fast off|critical|all`.
