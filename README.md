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
- `review`: check bid document against extracted requirements
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
