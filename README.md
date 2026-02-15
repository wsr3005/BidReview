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
  --bid "北京为是科技有限公司-安徽芜湖投标文件(1).docx" \
  --out "runs/demo" \
  --focus business \
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
- `annotate`: generate issue annotations sidecar files and annotated document copy (`.docx` / `.pdf`)
- `checklist`: export manual review list (`fail` + `high risk`)
- `report`: generate markdown summary report
- `run`: execute the full pipeline

## Notes

- Large files are handled in a staged, resumable pipeline.
- PDF parsing depends on `pypdf` (optional). If unavailable, use TXT/DOCX or install `pypdf`.
- DOCX parser uses built-in XML streaming and does not validate stamp/seal by design.
- If `--ai-provider deepseek` is enabled, the tool reads API key from:
  1) `DEEPSEEK_API_KEY`, 2) `--ai-api-key-file`, 3) `deepseek api key.txt`.
