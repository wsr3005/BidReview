from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bidagent.document import parse_page_range
from bidagent.pipeline import annotate, checklist, extract_req, ingest, report, review, run_pipeline


def _common_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--out", required=True, help="output directory")
    parent.add_argument("--resume", action="store_true", help="reuse existing stage outputs")
    parent.add_argument(
        "--focus",
        default="business",
        choices=["business", "all"],
        help="review scope",
    )
    parent.add_argument(
        "--page-range",
        default=None,
        help="optional page range for PDF: START-END",
    )
    parent.add_argument(
        "--ocr-mode",
        choices=["off", "auto", "tesseract"],
        default="off",
        help="OCR mode for image-only evidence in bid documents",
    )
    parent.add_argument("--workers", type=int, default=2, help="reserved for parallel stage workers")
    parent.add_argument("--batch-size", type=int, default=500, help="reserved for future batch operations")
    parent.add_argument(
        "--ai-provider",
        choices=["none", "deepseek"],
        default="none",
        help="optional LLM provider for secondary review",
    )
    parent.add_argument("--ai-model", default="deepseek-chat", help="LLM model name")
    parent.add_argument("--ai-api-key-file", default=None, help="path to API key text file")
    parent.add_argument("--ai-workers", type=int, default=4, help="parallel workers for LLM review")
    parent.add_argument(
        "--ai-base-url",
        default="https://api.deepseek.com/v1",
        help="LLM API base url",
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bidagent",
        description="Business-focused bid review CLI for large tender documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parent()

    ingest_parser = subparsers.add_parser("ingest", parents=[common], help="ingest documents")
    ingest_parser.add_argument("--tender", required=True, help="tender file path")
    ingest_parser.add_argument("--bid", required=True, help="bid file path")

    extract_parser = subparsers.add_parser(
        "extract-req", parents=[common], help="extract business requirements from tender"
    )
    extract_parser.add_argument("--tender", help="unused in this stage, kept for CLI consistency")

    subparsers.add_parser("review", parents=[common], help="review bid against requirements")
    annotate_parser = subparsers.add_parser("annotate", parents=[common], help="generate annotations sidecar")
    annotate_parser.add_argument("--bid-source", default=None, help="optional original bid file for annotated copy")
    subparsers.add_parser("report", parents=[common], help="generate markdown report")
    subparsers.add_parser("checklist", parents=[common], help="generate manual review checklist")

    run_parser = subparsers.add_parser("run", parents=[common], help="run full pipeline")
    run_parser.add_argument("--tender", required=True, help="tender file path")
    run_parser.add_argument("--bid", required=True, help="bid file path")

    return parser


def _print_result(result: dict) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    page_range = parse_page_range(args.page_range) if args.page_range else None
    ai_provider = None if args.ai_provider == "none" else args.ai_provider
    ai_api_key_file = Path(args.ai_api_key_file) if args.ai_api_key_file else None

    try:
        if args.command == "ingest":
            result = ingest(
                tender_path=Path(args.tender),
                bid_path=Path(args.bid),
                out_dir=out_dir,
                resume=args.resume,
                page_range=page_range,
                ocr_mode=args.ocr_mode,
            )
        elif args.command == "extract-req":
            result = extract_req(out_dir=out_dir, focus=args.focus, resume=args.resume)
        elif args.command == "review":
            result = review(
                out_dir=out_dir,
                resume=args.resume,
                ai_provider=ai_provider,
                ai_model=args.ai_model,
                ai_api_key_file=ai_api_key_file,
                ai_base_url=args.ai_base_url,
                ai_workers=args.ai_workers,
            )
        elif args.command == "annotate":
            bid_source = Path(args.bid_source) if getattr(args, "bid_source", None) else None
            result = annotate(out_dir=out_dir, resume=args.resume, bid_source=bid_source)
        elif args.command == "report":
            result = report(out_dir=out_dir)
        elif args.command == "checklist":
            result = checklist(out_dir=out_dir, resume=args.resume)
        elif args.command == "run":
            result = run_pipeline(
                tender_path=Path(args.tender),
                bid_path=Path(args.bid),
                out_dir=out_dir,
                focus=args.focus,
                resume=args.resume,
                page_range=page_range,
                ocr_mode=args.ocr_mode,
                ai_provider=ai_provider,
                ai_model=args.ai_model,
                ai_api_key_file=ai_api_key_file,
                ai_base_url=args.ai_base_url,
                ai_workers=args.ai_workers,
            )
        else:
            parser.error(f"unsupported command: {args.command}")
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
