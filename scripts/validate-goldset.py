from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bidagent.eval import validate_gold_rows
from bidagent.io_utils import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate gold set schema, labels, and stratification.")
    parser.add_argument(
        "--path",
        default="docs/goldset/l3-gold.jsonl",
        help="gold set JSONL path",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=200,
        help="minimum required rows",
    )
    parser.add_argument(
        "--min-per-tier",
        type=int,
        default=20,
        help="minimum rows required per tier",
    )
    parser.add_argument(
        "--min-per-status",
        type=int,
        default=5,
        help="minimum rows required per expected_status label",
    )
    parser.add_argument(
        "--allow-partial-statuses",
        action="store_true",
        help="allow a subset of expected_status labels (default requires all labels)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = Path(args.path)
    if not path.exists():
        print(f"gold set file not found: {path}", file=sys.stderr)
        return 1

    rows = list(read_jsonl(path))
    try:
        summary = validate_gold_rows(
            rows,
            min_rows=max(0, int(args.min_rows)),
            require_all_tiers=True,
            require_all_statuses=not bool(args.allow_partial_statuses),
            min_per_tier=max(0, int(args.min_per_tier)),
            min_per_status=max(0, int(args.min_per_status)),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = {
        "path": str(path),
        "summary": summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
