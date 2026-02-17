from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bidagent.io_utils import write_jsonl
from bidagent.pipeline import report


class ReportTests(unittest.TestCase):
    def test_consistency_section_shows_more_count_when_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "text": "必须提供营业执照",
                        "category": "资质与证照",
                        "rule_tier": "general",
                    }
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "risk",
                        "severity": "medium",
                        "reason": "待确认",
                        "evidence": [],
                    }
                ],
            )
            write_jsonl(
                out_dir / "consistency-findings.jsonl",
                [
                    {
                        "type": "bidder_name",
                        "status": "risk",
                        "severity": "high",
                        "reason": "字段 `bidder_name` 出现多个不同取值",
                        "values": [
                            {"value_raw_examples": ["公司A"], "count": 3},
                            {"value_raw_examples": ["公司B"], "count": 2},
                            {"value_raw_examples": ["公司C"], "count": 1},
                        ],
                    }
                ],
            )

            report(out_dir=out_dir)
            content = (out_dir / "review-report.md").read_text(encoding="utf-8")
            self.assertIn("公司A(count=3)", content)
            self.assertIn("公司B(count=2)", content)
            self.assertIn("(+1 more)", content)


if __name__ == "__main__":
    unittest.main()
