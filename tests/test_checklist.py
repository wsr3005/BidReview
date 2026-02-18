from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bidagent.io_utils import read_jsonl, write_jsonl
from bidagent.pipeline import checklist


class ChecklistTests(unittest.TestCase):
    def test_checklist_exports_blocking_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "text": "必须提供营业执照",
                        "category": "资质与证照",
                    },
                    {
                        "requirement_id": "R0002",
                        "text": "必须提供保证金",
                        "category": "保证金与担保",
                    },
                    {
                        "requirement_id": "R0003",
                        "text": "应响应付款条款",
                        "category": "付款与结算",
                    },
                    {
                        "requirement_id": "R0004",
                        "text": "必须提供营业执照",
                        "category": "资质与证照",
                    },
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {"requirement_id": "R0001", "status": "fail", "severity": "high", "reason": "缺失"},
                    {
                        "requirement_id": "R0002",
                        "status": "risk",
                        "severity": "high",
                        "reason": "证据弱",
                        "evidence": [{"location": {"block_index": 18, "page": 9}}],
                    },
                    {"requirement_id": "R0003", "status": "risk", "severity": "medium", "reason": "一般风险"},
                    {"requirement_id": "R0004", "status": "needs_ocr", "severity": "medium", "reason": "需OCR复核"},
                ],
            )

            result = checklist(out_dir)
            rows = list(read_jsonl(out_dir / "manual-review.jsonl"))

            self.assertEqual(result["manual_review"], 4)
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["requirement_id"] for row in rows}, {"R0001", "R0002", "R0003", "R0004"})
            self.assertTrue((out_dir / "manual-review.md").exists())


if __name__ == "__main__":
    unittest.main()
