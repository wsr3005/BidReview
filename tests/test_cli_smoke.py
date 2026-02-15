from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliSmokeTests(unittest.TestCase):
    def test_run_pipeline_with_txt_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            tender = base / "tender.txt"
            bid = base / "bid.txt"
            out = base / "out"

            tender.write_text(
                "\n".join(
                    [
                        "商务要求：投标人必须提供有效营业执照。",
                        "商务要求：投标人应提供近三年类似项目业绩。",
                        "技术参数必须满足性能指标。",
                    ]
                ),
                encoding="utf-8",
            )
            bid.write_text(
                "\n".join(
                    [
                        "我司已提供有效营业执照复印件。",
                        "技术方案详见第六章。",
                    ]
                ),
                encoding="utf-8",
            )

            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bidagent",
                    "run",
                    "--tender",
                    str(tender),
                    "--bid",
                    str(bid),
                    "--out",
                    str(out),
                    "--focus",
                    "business",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(process.returncode, 0, msg=process.stderr)
            self.assertTrue((out / "requirements.jsonl").exists())
            self.assertTrue((out / "findings.jsonl").exists())
            self.assertTrue((out / "annotations.jsonl").exists())
            self.assertTrue((out / "manual-review.jsonl").exists())
            self.assertTrue((out / "review-report.md").exists())


if __name__ == "__main__":
    unittest.main()
