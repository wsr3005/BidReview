from __future__ import annotations

import json
import os
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
                        # Use punctuation to help keyword extraction keep "营业执照" as a standalone keyword.
                        "商务要求：投标人必须提供：营业执照。",
                        "商务要求：投标人应提供：类似项目业绩。",
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
                    "--ai-provider",
                    "none",
                ],
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    # Offline CI/tests: review pipeline must still run deterministically without network.
                    "BIDAGENT_ALLOW_NO_AI": "1",
                },
            )

            self.assertEqual(process.returncode, 0, msg=process.stderr)
            self.assertTrue((out / "requirements.jsonl").exists())
            self.assertTrue((out / "findings.jsonl").exists())
            self.assertTrue((out / "annotations.jsonl").exists())
            self.assertTrue((out / "manual-review.jsonl").exists())
            self.assertTrue((out / "review-report.md").exists())
            self.assertTrue((out / "release" / "run-metadata.json").exists())
            self.assertTrue((out / "release" / "canary-result.json").exists())
            self.assertTrue((out / "release" / "release-trace.json").exists())
            run_metadata = json.loads((out / "release" / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertIn("model", run_metadata)
            self.assertIn("prompt", run_metadata)
            self.assertIn("strategy", run_metadata)
            report_text = (out / "review-report.md").read_text(encoding="utf-8")
            self.assertIn("| trace: clause=", report_text)
            self.assertIn("rule=keyword_match:r1-trace-v1", report_text)
            self.assertIn("| evidence:", report_text)
            self.assertIn("我司已提供有效营业执照复印件", report_text)


if __name__ == "__main__":
    unittest.main()
