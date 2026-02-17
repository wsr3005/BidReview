from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bidagent.eval import validate_gold_rows
from bidagent.io_utils import read_jsonl, write_jsonl


class GoldsetValidationTests(unittest.TestCase):
    def test_repo_goldset_meets_phase3_baseline(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        goldset_path = repo_root / "docs" / "goldset" / "l3-gold.jsonl"
        self.assertTrue(goldset_path.exists(), msg=f"missing goldset file: {goldset_path}")

        rows = list(read_jsonl(goldset_path))
        summary = validate_gold_rows(
            rows,
            min_rows=200,
            require_all_tiers=True,
            require_all_statuses=True,
            min_per_tier=20,
            min_per_status=5,
        )

        self.assertGreaterEqual(summary["total"], 200)
        self.assertSetEqual(set(summary["tier_counts"].keys()), {"hard_fail", "scored", "general"})
        self.assertSetEqual(
            set(summary["status_counts"].keys()),
            {"pass", "risk", "fail", "needs_ocr", "insufficient_evidence"},
        )

    def test_validate_goldset_script_outputs_summary(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "validate-goldset.py"
        goldset_path = repo_root / "docs" / "goldset" / "l3-gold.jsonl"

        process = subprocess.run(
            [sys.executable, str(script_path), "--path", str(goldset_path)],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,
        )

        self.assertEqual(process.returncode, 0, msg=process.stderr)
        payload = json.loads(process.stdout)
        self.assertEqual(payload.get("path"), str(goldset_path))
        self.assertGreaterEqual(((payload.get("summary") or {}).get("total") or 0), 200)

    def test_validate_goldset_script_rejects_invalid_labels(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "validate-goldset.py"

        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "bad-gold.jsonl"
            write_jsonl(
                bad_path,
                [
                    {"requirement_id": "R0001", "tier": "hard_fail", "expected_status": "fail"},
                    {"requirement_id": "R0002", "tier": "unknown", "expected_status": "maybe"},
                ],
            )

            process = subprocess.run(
                [sys.executable, str(script_path), "--path", str(bad_path), "--min-rows", "0"],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo_root,
            )

        self.assertNotEqual(process.returncode, 0)
        self.assertIn("gold set validation failed", process.stderr)


if __name__ == "__main__":
    unittest.main()
