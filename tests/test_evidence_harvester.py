from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bidagent.evidence_harvester import (
    build_evidence_index,
    harvest_task_evidence,
    harvest_task_evidence_packs,
    write_evidence_packs_jsonl,
)
from bidagent.io_utils import read_jsonl
from bidagent.models import Block, Location


class EvidenceHarvesterTests(unittest.TestCase):
    def test_build_evidence_index_classifies_block_type_and_hash(self) -> None:
        blocks = [
            {"doc_id": "bid", "text": "Business license is provided.", "location": {"block_index": 1, "page": 2}},
            {
                "doc_id": "bid",
                "text": "Payment table row with 30 days.",
                "location": {"block_index": 2, "page": 2, "section": "TableRow"},
            },
            {
                "doc_id": "bid",
                "text": "Scanned stamp text.",
                "location": {"block_index": 3, "page": 5, "section": "OCR_MEDIA"},
            },
        ]

        index_rows = build_evidence_index(blocks)
        self.assertEqual(len(index_rows), 3)
        self.assertEqual(index_rows[0]["block_type"], "text")
        self.assertEqual(index_rows[1]["block_type"], "table")
        self.assertEqual(index_rows[2]["block_type"], "ocr")
        self.assertEqual(len(index_rows[0]["excerpt_hash"]), 16)
        self.assertNotEqual(index_rows[0]["evidence_id"], index_rows[1]["evidence_id"])

    def test_harvest_task_evidence_returns_top_k_support_and_counter(self) -> None:
        blocks = [
            {
                "doc_id": "bid",
                "text": "We provide a valid business license and tax certificate for this bid.",
                "location": {"block_index": 1, "page": 1, "section": "Normal"},
            },
            {
                "doc_id": "bid",
                "text": "Business license see appendix A.",
                "location": {"block_index": 2, "page": 1, "section": "Normal"},
            },
            {
                "doc_id": "bid",
                "text": "No business license is available at submission time.",
                "location": {"block_index": 3, "page": 2, "section": "Normal"},
            },
        ]
        evidence_index = build_evidence_index(blocks)
        task = {
            "task_id": "T-R0001-01",
            "requirement_id": "R0001",
            "task_type": "evidence_check",
            "query": "Check whether bidder provide business license proof",
            "expected_logic": {"keywords": ["business", "license"]},
        }

        row = harvest_task_evidence(task, evidence_index, top_k=1, counter_k=1)
        self.assertEqual(row["task_id"], "T-R0001-01")
        self.assertEqual(len(row["evidence_pack"]), 1)
        self.assertEqual(len(row["counter_evidence_pack"]), 1)
        self.assertEqual(row["evidence_pack"][0]["location"]["block_index"], 1)
        self.assertEqual(row["counter_evidence_pack"][0]["location"]["block_index"], 3)
        self.assertIn("license", row["retrieval_trace"]["positive_terms"])
        self.assertGreater(row["counter_evidence_pack"][0]["score"], 0)

    def test_harvest_task_evidence_detects_chinese_counter_phrases(self) -> None:
        blocks = [
            {
                "doc_id": "bid",
                "text": "我司已提供营业执照复印件，材料齐全。",
                "location": {"block_index": 1, "page": 1, "section": "Normal"},
            },
            {
                "doc_id": "bid",
                "text": "经核查，营业执照未提供，存在缺失。",
                "location": {"block_index": 2, "page": 1, "section": "Normal"},
            },
        ]
        task = {
            "task_id": "T-R0007-01",
            "requirement_id": "R0007",
            "task_type": "evidence_check",
            "query": "核验是否提供营业执照",
            "expected_logic": {"keywords": ["营业执照"]},
        }

        row = harvest_task_evidence(task, build_evidence_index(blocks), top_k=1, counter_k=1)
        self.assertEqual(len(row["counter_evidence_pack"]), 1)
        self.assertEqual(row["counter_evidence_pack"][0]["location"]["block_index"], 2)
        self.assertIn("未提供", "".join(row["counter_evidence_pack"][0].get("matched_terms") or []))

    def test_harvest_task_evidence_filters_unrelated_counter_noise(self) -> None:
        blocks = [
            {
                "doc_id": "bid",
                "text": "我司已提供营业执照复印件，材料齐全。",
                "location": {"block_index": 1, "page": 1, "section": "Normal"},
            },
            {
                "doc_id": "bid",
                "text": "合同工期不满足30天，需调整施工计划。",
                "location": {"block_index": 2, "page": 2, "section": "Normal"},
            },
        ]
        task = {
            "task_id": "T-R0008-01",
            "requirement_id": "R0008",
            "task_type": "evidence_check",
            "query": "核验是否提供营业执照",
            "expected_logic": {"keywords": ["营业执照"]},
        }
        row = harvest_task_evidence(task, build_evidence_index(blocks), top_k=2, counter_k=2)
        self.assertEqual(len(row["evidence_pack"]), 1)
        self.assertEqual(row["evidence_pack"][0]["location"]["block_index"], 1)
        self.assertEqual(row["counter_evidence_pack"], [])

    def test_harvest_task_evidence_returns_empty_pack_without_terms(self) -> None:
        blocks = [
            {"doc_id": "bid", "text": "General statement with no link.", "location": {"block_index": 1, "page": 1}}
        ]
        task = {"task_id": "T-R0002-01", "requirement_id": "R0002", "query": ""}

        row = harvest_task_evidence(task, build_evidence_index(blocks))
        self.assertEqual(row["evidence_pack"], [])
        self.assertEqual(row["counter_evidence_pack"], [])
        self.assertEqual(row["retrieval_trace"]["candidate_blocks"], 1)

    def test_harvest_task_evidence_packs_for_multiple_tasks(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="Bidder submitted the authorization letter and signed it.",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="Payment term is 45 days after invoice.",
                location=Location(block_index=2, page=2, section="TableRow"),
            ),
        ]
        tasks = [
            {
                "task_id": "T-R0003-01",
                "requirement_id": "R0003",
                "query": "authorization letter",
                "expected_logic": {"keywords": ["authorization", "letter"]},
            },
            {
                "task_id": "T-R0004-01",
                "requirement_id": "R0004",
                "query": "payment term days",
                "expected_logic": {"keywords": ["payment", "days"]},
            },
        ]

        rows = harvest_task_evidence_packs(tasks, blocks, top_k=2, counter_k=1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["task_id"], "T-R0003-01")
        self.assertEqual(rows[1]["task_id"], "T-R0004-01")
        self.assertTrue(rows[0]["evidence_pack"])
        self.assertTrue(rows[1]["evidence_pack"])

    def test_write_evidence_packs_jsonl_validates_protocol(self) -> None:
        row = {
            "task_id": "T-R0005-01",
            "requirement_id": "R0005",
            "query": "license",
            "evidence_pack": [],
            "counter_evidence_pack": [],
            "retrieval_trace": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence-pack.jsonl"
            count = write_evidence_packs_jsonl(path, [row])
            loaded = list(read_jsonl(path))

        self.assertEqual(count, 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["task_id"], "T-R0005-01")

        invalid = [{"task_id": "T-R0006-01", "evidence_pack": [], "counter_evidence_pack": []}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence-pack-invalid.jsonl"
            with self.assertRaises(ValueError):
                write_evidence_packs_jsonl(path, invalid)


if __name__ == "__main__":
    unittest.main()
