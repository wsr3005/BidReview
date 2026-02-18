from __future__ import annotations

import unittest

from bidagent.evidence_index import (
    SOURCE_OCR,
    SOURCE_TABLE,
    SOURCE_TEXT,
    build_unified_evidence_index,
    retrieve_evidence_candidates,
)


class EvidenceIndexTests(unittest.TestCase):
    def test_build_unified_evidence_index_normalizes_source_type_aliases(self) -> None:
        rows = [
            {"doc_id": "bid", "text": "主体资格满足要求", "source_type": "text", "location": {"block_index": 1, "page": 1}},
            {
                "doc_id": "bid",
                "text": "项目|金额|税率",
                "source_type": "table",
                "location": {"block_index": 2, "page": 2},
            },
            {
                "doc_id": "bid",
                "text": "营业执照扫描件见附件",
                "source_type": "ocr_image",
                "location": {"block_index": 3, "page": 3},
            },
        ]

        indexed = build_unified_evidence_index(rows)
        self.assertEqual([row["source_type"] for row in indexed], [SOURCE_TEXT, SOURCE_TABLE, SOURCE_OCR])

    def test_build_unified_evidence_index_inferrs_source_types(self) -> None:
        rows = [
            {"doc_id": "bid", "text": "营业执照复印件", "location": {"block_index": 1, "section": "OCR_MEDIA", "page": 8}},
            {"doc_id": "bid", "text": "项目|金额|税率", "location": {"block_index": 2, "page": 5}},
            {"doc_id": "bid", "text": "我司承诺满足付款条件", "location": {"block_index": 3, "page": 9}},
        ]

        indexed = build_unified_evidence_index(rows)
        self.assertEqual(indexed[0]["source_type"], SOURCE_OCR)
        self.assertEqual(indexed[1]["source_type"], SOURCE_TABLE)
        self.assertEqual(indexed[2]["source_type"], SOURCE_TEXT)

    def test_build_unified_evidence_index_deduplicates_evidence_ids(self) -> None:
        rows = [
            {"doc_id": "bid", "text": "第一条证据", "location": {"block_index": 7, "page": 2}},
            {"doc_id": "bid", "text": "第二条证据", "location": {"block_index": 7, "page": 2}},
        ]

        indexed = build_unified_evidence_index(rows)
        self.assertEqual(indexed[0]["evidence_id"], "E-bid-p2-b7-text")
        self.assertEqual(indexed[1]["evidence_id"], "E-bid-p2-b7-text-n2")

    def test_retrieve_evidence_candidates_top_k_and_source_filter(self) -> None:
        indexed = build_unified_evidence_index(
            [
                {"doc_id": "bid", "text": "营业执照扫描件见附件", "source_type": "ocr", "location": {"block_index": 1, "page": 1}},
                {
                    "doc_id": "bid",
                    "text": "项目|金额|保证金|500000",
                    "source_type": "table",
                    "location": {"block_index": 2, "page": 2},
                },
                {"doc_id": "bid", "text": "企业基本信息说明", "source_type": "text", "location": {"block_index": 3, "page": 3}},
            ]
        )

        ranked = retrieve_evidence_candidates(indexed, "保证金 金额 500000", top_k=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["source_type"], SOURCE_TABLE)
        self.assertGreaterEqual(float(ranked[0]["score"]), float(ranked[1]["score"]))

        ocr_only = retrieve_evidence_candidates(indexed, "营业执照 扫描件", top_k=5, source_types=["ocr_image"])
        self.assertEqual(len(ocr_only), 1)
        self.assertEqual(ocr_only[0]["source_type"], SOURCE_OCR)

    def test_retrieve_evidence_candidates_returns_empty_when_query_blank(self) -> None:
        indexed = build_unified_evidence_index(
            [{"doc_id": "bid", "text": "测试内容", "location": {"block_index": 1, "page": 1}}]
        )
        self.assertEqual(retrieve_evidence_candidates(indexed, "   ", top_k=3), [])

    def test_retrieve_evidence_candidates_hybrid_semantic_boost(self) -> None:
        indexed = build_unified_evidence_index(
            [
                {
                    "doc_id": "bid",
                    "text": "法定代表人身份证明文件已提交并加盖公章。",
                    "location": {"block_index": 1, "page": 2},
                },
                {
                    "doc_id": "bid",
                    "text": "项目付款方式详见合同条款。",
                    "location": {"block_index": 2, "page": 3},
                },
            ]
        )
        ranked = retrieve_evidence_candidates(indexed, "法人身份证明", top_k=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["location"]["block_index"], 1)


if __name__ == "__main__":
    unittest.main()

