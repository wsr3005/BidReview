from __future__ import annotations

import unittest

from bidagent.doc_map import build_ingest_doc_map


class DocMapTests(unittest.TestCase):
    def test_build_ingest_doc_map_detects_anchors_and_offset(self) -> None:
        tender_rows = [
            {"doc_id": "tender", "text": "目录", "location": {"block_index": 1, "page": 1, "section": "Heading1"}},
            {"doc_id": "tender", "text": "第一章 投标人须知...............3", "location": {"block_index": 2, "page": 1}},
            {"doc_id": "tender", "text": "第二章 评标办法...............20", "location": {"block_index": 3, "page": 1}},
            {"doc_id": "tender", "text": "第三章 合同条款...............50", "location": {"block_index": 4, "page": 1}},
            {"doc_id": "tender", "text": "第一章 投标人须知", "location": {"block_index": 10, "page": 5}},
            {"doc_id": "tender", "text": "第二章 评标办法", "location": {"block_index": 30, "page": 22}},
            {"doc_id": "tender", "text": "第三章 合同条款", "location": {"block_index": 50, "page": 52}},
            {"doc_id": "tender", "text": "第四章 技术规范", "location": {"block_index": 70, "page": 80}},
        ]
        bid_rows = [
            {"doc_id": "bid", "text": "投标文件封面", "location": {"block_index": 1, "page": 1}},
            {"doc_id": "bid", "text": "法定代表人授权书", "location": {"block_index": 2, "page": 2}},
        ]

        payload = build_ingest_doc_map(tender_rows=tender_rows, bid_rows=bid_rows)
        self.assertEqual(payload.get("schema_version"), "doc-map-v1")
        docs = payload.get("docs") or []
        self.assertEqual(len(docs), 2)

        tender = next(item for item in docs if item.get("doc_id") == "tender")
        self.assertGreaterEqual(len(tender.get("anchors") or []), 4)
        sections = tender.get("sections") or []
        self.assertGreaterEqual(len(sections), 4)
        tags = {str(item.get("semantic_tag") or "") for item in sections}
        self.assertIn("evaluation_risk", tags)
        self.assertIn("business_contract", tags)
        self.assertIn("technical_spec", tags)

        offset = tender.get("page_offset") or {}
        self.assertEqual(offset.get("value"), 2)
        self.assertGreaterEqual(int(offset.get("votes") or 0), 2)
        self.assertGreaterEqual(int(offset.get("matched_entries") or 0), 3)

    def test_build_ingest_doc_map_falls_back_to_full_section(self) -> None:
        tender_rows = [
            {"doc_id": "tender", "text": "本项目采用公开招标方式。", "location": {"block_index": 1, "page": 1}},
            {"doc_id": "tender", "text": "投标人应在截止日前提交材料。", "location": {"block_index": 2, "page": 1}},
        ]
        bid_rows = [
            {"doc_id": "bid", "text": "我司承诺响应招标要求。", "location": {"block_index": 1, "page": 1}},
        ]

        payload = build_ingest_doc_map(tender_rows=tender_rows, bid_rows=bid_rows)
        tender = next(item for item in (payload.get("docs") or []) if item.get("doc_id") == "tender")
        sections = tender.get("sections") or []
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].get("title"), "全文")
        self.assertEqual((sections[0].get("range") or {}).get("start_block"), 1)
        self.assertEqual((sections[0].get("range") or {}).get("end_block"), 2)

    def test_build_ingest_doc_map_extracts_anchor_from_long_paragraph_heading(self) -> None:
        tender_rows = [
            {
                "doc_id": "tender",
                "text": (
                    "本章对评标流程进行说明。根据项目安排，第七章 评标办法与定标原则将明确"
                    "资格审查、否决条款与评分细则，投标人须严格遵循。"
                ),
                "location": {"block_index": 20, "page": 6, "section": "Normal"},
            },
            {
                "doc_id": "tender",
                "text": "评标委员会根据评标办法进行评分与否决项核验。",
                "location": {"block_index": 21, "page": 6, "section": "Normal"},
            },
        ]
        bid_rows = [{"doc_id": "bid", "text": "投标文件正文", "location": {"block_index": 1, "page": 1}}]

        payload = build_ingest_doc_map(tender_rows=tender_rows, bid_rows=bid_rows)
        tender = next(item for item in (payload.get("docs") or []) if item.get("doc_id") == "tender")
        anchors = tender.get("anchors") or []
        self.assertGreaterEqual(len(anchors), 1)
        sections = tender.get("sections") or []
        tags = {str(item.get("semantic_tag") or "") for item in sections}
        self.assertIn("evaluation_risk", tags)

    def test_build_ingest_doc_map_extracts_anchor_from_table_like_long_line(self) -> None:
        tender_rows = [
            {
                "doc_id": "tender",
                "text": "序号 | 所属章节 | 说明",
                "location": {"block_index": 8, "page": 2, "section": "Table"},
            },
            {
                "doc_id": "tender",
                "text": "1 | 第三章 评标办法及定标原则 | 本章用于资格审查与评分细则说明",
                "location": {"block_index": 9, "page": 2, "section": "Table"},
            },
            {
                "doc_id": "tender",
                "text": "2 | 第四章 合同条款 | 本章用于商务履约约定",
                "location": {"block_index": 10, "page": 2, "section": "Table"},
            },
        ]
        bid_rows = [{"doc_id": "bid", "text": "投标文件正文", "location": {"block_index": 1, "page": 1}}]

        payload = build_ingest_doc_map(tender_rows=tender_rows, bid_rows=bid_rows)
        tender = next(item for item in (payload.get("docs") or []) if item.get("doc_id") == "tender")
        anchors = tender.get("anchors") or []
        self.assertGreaterEqual(len(anchors), 2)
        tags = {str(item.get("semantic_tag") or "") for item in anchors}
        self.assertIn("evaluation_risk", tags)
        self.assertIn("business_contract", tags)
        block_type_counts = tender.get("block_type_counts") or {}
        self.assertGreaterEqual(int(block_type_counts.get("table") or 0), 2)


if __name__ == "__main__":
    unittest.main()
