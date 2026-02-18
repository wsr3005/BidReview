from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bidagent.entity_pool import build_entity_pool
from bidagent.pipeline import ingest


class EntityPoolTests(unittest.TestCase):
    def test_build_entity_pool_merges_ocr_confusion_aliases(self) -> None:
        rows = [
            {"doc_id": "tender", "text": "投标人：深圳ABCO有限公司"},
            {"doc_id": "bid", "text": "投标单位：深圳ABC0有限公司"},
            {"doc_id": "bid", "text": "法定代表人：张三"},
            {"doc_id": "bid", "text": "联系人：张三"},
        ]
        payload = build_entity_pool(rows)
        self.assertEqual(payload.get("schema_version"), "entity-pool-v1")
        entities = payload.get("entities") or []
        self.assertGreaterEqual(len(entities), 2)

        org_entities = [item for item in entities if item.get("entity_type") == "organization"]
        self.assertEqual(len(org_entities), 1)
        self.assertEqual(int(org_entities[0].get("mentions") or 0), 2)
        aliases = org_entities[0].get("aliases") or []
        self.assertGreaterEqual(len(aliases), 2)

        person_entities = [item for item in entities if item.get("entity_type") == "person"]
        self.assertEqual(len(person_entities), 1)
        self.assertEqual(int(person_entities[0].get("mentions") or 0), 2)

    def test_ingest_writes_entity_pool_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            tender = base / "tender.txt"
            bid = base / "bid.txt"
            out_dir = base / "out"
            tender.write_text("招标项目：深圳ABCO有限公司供应项目。", encoding="utf-8")
            bid.write_text("投标单位：深圳ABC0有限公司。法定代表人：张三。", encoding="utf-8")

            result = ingest(
                tender_path=tender,
                bid_path=bid,
                out_dir=out_dir,
                resume=False,
                ocr_mode="off",
            )

            self.assertIn("entity_pool", result)
            self.assertGreaterEqual(int((result.get("entity_pool") or {}).get("entities") or 0), 1)
            entity_pool_path = out_dir / "ingest" / "entity-pool.json"
            self.assertTrue(entity_pool_path.exists())
            payload = json.loads(entity_pool_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("schema_version"), "entity-pool-v1")


if __name__ == "__main__":
    unittest.main()
