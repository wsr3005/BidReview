from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import json

from bidagent.io_utils import read_jsonl, write_jsonl
from bidagent.models import Block, Finding, Location, Requirement
from bidagent.pipeline import extract_req, gate, ingest, plan_tasks, review, run_pipeline, verdict


class _DummyReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def __init__(self, *args, **kwargs) -> None:
        pass


class PipelineReviewTests(unittest.TestCase):
    def test_extract_req_uses_llm_schema_when_coverage_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(
                ingest_dir / "tender_blocks.jsonl",
                [
                    {
                        "doc_id": "tender",
                        "text": "商务要求：投标人必须提供有效营业执照复印件。",
                        "location": {"block_index": 1, "page": 1, "section": "Normal"},
                    }
                ],
            )

            rule_reqs = [
                Requirement("R0001", "规则条款A", "商务其他", True, ["规则"], []),
                Requirement("R0002", "规则条款B", "商务其他", True, ["规则"], []),
            ]
            llm_reqs = [
                Requirement("R0001", "投标人必须提供有效营业执照复印件。", "资质与证照", True, ["营业执照"], [])
            ]
            with (
                patch("bidagent.pipeline._load_api_key", return_value="sk-test"),
                patch("bidagent.pipeline.extract_requirements", return_value=rule_reqs),
                patch(
                    "bidagent.pipeline.extract_requirements_with_llm",
                    return_value=(llm_reqs, {"items_accepted": 1}),
                ),
            ):
                result = extract_req(
                    out_dir=out_dir,
                    focus="business",
                    resume=False,
                    ai_provider="deepseek",
                )

            rows = list(read_jsonl(out_dir / "requirements.jsonl"))
            self.assertEqual(result.get("extract_engine"), "llm_schema_validated")
            self.assertEqual(len(rows), 1)
            self.assertIn("营业执照", rows[0]["text"])
            self.assertTrue((out_dir / "requirements.atomic.jsonl").exists())
            self.assertGreaterEqual(result.get("atomic_requirements", 0), 1)

    def test_extract_req_falls_back_when_llm_coverage_low(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(
                ingest_dir / "tender_blocks.jsonl",
                [
                    {
                        "doc_id": "tender",
                        "text": "商务要求：投标人必须提供营业执照。商务要求：投标人必须提供授权书。",
                        "location": {"block_index": 1, "page": 1, "section": "Normal"},
                    }
                ],
            )

            rule_reqs = [
                Requirement("R0001", "投标人必须提供营业执照。", "资质与证照", True, ["营业执照"], []),
                Requirement("R0002", "投标人必须提供授权书。", "有效期与响应", True, ["授权书"], []),
                Requirement("R0003", "投标人必须提供承诺函。", "有效期与响应", True, ["承诺函"], []),
                Requirement("R0004", "投标人必须提供报价单。", "报价与税费", True, ["报价单"], []),
            ]
            with (
                patch("bidagent.pipeline._load_api_key", return_value="sk-test"),
                patch("bidagent.pipeline.extract_requirements", return_value=rule_reqs),
                patch(
                    "bidagent.pipeline.extract_requirements_with_llm",
                    return_value=([], {"items_accepted": 0}),
                ),
            ):
                result = extract_req(
                    out_dir=out_dir,
                    focus="business",
                    resume=False,
                    ai_provider="deepseek",
                )

            rows = list(read_jsonl(out_dir / "requirements.jsonl"))
            self.assertEqual(result.get("extract_engine"), "rule_fallback")
            self.assertEqual(len(rows), 4)
            self.assertTrue((out_dir / "requirements.atomic.jsonl").exists())
            self.assertGreaterEqual(result.get("atomic_requirements", 0), 1)

    def test_resume_ai_partial_llm_triggers_refill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {"requirement_id": "R0001", "text": "必须提供营业执照", "mandatory": True},
                    {"requirement_id": "R0002", "text": "必须提供保证金", "mandatory": True},
                ],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [{"doc_id": "bid", "text": "样例", "location": {"block_index": 1}}],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "risk",
                        "score": 1,
                        "severity": "medium",
                        "reason": "partial",
                        "llm": {"provider": "deepseek"},
                    },
                    {
                        "requirement_id": "R0002",
                        "status": "risk",
                        "score": 1,
                        "severity": "medium",
                        "reason": "partial",
                    },
                ],
            )

            fake_findings = [
                Finding("R0001", "risk", 1, "medium", "规则判定"),
                Finding("R0002", "risk", 1, "medium", "规则判定"),
            ]

            def _fake_apply(requirements, findings, reviewer, max_workers, **_kwargs):
                for item in findings:
                    item.llm = {"provider": "deepseek", "model": "deepseek-chat"}
                return findings

            with (
                patch("bidagent.pipeline._load_api_key", return_value="sk-test"),
                patch("bidagent.pipeline.DeepSeekReviewer", _DummyReviewer),
                patch("bidagent.pipeline.review_requirements", return_value=fake_findings) as mocked_review,
                patch("bidagent.pipeline.apply_llm_review", side_effect=_fake_apply) as mocked_apply,
            ):
                result = review(out_dir=out_dir, resume=True, ai_provider="deepseek")

            self.assertEqual(result["findings"], 2)
            self.assertTrue(mocked_review.called)
            self.assertTrue(mocked_apply.called)
            rows = list(read_jsonl(out_dir / "findings.jsonl"))
            self.assertTrue(all((row.get("llm") or {}).get("provider") == "deepseek" for row in rows))

    def test_resume_ai_full_llm_skips_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "pass",
                        "score": 3,
                        "severity": "none",
                        "reason": "ok",
                        "llm": {"provider": "deepseek"},
                    },
                    {
                        "requirement_id": "R0002",
                        "status": "risk",
                        "score": 1,
                        "severity": "medium",
                        "reason": "check",
                        "llm": {"provider": "deepseek"},
                    },
                ],
            )

            with patch("bidagent.pipeline.review_requirements") as mocked_review:
                result = review(out_dir=out_dir, resume=True, ai_provider="deepseek")

            self.assertEqual(result["findings"], 2)
            self.assertFalse(mocked_review.called)

    def test_ingest_appends_ocr_blocks_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            tender = base / "tender.txt"
            bid = base / "bid.txt"
            out_dir = base / "out"
            tender.write_text("商务要求：投标人必须提供营业执照。", encoding="utf-8")
            bid.write_text("我司已提交相关商务文件。", encoding="utf-8")

            fake_ocr_blocks = [
                Block(
                    doc_id="bid",
                    text="营业执照统一社会信用代码：9134XXXXXXXXXX",
                    location=Location(block_index=2, section="OCR_MEDIA"),
                )
            ]
            with (
                patch("bidagent.pipeline.iter_document_ocr_blocks", return_value=iter(fake_ocr_blocks)),
                patch(
                    "bidagent.pipeline.ocr_selfcheck",
                    return_value={"mode": "auto", "engine": "tesseract", "engine_available": True},
                ),
            ):
                result = ingest(
                    tender_path=tender,
                    bid_path=bid,
                    out_dir=out_dir,
                    resume=False,
                    ocr_mode="auto",
                )

            self.assertEqual(result["bid_ocr_blocks"], 1)
            self.assertIn("ocr", result)
            self.assertTrue((result.get("ocr") or {}).get("engine_available"))

            manifest = json.loads((out_dir / "ingest" / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("ocr", manifest)
            self.assertEqual((manifest.get("ocr") or {}).get("mode"), "auto")
            self.assertEqual(str(manifest.get("doc_map_schema_version")), "doc-map-v1")
            self.assertEqual(result["bid_blocks"], 2)
            rows = list(read_jsonl(out_dir / "ingest" / "bid_blocks.jsonl"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["location"]["section"], "OCR_MEDIA")
            self.assertEqual((result.get("doc_map") or {}).get("docs"), 2)
            self.assertTrue((out_dir / "ingest" / "doc-map.json").exists())
            self.assertTrue((out_dir / "ingest" / "entity-pool.json").exists())
            self.assertIn("entity_pool", result)

    def test_plan_tasks_uses_requirement_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "text": "投标人必须提供营业执照且保证金不少于50万元",
                        "category": "资质与证照",
                        "mandatory": True,
                        "keywords": ["营业执照", "保证金"],
                        "constraints": [{"type": "amount", "field": "保证金", "op": ">=", "value_fen": 50000000}],
                        "rule_tier": "hard_fail",
                    }
                ],
            )

            result = plan_tasks(out_dir=out_dir, resume=False)
            rows = list(read_jsonl(out_dir / "review-tasks.jsonl"))

            self.assertGreaterEqual(result["review_tasks"], 2)
            self.assertGreaterEqual(len(rows), 2)
            task_types = {str(row.get("task_type") or "") for row in rows}
            self.assertIn("evidence_check", task_types)
            self.assertIn("keyword_check", task_types)

    def test_gate_allows_auto_final_when_all_metrics_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            (out_dir / "eval").mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {"requirement_id": "R0001", "text": "必须提供营业执照", "rule_tier": "hard_fail", "mandatory": True},
                    {"requirement_id": "R0002", "text": "必须提供保证金", "rule_tier": "general", "mandatory": True},
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "fail",
                        "score": 0,
                        "severity": "high",
                        "reason": "缺失",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.95},
                        "evidence": [{"evidence_id": "E-1", "location": {"block_index": 10, "page": 2}}],
                    },
                    {
                        "requirement_id": "R0002",
                        "status": "pass",
                        "score": 3,
                        "severity": "none",
                        "reason": "满足",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.92},
                        "evidence": [{"evidence_id": "E-2", "location": {"block_index": 18, "page": 4}}],
                    },
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)
            verdict(out_dir=out_dir, resume=False)
            (out_dir / "eval" / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "hard_fail_recall": 1.0,
                            "false_positive_fail_rate": 0.0,
                            "false_positive_fail": 0,
                            "non_fail_total": 1,
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = gate(out_dir=out_dir, requested_release_mode="auto_final")
            self.assertEqual(result["release_mode"], "auto_final")
            self.assertIn("missing_rate", result.get("metrics") or {})
            self.assertTrue((out_dir / "gate-result.json").exists())

    def test_gate_forces_assist_only_when_metrics_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            (out_dir / "eval").mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [{"requirement_id": "R0001", "text": "必须提供营业执照", "rule_tier": "hard_fail", "mandatory": True}],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "fail",
                        "score": 0,
                        "severity": "high",
                        "reason": "缺失",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.95},
                        "evidence": [{"evidence_id": "E-1", "location": {"block_index": 10, "page": 2}}],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)
            verdict(out_dir=out_dir, resume=False)
            (out_dir / "eval" / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "hard_fail_recall": 0.2,
                            "false_positive_fail_rate": 0.0,
                            "false_positive_fail": 0,
                            "non_fail_total": 1,
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = gate(out_dir=out_dir, requested_release_mode="auto_final")
            self.assertEqual(result["release_mode"], "assist_only")
            failed = [item for item in result.get("checks", []) if not item.get("ok")]
            self.assertTrue(any(item.get("name") == "hard_fail_recall" for item in failed))

    def test_gate_threshold_overrides_allow_auto_final(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            (out_dir / "eval").mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [{"requirement_id": "R0001", "text": "必须提供营业执照", "rule_tier": "hard_fail", "mandatory": True}],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "fail",
                        "score": 0,
                        "severity": "high",
                        "reason": "缺失",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.95},
                        "evidence": [{"evidence_id": "E-1", "location": {"block_index": 10, "page": 2}}],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)
            verdict(out_dir=out_dir, resume=False)
            (out_dir / "eval" / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "hard_fail_recall": 0.95,
                            "false_positive_fail_rate": 0.0,
                            "false_positive_fail": 0,
                            "non_fail_total": 1,
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = gate(
                out_dir=out_dir,
                requested_release_mode="auto_final",
                threshold_overrides={"hard_fail_recall": 0.9},
                fail_fast="critical",
            )
            self.assertEqual(result["release_mode"], "auto_final")
            self.assertEqual(result["thresholds"]["hard_fail_recall"], 0.9)
            self.assertFalse((result.get("fail_fast") or {}).get("triggered"))

    def test_gate_fail_fast_critical_short_circuits_remaining_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            (out_dir / "eval").mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [{"requirement_id": "R0001", "text": "必须提供营业执照", "rule_tier": "hard_fail", "mandatory": True}],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "fail",
                        "score": 0,
                        "severity": "high",
                        "reason": "缺失",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.95},
                        "evidence": [{"evidence_id": "E-1", "location": {"block_index": 10, "page": 2}}],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)
            verdict(out_dir=out_dir, resume=False)
            (out_dir / "eval" / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "hard_fail_recall": 0.2,
                            "false_positive_fail_rate": 0.0,
                            "false_positive_fail": 0,
                            "non_fail_total": 1,
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = gate(out_dir=out_dir, requested_release_mode="auto_final", fail_fast="critical")

            self.assertEqual(result["release_mode"], "assist_only")
            fail_fast = result.get("fail_fast") or {}
            self.assertTrue(fail_fast.get("triggered"))
            self.assertEqual(fail_fast.get("triggered_by"), "hard_fail_recall")
            checks = result.get("checks") or []
            skipped = [item for item in checks if item.get("skipped")]
            self.assertTrue(skipped)
            self.assertTrue(any(item.get("name") == "false_positive_fail_rate" for item in skipped))

    def test_gate_evidence_traceability_requires_bid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            (out_dir / "eval").mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [{"requirement_id": "R0001", "text": "必须提供营业执照", "rule_tier": "hard_fail", "mandatory": True}],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "risk",
                        "score": 1,
                        "severity": "medium",
                        "reason": "仅命中弱证据，建议复核",
                        "decision_trace": {
                            "clause_source": {
                                "doc_id": "tender",
                                "location": {"block_index": 3, "page": 1, "section": "BODY"},
                            },
                            "evidence_refs": [],
                        },
                        "evidence": [],
                    }
                ],
            )
            write_jsonl(
                out_dir / "verdicts.jsonl",
                [
                    {
                        "task_id": "task-r1",
                        "requirement_id": "R0001",
                        "status": "risk",
                        "confidence": 0.72,
                        "reason": "证据不足",
                        "evidence_refs": [],
                        "counter_evidence_refs": [],
                        "model": {"provider": "rule_fallback", "name": "rule-only"},
                        "decision_trace": {"decision": {"status": "risk"}},
                    }
                ],
            )
            (out_dir / "eval" / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "hard_fail_recall": 1.0,
                            "false_positive_fail_rate": 0.0,
                            "false_positive_fail": 0,
                            "non_fail_total": 1,
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = gate(out_dir=out_dir, requested_release_mode="auto_final")
            self.assertEqual(result["release_mode"], "assist_only")
            self.assertAlmostEqual((result.get("metrics") or {}).get("evidence_traceability"), 0.0)
            self.assertAlmostEqual((result.get("metrics") or {}).get("llm_coverage"), 0.0)

    def test_verdict_harvests_evidence_refs_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [{"requirement_id": "R0001", "text": "必须提供营业执照", "mandatory": True}],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [
                    {
                        "doc_id": "bid",
                        "text": "我司已提供营业执照复印件，材料齐全。",
                        "location": {"block_index": 1, "page": 1, "section": "BODY"},
                    }
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "risk",
                        "score": 1,
                        "severity": "medium",
                        "reason": "检索到部分证据，建议人工复核",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.72},
                        "evidence": [],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)

            result = verdict(out_dir=out_dir, resume=False)
            self.assertEqual(result["verdicts"], 1)
            self.assertGreaterEqual(result.get("evidence_packs", 0), 1)
            rows = list(read_jsonl(out_dir / "verdicts.jsonl"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "risk")
            self.assertTrue(rows[0]["evidence_refs"])
            self.assertEqual(rows[0]["counter_evidence_refs"], [])
            self.assertTrue((out_dir / "evidence-packs.jsonl").exists())
            trace = rows[0].get("decision_trace") or {}
            floor = trace.get("status_floor") or {}
            self.assertTrue(floor.get("enabled"))
            self.assertEqual(floor.get("floor_status"), "risk")
            self.assertEqual((trace.get("evidence_index") or {}).get("unified_blocks_indexed"), 1)
            self.assertIsInstance((trace.get("evidence_harvest") or {}).get("query_terms"), list)

    def test_verdict_normalizes_legacy_insufficient_status_to_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [{"requirement_id": "R0001", "text": "必须提供营业执照", "mandatory": True}],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [
                    {
                        "doc_id": "bid",
                        "text": "本段未给出营业执照证明材料。",
                        "location": {"block_index": 1, "page": 1, "section": "BODY"},
                    }
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "insufficient_evidence",
                        "score": 0,
                        "severity": "medium",
                        "reason": "证据不足",
                        "llm": {"provider": "rule_fallback", "model": "rule-only", "confidence": 0.35},
                        "evidence": [],
                    }
                ],
            )

            plan_tasks(out_dir=out_dir, resume=False)
            result = verdict(out_dir=out_dir, resume=False)
            self.assertEqual(result["verdicts"], 1)
            rows = list(read_jsonl(out_dir / "verdicts.jsonl"))
            self.assertEqual(rows[0]["status"], "missing")

    def test_verdict_task_level_decision_not_blindly_inherits_requirement_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "text": "投标人必须提供营业执照",
                        "mandatory": True,
                        "keywords": ["营业执照"],
                        "constraints": [],
                        "rule_tier": "hard_fail",
                    }
                ],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [
                    {
                        "doc_id": "bid",
                        "text": "本段仅描述项目背景，无资质证明信息。",
                        "location": {"block_index": 1, "page": 1, "section": "BODY"},
                    }
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "pass",
                        "score": 3,
                        "severity": "none",
                        "reason": "旧结论误判为通过",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.92},
                        "evidence": [],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)

            result = verdict(out_dir=out_dir, resume=False)
            self.assertEqual(result["verdicts"], 1)
            rows = list(read_jsonl(out_dir / "verdicts.jsonl"))
            self.assertEqual(len(rows), 1)
            self.assertNotEqual(rows[0]["status"], "pass")
            self.assertIn(rows[0]["status"], {"risk", "needs_ocr", "missing", "fail"})

    def test_verdict_uses_task_level_llm_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "text": "投标人必须提供营业执照",
                        "mandatory": True,
                        "keywords": ["营业执照"],
                        "constraints": [],
                        "rule_tier": "hard_fail",
                    }
                ],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [
                    {
                        "doc_id": "bid",
                        "text": "投标文件附有营业执照相关说明。",
                        "location": {"block_index": 1, "page": 1, "section": "BODY"},
                    }
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "pass",
                        "score": 2,
                        "severity": "none",
                        "reason": "规则初判通过",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.86},
                        "evidence": [],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)

            with (
                patch("bidagent.pipeline._load_api_key", return_value="sk-test"),
                patch(
                    "bidagent.pipeline._PipelineDeepSeekTaskReviewer.review_task",
                    return_value={"status": "fail", "reason": "LLM任务判定失败", "confidence": 0.91},
                ) as mocked_task_review,
            ):
                result = verdict(
                    out_dir=out_dir,
                    resume=False,
                    ai_provider="deepseek",
                    ai_model="deepseek-chat",
                    ai_workers=16,
                )

            self.assertEqual(result["verdicts"], 1)
            self.assertTrue(mocked_task_review.called)
            rows = list(read_jsonl(out_dir / "verdicts.jsonl"))
            self.assertEqual(rows[0]["status"], "fail")
            self.assertIn("LLM任务判定失败", rows[0]["reason"])
            self.assertEqual((rows[0].get("model") or {}).get("provider"), "deepseek")
            trace = rows[0].get("decision_trace") or {}
            rate_limit = (trace.get("task_verdicts") or {}).get("llm_rate_limit") or {}
            self.assertEqual(rate_limit.get("strategy"), "deepseek_concurrency_cap_4")
            self.assertEqual(rate_limit.get("requested_workers"), 16)
            self.assertEqual(rate_limit.get("max_workers"), 4)

    def test_verdict_downgrades_unstable_pass_to_risk_on_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [{"requirement_id": "R0001", "text": "必须提供营业执照", "mandatory": True}],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [
                    {
                        "doc_id": "bid",
                        "text": "我司已提供营业执照复印件，符合要求。",
                        "location": {"block_index": 1, "page": 1, "section": "BODY"},
                    },
                    {
                        "doc_id": "bid",
                        "text": "材料清单显示营业执照未提供，存在缺失。",
                        "location": {"block_index": 2, "page": 1, "section": "BODY"},
                    },
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "pass",
                        "score": 3,
                        "severity": "none",
                        "reason": "匹配到充分证据",
                        "llm": {"provider": "deepseek", "model": "deepseek-chat", "confidence": 0.93},
                        "evidence": [
                            {
                                "evidence_id": "E-bid-p1-b1-sBODY",
                                "doc_id": "bid",
                                "location": {"block_index": 1, "page": 1, "section": "BODY"},
                                "excerpt": "我司已提供营业执照复印件，符合要求。",
                                "score": 3,
                            }
                        ],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)

            result = verdict(out_dir=out_dir, resume=False)
            self.assertEqual(result["verdicts"], 1)
            self.assertGreaterEqual(result.get("evidence_packs", 0), 1)
            rows = list(read_jsonl(out_dir / "verdicts.jsonl"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "risk")
            self.assertTrue(any("p1-b2" in token for token in rows[0]["counter_evidence_refs"]))
            self.assertIn("冲突证据", rows[0]["reason"])
            trace = rows[0].get("decision_trace") or {}
            audit = trace.get("counter_evidence_audit") or {}
            self.assertTrue(audit.get("conflict_detected"))
            self.assertIn(
                audit.get("action"),
                {None, "downgrade_pass_to_risk_conflict_second_pass"},
            )
            second_pass = audit.get("second_pass") or {}
            self.assertIn(second_pass.get("conflict_level"), {"strong", "weak", "none", None})
            self.assertEqual((trace.get("decision") or {}).get("status"), "risk")

    def test_verdict_cross_audit_downgrades_single_channel_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "text": "投标人必须提供营业执照复印件。",
                        "mandatory": True,
                        "rule_tier": "hard_fail",
                        "keywords": ["营业执照"],
                    }
                ],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [
                    {
                        "doc_id": "bid",
                        "block_id": "B-bid-1",
                        "block_type": "text",
                        "section_hint": "body",
                        "text": "我司已提供营业执照复印件，符合要求。",
                        "location": {"block_index": 1, "page": 1, "section": "BODY"},
                    }
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "pass",
                        "score": 3,
                        "severity": "none",
                        "reason": "匹配到证据",
                        "evidence": [
                            {
                                "evidence_id": "E-B-bid-1",
                                "block_id": "B-bid-1",
                                "doc_id": "bid",
                                "source_type": "text",
                                "location": {"block_index": 1, "page": 1, "section": "BODY"},
                                "excerpt": "我司已提供营业执照复印件，符合要求。",
                                "score": 3,
                            }
                        ],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)

            result = verdict(out_dir=out_dir, resume=False)
            self.assertEqual(result["verdicts"], 1)
            self.assertEqual(result.get("cross_audit"), 1)
            self.assertEqual(result.get("cross_audit_required"), 1)
            self.assertEqual(result.get("cross_audit_verified"), 0)

            rows = list(read_jsonl(out_dir / "verdicts.jsonl"))
            self.assertEqual(rows[0]["status"], "risk")
            trace = rows[0].get("decision_trace") or {}
            cross = trace.get("cross_audit") or {}
            self.assertTrue(cross.get("required"))
            self.assertFalse(cross.get("cross_verified"))
            self.assertEqual(cross.get("action"), "downgrade_pass_to_risk_cross_verification")

            audit_rows = list(read_jsonl(out_dir / "cross-audit.jsonl"))
            self.assertEqual(len(audit_rows), 1)
            self.assertEqual(audit_rows[0].get("status_after"), "risk")

    def test_verdict_early_exits_for_hard_fail_with_fail_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "text": "投标人必须提供有效营业执照，否则将被否决。",
                        "mandatory": True,
                        "rule_tier": "hard_fail",
                        "keywords": ["营业执照", "否决"],
                    }
                ],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [
                    {
                        "doc_id": "bid",
                        "text": "材料清单显示营业执照未提供，存在缺失。",
                        "location": {"block_index": 1, "page": 1, "section": "BODY"},
                    }
                ],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "fail",
                        "score": 0,
                        "severity": "high",
                        "reason": "命中明确反证",
                        "evidence": [
                            {
                                "evidence_id": "E-bid-p1-b1-sBODY",
                                "doc_id": "bid",
                                "location": {"block_index": 1, "page": 1, "section": "BODY"},
                                "excerpt": "材料清单显示营业执照未提供，存在缺失。",
                                "score": 9,
                            }
                        ],
                    }
                ],
            )
            plan_tasks(out_dir=out_dir, resume=False)

            result = verdict(out_dir=out_dir, resume=False)
            self.assertEqual(result["verdicts"], 1)
            rows = list(read_jsonl(out_dir / "verdicts.jsonl"))
            self.assertEqual(rows[0]["status"], "fail")
            trace = rows[0].get("decision_trace") or {}
            task_verdicts = trace.get("task_verdicts") or {}
            self.assertTrue(task_verdicts.get("early_exit_hard_fail"))

    def test_run_pipeline_writes_release_hardening_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            tender = base / "tender.txt"
            bid = base / "bid.txt"
            out_dir = base / "out"
            tender.write_text("商务要求：投标人必须提供营业执照。", encoding="utf-8")
            bid.write_text("我司已提供营业执照复印件。", encoding="utf-8")

            with patch.dict("os.environ", {"BIDAGENT_ALLOW_NO_AI": "1"}, clear=False):
                result = run_pipeline(
                    tender_path=tender,
                    bid_path=bid,
                    out_dir=out_dir,
                    focus="business",
                    resume=False,
                    ocr_mode="auto",
                    ai_provider=None,
                    release_mode="auto_final",
                )

            self.assertIn(result["gate"]["release_mode"], {"assist_only", "auto_final"})
            self.assertTrue((out_dir / "review-tasks.jsonl").exists())
            self.assertTrue((out_dir / "requirements.atomic.jsonl").exists())
            self.assertTrue((out_dir / "evidence-packs.jsonl").exists())
            self.assertTrue((out_dir / "cross-audit.jsonl").exists())
            self.assertTrue((out_dir / "verdicts.jsonl").exists())
            self.assertTrue((out_dir / "gate-result.json").exists())
            self.assertTrue((out_dir / "ingest" / "entity-pool.json").exists())
            self.assertTrue((out_dir / "release" / "run-metadata.json").exists())
            self.assertTrue((out_dir / "release" / "canary-result.json").exists())
            self.assertTrue((out_dir / "release" / "release-trace.json").exists())
            self.assertIn("run_metadata", result)
            self.assertIn("canary", result)
            self.assertIn("release_trace", result)
            self.assertIn("eval", result)
            self.assertIn("metrics_available", result["eval"])

            run_metadata = json.loads((out_dir / "release" / "run-metadata.json").read_text(encoding="utf-8"))
            self.assertIn("model", run_metadata)
            self.assertIn("prompt", run_metadata)
            self.assertIn("strategy", run_metadata)

            canary_result = json.loads((out_dir / "release" / "canary-result.json").read_text(encoding="utf-8"))
            self.assertEqual(canary_result["requested_release_mode"], "auto_final")
            self.assertIn(canary_result["status"], {"pass", "fail"})

            release_trace = json.loads((out_dir / "release" / "release-trace.json").read_text(encoding="utf-8"))
            self.assertIn("artifacts", release_trace)
            self.assertIsInstance(release_trace["artifacts"], list)
            self.assertGreaterEqual(len(release_trace["artifacts"]), 1)

    def test_run_pipeline_release_mode_follows_gate_threshold_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            tender = base / "tender.txt"
            bid = base / "bid.txt"
            out_dir = base / "out"
            tender.write_text("商务要求：投标人必须提供营业执照。", encoding="utf-8")
            bid.write_text("我司响应本项目商务要求。", encoding="utf-8")
            (out_dir / "eval").mkdir(parents=True, exist_ok=True)
            (out_dir / "eval" / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "hard_fail_recall": 1.0,
                            "false_positive_fail_rate": 0.0,
                            "false_positive_fail": 0,
                            "non_fail_total": 1,
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            def _fake_review(**kwargs):
                lane_out = Path(kwargs["out_dir"])
                write_jsonl(
                    lane_out / "findings.jsonl",
                    [
                        {
                            "requirement_id": "R0001",
                            "status": "pass",
                            "score": 2,
                            "severity": "none",
                            "reason": "LLM判定通过",
                            "llm": {
                                "provider": "deepseek",
                                "model": "deepseek-chat",
                                "prompt_version": "deepseek-review-v1",
                                "confidence": 0.92,
                            },
                            "decision_trace": {
                                "decision": {"status": "pass", "reason": "LLM判定通过", "source": "llm"}
                            },
                            "evidence": [],
                        }
                    ],
                )
                return {"findings": 1, "status_counts": {"pass": 1}}

            with patch("bidagent.pipeline.review", side_effect=_fake_review):
                result = run_pipeline(
                    tender_path=tender,
                    bid_path=bid,
                    out_dir=out_dir,
                    focus="business",
                    resume=False,
                    ocr_mode="auto",
                    ai_provider="deepseek",
                    release_mode="auto_final",
                    gate_threshold_overrides={"evidence_traceability": 0.0},
                    canary_min_streak=1,
                )

            self.assertEqual(result["gate"]["release_mode"], "auto_final")
            self.assertTrue(result["gate"]["eligible_for_auto_final"])
            self.assertEqual(result["release_mode"], "auto_final")

            canary_result = json.loads((out_dir / "release" / "canary-result.json").read_text(encoding="utf-8"))
            self.assertEqual(canary_result["status"], "pass")
            self.assertEqual(canary_result["release_mode"], "auto_final")

    def test_run_pipeline_auto_final_guard_requires_consecutive_core_pass_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            tender = base / "tender.txt"
            bid = base / "bid.txt"
            tender.write_text("商务要求：投标人必须提供营业执照。", encoding="utf-8")
            bid.write_text("我司已提供营业执照复印件。", encoding="utf-8")

            def _prepare_out(out_dir: Path) -> None:
                (out_dir / "eval").mkdir(parents=True, exist_ok=True)
                (out_dir / "eval" / "metrics.json").write_text(
                    json.dumps(
                        {
                            "metrics": {
                                "hard_fail_recall": 1.0,
                                "false_positive_fail_rate": 0.0,
                                "false_positive_fail": 0,
                                "non_fail_total": 1,
                            }
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            def _fake_review(**kwargs):
                lane_out = Path(kwargs["out_dir"])
                write_jsonl(
                    lane_out / "findings.jsonl",
                    [
                        {
                            "requirement_id": "R0001",
                            "status": "pass",
                            "score": 2,
                            "severity": "none",
                            "reason": "LLM判定通过",
                            "llm": {
                                "provider": "deepseek",
                                "model": "deepseek-chat",
                                "prompt_version": "deepseek-review-v1",
                                "confidence": 0.92,
                            },
                            "decision_trace": {
                                "decision": {"status": "pass", "reason": "LLM判定通过", "source": "llm"}
                            },
                            "evidence": [],
                        }
                    ],
                )
                return {"findings": 1, "status_counts": {"pass": 1}}

            outputs: list[dict] = []
            with patch("bidagent.pipeline.review", side_effect=_fake_review):
                for index in (1, 2, 3):
                    out_dir = base / f"out-{index}"
                    _prepare_out(out_dir)
                    outputs.append(
                        run_pipeline(
                            tender_path=tender,
                            bid_path=bid,
                            out_dir=out_dir,
                            focus="business",
                            resume=False,
                            ocr_mode="auto",
                            ai_provider=None,
                            release_mode="auto_final",
                            gate_threshold_overrides={"evidence_traceability": 0.0},
                            canary_min_streak=3,
                        )
                    )

            self.assertEqual(outputs[0]["release_mode"], "assist_only")
            self.assertEqual(outputs[1]["release_mode"], "assist_only")
            self.assertEqual(outputs[2]["release_mode"], "auto_final")
            self.assertEqual((outputs[0]["canary"].get("guard") or {}).get("streak_after"), 1)
            self.assertEqual((outputs[1]["canary"].get("guard") or {}).get("streak_after"), 2)
            self.assertEqual((outputs[2]["canary"].get("guard") or {}).get("streak_after"), 3)

            history = list(read_jsonl(base / "auto-final-history.jsonl"))
            self.assertEqual(len(history), 3)
            self.assertTrue(all(bool(row.get("core_ok")) for row in history))
            self.assertEqual(history[-1].get("streak_after"), 3)
            self.assertEqual(history[-1].get("final_release_mode"), "auto_final")


if __name__ == "__main__":
    unittest.main()
