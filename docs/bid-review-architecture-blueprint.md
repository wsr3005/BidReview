# Bid Review Architecture Blueprint

Date: 2026-02-18
Owner: Codex + Owner
Status: Draft for implementation

## 1) Objective

Build a production-grade bid review pipeline that is:

- structure-first (map before walk),
- context-aware (chapter-specific extraction and checks),
- atomized (checkable rule units),
- defensively auditable (cross-evidence + conservative verdict),
- and operationally efficient (cache + batching + resumable state).

## 2) Core Design Principles

1. Never treat a full tender as unordered chunks.
2. Never use one universal extractor prompt for all chapter types.
3. Never keep compound requirements as one undifferentiated rule.
4. Never issue `PASS` without verifiable positive evidence and conflict checks.
5. Every final decision must be traceable to source blocks/pages.

## 3) Gap Matrix (Current vs Target)

1. Global map:
- Current: no TOC tree/semantic map; retrieval is effectively global.
- Target: logical anchor tree + chapter semantic tags + page offset map.

2. Context-aware routing:
- Current: single extraction flow.
- Target: router dispatches chapter ranges to specialized extractors.

3. Atomization:
- Current: partial sentence split + constraints extraction.
- Target: strict atomic rule schema (subject/action/constraint/logic).

4. Defensive auditing:
- Current: counter-evidence downgrade exists; verdict floor now exists.
- Target: mandatory cross-form verification + strict missing-evidence policy.

5. Throughput/cost:
- Current: P2 extractor can be too slow on long docs.
- Target: batched extraction + semantic cache + async map-reduce execution.

## 4) L0-L3 Implementation Plan

## L0: Non-Structured Armor

1. Add `doc-map.json` after ingest:
- `anchors[]`: heading anchor candidates (chapter title, numbering pattern, page).
- `sections[]`: logical chapter ranges with semantic tags.
- `page_offset`: logical-vs-physical page mapping.

2. Enforce unified IR blocks:
- all parsers output: `block_id, doc_id, page_num, block_type, section_hint, content`.
- downstream modules only consume block ids and IR references.

3. Add initial global entity pool:
- normalize bidder/org/person aliases to stable `entity_id`.
- include fuzzy/OCR typo normalization for high-frequency business entities.

## L1: Context-Aware Extraction

1. Soft routing (not hard filtering):
- chapter tags apply positive weights; fallback to full-text retrieval.

2. Specialized extractors:
- `hf_extractor` for disqualifying clauses.
- `biz_extractor` for commercial terms.
- `tech_extractor` for key technical deviations.
- `appendix_extractor` default deny unless explicitly relevant.

3. P2 optimization:
- batch blocks by section windows (not one-call-per-block).
- per-batch timeout + deterministic fallback to rule extraction.

## L2: Rule Atomization and Fluff Control

1. Strict atomic schema:
- `subject`, `action`, `constraint`, `logic`, `evidence_expectation`, `tier`.

2. Compound-rule splitting:
- split one sentence into multiple atomic checks when multiple constraints exist.

3. Three-state extraction classification:
- `hard` (blocking-capable),
- `soft` (advisory/non-blocking),
- `fluff` (drop from compliance engine).

## L3: Defensive Auditing and Verdict

1. Four-state verdict state machine:
- `PASS`, `FAIL`, `MISSING`, `RISK`.

2. Cross-verification requirements:
- selected clause types must verify across at least two evidence channels where applicable
  (e.g., commitment letter vs deviation table vs attachment evidence).

3. Conservative policy:
- explicit contradictory evidence => `FAIL`.
- no required proof found => `MISSING`.
- ambiguous/partial/weak evidence => `RISK`.

4. Hard guardrails:
- verdict cannot be looser than earlier stage conservative status floor.

## 5) Report and Auditability Requirements

1. Every decision must include:
- decision status/reason,
- source block refs (`block_id`, `page_num`),
- evidence A/B excerpts and conclusion text.

2. Output artifacts:
- `doc-map.json`
- `requirements.atomic.jsonl`
- `evidence-packs.jsonl`
- `cross-audit.jsonl`
- `verdicts.jsonl`

3. UI/report requirement (future):
- clickable source references to page/blocks.
- if bbox is available, render page highlights.

## 6) Cost, Latency, and Reliability

1. Map-reduce async execution for independent atomic checks.
2. Semantic cache for repeated rule/evidence patterns across bids of same project.
3. Stage-level resumability:
- all intermediate states persisted to disk.
4. Early-exit critical fails:
- terminate expensive downstream steps on unambiguous blocking conditions.

## 7) Evaluation and Release Gates

1. Layered gold sets:
- L1 extraction set (amount/date/term extraction precision),
- L2 logic stress set (injected contradiction/omission cases),
- end-to-end blind comparison against expert labels.

2. Required online metrics:
- `hard_fail_recall`,
- `false_positive_fail_rate`,
- `missing_rate`,
- `cross_doc_conflict_recall`,
- `evidence_traceability`.

3. Promotion rule:
- no `auto_final` promotion unless all critical metrics pass threshold.

## 8) Immediate Next Sprint Scope

1. Implement `doc-map.json` (anchor tree + semantic tags + offsets).
2. Introduce soft-routing retrieval weights by section tags.
3. Refactor P2 to batched extraction with timeout fallback.
4. Add `MISSING` status through verdict + report + checklist + gate.

