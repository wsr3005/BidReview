# Task Card

## Metadata

- Task ID: TASK-2026-02-15-code-review-0a86369
- Owner: Codex
- Date: 2026-02-15
- Priority: High
- Risk level: Medium

## Objective

审查提交 `0a86369`（Word/PDF 批注导出）并记录可复现缺陷、影响范围和修复建议。

## Verification

- Commit inspected: `git show --stat --oneline 0a86369`
- Tests run: `python -m pytest -q` (`16 passed`)
- Targeted repros:
  - 相对路径 `bid_path` + 切换工作目录后执行 `annotate`
  - `findings` 全部无定位信息（无 `block_index/page`）时执行 `annotate`

## Findings

1. High: `annotate` 可能返回了不存在的 `annotated_copy` 路径。
   - Location: `bidagent/annotators.py:148`, `bidagent/annotators.py:149`, `bidagent/annotators.py:243`, `bidagent/annotators.py:244`, `bidagent/pipeline.py:257`, `bidagent/pipeline.py:272`
   - Root cause: 当没有可定位位置（DOCX 无 `block_index` / PDF 无 `page`）时，`annotate_docx_copy/annotate_pdf_copy` 会直接返回统计，不会生成输出文件；但 `pipeline.annotate` 仍无条件写入 `result["annotated_copy"]`。
   - Impact: 调用方拿到“看似成功”的路径，但文件不存在，导致后续打开/分发失败。
   - Repro result: `annotated_copy` 返回非空路径，`Path(result["annotated_copy"]).exists() == False`。

2. Medium: `manifest.json` 使用原始字符串路径，跨工作目录执行 `annotate` 时经常找不到源文档。
   - Location: `bidagent/pipeline.py:105`, `bidagent/pipeline.py:107`, `bidagent/pipeline.py:172`, `bidagent/pipeline.py:185`, `bidagent/pipeline.py:186`
   - Root cause: `ingest` 将 `bid_path` 以 `str(bid_path)` 写入 manifest；若是相对路径，后续 `_resolve_bid_source` 按当前进程工作目录解析，目录变化后解析失败。
   - Impact: 仅生成 sidecar，无法生成批注副本；用户会看到 `annotation_warning`，但实际数据可恢复性下降。
   - Repro result: 在 `ingest` 与 `annotate` 之间切换 `cwd`，出现 `Bid source file not found. Sidecar annotation files were generated.`。

3. Low: `annotated_paragraphs` 统计可能重复计数同一段落（长段被分块后）。
   - Location: `bidagent/annotators.py:173`, `bidagent/annotators.py:177`, `bidagent/annotators.py:184`, `bidagent/annotators.py:185`, `bidagent/annotators.py:186`
   - Root cause: 统计去重键使用 `block_index`（`idx`），而不是段落身份；同一段落拆成多个 block 时会被多次计入 `annotated_paragraphs`。
   - Impact: 指标值偏大，影响批注覆盖率度量准确性。

## Test Gaps

- `tests/test_annotate_output.py` 当前覆盖了“有定位能生成副本”和“完全找不到源文件”。
- 缺失：
  - “返回了 `annotated_copy` 但文件不存在”的防回归用例。
  - “manifest 中相对路径 + 切换 cwd” 的解析稳定性用例。

## Recommended Next Steps

1. 在 `pipeline.annotate` 中对 `output_path.exists()` 做最终校验；若未生成则返回 `annotated_copy=None` + 明确 warning。
2. `ingest` 将 `tender_path/bid_path` 持久化为绝对路径（或在 `_resolve_bid_source` 中按 `manifest.json` 所在目录解析相对路径）。
3. 将 `annotated_paragraphs` 的去重键改为段落对象身份（如 `id(paragraph)`），避免重复计数。
4. 为上述两个真实缺陷补充单测。

