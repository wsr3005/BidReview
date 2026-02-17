from __future__ import annotations

import io
import os
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterator

from bidagent.review import normalize_compact
from bidagent.document import split_text_blocks
from bidagent.models import Block, Location

OCR_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _resolve_tesseract_cmd() -> str | None:
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        p = Path(env_cmd)
        if p.exists():
            return str(p)
    which = shutil.which("tesseract")
    if which:
        return which
    for candidate in DEFAULT_TESSERACT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def ocr_selfcheck(mode: str) -> dict[str, Any]:
    if mode == "off":
        return {
            "mode": mode,
            "engine": None,
            "engine_available": False,
            "reason": "ocr_mode=off",
        }
    if mode not in {"auto", "tesseract"}:
        return {
            "mode": mode,
            "engine": None,
            "engine_available": False,
            "reason": "unknown ocr mode",
        }

    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ModuleNotFoundError as exc:
        return {
            "mode": mode,
            "engine": "tesseract",
            "engine_available": False,
            "reason": f"missing python deps: {exc.name}",
        }

    cmd = _resolve_tesseract_cmd()
    if not cmd:
        return {
            "mode": mode,
            "engine": "tesseract",
            "engine_available": False,
            "reason": "tesseract binary not found (install it or set TESSERACT_CMD)",
        }

    # Avoid calling pytesseract.get_tesseract_version() directly: on some Windows setups it can hang.
    version = None
    try:
        proc = subprocess.run(
            [cmd, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = (proc.stdout or proc.stderr or "").splitlines()[0].strip() if (proc.stdout or proc.stderr) else None
    except Exception:
        version = None

    if not version:
        return {
            "mode": mode,
            "engine": "tesseract",
            "engine_available": False,
            "reason": "tesseract binary detected but version check failed",
        }

    return {
        "mode": mode,
        "engine": "tesseract",
        "engine_available": True,
        "tesseract_version": version,
    }


def _load_tesseract_engine() -> Callable[[bytes], str] | None:
    try:
        from PIL import Image
        import pytesseract
    except ModuleNotFoundError:
        return None

    cmd = _resolve_tesseract_cmd()
    if not cmd:
        return None
    pytesseract.pytesseract.tesseract_cmd = cmd

    def _extract(image_bytes: bytes) -> str:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return pytesseract.image_to_string(image, lang="chi_sim+eng")

    return _extract


def load_ocr_engine(mode: str) -> Callable[[bytes], str] | None:
    if mode == "off":
        return None
    if mode in {"auto", "tesseract"}:
        return _load_tesseract_engine()
    return None


def iter_docx_ocr_blocks(
    path: Path,
    doc_id: str,
    start_index: int,
    ocr_mode: str = "auto",
    stats: dict[str, Any] | None = None,
    max_workers: int | None = None,
) -> Iterator[Block]:
    engine = load_ocr_engine(ocr_mode)
    if engine is None:
        if isinstance(stats, dict):
            stats.setdefault("engine_available", False)
        return

    current_index = start_index
    if max_workers is None:
        # Tesseract is CPU-heavy. Default to a conservative concurrency.
        cpu = os.cpu_count() or 2
        max_workers = max(1, min(4, cpu // 2))
    max_in_flight = max(2, int(max_workers) * 2)

    def _ocr_bytes(image_bytes: bytes) -> str:
        return engine(image_bytes) or ""

    seen_hashes: set[str] = set()
    with zipfile.ZipFile(path, "r") as archive:
        media_names = [name for name in archive.namelist() if name.startswith("word/media/")]
        image_names = [name for name in media_names if Path(name).suffix.lower() in OCR_IMAGE_EXTENSIONS]

        if isinstance(stats, dict):
            stats.setdefault("images_total", 0)
            stats.setdefault("images_skipped_duplicate", 0)
            stats.setdefault("images_succeeded", 0)
            stats.setdefault("images_failed", 0)

        # Tesseract is external-process based; threads are sufficient here.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight: dict[int, tuple[str, Any]] = {}
            next_seq = 0
            seq = 0

            def _drain_one(expected_seq: int) -> Iterator[str]:
                nonlocal in_flight
                name, fut = in_flight.pop(expected_seq)
                try:
                    text = fut.result()
                except Exception:  # noqa: BLE001
                    if isinstance(stats, dict):
                        stats["images_failed"] = int(stats.get("images_failed", 0)) + 1
                    return iter(())
                if isinstance(stats, dict):
                    stats["images_succeeded"] = int(stats.get("images_succeeded", 0)) + 1
                # Some images are logos/stamps with no useful text; filter noise.
                if not normalize_compact(text):
                    return iter(())
                return iter((text,))

            for name in image_names:
                image_bytes = archive.read(name)
                digest = hashlib.sha256(image_bytes).hexdigest()

                if digest in seen_hashes:
                    if isinstance(stats, dict):
                        stats["images_skipped_duplicate"] = int(stats.get("images_skipped_duplicate", 0)) + 1
                    continue
                seen_hashes.add(digest)

                if isinstance(stats, dict):
                    stats["images_total"] = int(stats.get("images_total", 0)) + 1

                fut = executor.submit(_ocr_bytes, image_bytes)
                in_flight[seq] = (name, fut)
                seq += 1

                # Keep in-flight bounded and preserve deterministic output order.
                while len(in_flight) >= max_in_flight and next_seq in in_flight:
                    for text in _drain_one(next_seq):
                        for chunk in split_text_blocks(text):
                            if isinstance(stats, dict):
                                stats["chars_total"] = int(stats.get("chars_total", 0)) + len(chunk)
                            current_index += 1
                            if isinstance(stats, dict):
                                stats["blocks_emitted"] = int(stats.get("blocks_emitted", 0)) + 1
                            yield Block(
                                doc_id=doc_id,
                                text=chunk,
                                location=Location(block_index=current_index, section="OCR_MEDIA"),
                            )
                    next_seq += 1

            # Drain remaining futures in submission order.
            while next_seq < seq:
                if next_seq not in in_flight:
                    next_seq += 1
                    continue
                for text in _drain_one(next_seq):
                    for chunk in split_text_blocks(text):
                        if isinstance(stats, dict):
                            stats["chars_total"] = int(stats.get("chars_total", 0)) + len(chunk)
                        current_index += 1
                        if isinstance(stats, dict):
                            stats["blocks_emitted"] = int(stats.get("blocks_emitted", 0)) + 1
                        yield Block(
                            doc_id=doc_id,
                            text=chunk,
                            location=Location(block_index=current_index, section="OCR_MEDIA"),
                        )
                next_seq += 1


def iter_pdf_ocr_blocks(
    path: Path,
    doc_id: str,
    start_index: int,
    page_range: tuple[int, int] | None = None,
    ocr_mode: str = "auto",
    stats: dict[str, Any] | None = None,
    max_workers: int | None = None,
) -> Iterator[Block]:
    engine = load_ocr_engine(ocr_mode)
    if engine is None:
        if isinstance(stats, dict):
            stats.setdefault("engine_available", False)
        return

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        return

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = max(1, min(4, cpu // 2))
    max_in_flight = max(2, int(max_workers) * 2)

    def _ocr_bytes(image_bytes: bytes) -> str:
        return engine(image_bytes) or ""

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    start_page, end_page = 1, total_pages
    if page_range:
        start_page, end_page = page_range
        end_page = min(end_page, total_pages)

    current_index = start_index
    if isinstance(stats, dict):
        stats.setdefault("images_total", 0)
        stats.setdefault("images_succeeded", 0)
        stats.setdefault("images_failed", 0)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight: dict[int, tuple[int, Any]] = {}
        next_seq = 0
        seq = 0

        def _drain_one(expected_seq: int) -> Iterator[tuple[int, str]]:
            page_no, fut = in_flight.pop(expected_seq)
            try:
                text = fut.result()
            except Exception:  # noqa: BLE001
                if isinstance(stats, dict):
                    stats["images_failed"] = int(stats.get("images_failed", 0)) + 1
                return iter(())
            if isinstance(stats, dict):
                stats["images_succeeded"] = int(stats.get("images_succeeded", 0)) + 1
            if not normalize_compact(text):
                return iter(())
            return iter(((page_no, text),))

        for page_no in range(start_page, end_page + 1):
            page = reader.pages[page_no - 1]
            try:
                images = list(page.images)
            except Exception:  # noqa: BLE001
                images = []
            for image in images:
                data = getattr(image, "data", None)
                if not isinstance(data, (bytes, bytearray)):
                    continue
                if isinstance(stats, dict):
                    stats["images_total"] = int(stats.get("images_total", 0)) + 1
                fut = executor.submit(_ocr_bytes, bytes(data))
                in_flight[seq] = (page_no, fut)
                seq += 1

                while len(in_flight) >= max_in_flight and next_seq in in_flight:
                    for page_for_text, text in _drain_one(next_seq):
                        for chunk in split_text_blocks(text):
                            if isinstance(stats, dict):
                                stats["chars_total"] = int(stats.get("chars_total", 0)) + len(chunk)
                            current_index += 1
                            if isinstance(stats, dict):
                                stats["blocks_emitted"] = int(stats.get("blocks_emitted", 0)) + 1
                            yield Block(
                                doc_id=doc_id,
                                text=chunk,
                                location=Location(block_index=current_index, page=page_for_text, section="OCR_MEDIA"),
                            )
                    next_seq += 1

        while next_seq < seq:
            if next_seq not in in_flight:
                next_seq += 1
                continue
            for page_for_text, text in _drain_one(next_seq):
                for chunk in split_text_blocks(text):
                    if isinstance(stats, dict):
                        stats["chars_total"] = int(stats.get("chars_total", 0)) + len(chunk)
                    current_index += 1
                    if isinstance(stats, dict):
                        stats["blocks_emitted"] = int(stats.get("blocks_emitted", 0)) + 1
                    yield Block(
                        doc_id=doc_id,
                        text=chunk,
                        location=Location(block_index=current_index, page=page_for_text, section="OCR_MEDIA"),
                    )
            next_seq += 1


def iter_document_ocr_blocks(
    path: Path,
    doc_id: str,
    start_index: int,
    page_range: tuple[int, int] | None = None,
    ocr_mode: str = "auto",
    stats: dict[str, Any] | None = None,
    max_workers: int | None = None,
) -> Iterator[Block]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        yield from iter_docx_ocr_blocks(
            path,
            doc_id=doc_id,
            start_index=start_index,
            ocr_mode=ocr_mode,
            stats=stats,
            max_workers=max_workers,
        )
        return
    if suffix == ".pdf":
        yield from iter_pdf_ocr_blocks(
            path,
            doc_id=doc_id,
            start_index=start_index,
            page_range=page_range,
            ocr_mode=ocr_mode,
            stats=stats,
            max_workers=max_workers,
        )
        return
