from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator

from bidagent.document import split_text_blocks
from bidagent.models import Block, Location

OCR_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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

    # Tesseract is an external binary; best-effort detect availability.
    version = None
    try:
        import pytesseract

        version = str(pytesseract.get_tesseract_version())
    except Exception:
        try:
            proc = subprocess.run(
                ["tesseract", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = (proc.stdout or proc.stderr or "").splitlines()[0].strip() if (proc.stdout or proc.stderr) else None
        except Exception:
            version = None

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
) -> Iterator[Block]:
    engine = load_ocr_engine(ocr_mode)
    if engine is None:
        if isinstance(stats, dict):
            stats.setdefault("engine_available", False)
        return

    current_index = start_index
    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if not name.startswith("word/media/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in OCR_IMAGE_EXTENSIONS:
                continue
            if isinstance(stats, dict):
                stats["images_total"] = int(stats.get("images_total", 0)) + 1
            image_bytes = archive.read(name)
            try:
                text = engine(image_bytes) or ""
                if isinstance(stats, dict):
                    stats["images_succeeded"] = int(stats.get("images_succeeded", 0)) + 1
            except Exception:  # noqa: BLE001
                if isinstance(stats, dict):
                    stats["images_failed"] = int(stats.get("images_failed", 0)) + 1
                continue
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


def iter_pdf_ocr_blocks(
    path: Path,
    doc_id: str,
    start_index: int,
    page_range: tuple[int, int] | None = None,
    ocr_mode: str = "auto",
    stats: dict[str, Any] | None = None,
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

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    start_page, end_page = 1, total_pages
    if page_range:
        start_page, end_page = page_range
        end_page = min(end_page, total_pages)

    current_index = start_index
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
            try:
                text = engine(bytes(data)) or ""
                if isinstance(stats, dict):
                    stats["images_succeeded"] = int(stats.get("images_succeeded", 0)) + 1
            except Exception:  # noqa: BLE001
                if isinstance(stats, dict):
                    stats["images_failed"] = int(stats.get("images_failed", 0)) + 1
                continue
            for chunk in split_text_blocks(text):
                if isinstance(stats, dict):
                    stats["chars_total"] = int(stats.get("chars_total", 0)) + len(chunk)
                current_index += 1
                if isinstance(stats, dict):
                    stats["blocks_emitted"] = int(stats.get("blocks_emitted", 0)) + 1
                yield Block(
                    doc_id=doc_id,
                    text=chunk,
                    location=Location(block_index=current_index, page=page_no, section="OCR_MEDIA"),
                )


def iter_document_ocr_blocks(
    path: Path,
    doc_id: str,
    start_index: int,
    page_range: tuple[int, int] | None = None,
    ocr_mode: str = "auto",
    stats: dict[str, Any] | None = None,
) -> Iterator[Block]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        yield from iter_docx_ocr_blocks(
            path,
            doc_id=doc_id,
            start_index=start_index,
            ocr_mode=ocr_mode,
            stats=stats,
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
        )
        return
