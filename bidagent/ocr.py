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
DEFAULT_TESSERACT_LANG = "chi_sim+eng"
DEFAULT_TESSERACT_CONFIG = "--oem 1 --psm 6"
OCR_BACKEND_CANDIDATES = ("paddle", "tesseract")
_TESSERACT_LANG_CACHE: tuple[str, str, frozenset[str]] | None = None


def _resolve_backend_preference(mode: str) -> str:
    if mode == "tesseract":
        return "tesseract"
    env_value = str(os.getenv("BIDAGENT_OCR_BACKEND") or "").strip().lower()
    if env_value in {"paddle", "tesseract"}:
        return env_value
    return "auto"


def _backend_order(mode: str) -> list[str]:
    preference = _resolve_backend_preference(mode)
    if preference == "paddle":
        return ["paddle", "tesseract"]
    if preference == "tesseract":
        return ["tesseract", "paddle"]
    # auto: prefer PaddleOCR-VL style backend when available, then fallback to tesseract.
    return ["paddle", "tesseract"]


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


def _resolve_tessdata_prefix() -> str | None:
    env_prefix = str(os.getenv("TESSDATA_PREFIX") or "").strip()
    if env_prefix and Path(env_prefix).exists():
        return str(Path(env_prefix).resolve())
    local_prefix = Path.cwd() / "tools" / "tessdata"
    if local_prefix.exists():
        return str(local_prefix.resolve())
    return None


def _tesseract_available_langs(cmd: str, tessdata_prefix: str | None = None) -> set[str]:
    global _TESSERACT_LANG_CACHE
    prefix_key = str(tessdata_prefix or "")
    cached = _TESSERACT_LANG_CACHE
    if cached and cached[0] == cmd and cached[1] == prefix_key:
        return set(cached[2])
    env = os.environ.copy()
    if tessdata_prefix:
        env["TESSDATA_PREFIX"] = str(tessdata_prefix)
    try:
        proc = subprocess.run(
            [cmd, "--list-langs"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            encoding="utf-8",
            errors="ignore",
            env=env,
        )
    except Exception:
        return set()
    payload = proc.stdout or proc.stderr or ""
    langs: set[str] = set()
    for line in payload.splitlines():
        text = str(line or "").strip()
        if not text:
            continue
        if text.lower().startswith("list of available languages"):
            continue
        langs.add(text)
    _TESSERACT_LANG_CACHE = (cmd, prefix_key, frozenset(langs))
    return langs


def _resolve_tesseract_lang(
    cmd: str,
    requested_lang: str,
    *,
    tessdata_prefix: str | None = None,
) -> tuple[str, dict[str, Any]]:
    available_langs = _tesseract_available_langs(cmd, tessdata_prefix=tessdata_prefix)
    requested_tokens = [token.strip() for token in str(requested_lang or "").split("+") if token.strip()]
    selected_tokens = [token for token in requested_tokens if token in available_langs] if available_langs else requested_tokens
    if not selected_tokens and available_langs:
        if "eng" in available_langs:
            selected_tokens = ["eng"]
        else:
            selected_tokens = [sorted(available_langs)[0]]
    effective_lang = "+".join(selected_tokens).strip() or str(requested_lang or "").strip() or "eng"
    meta = {
        "requested_lang": str(requested_lang or "").strip(),
        "effective_lang": effective_lang,
        "available_langs": sorted(available_langs),
        "lang_degraded": bool(selected_tokens) and "+".join(requested_tokens) != "+".join(selected_tokens),
        "tessdata_prefix": str(tessdata_prefix or ""),
    }
    return effective_lang, meta


def _tesseract_selfcheck() -> dict[str, Any]:
    payload: dict[str, Any] = {"engine": "tesseract"}

    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ModuleNotFoundError as exc:
        payload.update({"engine_available": False, "reason": f"missing python deps: {exc.name}"})
        return payload

    cmd = _resolve_tesseract_cmd()
    if not cmd:
        payload.update({"engine_available": False, "reason": "tesseract binary not found (install it or set TESSERACT_CMD)"})
        return payload

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
        payload.update({"engine_available": False, "reason": "tesseract binary detected but version check failed"})
        return payload

    requested = str(os.getenv("BIDAGENT_TESSERACT_LANG") or DEFAULT_TESSERACT_LANG).strip() or DEFAULT_TESSERACT_LANG
    tessdata_prefix = _resolve_tessdata_prefix()
    effective_lang, lang_meta = _resolve_tesseract_lang(cmd, requested, tessdata_prefix=tessdata_prefix)
    payload.update(
        {
            "engine_available": True,
            "tesseract_version": version,
            "requested_lang": requested,
            "effective_lang": effective_lang,
            "available_langs": lang_meta.get("available_langs"),
            "lang_degraded": bool(lang_meta.get("lang_degraded")),
            "tessdata_prefix": str(lang_meta.get("tessdata_prefix") or ""),
        }
    )
    return payload


def _paddle_selfcheck() -> dict[str, Any]:
    payload: dict[str, Any] = {"engine": "paddle"}
    try:
        import numpy as np
        from paddleocr import PaddleOCR, __version__ as paddleocr_version
    except ModuleNotFoundError as exc:
        payload.update({"engine_available": False, "reason": f"missing python deps: {exc.name}"})
        return payload

    try:
        ocr = PaddleOCR(lang="ch")
        probe = np.zeros((32, 128, 3), dtype="uint8")
        _ = list(ocr.predict(probe))
    except Exception as exc:  # noqa: BLE001
        payload.update({"engine_available": False, "paddleocr_version": paddleocr_version, "reason": f"runtime_error:{exc.__class__.__name__}"})
        return payload

    payload.update({"engine_available": True, "paddleocr_version": paddleocr_version})
    return payload


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

    reasons: list[str] = []
    for backend in _backend_order(mode):
        if backend == "paddle":
            check = _paddle_selfcheck()
        else:
            check = _tesseract_selfcheck()
        if bool(check.get("engine_available")):
            return {"mode": mode, **check}
        reason = str(check.get("reason") or "").strip()
        if reason:
            reasons.append(f"{backend}:{reason}")

    return {
        "mode": mode,
        "engine": _backend_order(mode)[0] if _backend_order(mode) else None,
        "engine_available": False,
        "reason": "; ".join(reasons) if reasons else "no OCR backend available",
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
    requested_lang = str(os.getenv("BIDAGENT_TESSERACT_LANG") or DEFAULT_TESSERACT_LANG).strip() or DEFAULT_TESSERACT_LANG
    tessdata_prefix = _resolve_tessdata_prefix()
    lang, lang_meta = _resolve_tesseract_lang(cmd, requested_lang, tessdata_prefix=tessdata_prefix)
    extra_config = str(os.getenv("BIDAGENT_TESSERACT_CONFIG") or DEFAULT_TESSERACT_CONFIG).strip()
    if tessdata_prefix:
        # Use project-local tessdata when available, so Chinese models can be versioned with the repo.
        # Prefer POSIX-style path to avoid Windows quoting edge cases in tesseract CLI parsing.
        tessdata_cli = str(Path(tessdata_prefix).as_posix())
        extra_config = f"{extra_config} --tessdata-dir {tessdata_cli}".strip()

    def _extract(image_bytes: bytes) -> str:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            return pytesseract.image_to_string(rgb, lang=lang, config=extra_config)

    setattr(
        _extract,
        "_meta",
        {
            "engine": "tesseract",
            "requested_lang": requested_lang,
            "effective_lang": lang,
            "available_langs": lang_meta.get("available_langs"),
            "lang_degraded": bool(lang_meta.get("lang_degraded")),
            "tessdata_prefix": str(lang_meta.get("tessdata_prefix") or ""),
        },
    )
    return _extract


def _load_paddle_engine() -> Callable[[bytes], str] | None:
    try:
        import numpy as np
        from PIL import Image
        from paddleocr import PaddleOCR
    except ModuleNotFoundError:
        return None

    try:
        ocr = PaddleOCR(lang="ch")
    except Exception:
        return None

    # Fail fast for environments where PaddleOCR initializes but cannot run inference.
    try:
        probe = np.zeros((32, 128, 3), dtype="uint8")
        _ = list(ocr.predict(probe))
    except Exception:
        return None

    def _extract(image_bytes: bytes) -> str:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            image_array = np.array(rgb)
        result = ocr.predict(image_array)
        lines: list[str] = []
        for page in result or []:
            if not page:
                continue
            if hasattr(page, "to_dict"):
                try:
                    page = page.to_dict()
                except Exception:  # noqa: BLE001
                    pass
            if isinstance(page, dict):
                rec_texts = page.get("rec_texts")
                if isinstance(rec_texts, list):
                    for text in rec_texts:
                        text = str(text or "").strip()
                        if text:
                            lines.append(text)
                continue
            # Backward-compatible structure for older PaddleOCR outputs.
            if isinstance(page, (list, tuple)):
                for item in page:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    rec = item[1]
                    if not isinstance(rec, (list, tuple)) or not rec:
                        continue
                    text = str(rec[0] or "").strip()
                    if text:
                        lines.append(text)
        return "\n".join(lines)

    return _extract


def _load_ocr_engines(mode: str) -> list[tuple[str, Callable[[bytes], str]]]:
    if mode == "off":
        return []
    if mode not in {"auto", "tesseract"}:
        return []
    engines: list[tuple[str, Callable[[bytes], str]]] = []
    for backend in _backend_order(mode):
        if backend == "paddle":
            engine = _load_paddle_engine()
        else:
            engine = _load_tesseract_engine()
        if engine is not None:
            engines.append((backend, engine))
    return engines


def load_ocr_engine(mode: str) -> Callable[[bytes], str] | None:
    engines = _load_ocr_engines(mode)
    return engines[0][1] if engines else None


def iter_docx_ocr_blocks(
    path: Path,
    doc_id: str,
    start_index: int,
    ocr_mode: str = "auto",
    stats: dict[str, Any] | None = None,
    max_workers: int | None = None,
) -> Iterator[Block]:
    engines = _load_ocr_engines(ocr_mode)
    if not engines:
        if isinstance(stats, dict):
            stats.setdefault("engine_available", False)
        return
    backend_names = [name for name, _ in engines]
    if isinstance(stats, dict):
        stats["engine_available"] = True
        stats["backend_order"] = backend_names
        stats.setdefault("backend_used_counts", {})
        stats.setdefault("backend_failures", {})
        stats.setdefault("backend_fallback_errors", {})
        stats.setdefault("sample_errors", [])
        backend_meta: dict[str, Any] = {}
        for name, engine in engines:
            meta = getattr(engine, "_meta", None)
            if isinstance(meta, dict):
                backend_meta[name] = meta
        if backend_meta:
            stats["backend_meta"] = backend_meta

    current_index = start_index
    if max_workers is None:
        # PaddleOCR object is not guaranteed thread-safe when shared.
        # Use single-thread by default when paddle backend is enabled.
        if "paddle" in backend_names:
            max_workers = 1
        else:
            cpu = os.cpu_count() or 2
            max_workers = max(1, min(4, cpu // 2))
    max_in_flight = max(2, int(max_workers) * 2)

    def _ocr_bytes(image_bytes: bytes) -> tuple[str, bool, str, list[str]]:
        errors: list[str] = []
        for backend_name, engine in engines:
            try:
                text = engine(image_bytes) or ""
                return text, True, backend_name, errors
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend_name}:{exc.__class__.__name__}")
        return "", False, "; ".join(errors), errors

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
                    text, ok, backend_or_error, fallback_errors = fut.result()
                except Exception:  # noqa: BLE001
                    if isinstance(stats, dict):
                        stats["images_failed"] = int(stats.get("images_failed", 0)) + 1
                    return iter(())
                if isinstance(stats, dict):
                    if ok:
                        stats["images_succeeded"] = int(stats.get("images_succeeded", 0)) + 1
                        used_counts = stats.setdefault("backend_used_counts", {})
                        used_counts[backend_or_error] = int(used_counts.get(backend_or_error, 0)) + 1
                        if fallback_errors:
                            fallback_counts = stats.setdefault("backend_fallback_errors", {})
                            for token in fallback_errors:
                                token = str(token or "").strip()
                                if not token:
                                    continue
                                fallback_counts[token] = int(fallback_counts.get(token, 0)) + 1
                    else:
                        stats["images_failed"] = int(stats.get("images_failed", 0)) + 1
                        failure_counts = stats.setdefault("backend_failures", {})
                        for token in fallback_errors or str(backend_or_error or "").split(";"):
                            token = str(token or "").strip()
                            if not token:
                                continue
                            failure_counts[token] = int(failure_counts.get(token, 0)) + 1
                        sample_errors = stats.setdefault("sample_errors", [])
                        if isinstance(sample_errors, list) and len(sample_errors) < 5:
                            sample_errors.append({"image": name, "error": backend_or_error})
                        return iter(())
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
                                block_type="ocr",
                                section_hint="OCR_MEDIA",
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
                            block_type="ocr",
                            section_hint="OCR_MEDIA",
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
    engines = _load_ocr_engines(ocr_mode)
    if not engines:
        if isinstance(stats, dict):
            stats.setdefault("engine_available", False)
        return
    backend_names = [name for name, _ in engines]
    if isinstance(stats, dict):
        stats["engine_available"] = True
        stats["backend_order"] = backend_names
        stats.setdefault("backend_used_counts", {})
        stats.setdefault("backend_failures", {})
        stats.setdefault("backend_fallback_errors", {})
        stats.setdefault("sample_errors", [])
        backend_meta: dict[str, Any] = {}
        for name, engine in engines:
            meta = getattr(engine, "_meta", None)
            if isinstance(meta, dict):
                backend_meta[name] = meta
        if backend_meta:
            stats["backend_meta"] = backend_meta

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        return

    if max_workers is None:
        if "paddle" in backend_names:
            max_workers = 1
        else:
            cpu = os.cpu_count() or 2
            max_workers = max(1, min(4, cpu // 2))
    max_in_flight = max(2, int(max_workers) * 2)

    def _ocr_bytes(image_bytes: bytes) -> tuple[str, bool, str, list[str]]:
        errors: list[str] = []
        for backend_name, engine in engines:
            try:
                text = engine(image_bytes) or ""
                return text, True, backend_name, errors
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend_name}:{exc.__class__.__name__}")
        return "", False, "; ".join(errors), errors

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
                text, ok, backend_or_error, fallback_errors = fut.result()
            except Exception:  # noqa: BLE001
                if isinstance(stats, dict):
                    stats["images_failed"] = int(stats.get("images_failed", 0)) + 1
                return iter(())
            if isinstance(stats, dict):
                if ok:
                    stats["images_succeeded"] = int(stats.get("images_succeeded", 0)) + 1
                    used_counts = stats.setdefault("backend_used_counts", {})
                    used_counts[backend_or_error] = int(used_counts.get(backend_or_error, 0)) + 1
                    if fallback_errors:
                        fallback_counts = stats.setdefault("backend_fallback_errors", {})
                        for token in fallback_errors:
                            token = str(token or "").strip()
                            if not token:
                                continue
                            fallback_counts[token] = int(fallback_counts.get(token, 0)) + 1
                else:
                    stats["images_failed"] = int(stats.get("images_failed", 0)) + 1
                    failure_counts = stats.setdefault("backend_failures", {})
                    for token in fallback_errors or str(backend_or_error or "").split(";"):
                        token = str(token or "").strip()
                        if not token:
                            continue
                        failure_counts[token] = int(failure_counts.get(token, 0)) + 1
                    sample_errors = stats.setdefault("sample_errors", [])
                    if isinstance(sample_errors, list) and len(sample_errors) < 5:
                        sample_errors.append({"page": page_no, "error": backend_or_error})
                    return iter(())
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
                                block_type="ocr",
                                section_hint="OCR_MEDIA",
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
                        block_type="ocr",
                        section_hint="OCR_MEDIA",
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
