"""
Document indexing pipeline for Maintenance Copilot.

This module is import-safe so the Streamlit app can call the indexer after
a user uploads manuals, SOPs, photos, or Word documents. It extracts native
text first and uses Tesseract OCR for scanned pages, embedded images, and
uploaded page photos.
"""

from __future__ import annotations

import os
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

import faiss
import fitz
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps
from sklearn.feature_extraction.text import HashingVectorizer

try:
    import pytesseract
except Exception:  # pragma: no cover - handled at runtime
    pytesseract = None


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
INPUT_FOLDER = BASE_DIR / "data_input"
INDEX_PATH = BASE_DIR / "maintenance_index.faiss"
MAPPING_PATH = BASE_DIR / "chunks_mapping.pkl"
CACHE_TRACKER_PATH = BASE_DIR / "processed_files.pkl"

PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_DOCUMENT_EXTENSIONS = (
    PDF_EXTENSIONS | WORD_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS
)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
if os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() not in {"1", "true", "yes"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
DEFAULT_CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", "800"))
DEFAULT_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2"))
DEFAULT_OCR_MODE = os.getenv("OCR_MODE", "auto").lower()
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "80"))
OCR_DPI = int(os.getenv("OCR_DPI", "220"))
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_CONFIG = os.getenv("OCR_TESSERACT_CONFIG", "--oem 1 --psm 6")

ProgressCallback = Callable[[dict[str, Any]], None]

_embedding_model: Any | None = None
_embedding_backend_name = ""
_tesseract_checked = False
_tesseract_ready = False
_tesseract_error = ""


class HashingEmbeddingModel:
    """Dependency-light embedding fallback for environments with broken Torch vision deps."""

    def __init__(self, n_features: int = 384) -> None:
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        del show_progress_bar
        if isinstance(texts, str):
            texts = [texts]
        return self.vectorizer.transform(texts).toarray().astype("float32")


def _emit(callback: ProgressCallback | None, event: str, **payload: Any) -> None:
    message = {"event": event, **payload}
    if callback:
        callback(message)
    else:
        text = payload.get("message") or event
        print(text)


def get_embedding_model() -> Any:
    global _embedding_model, _embedding_backend_name
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            _embedding_backend_name = f"sentence-transformers:{EMBEDDING_MODEL_NAME}"
        except Exception as exc:
            print(
                "SentenceTransformer could not start. "
                f"Using local hashing embeddings instead. Details: {exc}"
            )
            _embedding_model = HashingEmbeddingModel(
                n_features=int(os.getenv("HASH_EMBEDDING_FEATURES", "384"))
            )
            _embedding_backend_name = "hashing-fallback"
    return _embedding_model


def get_embedding_backend_name() -> str:
    if not _embedding_backend_name:
        get_embedding_model()
    return _embedding_backend_name


def _configure_tesseract() -> None:
    if pytesseract is None:
        return

    configured_cmd = os.getenv("TESSERACT_CMD")
    if configured_cmd:
        pytesseract.pytesseract.tesseract_cmd = configured_cmd
        return

    windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if windows_default.exists():
        pytesseract.pytesseract.tesseract_cmd = str(windows_default)


def get_ocr_status() -> dict[str, Any]:
    """
    Returns OCR availability. `ready=False` usually means the Python package
    exists but the native Tesseract executable is not installed or not on PATH.
    """
    global _tesseract_checked, _tesseract_ready, _tesseract_error

    if _tesseract_checked:
        return {"ready": _tesseract_ready, "error": _tesseract_error}

    _tesseract_checked = True
    if pytesseract is None:
        _tesseract_error = "pytesseract is not installed."
        return {"ready": False, "error": _tesseract_error}

    try:
        _configure_tesseract()
        pytesseract.get_tesseract_version()
        _tesseract_ready = True
    except Exception as exc:  # pragma: no cover - depends on local install
        _tesseract_ready = False
        _tesseract_error = str(exc)

    return {"ready": _tesseract_ready, "error": _tesseract_error}


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_into_sentences(text: str) -> list[str]:
    """
    Split text into sentence-like units while preserving technical fragments.
    Falls back to paragraph fragments for OCR text that has weak punctuation.
    """
    text = normalize_text(text)
    if not text:
        return []

    sentence_parts = re.split(r"(?<=[.!?])\s+", text)
    if len(sentence_parts) <= 1:
        sentence_parts = re.split(r"\n+|(?<=;)\s+", text)

    cleaned = [part.strip(" -\n\t") for part in sentence_parts if part.strip()]
    return cleaned


def _chunk_long_sentence(sentence: str, target_chunk_len: int) -> list[str]:
    if len(sentence) <= target_chunk_len * 1.4:
        return [sentence]

    chunks = []
    start = 0
    while start < len(sentence):
        end = min(start + target_chunk_len, len(sentence))
        split_at = sentence.rfind(" ", start, end)
        if split_at <= start:
            split_at = end
        chunks.append(sentence[start:split_at].strip())
        start = split_at
    return [chunk for chunk in chunks if chunk]


def chunk_page_text(
    text: str,
    filename: str,
    page_num: int | str,
    next_chunk_id: int,
    extraction_method: str,
    target_chunk_len: int,
    overlap_sentences: int,
) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    sentences: list[str] = []
    for sentence in split_text_into_sentences(text):
        sentences.extend(_chunk_long_sentence(sentence, target_chunk_len))

    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence)

        if current_len >= target_chunk_len:
            chunks.append(
                {
                    "chunk_id": next_chunk_id,
                    "source_file": filename,
                    "page": page_num,
                    "text": " ".join(current),
                    "extraction": extraction_method,
                }
            )
            next_chunk_id += 1
            current = (
                current[-overlap_sentences:]
                if len(current) > overlap_sentences
                else []
            )
            current_len = sum(len(item) for item in current)

    if current:
        chunks.append(
            {
                "chunk_id": next_chunk_id,
                "source_file": filename,
                "page": page_num,
                "text": " ".join(current),
                "extraction": extraction_method,
            }
        )
        next_chunk_id += 1

    return chunks, next_chunk_id


def extract_native_page_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda block: (block[1], block[0]))
    page_text = "\n".join(
        block[4].strip()
        for block in blocks
        if len(block) > 4 and block[4].strip()
    )
    return normalize_text(page_text)


def _render_page_image(page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def ocr_image(image: Image.Image) -> tuple[str, str | None]:
    status = get_ocr_status()
    if not status["ready"]:
        return "", status["error"] or "Tesseract OCR is not available."

    try:
        prepared = _preprocess_for_ocr(image.convert("RGB"))
        text = pytesseract.image_to_string(
            prepared,
            lang=OCR_LANG,
            config=OCR_CONFIG,
        )
        return normalize_text(text), None
    except Exception as exc:  # pragma: no cover - depends on local OCR
        return "", str(exc)


def extract_ocr_page_text(page: fitz.Page) -> tuple[str, str | None]:
    return ocr_image(_render_page_image(page))


def _merge_native_and_ocr_text(native_text: str, ocr_text: str) -> str:
    if not native_text:
        return ocr_text
    if not ocr_text:
        return native_text
    if ocr_text in native_text or native_text in ocr_text:
        return native_text if len(native_text) >= len(ocr_text) else ocr_text
    return f"{native_text}\n\nOCR text from page image:\n{ocr_text}"


def process_single_pdf(
    full_path: str | Path,
    filename: str,
    next_chunk_id: int,
    target_chunk_len: int = DEFAULT_CHUNK_LENGTH,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    ocr_mode: str = DEFAULT_OCR_MODE,
    ocr_min_text_chars: int = OCR_MIN_TEXT_CHARS,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Extract native text and optional OCR text from one PDF.

    OCR modes:
    - off: native text only
    - auto: OCR pages with little or no native text
    - always: combine native text and OCR text for every page
    """
    full_path = Path(full_path)
    file_chunks: list[dict[str, Any]] = []
    stats = {
        "pages": 0,
        "native_pages": 0,
        "ocr_pages": 0,
        "chunks": 0,
        "warnings": [],
    }

    try:
        with fitz.open(full_path) as doc:
            stats["pages"] = len(doc)

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                native_text = extract_native_page_text(page)
                use_ocr = (
                    ocr_mode == "always"
                    or (
                        ocr_mode == "auto"
                        and len(native_text) < ocr_min_text_chars
                    )
                )
                ocr_text = ""
                extraction_method = "native"

                if native_text:
                    stats["native_pages"] += 1

                if use_ocr:
                    _emit(
                        progress_callback,
                        "ocr_page",
                        message=f"OCR page {page_num} in {filename}",
                        file=filename,
                        page=page_num,
                    )
                    ocr_text, ocr_error = extract_ocr_page_text(page)
                    if ocr_error:
                        stats["warnings"].append(
                            f"{filename} page {page_num}: OCR skipped ({ocr_error})"
                        )
                    if ocr_text:
                        stats["ocr_pages"] += 1
                        extraction_method = "ocr" if not native_text else "native+ocr"

                page_text = _merge_native_and_ocr_text(native_text, ocr_text)
                if not page_text:
                    continue

                page_chunks, next_chunk_id = chunk_page_text(
                    page_text,
                    filename,
                    page_num,
                    next_chunk_id,
                    extraction_method,
                    target_chunk_len,
                    overlap_sentences,
                )
                file_chunks.extend(page_chunks)

    except Exception as exc:
        stats["warnings"].append(f"Failed to parse {filename}: {exc}")

    stats["chunks"] = len(file_chunks)
    return file_chunks, stats


def _docx_paragraph_text(element: ET.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append("\n")
    return normalize_text("".join(parts))


def _extract_docx_native_text(docx_path: Path) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    with zipfile.ZipFile(docx_path) as archive:
        if "word/document.xml" not in archive.namelist():
            return sections
        root = ET.fromstring(archive.read("word/document.xml"))

    body = next(
        (node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "body"),
        root,
    )
    section_id = 1
    for child in list(body):
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = _docx_paragraph_text(child)
            if text:
                sections.append((f"section {section_id}", text))
                section_id += 1
        elif tag == "tbl":
            rows = []
            for row in child.iter():
                if row.tag.rsplit("}", 1)[-1] == "tr":
                    cells = []
                    for cell in row:
                        if cell.tag.rsplit("}", 1)[-1] == "tc":
                            cell_text = _docx_paragraph_text(cell)
                            if cell_text:
                                cells.append(cell_text)
                    if cells:
                        rows.append(" | ".join(cells))
            if rows:
                sections.append((f"table {section_id}", "\n".join(rows)))
                section_id += 1
    return sections


def _extract_docx_images(docx_path: Path) -> list[tuple[str, Image.Image]]:
    images: list[tuple[str, Image.Image]] = []
    with zipfile.ZipFile(docx_path) as archive:
        media_files = [
            name for name in archive.namelist()
            if name.startswith("word/media/")
            and Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        for index, name in enumerate(media_files, 1):
            try:
                with archive.open(name) as file:
                    image = Image.open(file)
                    image.load()
                images.append((f"embedded image {index}", image.copy()))
            except Exception:
                continue
    return images


def process_single_docx(
    full_path: str | Path,
    filename: str,
    next_chunk_id: int,
    target_chunk_len: int = DEFAULT_CHUNK_LENGTH,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    ocr_mode: str = DEFAULT_OCR_MODE,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_path = Path(full_path)
    file_chunks: list[dict[str, Any]] = []
    stats = {
        "pages": 0,
        "native_pages": 0,
        "ocr_pages": 0,
        "chunks": 0,
        "warnings": [],
    }

    try:
        sections = _extract_docx_native_text(full_path)
        stats["pages"] = len(sections)
        for section_label, text in sections:
            stats["native_pages"] += 1
            section_chunks, next_chunk_id = chunk_page_text(
                text,
                filename,
                section_label,
                next_chunk_id,
                "docx-native",
                target_chunk_len,
                overlap_sentences,
            )
            file_chunks.extend(section_chunks)

        if ocr_mode != "off":
            for image_label, image in _extract_docx_images(full_path):
                _emit(
                    progress_callback,
                    "ocr_page",
                    message=f"OCR {image_label} in {filename}",
                    file=filename,
                    page=image_label,
                )
                ocr_text, ocr_error = ocr_image(image)
                if ocr_error:
                    stats["warnings"].append(
                        f"{filename} {image_label}: OCR skipped ({ocr_error})"
                    )
                    continue
                if not ocr_text:
                    continue
                stats["ocr_pages"] += 1
                section_chunks, next_chunk_id = chunk_page_text(
                    ocr_text,
                    filename,
                    image_label,
                    next_chunk_id,
                    "docx-image-ocr",
                    target_chunk_len,
                    overlap_sentences,
                )
                file_chunks.extend(section_chunks)

    except Exception as exc:
        stats["warnings"].append(f"Failed to parse {filename}: {exc}")

    stats["chunks"] = len(file_chunks)
    return file_chunks, stats


def process_single_image(
    full_path: str | Path,
    filename: str,
    next_chunk_id: int,
    target_chunk_len: int = DEFAULT_CHUNK_LENGTH,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_path = Path(full_path)
    file_chunks: list[dict[str, Any]] = []
    stats = {
        "pages": 1,
        "native_pages": 0,
        "ocr_pages": 0,
        "chunks": 0,
        "warnings": [],
    }

    try:
        image = Image.open(full_path)
        image.load()
        _emit(
            progress_callback,
            "ocr_page",
            message=f"OCR uploaded image: {filename}",
            file=filename,
            page="image",
        )
        ocr_text, ocr_error = ocr_image(image)
        if ocr_error:
            stats["warnings"].append(f"{filename}: OCR skipped ({ocr_error})")
        elif ocr_text:
            stats["ocr_pages"] = 1
            file_chunks, next_chunk_id = chunk_page_text(
                ocr_text,
                filename,
                "image",
                next_chunk_id,
                "image-ocr",
                target_chunk_len,
                overlap_sentences,
            )
    except Exception as exc:
        stats["warnings"].append(f"Failed to parse image {filename}: {exc}")

    stats["chunks"] = len(file_chunks)
    return file_chunks, stats


def process_single_text_file(
    full_path: str | Path,
    filename: str,
    next_chunk_id: int,
    target_chunk_len: int = DEFAULT_CHUNK_LENGTH,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_path = Path(full_path)
    stats = {
        "pages": 1,
        "native_pages": 0,
        "ocr_pages": 0,
        "chunks": 0,
        "warnings": [],
    }
    try:
        text = full_path.read_text(encoding="utf-8", errors="ignore")
        stats["native_pages"] = 1 if text.strip() else 0
        chunks, _ = chunk_page_text(
            text,
            filename,
            "text",
            next_chunk_id,
            "text-native",
            target_chunk_len,
            overlap_sentences,
        )
    except Exception as exc:
        chunks = []
        stats["warnings"].append(f"Failed to parse text file {filename}: {exc}")

    stats["chunks"] = len(chunks)
    return chunks, stats


def process_single_document(
    full_path: str | Path,
    filename: str,
    next_chunk_id: int,
    target_chunk_len: int = DEFAULT_CHUNK_LENGTH,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    ocr_mode: str = DEFAULT_OCR_MODE,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_path = Path(full_path)
    suffix = full_path.suffix.lower()

    if suffix in PDF_EXTENSIONS:
        return process_single_pdf(
            full_path,
            filename,
            next_chunk_id,
            target_chunk_len=target_chunk_len,
            overlap_sentences=overlap_sentences,
            ocr_mode=ocr_mode,
            progress_callback=progress_callback,
        )
    if suffix in WORD_EXTENSIONS:
        return process_single_docx(
            full_path,
            filename,
            next_chunk_id,
            target_chunk_len=target_chunk_len,
            overlap_sentences=overlap_sentences,
            ocr_mode=ocr_mode,
            progress_callback=progress_callback,
        )
    if suffix in IMAGE_EXTENSIONS:
        return process_single_image(
            full_path,
            filename,
            next_chunk_id,
            target_chunk_len=target_chunk_len,
            overlap_sentences=overlap_sentences,
            progress_callback=progress_callback,
        )
    if suffix in TEXT_EXTENSIONS:
        return process_single_text_file(
            full_path,
            filename,
            next_chunk_id,
            target_chunk_len=target_chunk_len,
            overlap_sentences=overlap_sentences,
        )

    return [], {
        "pages": 0,
        "native_pages": 0,
        "ocr_pages": 0,
        "chunks": 0,
        "warnings": [f"Unsupported document type: {filename}"],
    }


def load_existing_database() -> tuple[Any | None, list[dict[str, Any]], dict[str, Any]]:
    existing_index = None
    existing_chunks: list[dict[str, Any]] = []
    processed_files: dict[str, Any] = {}

    if all(path.exists() for path in [INDEX_PATH, MAPPING_PATH, CACHE_TRACKER_PATH]):
        try:
            existing_index = faiss.read_index(str(INDEX_PATH))
            with MAPPING_PATH.open("rb") as file:
                existing_chunks = pickle.load(file)
            with CACHE_TRACKER_PATH.open("rb") as file:
                processed_files = pickle.load(file)
        except Exception as exc:
            print(f"Cache read error ({exc}). Rebuilding from scratch.")

    return existing_index, existing_chunks, processed_files


def file_signature(path: Path, ocr_mode: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "ocr_mode": ocr_mode,
        "ocr_lang": OCR_LANG,
        "ocr_min_text_chars": OCR_MIN_TEXT_CHARS,
        "suffix": path.suffix.lower(),
        "embedding_backend": get_embedding_backend_name(),
        "extractor_version": 3,
    }


def signature_matches(previous: Any, current: dict[str, Any]) -> bool:
    if isinstance(previous, dict):
        return previous == current
    if isinstance(previous, (int, float)):
        return previous == current.get("mtime")
    return False


def reset_chunk_ids(chunks: list[dict[str, Any]]) -> None:
    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen = set()
    deduped: list[dict[str, Any]] = []

    for chunk in chunks:
        key = (
            chunk.get("source_file"),
            chunk.get("page"),
            chunk.get("text"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    return deduped, len(chunks) - len(deduped)


def encode_chunks(chunks: list[dict[str, Any]]) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
    if not texts:
        return np.empty((0, 0), dtype="float32")

    backend_name = get_embedding_backend_name()
    for chunk in chunks:
        chunk["embedding_backend"] = backend_name

    embeddings = get_embedding_model().encode(texts, show_progress_bar=False)
    matrix = np.array(embeddings).astype("float32")
    faiss.normalize_L2(matrix)
    return matrix


def rebuild_faiss_index(chunks: list[dict[str, Any]]) -> Any | None:
    matrix = encode_chunks(chunks)
    if matrix.size == 0:
        return None

    rebuilt_index = faiss.IndexFlatIP(matrix.shape[1])
    rebuilt_index.add(matrix)
    return rebuilt_index


def save_database(index: Any, chunks_data: list[dict[str, Any]], processed_files: dict[str, Any]) -> None:
    if index is None:
        raise RuntimeError("Cannot save an empty FAISS index.")

    faiss.write_index(index, str(INDEX_PATH))
    with MAPPING_PATH.open("wb") as file:
        pickle.dump(chunks_data, file)
    with CACHE_TRACKER_PATH.open("wb") as file:
        pickle.dump(processed_files, file)


def index_documents(
    input_folder: str | Path = INPUT_FOLDER,
    ocr_mode: str | None = None,
    force_rebuild: bool = False,
    target_chunk_len: int = DEFAULT_CHUNK_LENGTH,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Build or update the vector database from supported documents in `input_folder`.
    Returns a summary dictionary suitable for Streamlit status messages.
    """
    input_folder = Path(input_folder)
    input_folder.mkdir(parents=True, exist_ok=True)

    selected_ocr_mode = (ocr_mode or DEFAULT_OCR_MODE).lower()
    if selected_ocr_mode not in {"auto", "always", "off"}:
        selected_ocr_mode = "auto"

    document_files = sorted(
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    )

    summary: dict[str, Any] = {
        "files_seen": len(document_files),
        "files_processed": 0,
        "files_skipped": 0,
        "chunks_added": 0,
        "chunks_total": 0,
        "vectors_total": 0,
        "ocr_pages": 0,
        "document_types": {},
        "warnings": [],
    }

    if not document_files:
        _emit(
            progress_callback,
            "done",
            message="No supported documents found. Upload PDF, DOCX, TXT, or image files.",
        )
        return summary

    if force_rebuild:
        index = None
        chunks_data: list[dict[str, Any]] = []
        processed_files: dict[str, Any] = {}
        rebuild_required = True
    else:
        index, chunks_data, processed_files = load_existing_database()
        chunks_data, duplicate_count = deduplicate_chunks(chunks_data)
        rebuild_required = duplicate_count > 0

        if duplicate_count:
            summary["warnings"].append(f"Removed {duplicate_count} duplicate chunks.")

        if index is not None and index.ntotal != len(chunks_data):
            rebuild_required = True
            summary["warnings"].append("Index and metadata counts differed; rebuilding.")

    current_files = {path.name for path in document_files}
    stale_files = set(processed_files) - current_files
    if stale_files:
        chunks_data = [
            chunk for chunk in chunks_data
            if chunk.get("source_file") not in stale_files
        ]
        for filename in stale_files:
            processed_files.pop(filename, None)
        rebuild_required = True

    new_chunks: list[dict[str, Any]] = []

    for document_path in document_files:
        suffix = document_path.suffix.lower()
        summary["document_types"][suffix] = summary["document_types"].get(suffix, 0) + 1
        signature = file_signature(document_path, selected_ocr_mode)
        previous_signature = processed_files.get(document_path.name)

        if not force_rebuild and signature_matches(previous_signature, signature):
            summary["files_skipped"] += 1
            _emit(
                progress_callback,
                "skip_file",
                message=f"Skipping unchanged file: {document_path.name}",
                file=document_path.name,
            )
            continue

        _emit(
            progress_callback,
            "process_file",
            message=f"Processing document: {document_path.name}",
            file=document_path.name,
        )

        old_count = len(chunks_data)
        chunks_data = [
            chunk for chunk in chunks_data
            if chunk.get("source_file") != document_path.name
        ]
        if len(chunks_data) != old_count:
            rebuild_required = True

        file_chunks, file_stats = process_single_document(
            document_path,
            document_path.name,
            len(chunks_data) + len(new_chunks),
            target_chunk_len=target_chunk_len,
            overlap_sentences=overlap_sentences,
            ocr_mode=selected_ocr_mode,
            progress_callback=progress_callback,
        )

        summary["files_processed"] += 1
        summary["chunks_added"] += len(file_chunks)
        summary["ocr_pages"] += int(file_stats.get("ocr_pages", 0))
        summary["warnings"].extend(file_stats.get("warnings", []))

        if file_chunks:
            new_chunks.extend(file_chunks)
            processed_files[document_path.name] = signature

    if rebuild_required:
        chunks_data.extend(new_chunks)
        reset_chunk_ids(chunks_data)
        _emit(
            progress_callback,
            "embed",
            message=f"Rebuilding vector index for {len(chunks_data)} chunks.",
        )
        index = rebuild_faiss_index(chunks_data)
        if index is not None:
            save_database(index, chunks_data, processed_files)
    elif new_chunks:
        _emit(
            progress_callback,
            "embed",
            message=f"Embedding {len(new_chunks)} new chunks.",
        )
        matrix = encode_chunks(new_chunks)
        if matrix.size:
            if index is None:
                index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            chunks_data.extend(new_chunks)
            save_database(index, chunks_data, processed_files)

    summary["chunks_total"] = len(chunks_data)
    summary["vectors_total"] = index.ntotal if index is not None else 0

    _emit(
        progress_callback,
        "done",
        message=(
            f"Index ready: {summary['chunks_total']} chunks, "
            f"{summary['vectors_total']} vectors."
        ),
    )
    return summary


def index_pdfs(
    input_folder: str | Path = INPUT_FOLDER,
    ocr_mode: str | None = None,
    force_rebuild: bool = False,
    target_chunk_len: int = DEFAULT_CHUNK_LENGTH,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Backwards-compatible wrapper. The indexer now supports PDFs, DOCX,
    uploaded images/photos, TXT, and Markdown.
    """
    return index_documents(
        input_folder=input_folder,
        ocr_mode=ocr_mode,
        force_rebuild=force_rebuild,
        target_chunk_len=target_chunk_len,
        overlap_sentences=overlap_sentences,
        progress_callback=progress_callback,
    )


def _console_progress(event: dict[str, Any]) -> None:
    message = event.get("message")
    if message:
        print(message)


def main() -> None:
    summary = index_documents(progress_callback=_console_progress)
    print("\nSummary")
    for key, value in summary.items():
        if key == "warnings" and not value:
            continue
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
