"""Multi-format document text extraction, strict UTF-8 sanitization, and recursive chunking."""

import re
from pathlib import Path
from typing import Any, Callable, Optional

from pypdf import PdfReader

# Matches lone UTF-16 surrogates (U+D800..U+DFFF) and C0 control chars except \t, \n, \r
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str, strip: bool = True) -> str:
    """
    Sanitize extracted document text so it is guaranteed valid UTF-8 with no surrogates,
    null bytes, or broken control characters.
    """
    if not text:
        return ""

    cleaned = _SURROGATE_RE.sub("", text)
    cleaned = _CONTROL_CHAR_RE.sub(" ", cleaned)
    cleaned = cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() if strip else cleaned


def _paginate_plain_text(text: str, chars_per_page: int = 1800) -> tuple[int, list[dict[str, Any]]]:
    """Group long plain text into logical page-numbered sections for consistent citations."""
    clean = sanitize_text(text)
    if not clean:
        return 0, []

    paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]
    pages: list[dict[str, Any]] = []
    buffer: list[str] = []
    buffer_len = 0

    for para in paragraphs:
        if buffer and buffer_len + len(para) > chars_per_page:
            pages.append({"page": len(pages) + 1, "text": "\n\n".join(buffer)})
            buffer = [para]
            buffer_len = len(para)
        else:
            buffer.append(para)
            buffer_len += len(para) + 2

    if buffer:
        pages.append({"page": len(pages) + 1, "text": "\n\n".join(buffer)})

    return len(pages), pages


def extract_pdf_pages(
    file_path: str,
    ocr_callback: Optional[Callable[[str], list[dict[str, Any]]]] = None,
) -> tuple[int, list[dict[str, Any]]]:
    """
    Extract and sanitize text per page from a PDF file.
    Falls back to optional ocr_callback (e.g., multimodal Gemini extraction) for scanned PDFs.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypt_status = reader.decrypt("")
            if decrypt_status == 0:
                raise ValueError("PDF is password-protected and cannot be processed.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF is encrypted and could not be decrypted.") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("The uploaded PDF contains zero pages.")

    pages: list[dict[str, Any]] = []
    for idx, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        clean_text = sanitize_text(raw_text)
        if clean_text:
            pages.append({"page": idx + 1, "text": clean_text})

    if not pages and ocr_callback is not None:
        ocr_pages = ocr_callback(str(path))
        pages = [
            {"page": int(p.get("page", idx + 1)), "text": sanitize_text(p.get("text", ""))}
            for idx, p in enumerate(ocr_pages)
            if sanitize_text(p.get("text", ""))
        ]

    if not pages:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "If it is a scanned image PDF, configure GEMINI_API_KEY for multimodal OCR or run OCR first."
        )

    return total_pages, pages


def extract_docx_pages(file_path: str) -> tuple[int, list[dict[str, Any]]]:
    """Extract paragraphs and tables from a Word (.docx) file."""
    try:
        from docx import Document
    except ImportError as exc:
        raise ValueError("python-docx is not installed to read .docx files.") from exc

    try:
        doc = Document(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read DOCX file: {exc}") from exc

    parts: list[str] = []
    for para in doc.paragraphs:
        txt = sanitize_text(para.text)
        if txt:
            parts.append(txt)

    for table in doc.tables:
        for row in table.rows:
            cells = [sanitize_text(cell.text) for cell in row.cells if sanitize_text(cell.text)]
            if cells:
                parts.append(" | ".join(cells))

    total_pages, pages = _paginate_plain_text("\n\n".join(parts))
    if not pages:
        raise ValueError("The uploaded DOCX file contains no readable text.")
    return total_pages, pages


def extract_pptx_pages(file_path: str) -> tuple[int, list[dict[str, Any]]]:
    """Extract slide-by-slide text from a PowerPoint (.pptx) presentation."""
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise ValueError("python-pptx is not installed to read .pptx files.") from exc

    try:
        prs = Presentation(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read PPTX file: {exc}") from exc

    total_slides = len(prs.slides)
    if total_slides == 0:
        raise ValueError("The uploaded PPTX file contains zero slides.")

    pages: list[dict[str, Any]] = []
    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_lines: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                txt = sanitize_text(shape.text_frame.text)
                if txt:
                    slide_lines.append(txt)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [sanitize_text(c.text) for c in row.cells if sanitize_text(c.text)]
                    if cells:
                        slide_lines.append(" | ".join(cells))
        if getattr(slide, "has_notes_slide", False) and slide.notes_slide.notes_text_frame:
            notes = sanitize_text(slide.notes_slide.notes_text_frame.text)
            if notes:
                slide_lines.append(f"[Speaker Notes] {notes}")

        combined = sanitize_text("\n".join(slide_lines))
        if combined:
            pages.append({"page": slide_idx, "text": combined})

    if not pages:
        raise ValueError("No readable text was found in this PPTX presentation.")
    return total_slides, pages


def extract_text_pages(file_path: str) -> tuple[int, list[dict[str, Any]]]:
    """Extract text from .txt or .md files."""
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    total_pages, pages = _paginate_plain_text(raw)
    if not pages:
        raise ValueError("The uploaded text file is empty.")
    return total_pages, pages


def recursive_split_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""),
) -> list[str]:
    """
    Recursively split text into overlapping chunks without breaking words/sentences
    when natural boundaries are available.
    """
    clean = sanitize_text(text)
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    def _split_with_separators(segment: str, seps: tuple[str, ...]) -> list[str]:
        if len(segment) <= chunk_size:
            return [segment]

        if not seps:
            step = max(1, chunk_size - chunk_overlap)
            return [segment[i : i + chunk_size] for i in range(0, len(segment), step)]

        sep = seps[0]
        remaining_seps = seps[1:]

        if sep == "":
            step = max(1, chunk_size - chunk_overlap)
            return [segment[i : i + chunk_size] for i in range(0, len(segment), step)]

        if sep not in segment:
            return _split_with_separators(segment, remaining_seps)

        pieces = segment.split(sep)
        reconstructed: list[str] = []
        for idx, piece in enumerate(pieces):
            if idx < len(pieces) - 1 and sep in (". ", "? ", "! ", "; ", ", "):
                candidate = piece + sep.rstrip()
            else:
                candidate = piece
            if candidate.strip():
                reconstructed.append(candidate.strip())

        chunks: list[str] = []
        current = ""
        joiner = " " if sep in (". ", "? ", "! ", "; ", ", ", " ") else sep

        for part in reconstructed:
            if len(part) > chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(_split_with_separators(part, remaining_seps))
                continue

            proposed = f"{current}{joiner}{part}".strip() if current else part
            if len(proposed) <= chunk_size:
                current = proposed
            else:
                if current:
                    chunks.append(current.strip())
                if chunk_overlap > 0 and current:
                    tail = current[-chunk_overlap:].strip()
                    overlap_candidate = f"{tail}{joiner}{part}".strip()
                    current = overlap_candidate if len(overlap_candidate) <= chunk_size else part
                else:
                    current = part

        if current:
            chunks.append(current.strip())

        return chunks

    return [c for c in _split_with_separators(clean, separators) if c.strip()]


def process_document(
    file_path: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    ocr_callback: Optional[Callable[[str], list[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """
    Extract, sanitize, and chunk a PDF, DOCX, PPTX, TXT, or MD file into
    page/slide-attributed chunks.
    """
    ext = Path(file_path).suffix.lower().lstrip(".")
    if ext == "pdf":
        total_pages, pages = extract_pdf_pages(file_path, ocr_callback=ocr_callback)
    elif ext == "docx":
        total_pages, pages = extract_docx_pages(file_path)
    elif ext == "pptx":
        total_pages, pages = extract_pptx_pages(file_path)
    elif ext in {"txt", "md"}:
        total_pages, pages = extract_text_pages(file_path)
    else:
        raise ValueError(f"Unsupported file format: .{ext}")

    chunks: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    chunk_index = 0
    for page_item in pages:
        page_num = page_item["page"]
        page_text = page_item["text"]
        full_text_parts.append(f"[Page {page_num}]\n{page_text}")

        for piece in recursive_split_text(
            page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ):
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "page": page_num,
                    "text": piece,
                }
            )
            chunk_index += 1

    return {
        "file_type": ext,
        "page_count": total_pages,
        "chunks": chunks,
        "full_text": "\n\n".join(full_text_parts),
    }


def process_pdf(
    file_path: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    ocr_callback: Optional[Callable[[str], list[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for PDF processing."""
    return process_document(
        file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_callback=ocr_callback,
    )
