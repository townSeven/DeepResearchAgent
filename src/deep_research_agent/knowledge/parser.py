"""Parse private paper PDFs into deterministic, page-aware chunks."""

import hashlib

import fitz

from deep_research_agent.knowledge.models import PaperChunk


class PaperParseError(ValueError):
    """Raised when a PDF cannot produce searchable text chunks."""


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _split_page(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(text), step):
        content = text[start : start + chunk_size]
        if content.strip():
            chunks.append(content)
        if start + chunk_size >= len(text):
            break
    return chunks


def parse_pdf(
    file_name: str,
    pdf_bytes: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> list[PaperChunk]:
    """Extract deterministic chunks without crossing PDF page boundaries."""
    if not pdf_bytes:
        raise PaperParseError("PDF file is empty")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise PaperParseError("Chunk overlap must be smaller than chunk size")

    document_id = hashlib.sha256(pdf_bytes).hexdigest()
    chunks: list[PaperChunk] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                for content in _split_page(page_text, chunk_size, chunk_overlap):
                    chunk_index = len(chunks)
                    chunks.append(
                        PaperChunk(
                            chunk_id=_stable_id(
                                document_id,
                                str(page_number),
                                str(chunk_index),
                                content,
                            ),
                            document_id=document_id,
                            file_name=file_name,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            content=content,
                        )
                    )
    except PaperParseError:
        raise
    except Exception as exc:
        raise PaperParseError("Unable to read PDF") from exc

    if not chunks:
        raise PaperParseError("PDF contains no extractable text")
    return chunks
