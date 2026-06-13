"""Data contracts for private paper knowledge."""

from pydantic import BaseModel


class PaperChunk(BaseModel):
    """A citable text chunk extracted from a private paper."""

    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    chunk_index: int
    content: str
    source_type: str = "private_paper"


class PaperSearchResult(PaperChunk):
    """A private paper chunk returned by vector search."""

    score: float
