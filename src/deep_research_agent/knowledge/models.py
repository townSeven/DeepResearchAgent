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


class PaperUpload(BaseModel):
    """A private paper file submitted for ingestion."""

    file_name: str
    content_type: str
    data: bytes


class PaperInfo(BaseModel):
    """A stored private paper summary."""

    document_id: str
    file_name: str
    chunk_count: int


class PaperIngestionFailure(BaseModel):
    """A private paper that could not be ingested."""

    file_name: str
    error: str


class PaperIngestionSummary(BaseModel):
    """Per-file results from one ingestion batch."""

    ingested: list[PaperInfo] = []
    skipped: list[PaperInfo] = []
    failed: list[PaperIngestionFailure] = []
