"""Orchestrate private paper validation, embedding, and persistence."""

from deep_research_agent.knowledge.embedding import QwenEmbeddingClient
from deep_research_agent.knowledge.models import (
    PaperInfo,
    PaperIngestionFailure,
    PaperIngestionSummary,
    PaperUpload,
)
from deep_research_agent.knowledge.parser import parse_pdf
from deep_research_agent.knowledge.store import PaperVectorStore


class PaperKnowledgeService:
    """Manage private paper ingestion and listing."""

    def __init__(
        self,
        store: PaperVectorStore,
        embedding_client: QwenEmbeddingClient,
        chunk_size: int,
        chunk_overlap: int,
        max_size_mb: int,
    ) -> None:
        """Configure the ingestion dependencies and limits."""
        self.store = store
        self.embedding_client = embedding_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_size_bytes = max_size_mb * 1024 * 1024

    async def ingest_files(self, files: list[PaperUpload]) -> PaperIngestionSummary:
        """Ingest valid PDFs while isolating per-file failures."""
        summary = PaperIngestionSummary()
        for upload in files:
            try:
                self._validate_upload(upload)
                chunks = parse_pdf(
                    upload.file_name,
                    upload.data,
                    self.chunk_size,
                    self.chunk_overlap,
                )
                paper = PaperInfo(
                    document_id=chunks[0].document_id,
                    file_name=upload.file_name,
                    chunk_count=len(chunks),
                )
                if await self.store.has_document(paper.document_id):
                    summary.skipped.append(paper)
                    continue
                embeddings = await self.embedding_client.embed_documents(
                    [chunk.content for chunk in chunks]
                )
                added = await self.store.add_document(chunks, embeddings)
                (summary.ingested if added else summary.skipped).append(paper)
            except Exception as exc:
                summary.failed.append(
                    PaperIngestionFailure(file_name=upload.file_name, error=str(exc))
                )
        return summary

    async def list_papers(self) -> list[PaperInfo]:
        """Return summaries for all stored private papers."""
        return await self.store.list_papers()

    def _validate_upload(self, upload: PaperUpload) -> None:
        if upload.content_type != "application/pdf" or not upload.file_name.lower().endswith(
            ".pdf"
        ):
            raise ValueError("Only PDF files are supported")
        if not upload.data:
            raise ValueError("PDF file is empty")
        if len(upload.data) > self.max_size_bytes:
            raise ValueError("PDF file exceeds the configured size limit")
