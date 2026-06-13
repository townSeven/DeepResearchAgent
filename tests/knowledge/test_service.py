import fitz
import pytest

from deep_research_agent.knowledge.models import PaperInfo, PaperUpload
from deep_research_agent.knowledge.service import PaperKnowledgeService


def make_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class FakeEmbeddingClient:
    def __init__(self):
        self.requests = []

    async def embed_documents(self, texts):
        self.requests.append(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self):
        self.documents = {}

    async def has_document(self, document_id):
        return document_id in self.documents

    async def add_document(self, chunks, embeddings):
        document_id = chunks[0].document_id
        if document_id in self.documents:
            return False
        self.documents[document_id] = chunks
        return True

    async def list_papers(self):
        return [
            PaperInfo(
                document_id=document_id,
                file_name=chunks[0].file_name,
                chunk_count=len(chunks),
            )
            for document_id, chunks in self.documents.items()
        ]


@pytest.mark.asyncio
async def test_ingest_files_isolates_failures_and_reports_results():
    embedding_client = FakeEmbeddingClient()
    service = PaperKnowledgeService(
        store=FakeStore(),
        embedding_client=embedding_client,
        chunk_size=100,
        chunk_overlap=10,
        max_size_mb=1,
    )

    result = await service.ingest_files(
        [
            PaperUpload(
                file_name="valid.pdf",
                content_type="application/pdf",
                data=make_pdf("Private research evidence"),
            ),
            PaperUpload(
                file_name="notes.txt",
                content_type="text/plain",
                data=b"not a pdf",
            ),
        ]
    )

    assert [item.file_name for item in result.ingested] == ["valid.pdf"]
    assert result.ingested[0].chunk_count == 1
    assert result.skipped == []
    assert result.failed[0].file_name == "notes.txt"
    assert embedding_client.requests == [["Private research evidence"]]


@pytest.mark.asyncio
async def test_ingest_files_skips_duplicate_without_embedding_again():
    embedding_client = FakeEmbeddingClient()
    service = PaperKnowledgeService(
        store=FakeStore(),
        embedding_client=embedding_client,
        chunk_size=100,
        chunk_overlap=10,
        max_size_mb=1,
    )
    upload = PaperUpload(
        file_name="paper.pdf",
        content_type="application/pdf",
        data=make_pdf("Private research evidence"),
    )

    first = await service.ingest_files([upload])
    second = await service.ingest_files([upload])

    assert len(first.ingested) == 1
    assert len(second.skipped) == 1
    assert len(embedding_client.requests) == 1


@pytest.mark.asyncio
async def test_ingest_files_rejects_empty_and_oversized_files():
    service = PaperKnowledgeService(
        store=FakeStore(),
        embedding_client=FakeEmbeddingClient(),
        chunk_size=100,
        chunk_overlap=10,
        max_size_mb=1,
    )

    result = await service.ingest_files(
        [
            PaperUpload(file_name="empty.pdf", content_type="application/pdf", data=b""),
            PaperUpload(
                file_name="large.pdf",
                content_type="application/pdf",
                data=b"x" * (1024 * 1024 + 1),
            ),
        ]
    )

    assert result.ingested == []
    assert [item.file_name for item in result.failed] == ["empty.pdf", "large.pdf"]


@pytest.mark.asyncio
async def test_list_papers_returns_store_summary():
    store = FakeStore()
    service = PaperKnowledgeService(
        store=store,
        embedding_client=FakeEmbeddingClient(),
        chunk_size=100,
        chunk_overlap=10,
        max_size_mb=1,
    )
    await service.ingest_files(
        [
            PaperUpload(
                file_name="paper.pdf",
                content_type="application/pdf",
                data=make_pdf("Private research evidence"),
            )
        ]
    )

    papers = await service.list_papers()

    assert papers[0].file_name == "paper.pdf"
    assert papers[0].chunk_count == 1
