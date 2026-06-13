from fastapi.testclient import TestClient

from deep_research_agent.knowledge.models import (
    PaperInfo,
    PaperIngestionFailure,
    PaperIngestionSummary,
)
from deep_research_agent.server import app, get_paper_knowledge_service


class FakePaperKnowledgeService:
    def __init__(self):
        self.uploads = []

    async def ingest_files(self, uploads):
        self.uploads = uploads
        return PaperIngestionSummary(
            ingested=[
                PaperInfo(
                    document_id="paper-id",
                    file_name=uploads[0].file_name,
                    chunk_count=3,
                )
            ],
            failed=[
                PaperIngestionFailure(file_name=uploads[1].file_name, error="invalid PDF")
            ],
        )

    async def list_papers(self):
        return [
            PaperInfo(document_id="paper-id", file_name="paper.pdf", chunk_count=3)
        ]


def test_upload_private_papers_returns_per_file_summary():
    service = FakePaperKnowledgeService()
    app.dependency_overrides[get_paper_knowledge_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post(
            "/api/knowledge/papers",
            files=[
                ("files", ("paper.pdf", b"pdf bytes", "application/pdf")),
                ("files", ("broken.pdf", b"broken", "application/pdf")),
            ],
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["ingested"][0]["file_name"] == "paper.pdf"
    assert response.json()["failed"][0]["file_name"] == "broken.pdf"
    assert [upload.file_name for upload in service.uploads] == ["paper.pdf", "broken.pdf"]


def test_list_private_papers_returns_current_collection():
    service = FakePaperKnowledgeService()
    app.dependency_overrides[get_paper_knowledge_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.get("/api/knowledge/papers")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "papers": [
            {"document_id": "paper-id", "file_name": "paper.pdf", "chunk_count": 3}
        ]
    }
