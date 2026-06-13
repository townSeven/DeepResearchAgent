import pytest

from deep_research_agent.knowledge.models import PaperSearchResult
from deep_research_agent.knowledge.tools import create_private_paper_search_tool


class FakeEmbeddingClient:
    async def embed_query(self, query):
        if query == "fail":
            raise RuntimeError("provider unavailable")
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, results=None):
        self.results = results or []

    async def search(self, query_embedding, top_k):
        return self.results[:top_k]


@pytest.mark.asyncio
async def test_private_paper_search_tool_returns_citable_passages():
    result = PaperSearchResult(
        chunk_id="chunk-1",
        document_id="paper-1",
        file_name="private-paper.pdf",
        page_number=6,
        chunk_index=2,
        content="The proposed method aligns prompts with region embeddings.",
        score=0.91,
    )
    tool = create_private_paper_search_tool(
        store=FakeStore([result]),
        embedding_client=FakeEmbeddingClient(),
        default_top_k=5,
    )

    output = await tool.ainvoke({"query": "prompt region alignment", "top_k": 3})

    assert "[Private Paper: private-paper.pdf, Page 6, Chunk chunk-1]" in output
    assert "aligns prompts with region embeddings" in output
    assert tool.metadata["type"] == "search"
    assert tool.metadata["name"] == "private_paper_search"


@pytest.mark.asyncio
async def test_private_paper_search_tool_handles_empty_collection():
    tool = create_private_paper_search_tool(
        store=FakeStore(),
        embedding_client=FakeEmbeddingClient(),
    )

    output = await tool.ainvoke({"query": "missing evidence"})

    assert "No relevant private paper passages" in output


@pytest.mark.asyncio
async def test_private_paper_search_tool_degrades_on_failure():
    tool = create_private_paper_search_tool(
        store=FakeStore(),
        embedding_client=FakeEmbeddingClient(),
    )

    output = await tool.ainvoke({"query": "fail"})

    assert "Private paper search is unavailable" in output
    assert "provider unavailable" not in output


@pytest.mark.asyncio
async def test_private_paper_search_tool_reports_disabled_feature():
    tool = create_private_paper_search_tool(
        store=FakeStore(),
        embedding_client=FakeEmbeddingClient(),
        enabled=False,
    )

    output = await tool.ainvoke({"query": "private evidence"})

    assert "disabled" in output
