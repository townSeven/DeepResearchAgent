import pytest

from deep_research_agent.knowledge.models import PaperChunk
from deep_research_agent.knowledge.store import PaperVectorStore


def make_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    page_number: int = 1,
) -> PaperChunk:
    return PaperChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        file_name=f"{document_id}.pdf",
        page_number=page_number,
        chunk_index=page_number - 1,
        content=content,
    )


@pytest.mark.asyncio
async def test_store_persists_chunks_and_searches_in_similarity_order(tmp_path):
    chunks = [
        make_chunk("chunk-a", "paper-a", "alpha method"),
        make_chunk("chunk-b", "paper-a", "beta method", page_number=2),
    ]
    store = PaperVectorStore(tmp_path)

    added = await store.add_document(chunks, [[1.0, 0.0], [0.0, 1.0]])

    assert added is True
    reopened_store = PaperVectorStore(tmp_path)
    results = await reopened_store.search([1.0, 0.0], top_k=2)
    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-b"]
    assert results[0].file_name == "paper-a.pdf"
    assert results[0].content == "alpha method"
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_store_skips_an_existing_document(tmp_path):
    chunk = make_chunk("chunk-a", "paper-a", "alpha method")
    store = PaperVectorStore(tmp_path)

    assert await store.add_document([chunk], [[1.0, 0.0]]) is True
    assert await store.add_document([chunk], [[1.0, 0.0]]) is False
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_store_returns_empty_results_for_empty_collection(tmp_path):
    store = PaperVectorStore(tmp_path)

    assert await store.search([1.0, 0.0], top_k=5) == []


@pytest.mark.asyncio
async def test_store_validates_document_and_query_inputs(tmp_path):
    store = PaperVectorStore(tmp_path)
    chunk = make_chunk("chunk-a", "paper-a", "alpha method")

    with pytest.raises(ValueError):
        await store.add_document([], [])
    with pytest.raises(ValueError):
        await store.add_document([chunk], [])
    with pytest.raises(ValueError):
        await store.search([1.0, 0.0], top_k=0)
