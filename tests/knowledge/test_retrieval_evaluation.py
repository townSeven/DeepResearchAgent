import pytest

from deep_research_agent.knowledge.evaluation import (
    GoldEvidence,
    RetrievalEvalCase,
    evaluate_retrieval,
    render_markdown_report,
)
from deep_research_agent.knowledge.models import PaperSearchResult


class FakeEmbeddingClient:
    async def embed_query(self, query):
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, retrieved_ids, existing_ids):
        self.retrieved_ids = retrieved_ids
        self.existing_ids = set(existing_ids)

    async def existing_chunk_ids(self, chunk_ids):
        return set(chunk_ids) & self.existing_ids

    async def search(self, query_embedding, top_k):
        return [
            PaperSearchResult(
                chunk_id=chunk_id,
                document_id="paper",
                file_name="paper.pdf",
                page_number=1,
                chunk_index=index,
                content=chunk_id,
                score=1.0 - index / 10,
            )
            for index, chunk_id in enumerate(self.retrieved_ids[:top_k])
        ]


def make_case() -> RetrievalEvalCase:
    return RetrievalEvalCase(
        question="Which chunks contain the method?",
        gold_evidence=[
            GoldEvidence(document_id="paper", page_number=1, chunk_id="gold-a"),
            GoldEvidence(document_id="paper", page_number=2, chunk_id="gold-b"),
        ],
        index_version="chunk-1200-overlap-200",
    )


@pytest.mark.asyncio
async def test_evaluate_retrieval_computes_recall_and_hit_rate_at_k():
    report = await evaluate_retrieval(
        cases=[make_case()],
        store=FakeStore(["noise", "gold-a", "gold-b"], {"gold-a", "gold-b"}),
        embedding_client=FakeEmbeddingClient(),
        ks=(1, 3, 5),
    )

    assert report.metrics["recall@1"] == 0.0
    assert report.metrics["recall@3"] == 1.0
    assert report.metrics["recall@5"] == 1.0
    assert report.metrics["hit_rate@1"] == 0.0
    assert report.metrics["hit_rate@3"] == 1.0


@pytest.mark.asyncio
async def test_evaluate_retrieval_rejects_missing_gold_chunks():
    with pytest.raises(ValueError, match="gold-b"):
        await evaluate_retrieval(
            cases=[make_case()],
            store=FakeStore(["gold-a"], {"gold-a"}),
            embedding_client=FakeEmbeddingClient(),
        )


@pytest.mark.asyncio
async def test_markdown_report_contains_aggregate_metrics():
    report = await evaluate_retrieval(
        cases=[make_case()],
        store=FakeStore(["gold-a", "noise"], {"gold-a", "gold-b"}),
        embedding_client=FakeEmbeddingClient(),
    )

    markdown = render_markdown_report(report)

    assert "| Recall@1 |" in markdown
    assert "| Hit Rate@5 |" in markdown
    assert "Which chunks contain the method?" in markdown
