import json

import httpx
import pytest

from deep_research_agent.knowledge.embedding import EmbeddingError, QwenEmbeddingClient


@pytest.mark.asyncio
async def test_embed_documents_uses_default_dimension_and_restores_index_order():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload == {
            "model": "qwen3-vl-embedding",
            "input": {"contents": [{"text": "first"}, {"text": "second"}]},
            "parameters": {},
        }
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = QwenEmbeddingClient(api_key="test-key", http_client=http_client)

        embeddings = await client.embed_documents(["first", "second"])

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.asyncio
async def test_embed_query_returns_one_vector():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"output": {"embeddings": [{"index": 0, "embedding": [0.2, 0.8]}]}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = QwenEmbeddingClient(api_key="test-key", http_client=http_client)

        embedding = await client.embed_query("research question")

    assert embedding == [0.2, 0.8]


@pytest.mark.asyncio
async def test_embed_documents_batches_large_inputs():
    requested_batches = []

    async def handler(request: httpx.Request) -> httpx.Response:
        texts = [item["text"] for item in json.loads(request.content)["input"]["contents"]]
        requested_batches.append(texts)
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"index": index, "embedding": [float(index), 1.0]}
                        for index, _ in enumerate(texts)
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = QwenEmbeddingClient(
            api_key="test-key",
            http_client=http_client,
            batch_size=2,
        )

        embeddings = await client.embed_documents(["a", "b", "c"])

    assert requested_batches == [["a", "b"], ["c"]]
    assert len(embeddings) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_json",
    [
        {"output": {"embeddings": []}},
        {"output": {"embeddings": [{"index": 0, "embedding": []}]}},
        {
            "output": {
                "embeddings": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0]},
                ]
            }
        },
        {
            "output": {
                "embeddings": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 0, "embedding": [0.0, 1.0]},
                ]
            }
        },
    ],
)
async def test_embed_documents_rejects_malformed_responses(response_json):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_json)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = QwenEmbeddingClient(api_key="test-key", http_client=http_client)

        with pytest.raises(EmbeddingError):
            await client.embed_documents(["first", "second"])


@pytest.mark.asyncio
async def test_embedding_error_does_not_expose_api_key():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="provider failed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = QwenEmbeddingClient(api_key="secret-key", http_client=http_client)

        with pytest.raises(EmbeddingError) as error:
            await client.embed_query("question")

    assert "secret-key" not in str(error.value)
