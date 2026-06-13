"""Alibaba Model Studio embedding adapter for private paper retrieval."""

from typing import Any

import httpx

DEFAULT_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated safely."""


class QwenEmbeddingClient:
    """Generate text embeddings with the Qwen3-VL Embedding API."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-vl-embedding",
        base_url: str = DEFAULT_EMBEDDING_URL,
        http_client: httpx.AsyncClient | None = None,
        batch_size: int = 10,
    ) -> None:
        """Configure the provider credentials and optional HTTP transport."""
        if not api_key:
            raise EmbeddingError("DASHSCOPE_API_KEY is required")
        if batch_size < 1:
            raise EmbeddingError("Embedding batch size must be at least 1")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.http_client = http_client
        self.batch_size = batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Embedding inputs must not be empty")

        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(await self._embed_batch(batch))
        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise EmbeddingError("Qwen embedding batches have inconsistent dimensions")
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed one retrieval query."""
        embeddings = await self.embed_documents([query])
        return embeddings[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": {"contents": [{"text": text} for text in texts]},
            "parameters": {},
        }
        response_json = await self._post(payload)
        return self._parse_embeddings(response_json, len(texts))

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            if self.http_client is not None:
                response = await self.http_client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError("Qwen embedding request failed") from exc

    @staticmethod
    def _parse_embeddings(
        response_json: dict[str, Any],
        expected_count: int,
    ) -> list[list[float]]:
        try:
            items = response_json["output"]["embeddings"]
            ordered = sorted(items, key=lambda item: item["index"])
            indexes = [item["index"] for item in ordered]
            embeddings = [item["embedding"] for item in ordered]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("Qwen embedding response is malformed") from exc

        if indexes != list(range(expected_count)):
            raise EmbeddingError("Qwen embedding response indexes do not match input")
        if len(embeddings) != expected_count or not embeddings:
            raise EmbeddingError("Qwen embedding response count does not match input")
        dimensions = {len(embedding) for embedding in embeddings}
        if 0 in dimensions or len(dimensions) != 1:
            raise EmbeddingError("Qwen embedding vectors have inconsistent dimensions")
        return embeddings
