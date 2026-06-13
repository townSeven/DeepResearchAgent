"""Persistent Chroma storage for private paper chunks."""

import asyncio
from functools import lru_cache
from pathlib import Path

import chromadb

from deep_research_agent.knowledge.models import (
    PaperChunk,
    PaperInfo,
    PaperSearchResult,
)


class PaperVectorStore:
    """Store and retrieve citable private paper chunks."""

    def __init__(self, path: str | Path, collection_name: str = "private_papers") -> None:
        """Configure the persistent Chroma collection."""
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_document(
        self,
        chunks: list[PaperChunk],
        embeddings: list[list[float]],
    ) -> bool:
        """Persist a document unless its stable document ID already exists."""
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same non-zero length")
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("All chunks must belong to the same document")
        document_id = chunks[0].document_id
        if await self.has_document(document_id):
            return False

        await asyncio.to_thread(
            self.collection.add,
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "source_type": chunk.source_type,
                }
                for chunk in chunks
            ],
        )
        return True

    async def has_document(self, document_id: str) -> bool:
        """Return whether any chunks for a document already exist."""
        result = await asyncio.to_thread(
            self.collection.get,
            where={"document_id": document_id},
            limit=1,
            include=[],
        )
        return bool(result["ids"])

    async def count(self) -> int:
        """Return the number of stored chunks."""
        return await asyncio.to_thread(self.collection.count)

    async def existing_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return the requested chunk IDs that exist in the collection."""
        if not chunk_ids:
            return set()
        response = await asyncio.to_thread(
            self.collection.get,
            ids=chunk_ids,
            include=[],
        )
        return set(response["ids"])

    async def list_papers(self) -> list[PaperInfo]:
        """Aggregate stored chunks into paper summaries."""
        response = await asyncio.to_thread(self.collection.get, include=["metadatas"])
        papers: dict[str, PaperInfo] = {}
        for metadata in response["metadatas"] or []:
            document_id = metadata["document_id"]
            if document_id not in papers:
                papers[document_id] = PaperInfo(
                    document_id=document_id,
                    file_name=metadata["file_name"],
                    chunk_count=0,
                )
            papers[document_id].chunk_count += 1
        return sorted(papers.values(), key=lambda paper: paper.file_name.lower())

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[PaperSearchResult]:
        """Return the most similar private paper chunks."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if await self.count() == 0:
            return []

        response = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        return [
            PaperSearchResult(
                chunk_id=chunk_id,
                document_id=metadata["document_id"],
                file_name=metadata["file_name"],
                page_number=metadata["page_number"],
                chunk_index=metadata["chunk_index"],
                content=document,
                source_type=metadata["source_type"],
                score=1.0 - distance,
            )
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]


@lru_cache(maxsize=8)
def get_paper_vector_store(path: str) -> PaperVectorStore:
    """Reuse one persistent Chroma client for each configured path."""
    return PaperVectorStore(path)
