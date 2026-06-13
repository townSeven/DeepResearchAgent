"""Agent tools for searching private papers."""

from typing import Any

from langchain_core.tools import StructuredTool, tool


def create_private_paper_search_tool(
    store: Any,
    embedding_client: Any,
    default_top_k: int = 5,
    enabled: bool = True,
) -> StructuredTool:
    """Create a resilient private-paper search tool with injected dependencies."""

    @tool("search_private_papers")
    async def search_private_papers(query: str, top_k: int | None = None) -> str:
        """Search user-provided private papers for citable research evidence."""
        if not enabled:
            return "Private paper search is disabled."
        try:
            query_embedding = await embedding_client.embed_query(query)
            results = await store.search(query_embedding, top_k or default_top_k)
        except Exception:
            return "Private paper search is unavailable. Continue with other sources."
        if not results:
            return "No relevant private paper passages were found."
        return "\n\n".join(
            (
                f"[Private Paper: {result.file_name}, Page {result.page_number}, "
                f"Chunk {result.chunk_id}]\n{result.content}"
            )
            for result in results
        )

    search_private_papers.metadata = {
        **(search_private_papers.metadata or {}),
        "type": "search",
        "name": "private_paper_search",
    }
    return search_private_papers
