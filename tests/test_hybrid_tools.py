import pytest
from langchain_core.tools import tool

from deep_research_agent.prompts import (
    compress_research_system_prompt,
    final_report_generation_prompt,
    research_system_prompt,
)
from deep_research_agent.utils import get_all_tools, get_private_paper_search_tool


@tool
def fake_private_search(query: str) -> str:
    """Search fake private papers."""
    return query


@pytest.mark.asyncio
async def test_get_all_tools_registers_private_search_when_enabled(monkeypatch):
    async def no_search_tools(search_api):
        return []

    async def no_mcp_tools(config, existing_tool_names, include_minimax_search):
        return []

    monkeypatch.setattr("deep_research_agent.utils.get_search_tool", no_search_tools)
    monkeypatch.setattr("deep_research_agent.utils.load_mcp_tools", no_mcp_tools)
    monkeypatch.setattr(
        "deep_research_agent.utils.get_private_paper_search_tool",
        lambda configurable: fake_private_search,
    )

    tools = await get_all_tools({"configurable": {"private_papers_enabled": True}})

    assert "fake_private_search" in [item.name for item in tools]


@pytest.mark.asyncio
async def test_get_all_tools_does_not_register_private_search_when_disabled(monkeypatch):
    async def no_search_tools(search_api):
        return []

    async def no_mcp_tools(config, existing_tool_names, include_minimax_search):
        return []

    monkeypatch.setattr("deep_research_agent.utils.get_search_tool", no_search_tools)
    monkeypatch.setattr("deep_research_agent.utils.load_mcp_tools", no_mcp_tools)
    monkeypatch.setattr(
        "deep_research_agent.utils.get_private_paper_search_tool",
        lambda configurable: fake_private_search,
    )

    tools = await get_all_tools({"configurable": {"private_papers_enabled": False}})

    assert "fake_private_search" not in [item.name for item in tools]


def test_private_search_factory_requires_a_provider_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)

    assert get_private_paper_search_tool(object()) is None


def test_research_prompts_preserve_private_paper_citations():
    assert "search_private_papers" in research_system_prompt
    assert "Private Paper" in compress_research_system_prompt
    assert "Private Paper" in final_report_generation_prompt
