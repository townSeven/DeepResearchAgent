import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_research_agent.configuration import Configuration
from deep_research_agent.deep_researcher import _get_clarification_private_context
from deep_research_agent.prompts import clarify_with_user_instructions


class FakePrivatePaperTool:
    def __init__(self):
        self.input = None

    async def ainvoke(self, input):
        self.input = input
        return "[Private Paper: TopT.pdf, Page 1, Chunk chunk-1]\nTopT evidence"


@pytest.mark.asyncio
async def test_clarification_searches_private_papers_using_human_messages(monkeypatch):
    tool = FakePrivatePaperTool()
    monkeypatch.setattr(
        "deep_research_agent.deep_researcher.get_private_paper_search_tool",
        lambda configurable: tool,
    )

    context = await _get_clarification_private_context(
        [
            HumanMessage(content="研究 TopT"),
            AIMessage(content="请提供论文标题"),
            HumanMessage(content="论文已经上传"),
        ],
        Configuration(private_papers_top_k=3),
    )

    assert "TopT evidence" in context
    assert tool.input["top_k"] == 3
    assert "研究 TopT" in tool.input["query"]
    assert "论文已经上传" in tool.input["query"]
    assert "请提供论文标题" not in tool.input["query"]


@pytest.mark.asyncio
async def test_clarification_skips_private_search_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "deep_research_agent.deep_researcher.get_private_paper_search_tool",
        lambda configurable: pytest.fail("private search factory should not be called"),
    )

    context = await _get_clarification_private_context(
        [HumanMessage(content="研究 TopT")],
        Configuration(private_papers_enabled=False),
    )

    assert context == "Private-paper search is disabled."


def test_clarification_prompt_grounds_private_search_claims():
    assert "{private_paper_context}" in clarify_with_user_instructions
    assert "Do not claim that you searched" in clarify_with_user_instructions
