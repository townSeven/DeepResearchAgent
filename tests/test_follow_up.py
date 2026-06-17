from deep_research_agent.server import _follow_up_needs_research


def test_auto_follow_up_routes_explanations_to_direct_answer():
    assert _follow_up_needs_research("请总结报告第三部分") is False
    assert _follow_up_needs_research("Explain the main conclusion") is False


def test_auto_follow_up_routes_fresh_or_expanded_scope_to_research():
    assert _follow_up_needs_research("请结合最新数据重新验证结论") is True
    assert _follow_up_needs_research("Research two additional competitors") is True
    assert _follow_up_needs_research("请扩展到两个新的竞争对手") is True
