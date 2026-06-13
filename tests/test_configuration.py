from deep_research_agent.configuration import Configuration


def test_private_paper_configuration_defaults():
    configuration = Configuration()

    assert configuration.private_papers_enabled is True
    assert configuration.private_papers_top_k == 5
    assert configuration.knowledge_base_path == ".knowledge/private_papers"
    assert configuration.embedding_model == "qwen3-vl-embedding"
    assert configuration.paper_chunk_size == 1200
    assert configuration.paper_chunk_overlap == 200
    assert configuration.max_paper_size_mb == 25


def test_private_paper_configuration_can_be_overridden():
    configuration = Configuration.from_runnable_config(
        {
            "configurable": {
                "private_papers_enabled": False,
                "private_papers_top_k": 3,
                "knowledge_base_path": "/tmp/test-private-papers",
            }
        }
    )

    assert configuration.private_papers_enabled is False
    assert configuration.private_papers_top_k == 3
    assert configuration.knowledge_base_path == "/tmp/test-private-papers"
