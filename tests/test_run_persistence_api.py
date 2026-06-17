import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deep_research_agent import server
from deep_research_agent.persistence.store import SQLiteResearchStore
from deep_research_agent.server import RunMessageRequest


def configure_test_server(tmp_path, monkeypatch):
    store = SQLiteResearchStore(tmp_path / "research.sqlite")
    monkeypatch.setattr(server, "research_store", store)
    monkeypatch.setattr(server, "deep_researcher", object())
    monkeypatch.setattr(server, "checkpoint_saver", None)
    server.active_runs.clear()
    monkeypatch.setattr(
        server,
        "_start_task",
        lambda record: server.active_runs.__setitem__(record.id, record),
    )
    return store


def test_completed_run_survives_memory_reset_and_is_listed(tmp_path, monkeypatch):
    store = configure_test_server(tmp_path, monkeypatch)
    snapshot = make_snapshot("completed-run", "completed")
    snapshot["final_report"] = "Finished report"
    store.save_run(snapshot, {})
    restored = asyncio.run(server.get_run("completed-run"))
    listed = asyncio.run(server.list_runs())

    assert restored["final_report"] == "Finished report"
    assert listed["runs"][0]["id"] == "completed-run"


def test_delete_run_removes_snapshot_and_report_versions(tmp_path, monkeypatch):
    store = configure_test_server(tmp_path, monkeypatch)
    snapshot = make_snapshot("completed-run", "completed")
    store.save_run(snapshot, {})
    store.add_report(
        run_id="completed-run",
        title="Research title",
        content="Finished report",
        created_at="2026-06-15T01:00:00+00:00",
    )

    response = asyncio.run(server.delete_run("completed-run"))

    assert response == {"id": "completed-run", "deleted": True}
    assert store.get_run("completed-run") is None
    assert store.list_reports("completed-run") == []


def test_completed_run_accepts_follow_up_message(tmp_path, monkeypatch):
    store = configure_test_server(tmp_path, monkeypatch)
    store.save_run(make_snapshot("completed-run", "completed"), {})
    response = asyncio.run(
        server.continue_run(
            "completed-run",
            RunMessageRequest(message="Research the latest changes", mode="research"),
        )
    )

    assert response["status"] == "queued"
    assert any(
        message["content"] == "Research the latest changes"
        for message in server.research_store.get_run("completed-run")["messages"]
    )


def test_interrupted_run_requires_manual_retry(tmp_path, monkeypatch):
    store = configure_test_server(tmp_path, monkeypatch)
    snapshot = make_snapshot("interrupted-run", "running")
    store.save_run(snapshot, {})
    store.mark_active_runs_interrupted()

    class FakeCheckpointSaver:
        async def aget_tuple(self, config):
            return {"checkpoint": True}

    monkeypatch.setattr(server, "checkpoint_saver", FakeCheckpointSaver())
    restored = asyncio.run(server.get_run("interrupted-run"))
    retried = asyncio.run(server.retry_run("interrupted-run"))

    assert restored["status"] == "interrupted"
    assert retried["status"] == "queued"
    assert server.active_runs["interrupted-run"].resume_from_checkpoint is True
    assert server.active_runs["interrupted-run"].checkpoint_thread_id == "interrupted-run"


def test_run_persists_report_and_injects_explicitly_enabled_history(tmp_path, monkeypatch):
    store = configure_test_server(tmp_path, monkeypatch)
    store.add_report(
        run_id="older-run",
        title="Battery research",
        content="Battery manufacturing capacity was constrained.",
        created_at="2026-06-14T00:00:00+00:00",
    )

    class CapturingGraph:
        def __init__(self):
            self.messages = []

        async def ainvoke(self, inputs, config):
            self.messages = inputs["messages"]
            return {
                "messages": [*inputs["messages"], AIMessage(content="New report")],
                "final_report": "New report",
            }

    graph = CapturingGraph()
    monkeypatch.setattr(server, "deep_researcher", graph)
    record = server.RunRecord(
        id="new-run",
        title="Battery follow-up",
        messages=[HumanMessage(content="Research battery manufacturing capacity")],
        config={"research_history_enabled": True},
    )

    asyncio.run(server._run_graph(record))

    assert isinstance(graph.messages[0], SystemMessage)
    assert "Battery research" in str(graph.messages[0].content)
    assert store.list_reports("new-run")[0]["content"] == "New report"
    assert all(message["type"] != "system" for message in store.get_run("new-run")["messages"])


def test_direct_follow_up_does_not_launch_deep_research(tmp_path, monkeypatch):
    configure_test_server(tmp_path, monkeypatch)

    class FailingGraph:
        async def ainvoke(self, inputs, config):
            raise AssertionError("deep research should not run")

    async def fake_direct_answer(record):
        record.messages.append(AIMessage(content="Direct answer"))
        record.status = "completed"

    monkeypatch.setattr(server, "deep_researcher", FailingGraph())
    monkeypatch.setattr(server, "_run_direct_answer", fake_direct_answer)
    record = server.RunRecord(
        id="follow-up",
        title="Follow-up",
        messages=[HumanMessage(content="Summarize the conclusion")],
        config={},
        execution_mode="answer",
        final_report="Existing report",
    )

    asyncio.run(server._run_graph(record))

    assert record.status == "completed"
    assert record.messages[-1].content == "Direct answer"


def make_snapshot(run_id: str, status: str) -> dict:
    return {
        "id": run_id,
        "title": "Research title",
        "status": status,
        "messages": [{"type": "human", "content": "Question", "name": None}],
        "events": [],
        "final_report": None,
        "research_brief": None,
        "error": None,
        "language": "en",
        "created_at": "2026-06-15T00:00:00+00:00",
        "updated_at": "2026-06-15T00:00:00+00:00",
    }
