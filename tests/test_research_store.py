from deep_research_agent.persistence.store import SQLiteResearchStore


def make_snapshot(run_id: str, status: str, updated_at: str) -> dict:
    return {
        "id": run_id,
        "title": f"Research {run_id}",
        "status": status,
        "messages": [{"type": "human", "content": "Question", "name": None}],
        "events": [],
        "final_report": None,
        "research_brief": None,
        "error": None,
        "language": "en",
        "created_at": "2026-06-15T00:00:00+00:00",
        "updated_at": updated_at,
    }


def test_store_persists_runs_across_instances(tmp_path):
    path = tmp_path / "research.sqlite"
    store = SQLiteResearchStore(path)
    store.save_run(
        make_snapshot("run-1", "completed", "2026-06-15T01:00:00+00:00"),
        {"research_model": "test:model"},
    )

    reopened = SQLiteResearchStore(path)
    stored = reopened.get_run("run-1")

    assert stored is not None
    assert stored["title"] == "Research run-1"
    assert stored["messages"][0]["content"] == "Question"
    assert stored["config"] == {"research_model": "test:model"}


def test_store_filters_credentials_but_keeps_non_secret_token_limits(tmp_path):
    store = SQLiteResearchStore(tmp_path / "research.sqlite")
    store.save_run(
        make_snapshot("run-1", "queued", "2026-06-15T01:00:00+00:00"),
        {
            "api_key": "secret",
            "provider_api_key": "secret",
            "research_model_max_tokens": 10000,
            "mcp_config": {"env": {"ACCESS_TOKEN": "secret", "PUBLIC_SETTING": "kept"}},
        },
    )

    assert store.get_run("run-1")["config"] == {
        "research_model_max_tokens": 10000,
        "mcp_config": {"env": {"PUBLIC_SETTING": "kept"}},
    }


def test_store_lists_most_recent_runs_first(tmp_path):
    store = SQLiteResearchStore(tmp_path / "research.sqlite")
    store.save_run(make_snapshot("older", "completed", "2026-06-15T01:00:00+00:00"), {})
    store.save_run(make_snapshot("newer", "failed", "2026-06-15T02:00:00+00:00"), {})

    runs = store.list_runs()

    assert [run["id"] for run in runs] == ["newer", "older"]
    assert "messages" not in runs[0]
    assert "events" not in runs[0]


def test_store_marks_orphaned_active_runs_interrupted(tmp_path):
    store = SQLiteResearchStore(tmp_path / "research.sqlite")
    store.save_run(make_snapshot("running", "running", "2026-06-15T01:00:00+00:00"), {})
    store.save_run(make_snapshot("queued", "queued", "2026-06-15T01:00:00+00:00"), {})
    store.save_run(make_snapshot("done", "completed", "2026-06-15T01:00:00+00:00"), {})

    interrupted = store.mark_active_runs_interrupted()

    assert set(interrupted) == {"running", "queued"}
    assert store.get_run("running")["status"] == "interrupted"
    assert store.get_run("queued")["status"] == "interrupted"
    assert store.get_run("done")["status"] == "completed"


def test_store_versions_and_searches_final_reports(tmp_path):
    store = SQLiteResearchStore(tmp_path / "research.sqlite")
    store.add_report(
        run_id="run-1",
        title="Battery market",
        content="Solid-state battery costs and manufacturing capacity.",
        created_at="2026-06-15T01:00:00+00:00",
    )
    store.add_report(
        run_id="run-1",
        title="Battery market",
        content="Updated solid-state battery supplier comparison.",
        created_at="2026-06-15T02:00:00+00:00",
    )
    store.add_report(
        run_id="run-2",
        title="Unrelated",
        content="Cloud software pricing.",
        created_at="2026-06-15T03:00:00+00:00",
    )

    versions = store.list_reports("run-1")
    results = store.search_reports("solid-state battery manufacturing", exclude_run_id="new")

    assert [report["version"] for report in versions] == [2, 1]
    assert results[0]["run_id"] == "run-1"
    assert "manufacturing capacity" in results[0]["content"]


def test_store_deletes_run_and_reports(tmp_path):
    store = SQLiteResearchStore(tmp_path / "research.sqlite")
    store.save_run(make_snapshot("run-1", "completed", "2026-06-15T01:00:00+00:00"), {})
    store.add_report(
        run_id="run-1",
        title="Battery market",
        content="Report",
        created_at="2026-06-15T02:00:00+00:00",
    )

    assert store.delete_run("run-1") is True

    assert store.get_run("run-1") is None
    assert store.list_reports("run-1") == []
    assert store.delete_run("run-1") is False
