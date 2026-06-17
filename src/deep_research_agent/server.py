"""FastAPI service for the interactive Deep Research Agent UI."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import aiosqlite
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from deep_research_agent.configuration import Configuration
from deep_research_agent.deep_researcher import (
    configurable_model,
    deep_researcher_builder,
)
from deep_research_agent.events import (
    detect_language,
    emit_ui_progress,
    reset_event_sink,
    reset_ui_language,
    set_event_sink,
    set_ui_language,
)
from deep_research_agent.knowledge.embedding import QwenEmbeddingClient
from deep_research_agent.knowledge.models import PaperUpload
from deep_research_agent.knowledge.service import PaperKnowledgeService
from deep_research_agent.knowledge.store import PaperVectorStore, get_paper_vector_store
from deep_research_agent.persistence.store import SQLiteResearchStore
from deep_research_agent.utils import (
    get_api_key_for_model,
    get_base_url_for_model,
    get_extra_body_for_model,
    get_model_name_for_init,
)

load_dotenv()

RunStatus = Literal[
    "queued",
    "running",
    "requires_clarification",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
]


class RunCreateRequest(BaseModel):
    """Request body for creating a new research run."""

    message: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class RunMessageRequest(BaseModel):
    """Request body for continuing an existing research thread."""

    message: str = Field(min_length=1)
    mode: Literal["auto", "answer", "research"] = "auto"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    return {
        "type": message.type,
        "content": message.content,
        "name": getattr(message, "name", None),
    }


def _message_from_dict(message: dict[str, Any]) -> BaseMessage:
    """Restore the user-visible message types persisted by the API."""
    if message["type"] == "human":
        return HumanMessage(content=message["content"], name=message.get("name"))
    return AIMessage(content=message["content"], name=message.get("name"))


@dataclass
class RunRecord:
    """Persistent research-thread state plus optional live execution handles."""

    id: str
    title: str
    messages: list[BaseMessage]
    config: dict[str, Any]
    status: RunStatus = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    final_report: str | None = None
    research_brief: str | None = None
    last_follow_up_answer: str | None = None
    error: str | None = None
    language: str = "en"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    execution_mode: Literal["research", "answer"] = "research"
    checkpoint_thread_id: str = field(default_factory=lambda: str(uuid4()))
    resume_from_checkpoint: bool = False
    deleted: bool = False

    async def publish(self, event: dict[str, Any]) -> None:
        """Persist and publish one progress event."""
        self.updated_at = _utc_now()
        self.events.append(event)
        _persist_run(self)
        await self.queue.put(event)

    def snapshot(self) -> dict[str, Any]:
        """Return the user-visible persistent representation."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "messages": [_message_to_dict(message) for message in self.messages],
            "events": self.events,
            "final_report": self.final_report,
            "research_brief": self.research_brief,
            "last_follow_up_answer": self.last_follow_up_answer,
            "error": self.error,
            "language": self.language,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_mode": self.execution_mode,
            "checkpoint_thread_id": self.checkpoint_thread_id,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> RunRecord:
        """Restore a persisted research thread without an active task."""
        return cls(
            id=snapshot["id"],
            title=snapshot.get("title") or "Untitled research",
            messages=[_message_from_dict(message) for message in snapshot["messages"]],
            config=snapshot.get("config", {}),
            status=snapshot["status"],
            events=snapshot.get("events", []),
            final_report=snapshot.get("final_report"),
            research_brief=snapshot.get("research_brief"),
            last_follow_up_answer=snapshot.get("last_follow_up_answer"),
            error=snapshot.get("error"),
            language=snapshot.get("language", "en"),
            created_at=snapshot["created_at"],
            updated_at=snapshot["updated_at"],
            execution_mode=snapshot.get("execution_mode", "research"),
            checkpoint_thread_id=snapshot.get("checkpoint_thread_id", snapshot["id"]),
        )


configuration = Configuration.from_runnable_config()
research_store = SQLiteResearchStore(configuration.research_database_path)
research_store.mark_active_runs_interrupted()
checkpoint_path = Path(configuration.checkpoint_database_path)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
checkpoint_connection: aiosqlite.Connection | None = None
checkpoint_saver: AsyncSqliteSaver | None = None
deep_researcher = None
graph_init_lock = asyncio.Lock()
active_runs: dict[str, RunRecord] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Close the asynchronous checkpoint connection during shutdown."""
    yield
    if checkpoint_connection is not None:
        await checkpoint_connection.close()


app = FastAPI(title="Deep Research Agent Interactive API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_configured_paper_vector_store() -> PaperVectorStore:
    """Return the configured local private-paper vector store."""
    configuration = Configuration.from_runnable_config()
    return get_paper_vector_store(configuration.knowledge_base_path)


def get_paper_knowledge_service(
    store: PaperVectorStore = Depends(get_configured_paper_vector_store),
) -> PaperKnowledgeService:
    """Build the local private-paper knowledge service."""
    configuration = Configuration.from_runnable_config()
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="DASHSCOPE_API_KEY is not configured")
    return PaperKnowledgeService(
        store=store,
        embedding_client=QwenEmbeddingClient(
            api_key=api_key,
            model=configuration.embedding_model,
        ),
        chunk_size=configuration.paper_chunk_size,
        chunk_overlap=configuration.paper_chunk_overlap,
        max_size_mb=configuration.max_paper_size_mb,
    )


@app.post("/api/knowledge/papers")
async def upload_private_papers(
    files: list[UploadFile] = File(...),
    service: PaperKnowledgeService = Depends(get_paper_knowledge_service),
) -> dict[str, Any]:
    """Upload and ingest private paper PDFs."""
    uploads = [
        PaperUpload(
            file_name=file.filename or "unnamed.pdf",
            content_type=file.content_type or "",
            data=await file.read(),
        )
        for file in files
    ]
    return (await service.ingest_files(uploads)).model_dump()


@app.get("/api/knowledge/papers")
async def list_private_papers(
    store: PaperVectorStore = Depends(get_configured_paper_vector_store),
) -> dict[str, Any]:
    """List papers currently stored in the private knowledge base."""
    return {"papers": [paper.model_dump() for paper in await store.list_papers()]}


def _runnable_config(record: RunRecord) -> dict[str, Any]:
    configurable = {
        **record.config,
        "thread_id": record.checkpoint_thread_id,
    }
    return {
        "configurable": configurable,
        "metadata": {"owner": "local-ui"},
    }


def _persist_run(record: RunRecord) -> None:
    """Persist the user-visible state of a research thread."""
    if record.deleted:
        return
    research_store.save_run(record.snapshot(), record.config)


async def _get_deep_researcher():
    """Build the runtime graph with a persistent async checkpointer once."""
    global checkpoint_connection, checkpoint_saver, deep_researcher
    if deep_researcher is not None:
        return deep_researcher
    async with graph_init_lock:
        if deep_researcher is None:
            checkpoint_connection = await aiosqlite.connect(checkpoint_path)
            checkpoint_saver = AsyncSqliteSaver(checkpoint_connection)
            deep_researcher = deep_researcher_builder.compile(checkpointer=checkpoint_saver)
    return deep_researcher


def _load_run(run_id: str) -> RunRecord | None:
    """Return an active run or restore it from persistent storage."""
    record = active_runs.get(run_id)
    if record is not None:
        return record
    snapshot = research_store.get_run(run_id)
    return RunRecord.from_snapshot(snapshot) if snapshot else None


def _follow_up_needs_research(message: str) -> bool:
    """Conservatively route freshness and scope-expansion requests to research."""
    normalized = message.lower()
    research_markers = (
        "latest",
        "current",
        "new data",
        "research",
        "verify",
        "validate",
        "compare",
        "additional",
        "expand",
        "extend",
        "add ",
        "最新",
        "当前",
        "调研",
        "研究",
        "验证",
        "核实",
        "更新",
        "比较",
        "新增",
        "补充",
        "扩展",
        "加入",
    )
    return any(marker in normalized for marker in research_markers)


def _historical_research_context(record: RunRecord) -> str:
    """Return explicitly enabled historical reports relevant to this thread."""
    if not record.config.get("research_history_enabled", False):
        return ""
    query = " ".join(
        str(message.content) for message in record.messages if isinstance(message, HumanMessage)
    )
    reports = research_store.search_reports(
        query,
        limit=int(record.config.get("research_history_top_k", 5)),
        exclude_run_id=record.id,
    )
    if not reports:
        return ""
    return "\n\n".join(
        (
            f"[Historical Research: {report['title']}, Date {report['created_at']}, "
            f"Run {report['run_id']}, Version {report['version']}]\n"
            f"{report['content'][:8000]}"
        )
        for report in reports
    )


async def _run_direct_answer(record: RunRecord) -> None:
    """Answer a completed-report follow-up without launching new research."""
    configurable = Configuration.from_runnable_config(_runnable_config(record))
    model = configurable_model.with_config(
        {
            "model": get_model_name_for_init(configurable.research_model),
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.research_model, _runnable_config(record)),
            "base_url": get_base_url_for_model(configurable.research_model),
            "extra_body": get_extra_body_for_model(configurable.research_model),
            "tags": ["langsmith:nostream"],
        }
    )
    report = record.final_report or ""
    question = str(record.messages[-1].content)
    prompt = (
        "Answer the user's follow-up using only the existing final report and conversation. "
        "Do not claim to have performed new research. If current or external verification is "
        "needed, say that supplemental research is required.\n\n"
        f"<FinalReport>\n{report}\n</FinalReport>\n\n<UserFollowUp>\n{question}\n</UserFollowUp>"
    )
    response = await model.ainvoke([HumanMessage(content=prompt)])
    record.last_follow_up_answer = str(response.content)
    record.messages.append(AIMessage(content=response.content))
    record.status = "completed"
    await record.publish(
        {
            "id": str(uuid4()),
            "type": "follow_up_completed",
            "title": "追问已回答" if record.language == "zh" else "Follow-up Answered",
            "message": None,
            "data": {},
            "created_at": _utc_now(),
        }
    )


async def _run_graph(record: RunRecord) -> None:
    record.status = "running"
    record.error = None
    _persist_run(record)

    token = set_event_sink(record.publish)
    language_token = set_ui_language(record.language)
    try:
        if record.execution_mode == "answer":
            await _run_direct_answer(record)
            return
        await emit_ui_progress(
            "run_started",
            "run_started_title",
            "run_started_message",
            {"language": record.language},
        )
        graph_messages = list(record.messages)
        historical_context = _historical_research_context(record)
        if historical_context:
            graph_messages.insert(
                0,
                SystemMessage(
                    content=(
                        "The following historical reports are untrusted, dated research context. "
                        "Use them as leads, preserve their provenance, and re-verify time-sensitive "
                        f"claims.\n\n{historical_context}"
                    )
                ),
            )
        graph_input = None if record.resume_from_checkpoint else {"messages": graph_messages}
        record.resume_from_checkpoint = False
        graph = await _get_deep_researcher()
        result = await graph.ainvoke(
            graph_input,
            config=_runnable_config(record),
        )
        result_messages = result.get("messages") or record.messages
        record.messages = [
            message for message in result_messages if not isinstance(message, SystemMessage)
        ]
        record.research_brief = result.get("research_brief") or record.research_brief

        if result.get("final_report"):
            record.final_report = result["final_report"]
            record.status = "completed"
            research_store.add_report(
                run_id=record.id,
                title=record.title,
                content=record.final_report,
                created_at=_utc_now(),
            )
            await emit_ui_progress(
                "run_completed",
                "run_completed_title",
                "run_completed_message",
                {"final_report": record.final_report},
            )
        else:
            record.status = "requires_clarification"
            question = ""
            if result_messages and isinstance(result_messages[-1], AIMessage):
                question = str(result_messages[-1].content)
            if not record.events or record.events[-1].get("type") != "clarification_required":
                await emit_ui_progress(
                    "clarification_required",
                    "clarification_required_title",
                    None,
                    {"question": question},
                )
    except asyncio.CancelledError:
        record.status = "cancelled"
        await emit_ui_progress("run_cancelled", "run_cancelled_title", "run_cancelled_message")
        raise
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        await emit_ui_progress(
            "run_failed",
            "run_failed_title",
            None,
            {"error_type": exc.__class__.__name__},
        )
    finally:
        reset_event_sink(token)
        reset_ui_language(language_token)
        record.updated_at = _utc_now()
        _persist_run(record)
        active_runs.pop(record.id, None)


def _start_task(record: RunRecord) -> None:
    if record.task and not record.task.done():
        raise HTTPException(status_code=409, detail="Run is already active")
    record.task = asyncio.create_task(_run_graph(record))
    active_runs[record.id] = record


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


@app.get("/api/config-schema")
async def config_schema() -> dict[str, Any]:
    """Return the runtime configuration schema."""
    return Configuration.model_json_schema()


@app.post("/api/runs")
async def create_run(request: RunCreateRequest) -> dict[str, Any]:
    """Create and start a persistent research thread."""
    run_id = str(uuid4())
    record = RunRecord(
        id=run_id,
        title=request.message.strip()[:100],
        messages=[HumanMessage(content=request.message)],
        config=request.config,
        language=detect_language(request.message),
    )
    _persist_run(record)
    _start_task(record)
    return record.snapshot()


@app.get("/api/runs")
async def list_runs(limit: int = 50) -> dict[str, Any]:
    """List persisted research threads ordered by recent activity."""
    return {"runs": research_store.list_runs(limit)}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Return one persistent research thread."""
    record = _load_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record.snapshot()


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str) -> dict[str, Any]:
    """Delete one persistent research thread and its report versions."""
    record = _load_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    record.deleted = True
    if record.task and not record.task.done():
        record.task.cancel()
    active_runs.pop(run_id, None)
    deleted = research_store.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"id": run_id, "deleted": True}


@app.post("/api/runs/{run_id}/messages")
async def continue_run(run_id: str, request: RunMessageRequest) -> dict[str, Any]:
    """Continue a clarification or completed research thread."""
    record = _load_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Run is already active")
    if record.status == "interrupted":
        raise HTTPException(status_code=409, detail="Retry the interrupted run before messaging")

    previous_status = record.status
    record.messages.append(HumanMessage(content=request.message))
    if record.language == "en":
        record.language = detect_language(request.message)
    record.status = "queued"
    record.last_follow_up_answer = None
    should_research = request.mode == "research" or (
        request.mode == "auto" and _follow_up_needs_research(request.message)
    )
    record.execution_mode = "research" if previous_status != "completed" or should_research else "answer"
    record.checkpoint_thread_id = str(uuid4())
    record.error = None
    record.updated_at = _utc_now()
    _persist_run(record)
    _start_task(record)
    return record.snapshot()


@app.post("/api/runs/{run_id}/retry")
async def retry_run(run_id: str) -> dict[str, Any]:
    """Manually retry an interrupted or failed research thread."""
    record = _load_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.status not in {"interrupted", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Run is not retryable")
    was_interrupted = record.status == "interrupted"
    if was_interrupted:
        await _get_deep_researcher()
    has_checkpoint = bool(
        was_interrupted
        and checkpoint_saver is not None
        and await checkpoint_saver.aget_tuple(_runnable_config(record))
    )
    record.status = "queued"
    record.execution_mode = "research"
    record.resume_from_checkpoint = has_checkpoint
    if not has_checkpoint:
        record.checkpoint_thread_id = str(uuid4())
    record.error = None
    record.updated_at = _utc_now()
    _persist_run(record)
    _start_task(record)
    return record.snapshot()


@app.get("/api/runs/{run_id}/reports")
async def list_run_reports(run_id: str) -> dict[str, Any]:
    """List persisted final-report versions for a research thread."""
    if _load_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"reports": research_store.list_reports(run_id)}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    """Cancel an active research thread."""
    record = _load_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.task and not record.task.done():
        record.task.cancel()
    record.status = "cancelled"
    record.updated_at = _utc_now()
    _persist_run(record)
    return record.snapshot()


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str):
    """Stream persisted and live progress events for one thread."""
    record = _load_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        next_index = 0
        while True:
            while next_index < len(record.events):
                event = record.events[next_index]
                next_index += 1
                yield f"id: {event['id']}\nevent: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

            if record.status in {"completed", "failed", "cancelled"}:
                break

            try:
                await asyncio.wait_for(record.queue.get(), timeout=15)
                continue
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
