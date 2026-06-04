"""FastAPI service for the interactive Deep Research Agent UI."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from deep_research_agent.configuration import Configuration
from deep_research_agent.deep_researcher import deep_researcher
from deep_research_agent.events import (
    detect_language,
    emit_ui_progress,
    reset_event_sink,
    reset_ui_language,
    set_event_sink,
    set_ui_language,
)

load_dotenv()

RunStatus = Literal[
    "queued",
    "running",
    "requires_clarification",
    "completed",
    "failed",
    "cancelled",
]


class RunCreateRequest(BaseModel):
    """Request body for creating a new research run."""

    message: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class RunMessageRequest(BaseModel):
    """Request body for continuing a run after a clarification question."""

    message: str = Field(min_length=1)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    return {
        "type": message.type,
        "content": message.content,
        "name": getattr(message, "name", None),
    }


@dataclass
class RunRecord:
    """In-memory state for a single interactive run."""

    id: str
    messages: list[BaseMessage]
    config: dict[str, Any]
    status: RunStatus = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    final_report: str | None = None
    research_brief: str | None = None
    error: str | None = None
    language: str = "en"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    async def publish(self, event: dict[str, Any]) -> None:
        self.updated_at = _utc_now()
        self.events.append(event)
        await self.queue.put(event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "messages": [_message_to_dict(message) for message in self.messages],
            "events": self.events,
            "final_report": self.final_report,
            "research_brief": self.research_brief,
            "error": self.error,
            "language": self.language,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


runs: dict[str, RunRecord] = {}

app = FastAPI(title="Deep Research Agent Interactive API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _runnable_config(record: RunRecord) -> dict[str, Any]:
    configurable = {
        **record.config,
        "thread_id": record.id,
    }
    return {
        "configurable": configurable,
        "metadata": {"owner": "local-ui"},
    }


async def _run_graph(record: RunRecord) -> None:
    record.status = "running"

    token = set_event_sink(record.publish)
    language_token = set_ui_language(record.language)
    try:
        await emit_ui_progress(
            "run_started",
            "run_started_title",
            "run_started_message",
            {"language": record.language},
        )
        result = await deep_researcher.ainvoke(
            {"messages": record.messages},
            config=_runnable_config(record),
        )
        result_messages = result.get("messages") or record.messages
        record.messages = result_messages
        record.research_brief = result.get("research_brief") or record.research_brief

        if result.get("final_report"):
            record.final_report = result["final_report"]
            record.status = "completed"
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


def _start_task(record: RunRecord) -> None:
    if record.task and not record.task.done():
        raise HTTPException(status_code=409, detail="Run is already active")
    record.task = asyncio.create_task(_run_graph(record))


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config-schema")
async def config_schema() -> dict[str, Any]:
    return Configuration.model_json_schema()


@app.post("/api/runs")
async def create_run(request: RunCreateRequest) -> dict[str, Any]:
    run_id = str(uuid4())
    record = RunRecord(
        id=run_id,
        messages=[HumanMessage(content=request.message)],
        config=request.config,
        language=detect_language(request.message),
    )
    runs[run_id] = record
    _start_task(record)
    return record.snapshot()


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    record = runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record.snapshot()


@app.post("/api/runs/{run_id}/messages")
async def continue_run(run_id: str, request: RunMessageRequest) -> dict[str, Any]:
    record = runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.status != "requires_clarification":
        raise HTTPException(status_code=409, detail="Run is not waiting for clarification")

    record.messages.append(HumanMessage(content=request.message))
    if record.language == "en":
        record.language = detect_language(request.message)
    record.status = "queued"
    record.final_report = None
    record.error = None
    _start_task(record)
    return record.snapshot()


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    record = runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.task and not record.task.done():
        record.task.cancel()
    record.status = "cancelled"
    return record.snapshot()


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str):
    record = runs.get(run_id)
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
