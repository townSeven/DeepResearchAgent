"""Progress events for the interactive deep research UI."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from langchain_core.messages import BaseMessage

EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]

_event_sink: ContextVar[Optional[EventSink]] = ContextVar("event_sink", default=None)
_ui_language: ContextVar[str] = ContextVar("ui_language", default="en")

TEXT = {
    "zh": {
        "run_started_title": "开始研究",
        "run_started_message": "正在读取问题并准备研究流程。",
        "clarifying_title": "理解研究需求",
        "clarifying_message": "正在判断是否需要先向你追问细节。",
        "clarification_skipped_title": "跳过澄清",
        "clarification_skipped_message": "已根据配置直接进入研究计划生成。",
        "clarification_required_title": "需要补充信息",
        "clarification_required_message": "研究开始前需要更多细节。",
        "clarification_completed_title": "需求已确认",
        "research_brief_started_title": "制定研究计划",
        "research_brief_started_message": "正在把问题整理成可执行的研究 brief。",
        "research_brief_created_title": "研究 brief 已生成",
        "supervisor_planning_title": "规划研究路径",
        "supervisor_planning_message": "研究主管正在拆解问题并决定下一步搜索任务。",
        "research_plan_created_title": "已分派子研究",
        "research_plan_created_message": "已分派 {count} 个子研究任务。",
        "thinking_started_title": "正在思考",
        "thinking_started_message": "正在评估已有信息和下一步行动。",
        "thinking_completed_title": "正在思考",
        "research_phase_ready_title": "研究资料已足够",
        "research_phase_ready_message": "研究主管已准备进入最终报告生成。",
        "research_phase_completed_title": "研究阶段完成",
        "research_phase_completed_message": "已收集到足够资料，准备生成最终报告。",
        "research_tasks_started_title": "启动子研究",
        "research_tasks_started_message": "正在并行执行 {count} 个子研究任务。",
        "research_task_started_title": "子研究 {index} 开始",
        "research_task_started_message": "正在执行该子研究任务。",
        "research_task_completed_title": "子研究 {index} 完成",
        "research_task_completed_message": "该子研究已完成搜索和整理。",
        "researcher_thinking_title": "研究员正在分析",
        "researcher_thinking_message": "正在判断下一步需要搜索什么。",
        "search_started_title": "正在搜索网页",
        "search_started_message": "研究员正在搜索 {count} 个网页。",
        "search_completed_title": "网页搜索完成",
        "search_completed_message": "已完成 {count} 个网页搜索。",
        "tool_started_title": "正在调用工具",
        "tool_started_message": "正在调用 {tool_name}。",
        "tool_completed_title": "工具调用完成",
        "tool_completed_message": "{tool_name} 调用完成。",
        "tool_failed_title": "工具调用失败",
        "researcher_ready_to_compress_title": "准备整理研究结果",
        "researcher_ready_to_compress_message": "该子研究准备整理研究笔记。",
        "compressing_research_title": "整理子研究结果",
        "compressing_research_message": "正在把搜索记录压缩成结构化研究笔记。",
        "research_compressed_title": "子研究笔记已整理",
        "research_compressed_message": "该子研究的关键发现已压缩完成。",
        "writing_final_report_title": "生成最终报告",
        "writing_final_report_message": "正在综合所有研究笔记并撰写最终报告。",
        "final_report_created_title": "最终报告已生成",
        "final_report_created_message": "报告已完成，可以查看全文。",
        "run_completed_title": "研究完成",
        "run_completed_message": "最终报告已经生成。",
        "run_cancelled_title": "研究已停止",
        "run_cancelled_message": "当前研究任务已取消。",
        "run_failed_title": "研究失败",
    },
    "en": {
        "run_started_title": "Research Started",
        "run_started_message": "Reading the question and preparing the research workflow.",
        "clarifying_title": "Understanding The Request",
        "clarifying_message": "Checking whether a clarification is needed before research begins.",
        "clarification_skipped_title": "Clarification Skipped",
        "clarification_skipped_message": "Proceeding directly to the research plan based on configuration.",
        "clarification_required_title": "More Information Needed",
        "clarification_required_message": "More detail is needed before research begins.",
        "clarification_completed_title": "Request Confirmed",
        "research_brief_started_title": "Creating Research Plan",
        "research_brief_started_message": "Turning the question into an actionable research brief.",
        "research_brief_created_title": "Research Brief Created",
        "supervisor_planning_title": "Planning Research Path",
        "supervisor_planning_message": "The supervisor is breaking down the question and deciding the next research tasks.",
        "research_plan_created_title": "Sub-Research Assigned",
        "research_plan_created_message": "Assigned {count} sub-research tasks.",
        "thinking_started_title": "Thinking",
        "thinking_started_message": "Assessing current findings and next steps.",
        "thinking_completed_title": "Thinking",
        "research_phase_ready_title": "Research Is Sufficient",
        "research_phase_ready_message": "The supervisor is ready to move into final report generation.",
        "research_phase_completed_title": "Research Phase Complete",
        "research_phase_completed_message": "Enough material has been gathered. Preparing the final report.",
        "research_tasks_started_title": "Starting Sub-Research",
        "research_tasks_started_message": "Running {count} sub-research tasks in parallel.",
        "research_task_started_title": "Sub-Research {index} Started",
        "research_task_started_message": "Running this sub-research task.",
        "research_task_completed_title": "Sub-Research {index} Complete",
        "research_task_completed_message": "This sub-research task has completed search and synthesis.",
        "researcher_thinking_title": "Researcher Is Analyzing",
        "researcher_thinking_message": "Deciding what to search next.",
        "search_started_title": "Searching The Web",
        "search_started_message": "The researcher is searching {count} web pages.",
        "search_completed_title": "Web Search Complete",
        "search_completed_message": "Completed {count} web searches.",
        "tool_started_title": "Calling Tool",
        "tool_started_message": "Calling {tool_name}.",
        "tool_completed_title": "Tool Call Complete",
        "tool_completed_message": "{tool_name} completed.",
        "tool_failed_title": "Tool Call Failed",
        "researcher_ready_to_compress_title": "Preparing Research Notes",
        "researcher_ready_to_compress_message": "This sub-research task is preparing its notes.",
        "compressing_research_title": "Synthesizing Sub-Research",
        "compressing_research_message": "Compressing search records into structured research notes.",
        "research_compressed_title": "Sub-Research Notes Ready",
        "research_compressed_message": "Key findings from this sub-research task are ready.",
        "writing_final_report_title": "Writing Final Report",
        "writing_final_report_message": "Combining all research notes into the final report.",
        "final_report_created_title": "Final Report Created",
        "final_report_created_message": "The report is complete.",
        "run_completed_title": "Research Complete",
        "run_completed_message": "The final report has been generated.",
        "run_cancelled_title": "Research Stopped",
        "run_cancelled_message": "The current research task was cancelled.",
        "run_failed_title": "Research Failed",
    },
}


def set_event_sink(sink: Optional[EventSink]):
    """Set the event sink for the current async context."""
    return _event_sink.set(sink)


def reset_event_sink(token) -> None:
    """Reset the event sink context variable."""
    _event_sink.reset(token)


def set_ui_language(language: str):
    """Set the user-facing language for the current async context."""
    normalized = "zh" if language == "zh" else "en"
    return _ui_language.set(normalized)


def reset_ui_language(token) -> None:
    """Reset the user-facing language context variable."""
    _ui_language.reset(token)


def get_ui_language() -> str:
    """Return the active user-facing language."""
    return _ui_language.get()


def detect_language(text: str) -> str:
    """Detect whether user-facing UI should be Chinese or English."""
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in text if char.isascii() and char.isalpha())
    return "zh" if cjk_count >= max(2, latin_count // 4) else "en"


def ui_text(key: str, **kwargs: Any) -> str:
    """Return localized UI text for the active run."""
    language = get_ui_language()
    template = TEXT.get(language, TEXT["en"]).get(key) or TEXT["en"].get(key) or key
    return template.format(**kwargs)


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of common LangChain objects into JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseMessage):
        return {
            "type": value.type,
            "content": value.content,
            "name": getattr(value, "name", None),
            "tool_calls": getattr(value, "tool_calls", None),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


async def emit_progress(
    event_type: str,
    title: str,
    message: str | None = None,
    data: Optional[dict[str, Any]] = None,
) -> None:
    """Emit a progress event if the current run installed an event sink."""
    sink = _event_sink.get()
    if sink is None:
        return

    event = {
        "id": str(uuid4()),
        "type": event_type,
        "title": title,
        "message": message,
        "data": _json_safe(data or {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = sink(event)
    if inspect.isawaitable(result):
        await result


async def emit_ui_progress(
    event_type: str,
    title_key: str,
    message_key: str | None = None,
    data: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Emit a localized progress event."""
    await emit_progress(
        event_type,
        ui_text(title_key, **kwargs),
        ui_text(message_key, **kwargs) if message_key else None,
        data,
    )
