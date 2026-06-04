"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deep_research_agent.configuration import (
    Configuration,
)
from deep_research_agent.events import emit_progress, emit_ui_progress, get_ui_language, ui_text
from deep_research_agent.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from deep_research_agent.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from deep_research_agent.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_base_url_for_model,
    get_extra_body_for_model,
    get_model_name_for_init,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    strip_thinking_tags,
    think_tool,
    with_structured_output_for_model,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url", "extra_body"),
)


def _tool_call_summary(tool_calls: list[dict]) -> list[dict]:
    """Return compact tool call details for progress events."""
    return [
        {
            "id": tool_call.get("id"),
            "name": tool_call.get("name"),
            "args": tool_call.get("args", {}),
        }
        for tool_call in tool_calls
    ]


def _visible_tool_call_summary(tool_calls: list[dict]) -> list[dict]:
    """Return compact tool details without exposing long prompts or raw results."""
    summaries = []
    for tool_call in tool_calls:
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        summary = {"id": tool_call.get("id"), "name": name}
        if name == "ConductResearch":
            summary["research_topic"] = args.get("research_topic", "")
        elif name == "think_tool":
            summary["reflection"] = args.get("reflection", "")
        else:
            summary["query_count"] = _count_search_queries(args)
        summaries.append(summary)
    return summaries


def _count_search_queries(args) -> int:
    """Estimate how many web searches a tool call will perform."""
    if not isinstance(args, dict):
        return 1
    queries = args.get("queries")
    if isinstance(queries, list):
        return max(1, len(queries))
    for key in ("query", "q", "search_query", "keywords"):
        if args.get(key):
            return 1
    return 1


def _language_instruction() -> str:
    """Instruction appended to prompts to keep generated text in the user's language."""
    if get_ui_language() == "zh":
        return (
            "LANGUAGE LOCK: The user's target language is Chinese. All user-visible output "
            "must be written in Chinese, including research briefs, sub-research tasks, "
            "reflections, summaries, report titles, section headings, table headers, bullet "
            "points, and the final source-section heading. Translate English research notes "
            "and findings into Chinese. Preserve proper nouns, URLs, code identifiers, model "
            "names, and source titles only when translation would be unnatural. Before "
            "returning, check that no generated generic English phrases remain."
        )
    return (
        "LANGUAGE LOCK: Identify the target language from the latest human/user message in "
        "the conversation, then write all user-visible output in that same language. This "
        "includes research briefs, sub-research tasks, reflections, summaries, report titles, "
        "section headings, table headers, bullet points, and the final source-section heading. "
        "Ignore the language of prompt instructions, tool outputs, source snippets, citations, "
        "URLs, and previous assistant messages when deciding the target language. Translate "
        "research notes and findings into the target language unless they are proper nouns, "
        "URLs, code identifiers, model names, or source titles that should remain unchanged."
    )

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    await emit_ui_progress(
        "clarifying",
        "clarifying_title",
        "clarifying_message",
    )
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        await emit_ui_progress(
            "clarification_skipped",
            "clarification_skipped_title",
            "clarification_skipped_message",
        )
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": get_model_name_for_init(configurable.research_model),
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model),
        "extra_body": get_extra_body_for_model(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        with_structured_output_for_model(
            configurable_model,
            ClarifyWithUser,
            configurable.research_model,
        )
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    ) + "\n\n" + _language_instruction()
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        await emit_ui_progress(
            "clarification_required",
            "clarification_required_title",
            None,
            {"question": response.question},
        )
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        await emit_ui_progress(
            "clarification_completed",
            "clarification_completed_title",
            None,
            {"verification": response.verification},
        )
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    await emit_ui_progress(
        "research_brief_started",
        "research_brief_started_title",
        "research_brief_started_message",
    )
    research_model_config = {
        "model": get_model_name_for_init(configurable.research_model),
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model),
        "extra_body": get_extra_body_for_model(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        with_structured_output_for_model(
            configurable_model,
            ResearchQuestion,
            configurable.research_model,
        )
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    ) + "\n\n" + _language_instruction()
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    await emit_ui_progress(
        "research_brief_created",
        "research_brief_created_title",
        None,
        {"research_brief": response.research_brief},
    )
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    ) + "\n\n" + _language_instruction()
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    await emit_ui_progress(
        "supervisor_planning",
        "supervisor_planning_title",
        "supervisor_planning_message",
        {"iteration": state.get("research_iterations", 0) + 1},
    )
    research_model_config = {
        "model": get_model_name_for_init(configurable.research_model),
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model),
        "extra_body": get_extra_body_for_model(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    tool_calls = getattr(response, "tool_calls", [])
    conduct_topics = [
        tool_call["args"].get("research_topic")
        for tool_call in tool_calls
        if tool_call.get("name") == "ConductResearch"
    ]
    if conduct_topics:
        await emit_ui_progress(
            "research_plan_created",
            "research_plan_created_title",
            "research_plan_created_message",
            {"topics": conduct_topics, "tool_calls": _visible_tool_call_summary(tool_calls)},
            count=len(conduct_topics),
        )
    elif any(tool_call.get("name") == "ResearchComplete" for tool_call in tool_calls):
        await emit_ui_progress(
            "research_phase_ready",
            "research_phase_ready_title",
            "research_phase_ready_message",
        )
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations  # 研究轮数超过上限
    no_tool_calls = not most_recent_message.tool_calls      # 没有工具调用
    research_complete_tool_call = any(                      # 模型调用了ResearchComplete工具，表示研究完成
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:     # 满足上述任一条件则结束研究阶段
        await emit_ui_progress(
            "research_phase_completed",
            "research_phase_completed_title",
            "research_phase_completed_message",
            {
                "exceeded_allowed_iterations": exceeded_allowed_iterations,
                "no_tool_calls": no_tool_calls,
                "research_complete_tool_call": research_complete_tool_call,
            },
        )
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        await emit_ui_progress(
            "thinking_started",
            "thinking_started_title",
            "thinking_started_message",
            {"scope": "supervisor"},
        )
        await emit_ui_progress(
            "thinking_completed",
            "thinking_completed_title",
            None,
            {"scope": "supervisor", "reflection": reflection_content},
        )
        reflection_record = (
            f"思考记录：{reflection_content}"
            if get_ui_language() == "zh"
            else f"Reflection recorded: {reflection_content}"
        )
        all_tool_messages.append(ToolMessage(
            content=reflection_record,
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            await emit_ui_progress(
                "research_tasks_started",
                "research_tasks_started_title",
                "research_tasks_started_message",
                {
                    "overflow_count": len(overflow_conduct_research_calls),
                    "count": len(allowed_conduct_research_calls),
                },
                count=len(allowed_conduct_research_calls),
            )

            async def run_research_task(tool_call, index):
                topic = tool_call["args"]["research_topic"]
                await emit_ui_progress(
                    "research_task_started",
                    "research_task_started_title",
                    "research_task_started_message",
                    {"topic": topic, "index": index},
                    index=index + 1,
                )
                result = await researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=topic)
                    ],
                    "research_topic": topic
                }, config)
                await emit_ui_progress(
                    "research_task_completed",
                    "research_task_completed_title",
                    "research_task_completed_message",
                    {
                        "topic": topic,
                        "index": index,
                    },
                    index=index + 1,
                )
                return result
            
            # Execute research tasks in parallel
            research_tasks = [
                run_research_task(tool_call, index)
                for index, tool_call in enumerate(allowed_conduct_research_calls)
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    await emit_ui_progress(
        "researcher_thinking",
        "researcher_thinking_title",
        "researcher_thinking_message",
        {
            "topic": state.get("research_topic"),
            "iteration": state.get("tool_call_iterations", 0) + 1,
        },
    )
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": get_model_name_for_init(configurable.research_model),
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model),
        "extra_body": get_extra_body_for_model(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    ) + "\n\n" + _language_instruction()
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    tool_calls = getattr(response, "tool_calls", [])
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    tool_name = tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
    metadata = getattr(tool, "metadata", None) or {}
    is_search_tool = metadata.get("type") == "search" or "search" in tool_name
    is_think_tool = tool_name == "think_tool"
    query_count = _count_search_queries(args)
    if is_think_tool:
        reflection = args.get("reflection", "") if isinstance(args, dict) else ""
        await emit_ui_progress(
            "thinking_started",
            "thinking_started_title",
            "thinking_started_message",
            {"scope": "researcher"},
        )
        await emit_ui_progress(
            "thinking_completed",
            "thinking_completed_title",
            None,
            {"scope": "researcher", "reflection": reflection},
        )
        return (
            f"思考记录：{reflection}"
            if get_ui_language() == "zh"
            else f"Reflection recorded: {reflection}"
        )
    elif not is_search_tool:
        await emit_ui_progress(
            "tool_started",
            "tool_started_title",
            "tool_started_message",
            {"tool_name": tool_name},
            tool_name=tool_name,
        )
    try:
        result = await tool.ainvoke(args, config)
        if not is_search_tool and not is_think_tool:
            await emit_ui_progress(
                "tool_completed",
                "tool_completed_title",
                "tool_completed_message",
                {"tool_name": tool_name},
                tool_name=tool_name,
            )
        return result
    except Exception as e:
        await emit_ui_progress(
            "tool_failed",
            "tool_failed_title",
            None,
            {"tool_name": tool_name, "error": str(e)},
        )
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        await emit_ui_progress(
            "researcher_ready_to_compress",
            "researcher_ready_to_compress_title",
            "researcher_ready_to_compress_message",
            {"topic": state.get("research_topic")},
        )
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    search_query_count = 0
    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        metadata = getattr(tool, "metadata", None) or {}
        tool_name = tool.name if hasattr(tool, "name") else tool_call["name"]
        if metadata.get("type") == "search" or "search" in tool_name:
            search_query_count += _count_search_queries(tool_call.get("args", {}))

    if search_query_count:
        await emit_ui_progress(
            "search_started",
            "search_started_title",
            "search_started_message",
            {"query_count": search_query_count, "topic": state.get("research_topic")},
            count=search_query_count,
        )
    elif tool_calls:
        visible_tools = [
            tool_call for tool_call in tool_calls
            if tool_call.get("name") not in {"think_tool", "ResearchComplete"}
        ]
        if visible_tools:
            await emit_ui_progress(
                "tool_batch_started",
                "tool_started_title",
                "tool_started_message",
                {"tool_calls": _visible_tool_call_summary(visible_tools), "topic": state.get("research_topic")},
                tool_name=", ".join(tool_call.get("name", "") for tool_call in visible_tools),
            )
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    if search_query_count:
        await emit_ui_progress(
            "search_completed",
            "search_completed_title",
            "search_completed_message",
            {"query_count": search_query_count, "topic": state.get("research_topic")},
            count=search_query_count,
        )
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls     # 超过最大工具调用轮数
    research_complete_called = any(                             # 模型调用了ResearchComplete工具，表示研究完成
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:         # 满足上述任一条件则结束研究阶段，进入压缩阶段
        # End research and proceed to compression
        await emit_ui_progress(
            "researcher_ready_to_compress",
            "researcher_ready_to_compress_title",
            "researcher_ready_to_compress_message",
            {
                "topic": state.get("research_topic"),
                "exceeded_iterations": exceeded_iterations,
                "research_complete_called": research_complete_called,
            },
        )
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    await emit_ui_progress(
        "compressing_research",
        "compressing_research_title",
        "compressing_research_message",
        {"topic": state.get("research_topic")},
    )
    synthesizer_model = configurable_model.with_config({
        "model": get_model_name_for_init(configurable.compression_model),
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "base_url": get_base_url_for_model(configurable.compression_model),
        "extra_body": get_extra_body_for_model(configurable.compression_model),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str()) + "\n\n" + _language_instruction()
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            await emit_ui_progress(
                "research_compressed",
                "research_compressed_title",
                "research_compressed_message",
                {"topic": state.get("research_topic")},
            )
            return {
                "compressed_research": strip_thinking_tags(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    await emit_ui_progress(
        "writing_final_report",
        "writing_final_report_title",
        "writing_final_report_message",
        {"notes_count": len(notes)},
    )
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": get_model_name_for_init(configurable.final_report_model),
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "base_url": get_base_url_for_model(configurable.final_report_model),
        "extra_body": get_extra_body_for_model(configurable.final_report_model),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            ) + "\n\n" + _language_instruction()
            
            # Generate the final report with token streaming when supported.
            final_report_parts = []
            writer_model = configurable_model.with_config(writer_model_config)
            async for chunk in writer_model.astream([HumanMessage(content=final_report_prompt)]):
                chunk_content = getattr(chunk, "content", "")
                if isinstance(chunk_content, list):
                    chunk_text = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in chunk_content
                    )
                else:
                    chunk_text = str(chunk_content or "")
                if not chunk_text:
                    continue
                final_report_parts.append(chunk_text)
                await emit_progress(
                    "final_report_delta",
                    ui_text("writing_final_report_title"),
                    None,
                    {"delta": strip_thinking_tags(chunk_text)},
                )
            
            # Return successful report generation
            final_report_content = strip_thinking_tags("".join(final_report_parts))
            await emit_ui_progress(
                "final_report_created",
                "final_report_created_title",
                "final_report_created_message",
                {"final_report": final_report_content},
            )
            return {
                "final_report": final_report_content, 
                "messages": [AIMessage(content=final_report_content)],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
