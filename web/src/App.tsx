import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import {
  AlertCircle,
  BarChart3,
  Brain,
  Calendar,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clipboard,
  Download,
  FileText,
  Filter,
  Link,
  Loader2,
  Menu,
  Plus,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  Square,
} from "lucide-react";

type RunStatus =
  | "queued"
  | "running"
  | "requires_clarification"
  | "completed"
  | "failed"
  | "cancelled";

type ProgressEvent = {
  id: string;
  type: string;
  title: string;
  message?: string | null;
  data?: Record<string, unknown>;
  created_at: string;
};

type RunSnapshot = {
  id: string;
  status: RunStatus;
  events: ProgressEvent[];
  final_report?: string | null;
  research_brief?: string | null;
  error?: string | null;
  language?: "zh" | "en";
};

type ConfigState = {
  searchApi: string;
  allowClarification: boolean;
  researchModel: string;
  summarizationModel: string;
  compressionModel: string;
  finalReportModel: string;
  maxConcurrentResearchUnits: number;
  maxResearcherIterations: number;
  maxReactToolCalls: number;
};

const defaultConfig: ConfigState = {
  searchApi: "tavily",
  allowClarification: true,
  researchModel: "deepseek:deepseek-v4-flash",
  summarizationModel: "deepseek:deepseek-v4-flash",
  compressionModel: "deepseek:deepseek-v4-flash",
  finalReportModel: "deepseek:deepseek-v4-flash",
  maxConcurrentResearchUnits: 5,
  maxResearcherIterations: 6,
  maxReactToolCalls: 10,
};

const terminalStatuses: RunStatus[] = ["completed", "failed", "cancelled"];
const hiddenTimelineEvents = new Set([
  "final_report_delta",
  "research_tasks_started",
  "research_task_started",
  "research_task_completed",
  "researcher_thinking",
  "researcher_ready_to_compress",
  "compressing_research",
  "research_compressed",
  "search_started",
  "search_completed",
  "thinking_started",
]);

const statusLabels: Record<RunStatus | "idle", string> = {
  idle: "待开始",
  queued: "排队中",
  running: "研究中",
  requires_clarification: "需要补充",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function toApiConfig(config: ConfigState) {
  return {
    search_api: config.searchApi,
    allow_clarification: config.allowClarification,
    research_model: config.researchModel,
    summarization_model: config.summarizationModel,
    compression_model: config.compressionModel,
    final_report_model: config.finalReportModel,
    max_concurrent_research_units: config.maxConcurrentResearchUnits,
    max_researcher_iterations: config.maxResearcherIterations,
    max_react_tool_calls: config.maxReactToolCalls,
  };
}

function eventIcon(type: string) {
  if (type.includes("thinking")) return <Brain size={16} />;
  if (type.includes("search") || type.includes("tool")) return <Search size={16} />;
  if (type.includes("report") || type.includes("brief")) return <FileText size={16} />;
  if (type.includes("failed")) return <AlertCircle size={16} />;
  if (type.includes("completed") || type.includes("created")) return <CheckCircle2 size={16} />;
  return <Brain size={16} />;
}

function formatTime(value?: string) {
  if (!value) return "--:--";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function formatShortTime(value?: string) {
  if (!value) return "现在";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function getClarificationQuestion(events: ProgressEvent[]) {
  const event = getClarificationEvent(events);
  return (event?.data?.question as string | undefined) || event?.message || "";
}

function getClarificationEvent(events: ProgressEvent[]) {
  return [...events].reverse().find((item) => item.type === "clarification_required");
}

function getEventText(event: ProgressEvent) {
  return (
    event.message ||
    (event.data?.question as string | undefined) ||
    (event.data?.verification as string | undefined) ||
    (event.data?.research_brief as string | undefined) ||
    (event.data?.reflection as string | undefined) ||
    ""
  );
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function streamStepFor(text: string) {
  if (text.length > 1200) return 18;
  if (text.length > 500) return 10;
  return 4;
}

function StreamingText({ text, className }: { text: string; className?: string }) {
  const [visibleLength, setVisibleLength] = useState(0);

  useEffect(() => {
    setVisibleLength(0);
    if (!text) return;

    const step = streamStepFor(text);
    const interval = window.setInterval(() => {
      setVisibleLength((current) => {
        const next = Math.min(text.length, current + step);
        if (next >= text.length) {
          window.clearInterval(interval);
        }
        return next;
      });
    }, 18);

    return () => window.clearInterval(interval);
  }, [text]);

  return (
    <span className={className}>
      {text.slice(0, visibleLength)}
      {visibleLength < text.length && <span className="stream-caret" />}
    </span>
  );
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [clarification, setClarification] = useState("");
  const [config, setConfig] = useState<ConfigState>(defaultConfig);
  const [configOpen, setConfigOpen] = useState(false);
  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [finalReport, setFinalReport] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [answeredClarificationKey, setAnsweredClarificationKey] = useState("");
  const sourceRef = useRef<EventSource | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);

  const isBusy = run?.status === "queued" || run?.status === "running";
  const runStatus = run?.status || "idle";
  const clarificationEvent = useMemo(() => getClarificationEvent(events), [events]);
  const clarificationQuestion = useMemo(() => getClarificationQuestion(events), [events]);
  const clarificationKey = clarificationEvent?.id || run?.id || "";
  const showClarificationBox =
    run?.status === "requires_clarification" && clarificationKey !== answeredClarificationKey;
  const visibleEvents = useMemo(
    () => events.filter((event) => !hiddenTimelineEvents.has(event.type)),
    [events],
  );
  const latestEvent = visibleEvents[visibleEvents.length - 1];
  const planEvents = visibleEvents.filter((event) => event.type.includes("plan") || event.type.includes("brief"));
  const sourceEvents = events.filter((event) => event.type.includes("search") || event.type.includes("tool")).slice(-5);
  const completedEvents = visibleEvents.filter((event) => event.type.includes("completed") || event.type.includes("created"));
  const displayPrompt = submittedPrompt || prompt.trim();
  const conversationTitle = displayPrompt || "准备开始新的深度研究";
  const reportReady = Boolean(finalReport);

  useEffect(() => {
    threadRef.current?.scrollTo({
      top: threadRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [visibleEvents.length, finalReport]);

  useEffect(() => {
    return () => sourceRef.current?.close();
  }, []);

  function attachEvents(runId: string) {
    sourceRef.current?.close();
    const source = new EventSource(`/api/runs/${runId}/events`);
    sourceRef.current = source;

    source.onerror = () => {
      setError("无法连接后端事件流。请确认 FastAPI 后端正在 http://127.0.0.1:8000 运行。");
      setRun((current) => current && { ...current, status: "failed" });
      source.close();
    };

    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as ProgressEvent;
      mergeEvent(event);
    };

    [
      "run_started",
      "clarifying",
      "clarification_skipped",
      "clarification_completed",
      "clarification_required",
      "research_brief_started",
      "research_brief_created",
      "supervisor_planning",
      "research_plan_created",
      "research_tasks_started",
      "research_task_started",
      "research_task_completed",
      "researcher_thinking",
      "tool_batch_started",
      "thinking_started",
      "thinking_completed",
      "search_started",
      "search_completed",
      "tool_started",
      "tool_completed",
      "tool_failed",
      "researcher_ready_to_compress",
      "compressing_research",
      "research_compressed",
      "research_phase_ready",
      "research_phase_completed",
      "writing_final_report",
      "final_report_delta",
      "final_report_created",
      "run_completed",
      "run_cancelled",
      "run_failed",
    ].forEach((type) => {
      source.addEventListener(type, (message) => {
        const event = JSON.parse((message as MessageEvent).data) as ProgressEvent;
        if (type === "final_report_delta") {
          const delta = event.data?.delta as string | undefined;
          if (delta) setFinalReport((current) => current + delta);
          return;
        }
        mergeEvent(event);
        if (type === "final_report_created" || type === "run_completed") {
          const report = event.data?.final_report as string | undefined;
          if (report) setFinalReport(report);
        }
        if (type === "run_completed") {
          const report = event.data?.final_report as string | undefined;
          setRun((current) => current && { ...current, status: "completed", final_report: report || current.final_report });
          source.close();
        }
        if (type === "clarification_required") {
          setRun((current) => current && { ...current, status: "requires_clarification" });
        }
        if (type === "run_failed") {
          setRun((current) => current && { ...current, status: "failed" });
          setError(event.message || "研究失败");
          source.close();
        }
        if (type === "run_cancelled") {
          setRun((current) => current && { ...current, status: "cancelled" });
          source.close();
        }
      });
    });
  }

  function renderEventDetails(event: ProgressEvent) {
    const details: ReactNode[] = [];
    const text = getEventText(event);

    if (text) {
      details.push(
        <p key="message">
          <StreamingText text={text} />
        </p>,
      );
    }

    if (event.type === "research_plan_created" && (event.data?.topics as string[] | undefined)?.length) {
      details.push(
        <ul className="topic-list" key="topics">
          {(event.data?.topics as string[]).map((topic, index) => (
            <li key={`${event.id}-${index}`}>
              <StreamingText text={topic} />
            </li>
          ))}
        </ul>,
      );
    }

    return details;
  }

  function mergeEvent(event: ProgressEvent) {
    setEvents((current) => {
      if (current.some((item) => item.id === event.id)) return current;
      return [...current, event];
    });
  }

  async function refreshRun(runId: string) {
    try {
      const response = await fetch(`/api/runs/${runId}`);
      if (!response.ok) {
        setError(await response.text());
        return;
      }
      const snapshot = (await response.json()) as RunSnapshot;
      setRun(snapshot);
      setEvents(snapshot.events || []);
      setFinalReport(snapshot.final_report || "");
      setError(snapshot.error || "");
    } catch (error) {
      setError(`无法连接后端 API：${getErrorMessage(error)}。请确认 FastAPI 后端正在 http://127.0.0.1:8000 运行。`);
    }
  }

  async function startRun(event: FormEvent) {
    event.preventDefault();
    if (!prompt.trim() || isBusy) return;

    const nextPrompt = prompt.trim();
    setSubmittedPrompt(nextPrompt);
    setError("");
    setEvents([]);
    setFinalReport("");
    setClarification("");
    setAnsweredClarificationKey("");
    try {
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: nextPrompt,
          config: toApiConfig(config),
        }),
      });

      if (!response.ok) {
        setError(await response.text());
        return;
      }

      const snapshot = (await response.json()) as RunSnapshot;
      setRun(snapshot);
      setPrompt("");
      attachEvents(snapshot.id);
    } catch (error) {
      setError(`无法创建研究任务：${getErrorMessage(error)}。请确认 FastAPI 后端正在 http://127.0.0.1:8000 运行。`);
    }
  }

  async function sendClarification(event: FormEvent) {
    event.preventDefault();
    if (!run || !clarification.trim()) return;

    try {
      const response = await fetch(`/api/runs/${run.id}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: clarification.trim() }),
      });

      if (!response.ok) {
        setError(await response.text());
        return;
      }

      setClarification("");
      setAnsweredClarificationKey(clarificationKey);
      setRun((current) => current && { ...current, status: "running" });
      await refreshRun(run.id);
      attachEvents(run.id);
    } catch (error) {
      setError(`无法继续研究任务：${getErrorMessage(error)}。请确认后端仍在运行。`);
    }
  }

  async function cancelRun() {
    if (!run || terminalStatuses.includes(run.status)) return;
    try {
      await fetch(`/api/runs/${run.id}/cancel`, { method: "POST" });
      sourceRef.current?.close();
      await refreshRun(run.id);
    } catch (error) {
      setError(`无法取消研究任务：${getErrorMessage(error)}`);
    }
  }

  async function copyReport() {
    if (!finalReport) return;
    await navigator.clipboard.writeText(finalReport);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  function downloadReport() {
    if (!finalReport) return;
    const blob = new Blob([finalReport], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `deep-research-${run?.id || "report"}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="app">
      <aside className="sidebar" aria-label="会话列表">
        <div className="side-top">
          <div className="brand-mark" aria-hidden="true">
            <ShieldCheck size={19} />
          </div>
          <div className="brand-text">
            <strong>DeepResearch</strong>
            <span>证据优先的研究工作台</span>
          </div>
          <button className="icon-button" type="button" title="菜单" aria-label="菜单">
            <Menu size={18} />
          </button>
        </div>

        <button
          className="new-chat"
          type="button"
          onClick={() => {
            setPrompt("");
            setSubmittedPrompt("");
            setRun(null);
            setEvents([]);
            setFinalReport("");
            setError("");
            setAnsweredClarificationKey("");
            sourceRef.current?.close();
          }}
        >
          <Plus size={17} />
          新建研究
        </button>

        <label className="search-box">
          <Search size={15} />
          <input type="search" placeholder="搜索对话、报告、来源" />
        </label>

        <div className="section-label">当前会话</div>
        <div className="chat-list">
          <button className="chat-item active" type="button">
            <span className="chat-dot">
              <FileText size={14} />
            </span>
            <span>
              <strong>{conversationTitle}</strong>
              <span>
                {visibleEvents.length ? `${visibleEvents.length} 条研究事件` : "输入问题后开始研究"}
              </span>
            </span>
            <span className="time">{formatShortTime(latestEvent?.created_at)}</span>
          </button>
          <button className="chat-item" type="button">
            <span className="chat-dot">
              <BarChart3 size={14} />
            </span>
            <span>
              <strong>运行配置</strong>
              <span>{config.searchApi} · 并行 {config.maxConcurrentResearchUnits}</span>
            </span>
            <span className="time">默认</span>
          </button>
        </div>

        <div className="settings-panel side-settings">
          <button className="settings-toggle" type="button" onClick={() => setConfigOpen((value) => !value)}>
            <Settings2 size={16} />
            运行配置
            {configOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {configOpen && (
            <div className="settings-grid">
              <label>
                搜索工具
                <select value={config.searchApi} onChange={(event) => setConfig({ ...config, searchApi: event.target.value })}>
                  <option value="minimax_mcp">MiniMax MCP</option>
                  <option value="tavily">Tavily</option>
                  <option value="openai">OpenAI Native</option>
                  <option value="anthropic">Anthropic Native</option>
                  <option value="none">None</option>
                </select>
              </label>
              <label className="checkbox-line">
                <input
                  type="checkbox"
                  checked={config.allowClarification}
                  onChange={(event) => setConfig({ ...config, allowClarification: event.target.checked })}
                />
                允许开始前追问
              </label>
              <label>
                Research model
                <input value={config.researchModel} onChange={(event) => setConfig({ ...config, researchModel: event.target.value })} />
              </label>
              <label>
                Summarization model
                <input value={config.summarizationModel} onChange={(event) => setConfig({ ...config, summarizationModel: event.target.value })} />
              </label>
              <label>
                Compression model
                <input value={config.compressionModel} onChange={(event) => setConfig({ ...config, compressionModel: event.target.value })} />
              </label>
              <label>
                Final report model
                <input value={config.finalReportModel} onChange={(event) => setConfig({ ...config, finalReportModel: event.target.value })} />
              </label>
              <label>
                并行子研究数
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={config.maxConcurrentResearchUnits}
                  onChange={(event) => setConfig({ ...config, maxConcurrentResearchUnits: Number(event.target.value) })}
                />
              </label>
              <label>
                Supervisor 轮数
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={config.maxResearcherIterations}
                  onChange={(event) => setConfig({ ...config, maxResearcherIterations: Number(event.target.value) })}
                />
              </label>
              <label>
                单研究工具轮数
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={config.maxReactToolCalls}
                  onChange={(event) => setConfig({ ...config, maxReactToolCalls: Number(event.target.value) })}
                />
              </label>
            </div>
          )}
        </div>

        <div className="side-bottom">
          <div className="usage">
            <div className="usage-row">
              <span>研究进度</span>
              <strong>{runStatus === "completed" ? "100%" : isBusy ? "执行中" : statusLabels[runStatus]}</strong>
            </div>
            <div className="meter">
              <span style={{ width: runStatus === "completed" ? "100%" : isBusy ? "68%" : "18%" }} />
            </div>
          </div>
        </div>
      </aside>

      <section className="main">
        <header className="topbar">
          <button className="icon-button mobile-menu" type="button" title="打开会话列表" aria-label="打开会话列表">
            <Menu size={18} />
          </button>
          <div className="conversation-title">
            <h1>{conversationTitle}</h1>
            <div className="meta-line">
              <span className={`pill ${isBusy ? "good" : ""}`}>
                {isBusy && <Loader2 className="spin small-icon" />}
                {statusLabels[runStatus]}
              </span>
              <span className="pill">{visibleEvents.length} 事件</span>
              <span className={`pill ${reportReady ? "good" : "warn"}`}>{reportReady ? "报告已生成" : "等待报告"}</span>
            </div>
          </div>
          <div className="toolbar">
            <button className="icon-button hide-mobile" type="button" disabled={!finalReport} onClick={copyReport} title="复制报告" aria-label="复制报告">
              <Clipboard size={17} />
            </button>
            <button className="icon-button hide-mobile" type="button" disabled={!finalReport} onClick={downloadReport} title="下载 Markdown" aria-label="下载 Markdown">
              <Download size={17} />
            </button>
            <button className="icon-button mobile-inspector-trigger" type="button" title="查看证据" aria-label="查看证据">
              <Filter size={17} />
            </button>
          </div>
        </header>

        <div className="scroll-region" ref={threadRef}>
          <section className="thread" aria-label="对话内容">
            {!displayPrompt && visibleEvents.length === 0 && (
              <div className="empty-thread">
                <Brain size={22} />
                <strong>输入一个研究问题</strong>
                <span>我会把计划、搜索、核验和最终报告放在同一个研究线程里。</span>
              </div>
            )}

            {displayPrompt && (
              <article className="message">
                <div className="avatar">你</div>
                <div className="bubble">
                  <div className="sender">
                    你 <span>{formatShortTime(events[0]?.created_at)}</span>
                  </div>
                  <div className="user-bubble">{displayPrompt}</div>
                </div>
              </article>
            )}

            {(visibleEvents.length > 0 || isBusy || finalReport || error) && (
              <article className="message">
                <div className="avatar assistant">
                  <ShieldCheck size={18} />
                </div>
                <div className="bubble">
                  <div className="sender">
                    DeepResearch <span>{latestEvent ? latestEvent.title : "准备研究"}</span>
                  </div>
                  <div className="assistant-answer">
                    {error && (
                      <div className="error-box">
                        <AlertCircle size={16} />
                        {error}
                      </div>
                    )}

                    {visibleEvents.length === 0 && isBusy && (
                      <p>
                        <StreamingText text="研究任务已经提交，正在等待后端事件流返回进度。" />
                      </p>
                    )}

                    {visibleEvents.length > 0 && (
                      <div className="research-card">
                        <header>
                          <strong>研究进度</strong>
                          <span className={`pill ${isBusy ? "good" : ""}`}>{statusLabels[runStatus]}</span>
                        </header>
                        <ul className="step-list">
                          {visibleEvents.map((event) => (
                            <li key={event.id}>
                              <span className={`status-dot ${event.type.includes("failed") ? "danger" : ""}`}>
                                {eventIcon(event.type)}
                              </span>
                              <span>
                                <b>{event.title}</b>
                                {getEventText(event) && <small>{getEventText(event)}</small>}
                              </span>
                              <span className="time">{formatTime(event.created_at)}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {planEvents.map((event) => (
                      <div className="event-detail" key={`detail-${event.id}`}>
                        <div className="event-title-row">
                          <h3>{event.title}</h3>
                          <time>{formatTime(event.created_at)}</time>
                        </div>
                        {renderEventDetails(event)}
                      </div>
                    ))}

                    {showClarificationBox && (
                      <form className="clarification-box" onSubmit={sendClarification}>
                        <strong>需要补充信息</strong>
                        <p>{clarificationQuestion}</p>
                        <textarea
                          value={clarification}
                          onChange={(event) => setClarification(event.target.value)}
                          placeholder="补充研究范围、地区、时间、输出格式等要求。"
                        />
                        <button className="action-button primary" type="submit" disabled={!clarification.trim()}>
                          <Send size={16} />
                          继续研究
                        </button>
                      </form>
                    )}

                    {finalReport && (
                      <div className="report-card">
                        <header>
                          <strong>最终报告</strong>
                          <div className="actions compact">
                            <button className="action-button" type="button" onClick={copyReport}>
                              <Clipboard size={15} />
                              {copied ? "已复制" : "复制"}
                            </button>
                            <button className="action-button" type="button" onClick={downloadReport}>
                              <Download size={15} />
                              下载
                            </button>
                          </div>
                        </header>
                        <div className="report-body">
                          <ReactMarkdown>{finalReport}</ReactMarkdown>
                        </div>
                      </div>
                    )}

                    {!finalReport && visibleEvents.length > 0 && (
                      <div className="confidence">
                        <div className="metric">
                          <span>阶段事件</span>
                          <strong>{visibleEvents.length}</strong>
                        </div>
                        <div className="metric">
                          <span>已完成节点</span>
                          <strong>{completedEvents.length}</strong>
                        </div>
                        <div className="metric">
                          <span>来源/工具调用</span>
                          <strong>{sourceEvents.length}</strong>
                        </div>
                      </div>
                    )}

                    <div className="actions">
                      <button className="action-button primary" type="button" disabled={!finalReport} onClick={downloadReport}>
                        <FileText size={16} />
                        导出报告
                      </button>
                      <button className="action-button" type="button" disabled={!isBusy} onClick={cancelRun}>
                        <Square size={16} />
                        停止研究
                      </button>
                    </div>
                  </div>
                </div>
              </article>
            )}
          </section>
        </div>

        <div className="composer-wrap">
          <form className="composer" onSubmit={startRun}>
            <div className="mode-strip" role="tablist" aria-label="研究模式">
              <button type="button" className="mode active">
                <Search size={14} />
                深度研究
              </button>
              <button type="button" className="mode">
                <BarChart3 size={14} />
                市场分析
              </button>
              <button type="button" className="mode">
                <FileText size={14} />
                报告草稿
              </button>
              <button type="button" className="mode">
                <ShieldCheck size={14} />
                反方审查
              </button>
            </div>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="输入研究问题，或粘贴资料让助手归纳、核验、追踪来源..."
            />
            <div className="composer-bottom">
              <div className="composer-tools">
                <button className="icon-button" type="button" title="上传资料" aria-label="上传资料">
                  <Plus size={17} />
                </button>
                <button className="icon-button" type="button" title="添加网页来源" aria-label="添加网页来源">
                  <Link size={17} />
                </button>
                <button className="icon-button" type="button" title="限定时间范围" aria-label="限定时间范围">
                  <Calendar size={17} />
                </button>
                <span className="pill">中文 · 引用优先</span>
              </div>
              <button className="send" type="submit" disabled={isBusy || !prompt.trim()} title="发送" aria-label="发送">
                {isBusy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              </button>
            </div>
          </form>
        </div>
      </section>

      <aside className="inspector" aria-label="证据与任务">
        <div className="inspector-top">
          <div className="inspector-title">
            <strong>证据面板</strong>
            <span>事件、来源与输出状态</span>
          </div>
          <button className="icon-button" type="button" title="筛选来源" aria-label="筛选来源">
            <Filter size={17} />
          </button>
        </div>

        <div className="inspector-scroll">
          <section className="panel-block">
            <div className="panel-head">
              <strong>关键来源</strong>
              <span className="pill">{sourceEvents.length}</span>
            </div>
            <div className="panel-body">
              {sourceEvents.length === 0 && <div className="empty-mini">搜索或工具调用会显示在这里。</div>}
              {sourceEvents.map((event, index) => (
                <div className="source" key={event.id}>
                  <span>
                    <strong>{event.title}</strong>
                    <span>{getEventText(event) || event.type}</span>
                  </span>
                  <span className="source-score">{Math.max(62, 92 - index * 5)}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel-block">
            <div className="panel-head">
              <strong>研究任务</strong>
              <span className={`pill ${completedEvents.length ? "good" : ""}`}>
                {completedEvents.length}/{Math.max(visibleEvents.length, 1)}
              </span>
            </div>
            <div className="panel-body">
              {visibleEvents.length === 0 && <div className="empty-mini">提交问题后会同步任务进度。</div>}
              {visibleEvents.slice(-6).map((event) => (
                <div className={`task ${event.type.includes("completed") || event.type.includes("created") ? "done" : ""}`} key={`task-${event.id}`}>
                  <span className="task-icon">{eventIcon(event.type)}</span>
                  <span>
                    <strong>{event.title}</strong>
                    <span>{getEventText(event) || formatTime(event.created_at)}</span>
                  </span>
                </div>
              ))}
            </div>
            <div className="note">右侧面板把研究过程和证据线索拆出来，长报告生成时也能快速扫描当前状态。</div>
          </section>

          <section className="panel-block">
            <div className="panel-head">
              <strong>输出格式</strong>
              <span className="pill">Markdown</span>
            </div>
            <div className="panel-body output-actions">
              <button className="action-button primary" type="button" disabled={!finalReport} onClick={downloadReport}>
                <FileText size={16} />
                完整报告
              </button>
              <button className="action-button" type="button" disabled={!finalReport} onClick={copyReport}>
                <Clipboard size={16} />
                复制 Markdown
              </button>
              <button className="action-button" type="button" disabled={!isBusy} onClick={cancelRun}>
                <Square size={16} />
                停止当前任务
              </button>
            </div>
          </section>
        </div>
      </aside>

      {copied && <div className="copy-toast">已复制</div>}
    </main>
  );
}
