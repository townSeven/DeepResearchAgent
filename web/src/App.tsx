import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import {
  AlertCircle,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clipboard,
  Download,
  FileText,
  Loader2,
  Search,
  Send,
  Settings2,
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
  searchApi: "minimax_mcp",
  allowClarification: true,
  researchModel: "minimax:MiniMax-M2.7",
  summarizationModel: "deepseek:deepseek-v4-flash",
  compressionModel: "minimax:MiniMax-M2.7",
  finalReportModel: "minimax:MiniMax-M2.7",
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
  if (type.includes("search")) return <Search size={16} />;
  if (type.includes("report")) return <FileText size={16} />;
  if (type.includes("failed")) return <AlertCircle size={16} />;
  if (type.includes("completed") || type.includes("created")) return <CheckCircle2 size={16} />;
  return <Brain size={16} />;
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function getClarificationQuestion(events: ProgressEvent[]) {
  const event = [...events].reverse().find((item) => item.type === "clarification_required");
  return (event?.data?.question as string | undefined) || event?.message || "";
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
  const [clarification, setClarification] = useState("");
  const [config, setConfig] = useState<ConfigState>(defaultConfig);
  const [configOpen, setConfigOpen] = useState(false);
  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [finalReport, setFinalReport] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);

  const isBusy = run?.status === "queued" || run?.status === "running";
  const clarificationQuestion = useMemo(() => getClarificationQuestion(events), [events]);
  const visibleEvents = useMemo(
    () => events.filter((event) => !hiddenTimelineEvents.has(event.type)),
    [events],
  );

  useEffect(() => {
    timelineRef.current?.scrollTo({
      top: timelineRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [events.length]);

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
    const text =
      event.message ||
      (event.data?.question as string | undefined) ||
      (event.data?.verification as string | undefined) ||
      (event.data?.research_brief as string | undefined) ||
      (event.data?.reflection as string | undefined);

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

    setError("");
    setEvents([]);
    setFinalReport("");
    setClarification("");
    try {
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: prompt.trim(),
          config: toApiConfig(config),
        }),
      });

      if (!response.ok) {
        setError(await response.text());
        return;
      }

      const snapshot = (await response.json()) as RunSnapshot;
      setRun(snapshot);
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
    <main className="app-shell">
      <section className="control-pane">
        <div className="brand-row">
          <div>
            <h1>Deep Research Agent</h1>
            <p>交互式深度研究工作台</p>
          </div>
          <span className={`status-pill ${run?.status || "idle"}`}>
            {run?.status || "idle"}
          </span>
        </div>

        <form className="prompt-form" onSubmit={startRun}>
          <label htmlFor="prompt">研究问题</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="例如：调研 2026 年企业级 AI Agent 平台的主要技术趋势和代表产品。"
          />
          <div className="form-actions">
            <button className="primary-button" type="submit" disabled={isBusy || !prompt.trim()}>
              {isBusy ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
              开始研究
            </button>
            <button className="icon-button" type="button" disabled={!isBusy} onClick={cancelRun} title="停止研究">
              <Square size={16} />
            </button>
          </div>
        </form>

        <div className="settings-panel">
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

        {run?.status === "requires_clarification" && (
          <form className="clarification-box" onSubmit={sendClarification}>
            <strong>需要补充信息</strong>
            <p>{clarificationQuestion}</p>
            <textarea
              value={clarification}
              onChange={(event) => setClarification(event.target.value)}
              placeholder="补充研究范围、地区、时间、输出格式等要求。"
            />
            <button className="primary-button" type="submit" disabled={!clarification.trim()}>
              <Send size={16} />
              继续研究
            </button>
          </form>
        )}

        {error && (
          <div className="error-box">
            <AlertCircle size={16} />
            {error}
          </div>
        )}
      </section>

      <section className="timeline-pane">
        <div className="section-header">
          <div>
            <h2>研究进度</h2>
            <p>{visibleEvents.length ? `${visibleEvents.length} 条事件` : "等待开始"}</p>
          </div>
          {isBusy && <Loader2 className="spin subtle-loader" size={20} />}
        </div>
        <div className="timeline" ref={timelineRef}>
          {visibleEvents.length === 0 && <div className="empty-state">提交问题后，这里会实时显示计划、搜索、整理和报告生成状态。</div>}
          {visibleEvents.map((event) => (
            <article className={`timeline-item ${event.type}`} key={event.id}>
              <div className="event-icon">{eventIcon(event.type)}</div>
              <div className="event-body">
                <div className="event-title-row">
                  <h3>{event.title}</h3>
                  <time>{formatTime(event.created_at)}</time>
                </div>
                {renderEventDetails(event)}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="report-pane">
        <div className="section-header">
          <div>
            <h2>最终报告</h2>
            <p>{finalReport ? "Markdown 已生成" : "报告会在研究完成后出现"}</p>
          </div>
          <div className="report-actions">
            <button className="icon-button" type="button" disabled={!finalReport} onClick={copyReport} title="复制报告">
              <Clipboard size={16} />
            </button>
            <button className="icon-button" type="button" disabled={!finalReport} onClick={downloadReport} title="下载 Markdown">
              <Download size={16} />
            </button>
          </div>
        </div>
        {copied && <div className="copy-toast">已复制</div>}
        <div className="report-body">
          {finalReport ? <ReactMarkdown>{finalReport}</ReactMarkdown> : <div className="empty-state">研究完成后会在这里渲染完整报告。</div>}
        </div>
      </section>
    </main>
  );
}
