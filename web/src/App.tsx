import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
  Trash2,
} from "lucide-react";

type RunStatus =
  | "queued"
  | "running"
  | "requires_clarification"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

type RunMessage = {
  type: string;
  content: string;
};

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
  title?: string;
  status: RunStatus;
  messages?: RunMessage[];
  events: ProgressEvent[];
  final_report?: string | null;
  research_brief?: string | null;
  last_follow_up_answer?: string | null;
  error?: string | null;
  language?: "zh" | "en";
  updated_at?: string;
};

type RunSummary = Pick<RunSnapshot, "id" | "title" | "status" | "final_report" | "updated_at">;

type PaperInfo = {
  document_id: string;
  file_name: string;
  chunk_count: number;
};

type PaperIngestionSummary = {
  ingested: PaperInfo[];
  skipped: PaperInfo[];
  failed: { file_name: string; error: string }[];
};

type ConfigState = {
  searchApi: string;
  allowClarification: boolean;
  reuseHistoricalResearch: boolean;
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
  reuseHistoricalResearch: false,
  researchModel: "deepseek:deepseek-v4-flash",
  summarizationModel: "deepseek:deepseek-v4-flash",
  compressionModel: "deepseek:deepseek-v4-flash",
  finalReportModel: "deepseek:deepseek-v4-flash",
  maxConcurrentResearchUnits: 5,
  maxResearcherIterations: 6,
  maxReactToolCalls: 10,
};

const terminalStatuses: RunStatus[] = ["completed", "failed", "cancelled", "interrupted"];
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
  interrupted: "已中断",
};

function toApiConfig(config: ConfigState) {
  return {
    search_api: config.searchApi,
    allow_clarification: config.allowClarification,
    research_history_enabled: config.reuseHistoricalResearch,
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

function getSubResearchTitle(topic: string) {
  const bracketTitle = topic.match(/^【([^】]+)】/)?.[1] || topic.match(/^\[([^\]]+)\]/)?.[1];
  if (bracketTitle) return bracketTitle.trim();
  const firstClause = topic.split(/[。！？.!?\n]/)[0]?.trim();
  return firstClause.length > 80 ? `${firstClause.slice(0, 80)}...` : firstClause || topic;
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function splitMarkdownTableCells(line: string) {
  const trimmed = line.trim();
  const withoutLeadingPipe = trimmed.startsWith("|") ? trimmed.slice(1) : trimmed;
  const withoutOuterPipes = withoutLeadingPipe.endsWith("|") ? withoutLeadingPipe.slice(0, -1) : withoutLeadingPipe;
  return withoutOuterPipes.split("|").map((cell) => cell.trim());
}

function isMarkdownTableSeparator(line: string) {
  const cells = splitMarkdownTableCells(line);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function isMarkdownTableRow(line: string) {
  const trimmed = line.trim();
  if (!trimmed || /^#{1,6}\s/.test(trimmed) || /^[-*+]\s/.test(trimmed)) return false;
  return splitMarkdownTableCells(trimmed).length > 1;
}

function formatMarkdownTableRow(cells: string[]) {
  const normalizedCells = cells.map((cell, index) => (index === 0 ? cell.replace(/^[,，]\s*/, "") : cell));
  return `| ${normalizedCells.map((cell) => cell || " ").join(" | ")} |`;
}

function startsWithRatingCell(cells: string[]) {
  return Boolean(cells[0]?.trim().startsWith("⭐"));
}

function findNextNonEmptyLineIndex(lines: string[], startIndex: number) {
  for (let index = startIndex; index < lines.length; index += 1) {
    if (lines[index].trim()) return index;
  }
  return -1;
}

function pushPaddedTableRow(rows: string[], cells: string[], columnCount: number) {
  if (!cells.length) return;
  rows.push(formatMarkdownTableRow([...cells, ...Array.from({ length: Math.max(0, columnCount - cells.length) }, () => "")]));
  cells.length = 0;
}

function normalizeMarkdownTables(markdown: string) {
  const lines = markdown.split("\n");
  const normalized: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const nextLine = lines[index + 1];
    if (!isMarkdownTableRow(line) || !nextLine || !isMarkdownTableSeparator(nextLine)) {
      normalized.push(line);
      continue;
    }

    const headerCells = splitMarkdownTableCells(line);
    const columnCount = headerCells.length;
    normalized.push(formatMarkdownTableRow(headerCells));
    normalized.push(formatMarkdownTableRow(Array.from({ length: columnCount }, () => "---")));
    index += 1;

    const pendingCells: string[] = [];
    while (index + 1 < lines.length) {
      const candidate = lines[index + 1];
      if (!candidate.trim()) {
        const nextIndex = findNextNonEmptyLineIndex(lines, index + 2);
        const nextCandidate = nextIndex >= 0 ? lines[nextIndex] : "";
        if (pendingCells.length) {
          pushPaddedTableRow(normalized, pendingCells, columnCount);
        }
        if (nextCandidate && (isMarkdownTableRow(nextCandidate) || isMarkdownTableRow(lines[findNextNonEmptyLineIndex(lines, nextIndex + 1)] || ""))) {
          index += 1;
          continue;
        }
        break;
      }
      if (!isMarkdownTableRow(candidate)) {
        const nextIndex = findNextNonEmptyLineIndex(lines, index + 2);
        const nextCandidate = nextIndex >= 0 ? lines[nextIndex] : "";
        if (nextCandidate && isMarkdownTableRow(nextCandidate)) {
          pendingCells.push(candidate.trim());
          index += 1;
          continue;
        }
        if (pendingCells.length) {
          pushPaddedTableRow(normalized, pendingCells, columnCount);
          index += 1;
          continue;
        }
        break;
      }
      if (isMarkdownTableSeparator(candidate)) {
        index += 1;
        continue;
      }

      const cells = splitMarkdownTableCells(candidate);
      if (!pendingCells.length && startsWithRatingCell(cells) && cells.length <= columnCount - 2) {
        pendingCells.push(...Array.from({ length: columnCount - cells.length }, () => ""));
      }
      if (pendingCells.length === 1 && startsWithRatingCell(cells) && cells.length <= columnCount - 2) {
        pendingCells.push(...Array.from({ length: columnCount - 1 - cells.length }, () => ""));
      }
      pendingCells.push(...cells);
      index += 1;

      while (pendingCells.length >= columnCount) {
        if (!pendingCells[0] && pendingCells.length > columnCount) {
          pendingCells.shift();
          continue;
        }
        normalized.push(formatMarkdownTableRow(pendingCells.splice(0, columnCount)));
      }
    }

    if (pendingCells.length >= Math.ceil(columnCount / 2)) {
      normalized.push(formatMarkdownTableRow([...pendingCells, ...Array.from({ length: columnCount - pendingCells.length }, () => "")]));
    }
  }

  return normalized.join("\n");
}

function normalizeStreamingMarkdown(markdown: string) {
  let text = markdown.replace(/\r\n?/g, "\n");

  // Streaming model output often arrives as "text## Heading" or "#Heading".
  text = text.replace(/([^\n#\\])(?=#{2,6}(?!#))/g, "$1\n\n");
  text = text.replace(/^(#{1,6})(?=[^\s#])/gm, "$1 ");
  text = text.replace(/^(#{1,6}\s+)\\?#\s*(.+)$/gm, "$1$2");
  text = text.replace(/^\\?#{1,6}\s*([一二三四五六七八九十]+、\s*.+)$/gm, "## $1");
  text = text.replace(/^\\?#{1,6}\s*(\d+(?:\.\d+)+\s*.+)$/gm, "### $1");
  text = text.replace(/^\\(#{1,6}\s+.+)$/gm, "$1");

  // Keep headings and following tables as separate Markdown blocks.
  text = text.replace(/^(#{1,6}\s[^\n|]+?)\s+(\|[^\n]+\|)$/gm, "$1\n\n$2");
  text = text.replace(/\|\s*\|\s*(?=[\p{L}\p{N}_.,，、(（【[][^|\n]{0,100}\|)/gu, "|\n| ");

  return normalizeMarkdownTables(text);
}

function MarkdownContent({ children }: { children: string }) {
  const normalized = useMemo(() => normalizeStreamingMarkdown(children), [children]);
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalized}</ReactMarkdown>;
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

type StageState = "waiting" | "running" | "completed" | "failed";

type ResearchStage = {
  title: string;
  description: string;
  state: StageState;
};

type AgentState = {
  id: string;
  name: string;
  role: string;
  state: StageState;
};

const stageDefinitions = [
  {
    title: "明确研究目标",
    description: "理解问题、确认范围并形成研究计划。",
    active: ["run_started", "clarifying", "research_brief_started"],
    done: ["research_brief_created", "research_plan_created"],
  },
  {
    title: "收集 Web 信息",
    description: "并行检索公开网页与实时资料。",
    active: ["search_started"],
    done: ["search_completed", "research_phase_completed"],
  },
  {
    title: "检索私有论文库",
    description: "联合本地论文和专业资料进行检索。",
    active: ["tool_started", "tool_batch_started"],
    done: ["tool_completed", "research_phase_completed"],
  },
  {
    title: "聚合与压缩证据",
    description: "整理研究笔记并提炼关键证据。",
    active: ["researcher_ready_to_compress", "compressing_research"],
    done: ["research_compressed", "research_phase_completed"],
  },
  {
    title: "生成报告",
    description: "综合研究结论并生成最终报告。",
    active: ["writing_final_report", "final_report_delta"],
    done: ["final_report_created", "run_completed"],
  },
];

function deriveStages(events: ProgressEvent[], runStatus: RunStatus | "idle"): ResearchStage[] {
  const types = new Set(events.map((event) => event.type));
  const failed = runStatus === "failed";
  let activeFound = false;

  return stageDefinitions.map((stage, index) => {
    const completed = stage.done.some((type) => types.has(type));
    const active = stage.active.some((type) => types.has(type));
    let state: StageState = completed ? "completed" : active ? "running" : "waiting";

    if (runStatus === "completed") state = "completed";
    if (failed && !activeFound && (active || (!completed && index === 0))) state = "failed";
    if (state === "running" || state === "failed") activeFound = true;
    return { title: stage.title, description: stage.description, state };
  });
}

function deriveAgents(events: ProgressEvent[], runStatus: RunStatus | "idle"): AgentState[] {
  const agents = new Map<string, AgentState>();
  const supervisorRunning = events.some((event) =>
    ["supervisor_planning", "research_plan_created", "research_tasks_started"].includes(event.type),
  );
  agents.set("supervisor", {
    id: "supervisor",
    name: "Supervisor",
    role: "任务与调度",
    state: runStatus === "completed" ? "completed" : supervisorRunning ? "running" : "waiting",
  });

  for (const event of events) {
    const topic = event.data?.topic as string | undefined;
    const index = event.data?.index as number | undefined;
    if (index === undefined && !topic) continue;
    const id = `researcher-${topic || index}`;
    const previous = agents.get(id);
    let state = previous?.state || "waiting";
    if (event.type === "research_task_started" || event.type === "researcher_thinking" || event.type === "search_started" || event.type === "compressing_research") {
      state = "running";
    }
    if (event.type === "research_task_completed" || event.type === "research_compressed") state = "completed";
    if (event.type.includes("failed")) state = "failed";
    agents.set(id, {
      id,
      name: previous?.name || `Researcher ${(index ?? agents.size - 1) + 1}`,
      role: topic || previous?.role || "研究任务",
      state,
    });
  }

  return [...agents.values()];
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
  const [papers, setPapers] = useState<PaperInfo[]>([]);
  const [papersLoading, setPapersLoading] = useState(true);
  const [papersUploading, setPapersUploading] = useState(false);
  const [paperStatus, setPaperStatus] = useState("");
  const [runHistory, setRunHistory] = useState<RunSummary[]>([]);
  const [historyQuery, setHistoryQuery] = useState("");
  const [deletingRunId, setDeletingRunId] = useState("");
  const [expandedStages, setExpandedStages] = useState<Set<number>>(new Set());
  const [now, setNow] = useState(() => Date.now());
  const sourceRef = useRef<EventSource | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);
  const paperInputRef = useRef<HTMLInputElement | null>(null);

  const isBusy = run?.status === "queued" || run?.status === "running";
  const canCancel = Boolean(run && !terminalStatuses.includes(run.status));
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
  const planEvents = visibleEvents.filter((event) => event.type === "research_plan_created");
  const sourceEvents = events.filter((event) => event.type.includes("search") || event.type.includes("tool")).slice(-5);
  const completedEvents = visibleEvents.filter((event) => event.type.includes("completed") || event.type.includes("created"));
  const displayPrompt = submittedPrompt;
  const conversationTitle = run?.title || submittedPrompt || "准备开始新的深度研究";
  const reportReady = Boolean(finalReport);
  const latestFollowUp = run?.last_follow_up_answer;
  const stages = useMemo(() => deriveStages(events, runStatus), [events, runStatus]);
  const agents = useMemo(() => deriveAgents(events, runStatus), [events, runStatus]);
  const completedSubResearchTopics = useMemo(
    () =>
      new Set(
        events
          .filter((event) => event.type === "research_task_completed")
          .map((event) => event.data?.topic as string | undefined)
          .filter((topic): topic is string => Boolean(topic)),
      ),
    [events],
  );
  const filteredHistory = useMemo(
    () => runHistory.filter((item) => (item.title || "").toLowerCase().includes(historyQuery.trim().toLowerCase())),
    [historyQuery, runHistory],
  );
  const progress = useMemo(() => {
    if (runStatus === "completed") return 100;
    const completed = stages.filter((stage) => stage.state === "completed").length;
    const running = stages.some((stage) => stage.state === "running");
    return Math.min(95, Math.round(((completed + (running ? 0.55 : 0)) / stages.length) * 100));
  }, [runStatus, stages]);
  const currentStage = stages.find((stage) => stage.state === "running" || stage.state === "failed");
  const elapsedEnd = isBusy || !run?.updated_at ? now : new Date(run.updated_at).getTime();
  const elapsed = events[0]?.created_at ? Math.max(0, Math.floor((elapsedEnd - new Date(events[0].created_at).getTime()) / 1000)) : 0;
  const elapsedLabel = `${String(Math.floor(elapsed / 3600)).padStart(2, "0")}:${String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;

  useEffect(() => {
    if (!isBusy) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [isBusy]);

  useEffect(() => {
    threadRef.current?.scrollTo({
      top: threadRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [visibleEvents.length, finalReport]);

  useEffect(() => {
    return () => sourceRef.current?.close();
  }, []);

  useEffect(() => {
    void refreshPapers();
    void refreshRunHistory();
  }, []);

  async function refreshRunHistory() {
    try {
      const response = await fetch("/api/runs");
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as { runs: RunSummary[] };
      setRunHistory(data.runs);
    } catch (error) {
      setError(`无法读取历史任务：${getErrorMessage(error)}`);
    }
  }

  async function refreshPapers() {
    setPapersLoading(true);
    try {
      const response = await fetch("/api/knowledge/papers");
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as { papers: PaperInfo[] };
      setPapers(data.papers);
    } catch (error) {
      setPaperStatus(`无法读取论文库：${getErrorMessage(error)}`);
    } finally {
      setPapersLoading(false);
    }
  }

  async function uploadPapers(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (!files.length || papersUploading) return;

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    setPapersUploading(true);
    setPaperStatus(`正在解析并入库 ${files.length} 篇论文…`);
    try {
      const response = await fetch("/api/knowledge/papers", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(await response.text());
      const result = (await response.json()) as PaperIngestionSummary;
      setPaperStatus(
        `已入库 ${result.ingested.length}，跳过重复 ${result.skipped.length}，失败 ${result.failed.length}`,
      );
      await refreshPapers();
    } catch (error) {
      setPaperStatus(`论文入库失败：${getErrorMessage(error)}`);
    } finally {
      setPapersUploading(false);
    }
  }

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
      "follow_up_completed",
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
          void refreshRun(runId);
        }
        if (type === "follow_up_completed") {
          source.close();
          void refreshRun(runId);
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
          {(event.data?.topics as string[]).map((topic, index) => {
            const completed = runStatus === "completed" || completedSubResearchTopics.has(topic);
            return (
              <li key={`${event.id}-${index}`}>
                <span className={`subtask-progress ${completed ? "done" : "running"}`} aria-label={completed ? "已完成" : "进行中"}>
                  {completed && <CheckCircle2 size={15} />}
                </span>
                <StreamingText text={getSubResearchTitle(topic)} />
              </li>
            );
          })}
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
      setSubmittedPrompt(snapshot.messages?.find((message) => message.type === "human")?.content || "");
      await refreshRunHistory();
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
      await refreshRunHistory();
    } catch (error) {
      setError(`无法创建研究任务：${getErrorMessage(error)}。请确认 FastAPI 后端正在 http://127.0.0.1:8000 运行。`);
    }
  }

  async function sendFollowUp(event: FormEvent) {
    event.preventDefault();
    if (!run || !prompt.trim() || isBusy) return;
    const message = prompt.trim();
    setPrompt("");
    setError("");
    try {
      const response = await fetch(`/api/runs/${run.id}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, mode: "auto" }),
      });
      if (!response.ok) throw new Error(await response.text());
      const snapshot = (await response.json()) as RunSnapshot;
      setRun(snapshot);
      attachEvents(run.id);
    } catch (error) {
      setError(`无法继续研究任务：${getErrorMessage(error)}`);
    }
  }

  async function retryRun() {
    if (!run || run.status !== "interrupted") return;
    try {
      const response = await fetch(`/api/runs/${run.id}/retry`, { method: "POST" });
      if (!response.ok) throw new Error(await response.text());
      setRun((await response.json()) as RunSnapshot);
      attachEvents(run.id);
    } catch (error) {
      setError(`无法恢复研究任务：${getErrorMessage(error)}`);
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

  async function deleteHistoryRun(runId: string, title?: string) {
    if (deletingRunId) return;
    const confirmed = window.confirm(`删除“${title || "未命名研究"}”？此操作会移除该研究会话和已保存报告。`);
    if (!confirmed) return;

    setDeletingRunId(runId);
    setError("");
    try {
      const response = await fetch(`/api/runs/${runId}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await response.text());
      if (run?.id === runId) {
        sourceRef.current?.close();
        setRun(null);
        setEvents([]);
        setFinalReport("");
        setSubmittedPrompt("");
        setAnsweredClarificationKey("");
      }
      setRunHistory((current) => current.filter((item) => item.id !== runId));
    } catch (error) {
      setError(`无法删除历史研究：${getErrorMessage(error)}`);
    } finally {
      setDeletingRunId("");
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
            <strong>Deep Research Agent</strong>
            <span>智能研究工作台</span>
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
          新建任务
        </button>

        <label className="search-box">
          <Search size={15} />
          <input type="search" value={historyQuery} onChange={(event) => setHistoryQuery(event.target.value)} placeholder="搜索历史研究" />
        </label>

        <div className="section-label">历史研究</div>
        <div className="chat-list">
          {filteredHistory.map((item) => (
            <div
              className={`chat-item ${run?.id === item.id ? "active" : ""}`}
              key={item.id}
            >
              <button
                className="chat-item-main"
                type="button"
                onClick={() => {
                  sourceRef.current?.close();
                  void refreshRun(item.id);
                  if (item.status === "queued" || item.status === "running") attachEvents(item.id);
                }}
              >
                <span className="chat-dot"><FileText size={14} /></span>
                <span>
                  <strong>{item.title || "未命名研究"}</strong>
                  <span>{statusLabels[item.status]}</span>
                </span>
                <span className="time">{formatShortTime(item.updated_at)}</span>
              </button>
              <button
                className="chat-delete"
                type="button"
                disabled={deletingRunId === item.id}
                title="删除历史研究"
                aria-label={`删除历史研究：${item.title || "未命名研究"}`}
                onClick={() => void deleteHistoryRun(item.id, item.title)}
              >
                {deletingRunId === item.id ? <Loader2 className="spin" size={14} /> : <Trash2 size={14} />}
              </button>
            </div>
          ))}
          {!filteredHistory.length && <span className="paper-empty">暂无匹配的历史研究。</span>}
        </div>

        <section className="paper-library" aria-labelledby="paper-library-title">
          <div className="paper-library-head">
            <div>
              <strong id="paper-library-title">论文库</strong>
              <span>{papers.length} 篇已入库</span>
            </div>
            <button
              className="paper-upload-button"
              type="button"
              disabled={papersUploading}
              onClick={() => paperInputRef.current?.click()}
            >
              {papersUploading ? <Loader2 className="spin small-icon" /> : <Plus size={14} />}
              上传
            </button>
            <input
              ref={paperInputRef}
              className="visually-hidden"
              type="file"
              accept="application/pdf,.pdf"
              multiple
              onChange={uploadPapers}
            />
          </div>
          <div className="paper-list" aria-live="polite">
            {papersLoading && <span className="paper-empty">正在读取论文库…</span>}
            {!papersLoading && !papers.length && (
              <span className="paper-empty">上传本地论文后，研究员可联合私有证据与公开资料。</span>
            )}
            {papers.map((paper) => (
              <div className="paper-item" key={paper.document_id} title={paper.file_name}>
                <FileText size={14} />
                <span>{paper.file_name}</span>
                <small>{paper.chunk_count} chunks</small>
              </div>
            ))}
          </div>
          {paperStatus && <p className="paper-status" aria-live="polite">{paperStatus}</p>}
          <p className="paper-privacy">论文片段与检索问题会发送至阿里云百炼生成向量。</p>
        </section>

        <div className={`settings-panel side-settings ${configOpen ? "open" : ""}`}>
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
              <label className="checkbox-line">
                <input
                  type="checkbox"
                  checked={config.reuseHistoricalResearch}
                  onChange={(event) => setConfig({ ...config, reuseHistoricalResearch: event.target.checked })}
                />
                复用历史研究成果
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
                          <span className={`pill ${isBusy ? "good" : ""}`}>{progress}%</span>
                        </header>
                        <div className="stage-stack">
                          {stages.map((stage, index) => {
                            const definition = stageDefinitions[index];
                            const related = events.filter((event) =>
                              [...definition.active, ...definition.done].includes(event.type),
                            );
                            const expanded = expandedStages.has(index);
                            return (
                              <div className={`stage-row ${stage.state}`} key={stage.title}>
                                <button type="button" onClick={() => setExpandedStages((current) => {
                                  const next = new Set(current);
                                  next.has(index) ? next.delete(index) : next.add(index);
                                  return next;
                                })}>
                                  <span className="stage-number">
                                    {stage.state === "completed" ? <CheckCircle2 size={17} /> : stage.state === "running" ? <Loader2 className="spin" size={17} /> : index + 1}
                                  </span>
                                  <span><b>{stage.title}</b><small>{stage.description}</small></span>
                                  <time>{formatShortTime(related[related.length - 1]?.created_at)}</time>
                                  {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                </button>
                                {expanded && related.length > 0 && (
                                  <ul className="step-list">
                                    {related.slice(-6).map((event) => (
                                      <li key={event.id}>
                                        <span className={`status-dot ${event.type.includes("failed") ? "danger" : ""}`}>{eventIcon(event.type)}</span>
                                        <span><b>{event.title}</b>{getEventText(event) && <small>{getEventText(event)}</small>}</span>
                                        <span className="time">{formatTime(event.created_at)}</span>
                                      </li>
                                    ))}
                                  </ul>
                                )}
                              </div>
                            );
                          })}
                        </div>
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
                          <MarkdownContent>{finalReport}</MarkdownContent>
                        </div>
                      </div>
                    )}

                    {latestFollowUp && (
                      <div className="event-detail">
                        <div className="event-title-row"><h3>追问回答</h3></div>
                        <MarkdownContent>{latestFollowUp}</MarkdownContent>
                      </div>
                    )}

                    {run?.status === "interrupted" && (
                      <div className="clarification-box">
                        <strong>研究任务已中断</strong>
                        <p>该任务不会自动恢复。手动恢复会优先从最近的研究检查点继续。</p>
                        <button className="action-button primary" type="button" onClick={retryRun}>
                          继续恢复
                        </button>
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
                        下载报告
                      </button>
                      <button className="action-button" type="button" disabled={!canCancel} onClick={cancelRun}>
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
          <form className="composer" onSubmit={run?.status === "completed" ? sendFollowUp : startRun}>
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
              placeholder={run?.status === "completed" ? "继续追问；涉及最新信息时会自动启动补充研究..." : "输入研究问题，或粘贴资料让助手归纳、核验、追踪来源..."}
            />
            <div className="composer-bottom">
              <div className="composer-tools">
                <button className="icon-button" type="button" title="上传论文" aria-label="上传论文" onClick={() => paperInputRef.current?.click()}>
                  <Plus size={17} />
                </button>
                <button className={`model-config-button ${configOpen ? "active" : ""}`} type="button" onClick={() => setConfigOpen((value) => !value)}>
                  <Settings2 size={15} />
                  模型配置
                  <ChevronDown size={14} />
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

      <aside className="inspector" aria-label="计划与运行状态">
        <div className="inspector-scroll">
          {(submittedPrompt || run || events.length > 0) && (
            <section className="panel-block plan-panel">
              <div className="panel-head">
                <strong><Brain size={16} />计划</strong>
                <ChevronUp size={16} />
              </div>
              <ol className="plan-list">
                {stages.map((stage, index) => (
                  <li className={stage.state} key={stage.title}>
                    <span>{stage.state === "completed" ? <CheckCircle2 size={15} /> : index + 1}</span>
                    <div><strong>{stage.title}</strong><small>{stage.state === "completed" ? "已完成" : stage.state === "running" ? "进行中" : stage.state === "failed" ? "失败" : "等待中"}</small></div>
                  </li>
                ))}
              </ol>
            </section>
          )}

          <section className="panel-block runtime-panel">
            <div className="panel-head">
              <strong><ShieldCheck size={16} />运行状态</strong>
              <ChevronUp size={16} />
            </div>
            <div className="runtime-body">
              <div><span>当前阶段</span><strong className="current-stage">{currentStage?.title || statusLabels[runStatus]}</strong></div>
              <div className="progress-line"><span>总体进度</span><i><b style={{ width: `${progress}%` }} /></i><strong>{progress}%</strong></div>
              <div><span>活跃 Agent</span><strong>{agents.filter((agent) => agent.state === "running").length} / {agents.length}</strong></div>
              <div><span>已用时间</span><strong>{elapsedLabel}</strong></div>
              {canCancel && <button className="action-button" type="button" onClick={cancelRun}><Square size={14} />停止研究</button>}
              <p>研究过程中，可随时查看任务进度或停止当前任务。</p>
            </div>
          </section>
        </div>
      </aside>

      {copied && <div className="copy-toast">已复制</div>}
    </main>
  );
}
