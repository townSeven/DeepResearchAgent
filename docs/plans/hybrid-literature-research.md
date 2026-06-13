# Implementation Plan: Hybrid Literature Research

## Status

- Spec: `docs/specs/hybrid-literature-research.md`
- State: Tasks 1-10 implemented; Task 11 local evaluation runs pending
- Scope: MVP

## Overview

在现有 Deep Research Agent 中增加私有论文知识库。用户上传合法持有的文本型 PDF 后，系统通过 PyMuPDF 解析和切分，调用阿里云百炼 `qwen3-vl-embedding` API 生成默认维度向量，持久化到 ChromaDB。Researcher 获得 `search_private_papers` 工具后，可以自主联合私有论文与 Web Search 生成带页码引用的研究报告。

计划优先建立一条可运行的纵向链路：

```text
PDF bytes
  -> page-aware chunks
  -> qwen3-vl-embedding
  -> ChromaDB
  -> private-paper search result
  -> Agent tool
  -> hybrid research report
```

UI、评测和文档在核心链路验证后接入。

## Architecture Decisions

- 使用单个本地持久化 Chroma collection，collection 名固定，运行目录默认为 `.knowledge/private_papers`。
- 使用阿里云百炼 `qwen3-vl-embedding` API 默认向量维度；不在代码中写死维度。
- Embedding 访问封装为独立 adapter，不让 Chroma、Agent tool 或 API endpoint 依赖百炼响应结构。
- `document_id` 基于原始 PDF bytes 的 SHA-256；重复上传通过 `document_id` 判断。
- `chunk_id` 基于 `document_id + page_number + chunk_index + chunk_content` 生成稳定哈希。
- MVP 按页提取文本，并在每页内部切分，避免 chunk 跨页，从而保证页码引用明确。
- Chroma 同步调用和 PDF 解析在异步服务中通过线程卸载，避免阻塞 FastAPI 与研究任务事件循环。
- `search_private_papers` 作为普通 LangChain 搜索工具接入现有 `get_all_tools()`，不修改 LangGraph 状态结构和主图。
- 私有论文检索失败采用降级返回，不让整个研究 run 失败。
- MVP 不实现删除论文、OCR、Reranker、多知识库或权限控制。

## Dependency Graph

```text
Task 1: Config and dependencies
  |
  +--> Task 2: Paper parsing contracts
  |
  +--> Task 3: Qwen embedding adapter
          |
Task 2 + Task 3
          |
          +--> Task 4: Chroma storage and retrieval
                    |
                    +--> Task 5: Ingestion service
                    |         |
                    |         +--> Task 6: Knowledge API
                    |
                    +--> Task 7: Private-paper Agent tool
                              |
                              +--> Task 8: Agent registration and prompts

Task 6 --> Task 9: Upload and paper-list UI
Task 4 --> Task 10: Retrieval evaluation
Task 8 + Task 9 + Task 10 --> Task 11: End-to-end demo and documentation
```

## Task List

### Phase 1: Technical Foundation

## Task 1: Add knowledge-base configuration and dependencies

**Description:** Establish the feature's configuration boundary and install only the dependencies required by the approved MVP.

**Acceptance criteria:**

- [ ] `Configuration` exposes private-paper enablement, Top-K, Chroma path, embedding model, chunk settings and upload size limit with approved defaults.
- [ ] `DASHSCOPE_API_KEY` is documented in `.env.example`; `.knowledge/` is ignored by Git.
- [ ] ChromaDB and multipart upload dependencies install successfully without breaking current imports.

**Verification:**

- [ ] Run `uv sync`.
- [ ] Run `uv run python -c "from deep_research_agent.configuration import Configuration; print(Configuration())"`.
- [ ] Run `git status --short` and confirm `.knowledge/` would not be tracked.

**Dependencies:** None

**Files likely touched:**

- `pyproject.toml`
- `uv.lock`
- `.env.example`
- `.gitignore`
- `src/deep_research_agent/configuration.py`

**Estimated scope:** Medium

## Task 2: Define paper models and page-aware PDF parsing

**Description:** Build deterministic parsing and chunking from PDF bytes to typed paper chunks while retaining page-level citation metadata.

**Acceptance criteria:**

- [ ] Text PDF pages are extracted with 1-based page numbers; empty pages are skipped.
- [ ] Chunks stay within one page and respect configured size and overlap.
- [ ] Identical PDF bytes and chunk settings produce stable `document_id` and `chunk_id` values.
- [ ] Empty, unreadable and textless PDFs return explicit domain errors.

**Verification:**

- [ ] Run `uv run pytest tests/knowledge/test_parser.py -q`.
- [ ] Run `uv run ruff check src/deep_research_agent/knowledge/models.py src/deep_research_agent/knowledge/parser.py tests/knowledge/test_parser.py`.

**Dependencies:** Task 1

**Files likely touched:**

- `src/deep_research_agent/knowledge/__init__.py`
- `src/deep_research_agent/knowledge/models.py`
- `src/deep_research_agent/knowledge/parser.py`
- `tests/knowledge/test_parser.py`

**Estimated scope:** Medium

## Task 3: Build the Qwen3-VL Embedding adapter

**Description:** Isolate the Alibaba Model Studio API contract behind an async adapter supporting document batches and single-query embedding.

**Acceptance criteria:**

- [ ] Adapter sends text inputs to `qwen3-vl-embedding` using `DASHSCOPE_API_KEY` and does not specify a vector dimension.
- [ ] Adapter validates response count, vector shape consistency and empty vectors before returning plain `list[list[float]]`.
- [ ] API errors, timeouts and malformed responses become explicit embedding errors without exposing secrets.
- [ ] Tests mock HTTP behavior and cover batching, query embedding and failure paths.

**Verification:**

- [ ] Run `uv run pytest tests/knowledge/test_embedding.py -q`.
- [ ] Run `uv run ruff check src/deep_research_agent/knowledge/embedding.py tests/knowledge/test_embedding.py`.
- [ ] Optional manual smoke check with a real `DASHSCOPE_API_KEY`: embed one short query and print only vector length.

**Dependencies:** Task 1

**Files likely touched:**

- `src/deep_research_agent/knowledge/embedding.py`
- `tests/knowledge/test_embedding.py`

**Estimated scope:** Medium

### Checkpoint: Foundation

- [ ] Tasks 1-3 tests pass.
- [ ] Existing application imports still work.
- [ ] PDF input produces deterministic citable chunks.
- [ ] Mocked embedding contract is stable; optional real API smoke check confirms response compatibility.

### Phase 2: Searchable Knowledge Base

## Task 4: Persist and retrieve chunks with ChromaDB

**Description:** Implement the storage boundary that writes precomputed Qwen vectors and returns typed, citable search results.

**Acceptance criteria:**

- [ ] Chunks and precomputed embeddings persist in the configured Chroma path and survive store re-instantiation.
- [ ] Re-ingesting an existing `document_id` does not duplicate chunks.
- [ ] Vector query returns ordered `PaperSearchResult` values containing chunk ID, document ID, file name, page, content and score.
- [ ] Empty collections and invalid Top-K values have defined behavior.

**Verification:**

- [ ] Run `uv run pytest tests/knowledge/test_store.py -q`.
- [ ] Run `uv run ruff check src/deep_research_agent/knowledge/store.py tests/knowledge/test_store.py`.
- [ ] Manual local check: write chunks, recreate store object, retrieve the expected chunk.

**Dependencies:** Tasks 2 and 3

**Files likely touched:**

- `src/deep_research_agent/knowledge/store.py`
- `tests/knowledge/test_store.py`

**Estimated scope:** Medium

## Task 5: Create the ingestion and paper-list service

**Description:** Orchestrate validation, parsing, embedding and persistence for a batch of uploaded PDFs while isolating per-file failures.

**Acceptance criteria:**

- [ ] Service validates PDF type, non-empty bytes and 25 MB limit before processing.
- [ ] A batch returns separate `ingested`, `skipped` and `failed` entries; one failed file does not fail the batch.
- [ ] Duplicate documents are reported as skipped without making another embedding request.
- [ ] Paper listing aggregates stored chunks into file name, document ID and chunk count.

**Verification:**

- [ ] Run `uv run pytest tests/knowledge/test_service.py -q`.
- [ ] Run `uv run ruff check src/deep_research_agent/knowledge/service.py tests/knowledge/test_service.py`.

**Dependencies:** Task 4

**Files likely touched:**

- `src/deep_research_agent/knowledge/service.py`
- `tests/knowledge/test_service.py`

**Estimated scope:** Medium

## Task 6: Expose upload and paper-list API endpoints

**Description:** Add the smallest FastAPI surface needed by the UI and manual demonstrations.

**Acceptance criteria:**

- [ ] `POST /api/knowledge/papers` accepts one or more multipart PDF files and returns the ingestion summary contract.
- [ ] `GET /api/knowledge/papers` returns the current paper list and chunk counts.
- [ ] Validation and service errors map to stable, understandable HTTP responses.
- [ ] Existing run and event-stream endpoints remain unchanged.

**Verification:**

- [ ] Run `uv run pytest tests/test_knowledge_api.py -q`.
- [ ] Run `uv run ruff check src/deep_research_agent/server.py tests/test_knowledge_api.py`.
- [ ] Manual check with FastAPI docs or `curl`: upload a small PDF, then list it.

**Dependencies:** Task 5

**Files likely touched:**

- `src/deep_research_agent/server.py`
- `tests/test_knowledge_api.py`

**Estimated scope:** Small

### Checkpoint: Searchable Knowledge Base

- [ ] Tasks 4-6 tests pass.
- [ ] A PDF can be uploaded through HTTP and retrieved after service/store re-instantiation.
- [ ] Duplicate upload is skipped.
- [ ] No real private paper or `.knowledge/` data is tracked by Git.

### Phase 3: Hybrid Agent Research

## Task 7: Implement the citable private-paper search tool

**Description:** Wrap the knowledge store search path as a resilient LangChain tool suitable for Researcher calls.

**Acceptance criteria:**

- [ ] `search_private_papers(query, top_k)` returns citable passages in the approved file/page/chunk format.
- [ ] Tool metadata marks it as a search tool so existing progress handling recognizes it.
- [ ] Empty knowledge base, disabled feature, missing key and retrieval failures return clear degradation messages instead of raising into the graph.
- [ ] Tool tests use fake embedding/store dependencies and require no network access.

**Verification:**

- [ ] Run `uv run pytest tests/knowledge/test_tools.py -q`.
- [ ] Run `uv run ruff check src/deep_research_agent/knowledge/tools.py tests/knowledge/test_tools.py`.

**Dependencies:** Task 4

**Files likely touched:**

- `src/deep_research_agent/knowledge/tools.py`
- `tests/knowledge/test_tools.py`

**Estimated scope:** Small

## Task 8: Register hybrid search and guide Agent source selection

**Description:** Make private-paper search available to Researcher agents and update prompts so the model understands when to use private, public or both sources.

**Acceptance criteria:**

- [ ] `get_all_tools()` includes `search_private_papers` only when private-paper search is enabled.
- [ ] Researcher prompts describe private-paper, Web Search and combined-source use cases without forcing unnecessary private searches.
- [ ] Final report instructions require private claims to preserve file name and page citations.
- [ ] Existing Web-only tool assembly and research flow remain functional when the feature is disabled or the collection is empty.

**Verification:**

- [ ] Run `uv run pytest tests/test_hybrid_tools.py -q`.
- [ ] Run `uv run ruff check src/deep_research_agent/utils.py src/deep_research_agent/prompts.py tests/test_hybrid_tools.py`.
- [ ] Manual graph check: inspect one run where the Researcher calls both search tools.

**Dependencies:** Task 7

**Files likely touched:**

- `src/deep_research_agent/utils.py`
- `src/deep_research_agent/prompts.py`
- `tests/test_hybrid_tools.py`

**Estimated scope:** Medium

### Checkpoint: Hybrid Research

- [ ] Tasks 7-8 tests pass.
- [ ] Web-only mode still works.
- [ ] One manual task calls private-paper search and Web Search.
- [ ] Private tool output contains correct file name and page citations.

### Phase 4: User Flow and Evidence

## Task 9: Add the minimal paper upload and list UI

**Description:** Extend the existing single-page React UI with a compact private-paper knowledge-base panel without creating a separate chat product.

**Acceptance criteria:**

- [ ] User can select one or more PDFs, upload them and see per-file ingested/skipped/failed feedback.
- [ ] UI fetches and displays current paper names and chunk counts.
- [ ] UI states that paper chunks and search queries are sent to Alibaba Model Studio for embeddings.
- [ ] Existing research submission, progress timeline and report UI continue to work.

**Verification:**

- [ ] Run `cd web && npm run build`.
- [ ] Manual browser check: upload, duplicate upload, list refresh and research submission.

**Dependencies:** Task 6

**Files likely touched:**

- `web/src/App.tsx`
- `web/src/styles.css`

**Estimated scope:** Medium

## Task 10: Add reproducible private-retrieval evaluation

**Description:** Create a small offline evaluation runner that measures the retriever independently from final-report quality.

**Acceptance criteria:**

- [ ] JSONL schema supports question, gold document/page/chunk IDs and index/chunking version metadata.
- [ ] Runner computes per-question and aggregate Recall@1, Recall@3, Recall@5 and Hit Rate@K.
- [ ] Runner validates missing gold chunks and reports them instead of silently producing misleading scores.
- [ ] At least 15 locally maintained evaluation questions can be executed without committing private paper contents.

**Verification:**

- [ ] Run `uv run pytest tests/knowledge/test_retrieval_evaluation.py -q`.
- [ ] Run `uv run python tests/evaluate_private_retrieval.py --help`.
- [ ] Run the evaluation against the local demo collection and save a Markdown result table.

**Dependencies:** Task 4

**Files likely touched:**

- `tests/evaluate_private_retrieval.py`
- `tests/knowledge/test_retrieval_evaluation.py`
- `tests/fixtures/retrieval_eval.example.jsonl`
- `docs/evaluations/private-retrieval.md`

**Estimated scope:** Medium

## Task 11: Validate the end-to-end demo and document project evidence

**Description:** Prove the complete portfolio story and document reproducible evidence for reviewers and interviewers.

**Acceptance criteria:**

- [ ] At least 5 research tasks are run in Web-only and Hybrid modes and evaluated for private evidence coverage, citation correctness, report completeness and tool selection.
- [ ] At least one Hybrid report contains a verified private file/page citation and a public URL citation.
- [ ] README explains architecture, data boundary, setup, upload/research workflow, evaluation method and measured results.
- [ ] Spec and plan statuses reflect the delivered MVP and any approved deviations.

**Verification:**

- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check src tests`.
- [ ] Run `cd web && npm run build`.
- [ ] Manually verify the README demo steps from a clean local knowledge-base directory.

**Dependencies:** Tasks 8, 9 and 10

**Files likely touched:**

- `README.md`
- `docs/evaluations/hybrid-research.md`
- `docs/specs/hybrid-literature-research.md`
- `docs/plans/hybrid-literature-research.md`

**Estimated scope:** Medium

### Checkpoint: Complete MVP

- [ ] All automated tests pass.
- [ ] Frontend builds cleanly.
- [ ] Upload-to-hybrid-report flow works end to end.
- [ ] Retrieval metrics and end-to-end comparison results are documented.
- [ ] No secrets, real private papers or local vector data are tracked.
- [ ] Project is ready for code review and resume evidence extraction.

## Fastest Useful Delivery Path

For immediate portfolio value, implement in this order:

```text
Tasks 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
```

At that point, the visible MVP works end to end. Tasks 10-11 are still required before claiming measured retrieval or report-quality improvements on a resume.

## Parallelization Opportunities

- Tasks 2 and 3 can run in parallel after Task 1 because parsing and Embedding adapters share only typed contracts.
- Task 9 can begin after Task 6 while Tasks 7-8 are being implemented.
- Task 10 can begin after Task 4 while ingestion API and Agent integration continue.
- Task 11 must wait for Agent integration, UI and evaluation outputs.

Coordination rule: define and stabilize models and API response contracts before parallel work consumes them.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 百炼 `qwen3-vl-embedding` API contract or batching behavior differs from assumptions | High | Validate adapter contract in Task 3 before Chroma integration; isolate provider-specific code |
| Default embedding dimension changes or differs across calls | High | Never hardcode dimension; validate consistent vector lengths before persistence |
| Chroma synchronous calls block async Agent execution | Medium | Isolate store calls and offload blocking work from the event loop |
| Stable chunk IDs change when chunking configuration changes | Medium | Record index/chunking version in evaluation data and rebuild gold labels after changes |
| Double-column PDFs produce poor reading order | Medium | Document limitation, use text-layer demo PDFs, keep parser replaceable |
| Agent ignores private search or overuses it | High | Add explicit source-selection prompts, tool-selection tests and end-to-end evaluation |
| Model cites private evidence incorrectly after compression/report generation | High | Preserve citation markers in tool output and prompts; manually verify citations in demo evaluation |
| Private content is unintentionally disclosed | High | Document that chunks go only to百炼 Embedding, never Web Search; do not commit papers or local index |
| Current frontend proxy points to a remote backend | Medium | Preserve the user's existing change; use an explicitly selected local/remote backend during manual verification |
| Small evaluation set produces unstable claims | Medium | Separate retriever and report metrics, use at least 15 retrieval questions, report raw counts with percentages |

## Deferred Decisions

- Which locally held paper will be used for the final portfolio demonstration.
- Whether a future iteration should add Qwen3-VL multimodal page-image embedding for scanned or visually complex papers.
- Whether retrieval metrics justify adding a Reranker after the MVP.

## Plan Approval Checklist

- [ ] Task order matches the desired fastest-MVP priority.
- [ ] ChromaDB and `qwen3-vl-embedding` adapter boundaries are acceptable.
- [ ] No task exceeds approximately five files.
- [ ] Evaluation scope is sufficient before making resume claims.
- [ ] Human approves implementation to begin with Task 1.
