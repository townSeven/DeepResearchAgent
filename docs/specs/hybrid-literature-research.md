# Spec: Hybrid Literature Research

## Status

- Phase: Plan
- State: Approved
- Feature name: Hybrid Literature Research

## Assumptions To Confirm

1. MVP 面向单用户本地作品集演示，不实现账号、租户和论文访问权限隔离。
2. 用户只上传自己有权使用的论文 PDF，包括未发表稿件与从受限数据库合法下载的文件；系统不抓取或绕过知网等数据库权限。
3. MVP 只支持存在可提取文本层的 PDF，不支持扫描件 OCR、公式理解、图片和复杂表格解析。
4. 所有私有论文写入一个本地持久化 Chroma collection，不支持创建和切换多个知识库。
5. Embedding 使用阿里云百炼平台提供的 `qwen3-vl-embedding` API，通过 `DASHSCOPE_API_KEY` 配置，并采用 API 默认向量维度。
6. 检索采用向量相似度 Top-K；MVP 不实现 BM25、Hybrid Search、Reranker 和 Query Rewrite。
7. 私有论文检索作为 Researcher 的并列工具接入，由模型自主选择私有知识库、Web Search 或同时使用两者。
8. Web UI 只新增论文上传、入库状态和已入库论文列表，不新增独立的 PDF Chat 页面。
9. MVP 使用人工准备的小型问题集比较 `Web only` 与 `Web + private papers`，证明私有文献覆盖度和引用正确性有所提升。

## Objective

为现有 Deep Research Agent 增加私有论文知识库，使科研人员能够上传未发表论文、本地论文和从受限数据库合法获得的文献，并让多智能体联合私有文献与公开 Web Search 开展科研课题调研。

该功能不是一个独立的 PDF 问答应用。它延续现有的任务澄清、研究规划、并行 Researcher、笔记压缩和报告生成流程，只增加新的文献入库链路与 `search_private_papers` 研究工具。

### Target User

需要研究同时包含公开论文和本地私有资料的科研人员，例如：

- 调研包含实验室未发表工作的研究方向；
- 联合本地下载论文与互联网最新资料生成文献综述；
- 比较私有论文方法与公开研究，并识别研究空白。

### Primary User Story

作为科研人员，我希望上传本地论文并发起一个研究课题，让 Deep Research Agent 自主联合私有论文与公开资料进行调研，从而获得覆盖完整、引用可追溯的研究报告。

### Demo Scenario

1. 用户上传一篇无法被 Web Search 检索到的私有论文 PDF。
2. 系统解析论文、切分文本、生成 Embedding，并写入本地向量数据库。
3. 用户提交一个必须使用该论文才能完整回答的科研课题。
4. Researcher 根据子问题调用 `search_private_papers`、Web Search 或两者。
5. 最终报告同时引用私有论文页码与公开网页 URL。

## Functional Requirements

### FR-1: Private Paper Ingestion

- Web UI 支持选择并上传单个或多个 PDF 文件。
- 服务端拒绝非 PDF 文件、空文件与超过大小限制的文件。
- 服务端使用 PyMuPDF 提取逐页文本，保留论文文件名和页码。
- 文本按可配置 chunk 大小和 overlap 切分。
- 每个 chunk 生成 Embedding 并写入 Chroma。
- 重复上传同一内容的论文时，不重复创建 chunks。
- 入库接口返回成功、跳过和失败文件的摘要。

### FR-2: Paper Metadata

每个 chunk 至少保存：

| Field | Description |
|---|---|
| `chunk_id` | chunk 唯一标识 |
| `document_id` | 基于文件内容生成的稳定标识 |
| `file_name` | 原始 PDF 文件名 |
| `page_number` | 1-based 页码 |
| `chunk_index` | 文档内 chunk 顺序 |
| `content` | chunk 原文 |
| `source_type` | 固定为 `private_paper` |

MVP 不要求可靠提取标题、作者、年份；文件名作为论文展示名称。

### FR-3: Private Paper Retrieval

新增研究工具：

```python
@tool
async def search_private_papers(query: str, top_k: int = 5) -> str:
    """Search user-provided private papers and return citable passages."""
```

工具行为：

- 对 query 生成 Embedding，并从 Chroma 检索 Top-K chunks。
- 返回包含论文文件名、页码、相关文本和稳定引用标识的结构化可读结果。
- 无知识库、无结果或 Embedding 失败时返回可理解的降级信息，不中断研究流程。
- 工具 metadata 标记为搜索工具，使现有进度事件与工具执行逻辑能够识别。

期望返回格式：

```text
[Private Paper: paper-a.pdf, Page 6, Chunk abc123]
<retrieved passage>
```

### FR-4: Agent Source Selection

- `search_private_papers` 加入现有 `get_all_tools()` 工具集合。
- Researcher prompt 明确说明各类工具的用途：
  - 私有论文中的方法、实验结果和内部研究信息优先使用 `search_private_papers`；
  - 公开现状、最新论文和外部事实使用 Web Search；
  - 对比、综述和研究空白问题应考虑联合使用两类来源。
- Agent 不强制每个子任务调用私有论文检索；工具选择应由研究问题决定。
- 现有 Web-only 工作流在未上传论文时仍然正常工作。

### FR-5: Traceable Final Report

- 私有论文结论必须带有文件名和页码引用。
- Web 来源继续使用现有 URL 引用方式。
- 最终报告能够清晰区分私有论文来源和公开来源。
- 不得为检索结果中不存在的私有论文或页码生成引用。

### FR-6: Minimal Web UI

- 增加“私有论文知识库”区域。
- 用户可以上传 PDF，并看到解析、入库成功或失败状态。
- 用户可以查看已入库论文的文件名与 chunk 数量。
- 研究执行时间线能够显示私有论文检索事件。
- 不实现文档预览、chunk 编辑和独立知识库问答页面。

### FR-7: MVP Evaluation

评测分为两个互不混淆的层级：

1. **Retriever evaluation**：准备至少 15 个私有论文检索问题，评估 `search_private_papers` 的 Recall@K 和 Hit Rate@K。
2. **End-to-end evaluation**：准备至少 5 个必须依赖私有论文才能完整回答的研究任务，分别执行：
   - Baseline: Web Search only
   - Treatment: Web Search + private papers

至少记录：

| Metric | Definition |
|---|---|
| Retrieval Recall@K | 私有论文 Top-K 检索结果对人工标注 gold chunks 的召回率 |
| Private evidence coverage | 最终报告是否覆盖问题要求的私有论文关键证据 |
| Citation correctness | 私有论文引用的文件名、页码和结论是否一致 |
| Report completeness | 报告是否覆盖预定义要点 |
| Tool selection accuracy | 需要私有信息的问题是否调用了私有论文检索 |

评测结果应形成可展示的 Markdown 表格，不要求建设完整评测平台。

#### Retrieval Recall@K Evaluation

召回率只评估 `search_private_papers` 的检索能力，不直接评估最终报告。评测数据集至少包含：

```json
{
  "question": "论文提出的 Prompt-Region Alignment 如何训练？",
  "gold_evidence": [
    {
      "document_id": "paper-document-id",
      "page_number": 6,
      "chunk_id": "stable-gold-chunk-id"
    }
  ]
}
```

Gold evidence 在 PDF 解析和 chunking 策略固定后，通过人工阅读论文并标注能够回答问题的相关 chunks 建立。一个问题允许对应多个 gold chunks。评测数据需记录 chunking 配置或索引版本；当 chunking 策略变化时，必须重新生成和检查 gold chunk IDs。

检索问题至少覆盖：

- 可直接从论文原文定位的事实问题；
- 使用同义表达或改写后的语义问题；
- 答案分布在多个 chunks 或多篇论文中的问题。

对问题集合 \(Q\)，定义：

```text
Recall@K(q) = |retrieved_top_k(q) ∩ gold_chunks(q)| / |gold_chunks(q)|
Mean Recall@K = 所有问题 Recall@K(q) 的平均值
```

MVP 至少报告 `Recall@1`、`Recall@3` 和 `Recall@5`。同时报告 `Hit Rate@K`，用于表示 Top-K 中是否至少命中一个 gold chunk：

```text
Hit Rate@K(q) = 1 if retrieved_top_k(q) ∩ gold_chunks(q) is not empty else 0
```

由于 chunk 边界可能导致同一证据跨越相邻 chunks，人工标注时应将所有能够独立支撑答案的相邻 chunks 纳入 gold evidence，而不是只标注一个任意 chunk。

Web-only 模式不存在私有论文检索结果，因此不参与 Retrieval Recall@K 对比；它只与 Hybrid 模式比较最终报告的 private evidence coverage、citation correctness 和 report completeness。

## Non-Functional Requirements

- 私有 PDF 文件和 Chroma 向量数据保存在本地，不上传至 Web Search 服务。
- 为生成向量，解析后的论文 chunks 和检索 query 会发送至阿里云百炼 `qwen3-vl-embedding` API；UI 和 README 必须明确说明该数据边界。
- 单个文件解析或入库失败不应导致同批其他文件失败。
- 私有论文检索失败不应导致整个 Deep Research run 失败。
- 所有新增配置均提供可运行默认值，并支持通过环境变量或 RunnableConfig 覆盖。
- MVP 目标数据规模为不超过 50 篇、每篇不超过 100 页的文本型 PDF。

## Tech Stack

### Existing

- Python 3.10+
- LangGraph and LangChain tools
- FastAPI
- React 18, TypeScript, Vite
- PyMuPDF
- pytest

### Proposed Additions

- ChromaDB: 本地持久化向量存储
- 阿里云百炼 `qwen3-vl-embedding` API: Embedding 模型
- `python-multipart`: FastAPI PDF 上传解析

新增依赖必须在实现前确认。

## Proposed Interfaces

### API

```text
POST   /api/knowledge/papers
GET    /api/knowledge/papers
DELETE /api/knowledge/papers/{document_id}  # Optional, not required for MVP acceptance
```

`POST /api/knowledge/papers` 接受 multipart PDF 文件列表并返回：

```json
{
  "ingested": [{"document_id": "...", "file_name": "paper.pdf", "chunk_count": 42}],
  "skipped": [],
  "failed": []
}
```

### Configuration

建议新增配置：

| Field | Default |
|---|---|
| `private_papers_enabled` | `true` |
| `private_papers_top_k` | `5` |
| `knowledge_base_path` | `.knowledge/private_papers` |
| `embedding_model` | `qwen3-vl-embedding` |
| `embedding_api_key_env` | `DASHSCOPE_API_KEY` |
| `paper_chunk_size` | `1200` characters |
| `paper_chunk_overlap` | `200` characters |
| `max_paper_size_mb` | `25` |

## Commands

```bash
# Install Python dependencies
uv sync

# Run backend
uv run uvicorn deep_research_agent.server:app --reload --port 8000

# Run LangGraph development server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking

# Run Python tests
uv run pytest

# Run Python lint
uv run ruff check src tests

# Run frontend
cd web && npm run dev

# Build frontend
cd web && npm run build
```

## Project Structure

```text
DeepResearchAgent/
  docs/specs/
    hybrid-literature-research.md   # This specification
  src/deep_research_agent/
    knowledge/
      models.py                     # Paper/chunk/result data contracts
      parser.py                     # PDF parsing and chunking
      store.py                      # Chroma and embedding operations
      service.py                    # Ingestion/list orchestration
      tools.py                      # search_private_papers tool
    configuration.py               # Knowledge-base configuration
    utils.py                       # Register private search tool
    prompts.py                     # Hybrid source-selection guidance
    server.py                      # Knowledge API endpoints
  tests/
    knowledge/
      test_parser.py
      test_store.py
      test_tools.py
    test_knowledge_api.py
    fixtures/papers/
    fixtures/retrieval_eval.jsonl
    evaluate_private_retrieval.py
  web/src/
    App.tsx
    styles.css
```

Runtime Chroma data lives under `.knowledge/` and must be ignored by Git.

## Code Style

- 延续现有异步 Python、Pydantic 数据模型和 LangChain `@tool` 模式。
- 公共函数使用类型标注和简短 docstring。
- 知识库逻辑保持在 `knowledge/` 模块内；`server.py` 和 `utils.py` 只负责接线。
- 不把 Chroma 返回对象直接泄漏给 Agent 或 API。

示例：

```python
class PaperSearchResult(BaseModel):
    """A citable passage retrieved from a private paper."""

    document_id: str
    file_name: str
    page_number: int
    content: str
    score: float


async def search(self, query: str, top_k: int) -> list[PaperSearchResult]:
    """Return the most relevant private-paper passages."""
```

## Testing Strategy

### Unit Tests

- PDF 文本按页提取并保留正确页码。
- chunk 边界、overlap 和稳定 ID 符合预期。
- 重复论文不会重复入库。
- 检索结果正确映射文件名、页码、内容和 score。
- `search_private_papers` 正确格式化引用，并对空库和异常降级。

Embedding 和 Chroma 外部行为在单元测试中使用 fake 或 mock，避免依赖网络与真实 API key。

### Integration Tests

- 使用小型文本 PDF fixture 验证上传、入库、列表和检索闭环。
- 验证 `get_all_tools()` 在功能启用时包含 `search_private_papers`。
- 验证未上传论文时现有 Web-only research 工具集合仍可使用。

### Manual Acceptance

- 上传一篇私有测试论文。
- 发起一个公开网络无法完整回答的问题。
- 在执行时间线中观察私有论文检索。
- 检查最终报告包含正确文件名和页码引用。
- 对照 PDF 原文确认至少一个私有结论与引用一致。

### Coverage Expectation

新增 `knowledge/` 核心模块的正常路径和主要失败路径必须有自动化测试。MVP 不设置全仓覆盖率门槛。

## Boundaries

### Always Do

- 对上传文件做类型、大小和空内容校验。
- 保留私有证据的文件名与页码。
- 对解析、Embedding、存储和检索失败进行可理解的降级处理。
- 在提交实现前运行相关 pytest、Ruff 和前端 build。
- 使用合法获得的论文作为演示和测试材料。
- 在 UI 和 README 中说明论文 chunks 会发送至阿里云百炼 Embedding API。

### Ask First

- 增加 ChromaDB、`python-multipart` 等新依赖。
- 更换阿里云百炼 `qwen3-vl-embedding` Embedding 服务或改变向量维度。
- 修改现有 Deep Research 图结构或 Agent 状态模型。
- 扩展到多知识库、用户权限或云端持久化。

### Never Do

- 抓取或绕过知网等受限数据库的访问控制。
- 将用户上传的 PDF 原文发送给 Web Search 服务。
- 在仓库中提交真实私有论文、API key 或 `.knowledge/` 数据。
- 为未检索到的私有证据伪造文件名或页码引用。
- 为追求 MVP 完整度而加入 OCR、知识图谱或复杂表格解析。

## Success Criteria

- [ ] 用户能够通过 Web UI 上传至少一个文本型 PDF，并看到入库结果。
- [ ] 系统能够持久化论文 chunks，并在服务重启后继续检索。
- [ ] 重复上传相同 PDF 不会产生重复 chunks。
- [ ] Researcher 可调用 `search_private_papers`，并获得带文件名和页码的结果。
- [ ] Agent 能在一个演示任务中同时调用私有论文检索和 Web Search。
- [ ] 最终报告同时包含至少一个正确的私有论文页码引用和一个公开 URL 引用。
- [ ] 未上传论文或私有检索失败时，现有 Web-only 研究流程仍可完成。
- [ ] 至少 15 个检索问题完成 Recall@1、Recall@3、Recall@5 和 Hit Rate@K 评测。
- [ ] 至少 5 个完整研究任务的对比实验表明 Hybrid 模式比 Web-only 具有更高的私有证据覆盖率。
- [ ] 新增核心模块自动化测试通过，`uv run ruff check src tests` 与 `cd web && npm run build` 通过。
- [ ] README 包含功能说明、架构图或流程图、运行步骤、演示案例和评测结果。

## Not Doing In MVP

- 知网等数据库自动抓取或登录集成
- 扫描件 OCR、公式理解、图片和复杂表格解析
- 多用户、多租户、权限控制和云端知识库
- 多 collection 管理和知识库切换
- 独立 PDF Chat 页面
- BM25、Hybrid Search、Reranker、Query Rewrite
- 引用关系图谱、论文推荐和研究趋势可视化
- 自动可靠提取标题、作者、年份等论文元数据
- 大规模 RAG 评测平台

## Confirmed Decisions

- 使用 ChromaDB 作为本地持久化向量数据库。
- 使用阿里云百炼 `qwen3-vl-embedding` API，并采用默认向量维度。
- MVP 不要求删除已入库论文。
- 单个 PDF 大小限制暂定为 25 MB。

## Remaining Open Questions

1. 演示私有论文使用你自己的 ToPT 论文，还是另行准备不会提交到公开仓库的测试论文？
