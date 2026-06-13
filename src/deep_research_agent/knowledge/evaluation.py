"""Evaluate private paper retrieval against manually labeled evidence."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from deep_research_agent.configuration import Configuration
from deep_research_agent.knowledge.embedding import QwenEmbeddingClient
from deep_research_agent.knowledge.store import PaperVectorStore


class GoldEvidence(BaseModel):
    """One relevant private-paper chunk for an evaluation question."""

    document_id: str
    page_number: int
    chunk_id: str


class RetrievalEvalCase(BaseModel):
    """A retrieval question and its manually labeled relevant chunks."""

    question: str
    gold_evidence: list[GoldEvidence] = Field(min_length=1)
    index_version: str


class RetrievalEvalReport(BaseModel):
    """Aggregate and per-question retrieval evaluation results."""

    metrics: dict[str, float]
    cases: list[dict[str, Any]]


async def evaluate_retrieval(
    cases: list[RetrievalEvalCase],
    store: Any,
    embedding_client: Any,
    ks: tuple[int, ...] = (1, 3, 5),
) -> RetrievalEvalReport:
    """Compute mean Recall@K and Hit Rate@K for labeled questions."""
    if not cases or not ks or min(ks) < 1:
        raise ValueError("Evaluation cases and positive K values are required")

    metric_values = {
        **{f"recall@{k}": [] for k in ks},
        **{f"hit_rate@{k}": [] for k in ks},
    }
    case_results = []
    for case in cases:
        gold_ids = {evidence.chunk_id for evidence in case.gold_evidence}
        existing_ids = await store.existing_chunk_ids(list(gold_ids))
        missing_ids = sorted(gold_ids - existing_ids)
        if missing_ids:
            raise ValueError(f"Gold chunks are missing from the index: {', '.join(missing_ids)}")

        query_embedding = await embedding_client.embed_query(case.question)
        results = await store.search(query_embedding, top_k=max(ks))
        retrieved_ids = [result.chunk_id for result in results]
        case_metrics = {}
        for k in ks:
            hits = gold_ids.intersection(retrieved_ids[:k])
            recall = len(hits) / len(gold_ids)
            hit_rate = float(bool(hits))
            case_metrics[f"recall@{k}"] = recall
            case_metrics[f"hit_rate@{k}"] = hit_rate
            metric_values[f"recall@{k}"].append(recall)
            metric_values[f"hit_rate@{k}"].append(hit_rate)
        case_results.append(
            {
                "question": case.question,
                "gold_chunk_ids": sorted(gold_ids),
                "retrieved_chunk_ids": retrieved_ids,
                **case_metrics,
            }
        )

    metrics = {
        name: sum(values) / len(values)
        for name, values in metric_values.items()
    }
    return RetrievalEvalReport(metrics=metrics, cases=case_results)


def load_cases(path: Path) -> list[RetrievalEvalCase]:
    """Load retrieval evaluation cases from JSONL."""
    return [
        RetrievalEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_markdown_report(report: RetrievalEvalReport) -> str:
    """Render retrieval metrics and per-question results as Markdown."""
    lines = [
        "# Private Paper Retrieval Evaluation",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for name, value in sorted(report.metrics.items()):
        label = name.replace("_", " ").title().replace("@", "@")
        lines.append(f"| {label} | {value:.3f} |")
    lines.extend(["", "## Per-Question Results", ""])
    for case in report.cases:
        lines.append(f"- **{case['question']}**")
        lines.append(f"  - Retrieved: {', '.join(case['retrieved_chunk_ids'])}")
    return "\n".join(lines) + "\n"


async def _run_cli(args: argparse.Namespace) -> str:
    configuration = Configuration.from_runnable_config()
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY or QWEN_API_KEY is required")
    report = await evaluate_retrieval(
        cases=load_cases(args.dataset),
        store=PaperVectorStore(configuration.knowledge_base_path),
        embedding_client=QwenEmbeddingClient(api_key, model=configuration.embedding_model),
    )
    return render_markdown_report(report)


def main() -> None:
    """Run private-paper retrieval evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL file containing labeled questions")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()
    load_dotenv()
    markdown = asyncio.run(_run_cli(args))
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
