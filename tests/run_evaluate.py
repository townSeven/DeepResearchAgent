import argparse
import asyncio
import os
import sys
from pathlib import Path
import uuid

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATASET_NAME = "Deep Research Bench"
DEFAULT_PUBLIC_DATASET_URL = "https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d"

def parse_example_indexes(value: str) -> list[int]:
    """Parse one-based example indexes such as '3,4,12-17' into zero-based indexes."""
    indexes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end < start:
                raise argparse.ArgumentTypeError(f"Invalid example index range: {part}")
            indexes.extend(range(start - 1, end))
        else:
            index = int(part)
            if index <= 0:
                raise argparse.ArgumentTypeError(f"Example indexes are one-based: {part}")
            indexes.append(index - 1)
    if not indexes:
        raise argparse.ArgumentTypeError("At least one example index is required")
    return list(dict.fromkeys(indexes))

def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Research Agent evaluation.")
    parser.add_argument("--model", help="Temporarily use one model for all agent stages, e.g. qwen:qwen-plus.")
    parser.add_argument("--summarization-model")
    parser.add_argument("--research-model")
    parser.add_argument("--compression-model")
    parser.add_argument("--final-report-model")
    parser.add_argument("--eval-model", help="LLM-as-judge model. Defaults to the manual eval_model below.")
    parser.add_argument("--dataset-name", default=os.getenv("DRA_DATASET_NAME", DEFAULT_DATASET_NAME))
    parser.add_argument("--dataset-url", default=os.getenv("DRA_DATASET_URL", DEFAULT_PUBLIC_DATASET_URL), help="Public LangSmith dataset URL to clone if the dataset is not in your workspace.")
    parser.add_argument("--experiment-prefix")
    parser.add_argument("--max-concurrency", type=int, default=int(os.getenv("DRA_MAX_CONCURRENCY", "10")))
    parser.add_argument("--max-examples", type=int, default=int(os.getenv("DRA_MAX_EXAMPLES", "0")), help="Limit examples for smoke tests. Use 0 for the full dataset.")
    parser.add_argument("--start-index", type=int, default=int(os.getenv("DRA_START_INDEX", "0")), help="Zero-based dataset offset for batched evaluation.")
    parser.add_argument("--example-indexes", type=parse_example_indexes, default=None, help="One-based dataset row indexes to run, e.g. '3,4,12-17'. Takes precedence over batch slicing.")
    parser.add_argument("--experiment", help="Existing LangSmith experiment id/name to extend. Omit to create a new retry experiment.")
    parser.add_argument("--max-research-units", type=int, default=int(os.getenv("DRA_MAX_RESEARCH_UNITS", "10")))
    parser.add_argument("--max-researcher-iterations", type=int, default=int(os.getenv("DRA_MAX_RESEARCHER_ITERATIONS", "6")))
    parser.add_argument("--max-react-tool-calls", type=int, default=int(os.getenv("DRA_MAX_REACT_TOOL_CALLS", "10")))
    return parser.parse_args()

def configure_from_args(args):
    global dataset_name, dataset_url, max_concurrency, max_examples, start_index, example_indexes
    global experiment_prefix, experiment
    global max_concurrent_research_units, max_researcher_iterations, max_react_tool_calls
    global summarization_model, summarization_model_max_tokens
    global research_model, research_model_max_tokens
    global compression_model, compression_model_max_tokens
    global final_report_model, final_report_model_max_tokens
    global eval_model

    if args.example_indexes and (args.max_examples > 0 or args.start_index > 0):
        raise ValueError("--example-indexes cannot be combined with --max-examples or --start-index")

    dataset_name = args.dataset_name
    dataset_url = args.dataset_url
    max_concurrency = args.max_concurrency
    max_examples = args.max_examples
    start_index = args.start_index
    example_indexes = args.example_indexes
    experiment = args.experiment
    max_concurrent_research_units = args.max_research_units
    max_researcher_iterations = args.max_researcher_iterations
    max_react_tool_calls = args.max_react_tool_calls
    experiment_prefix = args.experiment_prefix or experiment_prefix
    summarization_model = args.summarization_model or args.model or summarization_model
    research_model = args.research_model or args.model or research_model
    compression_model = args.compression_model or args.model or compression_model
    final_report_model = args.final_report_model or args.model or final_report_model
    eval_model = args.eval_model or eval_model
    os.environ["SUMMARIZATION_MODEL"] = summarization_model
    os.environ["RESEARCH_MODEL"] = research_model
    os.environ["COMPRESSION_MODEL"] = compression_model
    os.environ["FINAL_REPORT_MODEL"] = final_report_model
    os.environ["EVAL_MODEL"] = eval_model

# NOTE: Configure the right dataset and evaluators
dataset_name = DEFAULT_DATASET_NAME
dataset_url = DEFAULT_PUBLIC_DATASET_URL
# NOTE: Configure the right parameters for the experiment, these will be logged in the metadata
max_structured_output_retries = 3
allow_clarification = False
max_concurrent_research_units = 10
search_api = "tavily" # NOTE: We use Tavily to stay consistent
max_researcher_iterations = 6
max_react_tool_calls = 10

# Model configuration
# Edit these directly when mixing providers. Each provider uses its own API key
# from .env, for example MINIMAX_API_KEY for minimax:* and DEEPSEEK_API_KEY for deepseek:*.
summarization_model = "deepseek:deepseek-v4-flash"
summarization_model_max_tokens = 8192
research_model = "deepseek:deepseek-v4-flash"
research_model_max_tokens = 10000
compression_model = "deepseek:deepseek-v4-flash"
compression_model_max_tokens = 10000
final_report_model = "deepseek:deepseek-v4-flash"
final_report_model_max_tokens = 10000
eval_model = "deepseek:deepseek-v4-flash"
max_concurrency = 10
max_examples = 0
start_index = 0
example_indexes = None
experiment = None
experiment_prefix = "DRA Mixed Models, Tavily Search"

async def target(
    inputs: dict,
):
    from langgraph.checkpoint.memory import MemorySaver
    from deep_research_agent.deep_researcher import deep_researcher_builder

    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    # NOTE: Configure the right dataset and evaluators
    config["configurable"]["max_structured_output_retries"] = max_structured_output_retries
    config["configurable"]["allow_clarification"] = allow_clarification
    config["configurable"]["max_concurrent_research_units"] = max_concurrent_research_units
    config["configurable"]["search_api"] = search_api
    config["configurable"]["max_researcher_iterations"] = max_researcher_iterations
    config["configurable"]["max_react_tool_calls"] = max_react_tool_calls
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_model_max_tokens"] = summarization_model_max_tokens
    config["configurable"]["research_model"] = research_model
    config["configurable"]["research_model_max_tokens"] = research_model_max_tokens
    config["configurable"]["compression_model"] = compression_model
    config["configurable"]["compression_model_max_tokens"] = compression_model_max_tokens
    config["configurable"]["final_report_model"] = final_report_model
    config["configurable"]["final_report_model_max_tokens"] = final_report_model_max_tokens
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config
    )
    return final_state

async def main():
    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError
    from tests.evaluators import eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness

    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        print(f"Dataset '{dataset_name}' not found in this LangSmith workspace. Cloning from public dataset...")
        dataset = client.clone_public_dataset(dataset_url, dataset_name=dataset_name)

    evaluators = [eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness]
    data = dataset_name
    if example_indexes:
        examples = list(client.list_examples(dataset_id=dataset.id, limit=max(example_indexes) + 1))
        if len(examples) <= max(example_indexes):
            raise ValueError(f"Dataset has only {len(examples)} examples; requested index {max(example_indexes) + 1}")
        data = [examples[index] for index in example_indexes]
        print(f"Running selected dataset rows: {', '.join(str(index + 1) for index in example_indexes)}")
    elif max_examples > 0 or start_index > 0:
        limit = start_index + max_examples if max_examples > 0 else None
        examples = list(client.list_examples(dataset_id=dataset.id, limit=limit))
        data = examples[start_index:] if max_examples == 0 else examples[start_index:start_index + max_examples]

    evaluate_kwargs = {
        "experiment": experiment,
    } if experiment else {
        "experiment_prefix": experiment_prefix,
    }
    print(
        "Effective models: "
        f"summarization={summarization_model}, "
        f"research={research_model}, "
        f"compression={compression_model}, "
        f"final_report={final_report_model}, "
        f"eval={eval_model}"
    )

    return await client.aevaluate(
        target,
        data=data,
        evaluators=evaluators,
        **evaluate_kwargs,
        max_concurrency=max_concurrency,
        metadata={
            "max_structured_output_retries": max_structured_output_retries,
            "allow_clarification": allow_clarification,
            "max_concurrent_research_units": max_concurrent_research_units,
            "search_api": search_api,
            "max_researcher_iterations": max_researcher_iterations,
            "max_react_tool_calls": max_react_tool_calls,
            "summarization_model": summarization_model,
            "summarization_model_max_tokens": summarization_model_max_tokens,
            "research_model": research_model,
            "research_model_max_tokens": research_model_max_tokens,
            "compression_model": compression_model,
            "compression_model_max_tokens": compression_model_max_tokens,
            "final_report_model": final_report_model,
            "final_report_model_max_tokens": final_report_model_max_tokens,
            "eval_model": eval_model,
            "start_index": start_index,
            "max_examples": max_examples,
            "example_indexes": [index + 1 for index in example_indexes] if example_indexes else None,
        }
    )

if __name__ == "__main__":
    configure_from_args(parse_args())
    results = asyncio.run(main())
    print(results)
