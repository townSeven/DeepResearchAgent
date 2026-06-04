#!/usr/bin/env python3
"""Extract data from LangSmith and save to JSONL file with configurable dataset."""

import os
import json
import argparse
import re
from datetime import datetime, timezone
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ID_SOURCE_JSONL = "tests/expt_results/deep_research_bench_gpt-5.jsonl"

def strip_thinking_tags(text):
    """Remove provider-injected thinking blocks from exported reports."""
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.IGNORECASE | re.DOTALL).strip()

def load_prompt_id_map(path):
    """Load canonical benchmark ids from an existing JSONL result file."""
    if not path or not os.path.exists(path):
        return {}

    prompt_id_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt")
            benchmark_id = item.get("id")
            if prompt is not None and benchmark_id is not None:
                prompt_id_map[prompt] = benchmark_id
    return prompt_id_map

def message_content(message):
    """Extract content from LangSmith message dicts in either stored format."""
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")

def prompt_from_inputs(inputs):
    """Extract the original user prompt from a run or example input payload."""
    payload = inputs.get("inputs", inputs) if inputs else {}
    messages = payload.get("messages", [])
    if not messages:
        return ""
    return message_content(messages[0])

def benchmark_id_for_run(run, example, prompt_id_map, fallback_id):
    """Resolve benchmark id from metadata, canonical prompt map, or fallback order."""
    metadata = (example.metadata or {}) if example else {}
    if "id" in metadata:
        return metadata["id"]

    prompt = prompt_from_inputs(run.inputs)
    if prompt in prompt_id_map:
        return prompt_id_map[prompt]

    return fallback_id

def run_timestamp(run):
    """Return the best available timestamp for choosing the newest run."""
    timestamp = (
        getattr(run, "end_time", None)
        or getattr(run, "start_time", None)
        or getattr(run, "created_at", None)
    )
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.timestamp()
    return 0.0

def split_project_names(project_names):
    """Allow --project-name to be repeated or comma-separated."""
    names = []
    for value in project_names:
        names.extend(name.strip() for name in value.split(",") if name.strip())
    return names


def extract_langsmith_data(project_names, model_name, dataset_name, api_key, id_source_jsonl):
    """Extract data from LangSmith and save to JSONL file."""
    project_names = split_project_names(project_names)
    print(f"Extracting data from LangSmith projects: {', '.join(project_names)}")
    print(f"Using dataset: {dataset_name}")
    
    client = Client(api_key=api_key)
    prompt_id_map = load_prompt_id_map(id_source_jsonl)
    if prompt_id_map:
        print(f"Loaded {len(prompt_id_map)} prompt->id mappings from {id_source_jsonl}")

    examples_dict = {}
    fallback_ids = {}
    latest_runs_by_example = {}
    total_output_runs = 0

    for project_name in project_names:
        # Read project to get reference dataset id
        project_data = client.read_project(project_name=project_name)

        # Read reference dataset to get examples
        examples = list(client.list_examples(dataset_id=project_data.reference_dataset_id))
        for index, example in enumerate(examples, start=1):
            examples_dict[example.id] = example
            fallback_ids.setdefault(example.id, index)

        # Read project runs to get runs inputs, outputs, and reference_example_ids
        output_runs = client.list_runs(
            project_name=project_name,
            is_root=True
        )

        for run in output_runs:
            if run.outputs is None or run.outputs.get("final_report") is None:
                continue

            total_output_runs += 1
            example_key = run.reference_example_id or prompt_from_inputs(run.inputs)
            existing_run = latest_runs_by_example.get(example_key)
            if existing_run is None or run_timestamp(run) > run_timestamp(existing_run):
                latest_runs_by_example[example_key] = run

    runs = list(latest_runs_by_example.values())
    duplicate_count = total_output_runs - len(runs)

    output_jsonl = [
        {
            "id": benchmark_id_for_run(
                run,
                examples_dict.get(run.reference_example_id),
                prompt_id_map,
                fallback_ids.get(run.reference_example_id),
            ),
            "prompt": prompt_from_inputs(run.inputs),
            "article": strip_thinking_tags(run.outputs["final_report"]),
        } for run in runs
    ]
    output_jsonl.sort(key=lambda item: (item["id"] is None, item["id"]))
    
    # Write output_jsonl to JSONL file in tests/expt_results directory
    output_file_path = f"tests/expt_results/{dataset_name}_{model_name}.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in output_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Data written to {output_file_path}")
    print(f"Total records: {len(output_jsonl)}")
    print(f"Skipped duplicate older runs: {duplicate_count}")
    return output_file_path


def main():
    parser = argparse.ArgumentParser(description='Extract data from LangSmith project')
    parser.add_argument('--project-name', required=True, action='append', help='LangSmith project name. May be repeated or comma-separated to merge retry experiments.')
    parser.add_argument('--model-name', required=True, help='Model name for output filename')
    parser.add_argument('--dataset-name', required=True, help='Dataset name for output filename')
    parser.add_argument('--api-key', help='LangSmith API key (defaults to LANGSMITH_API_KEY env var)')
    parser.add_argument('--id-source-jsonl', default=DEFAULT_ID_SOURCE_JSONL, help='Existing Deep Research Bench JSONL used to recover canonical ids when LangSmith metadata has no id')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('LANGSMITH_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided via --api-key or LANGSMITH_API_KEY environment variable")
    
    extract_langsmith_data(
        project_names=args.project_name,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        api_key=api_key,
        id_source_jsonl=args.id_source_jsonl,
    )


if __name__ == "__main__":
    main()
