#!/usr/bin/env python3
"""Replace JSONL article fields with direct DeepSeek chat-completion outputs."""

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "tests" / "expt_results" / "deep_research_bench_deepseek_raw.jsonl"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send each JSONL row's prompt directly to DeepSeek and replace that "
            "row's article field with the returned content."
        )
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help="Input JSONL file. Defaults to tests/expt_results/deep_research_bench_deepseek_raw.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        help="Output JSONL file. Defaults to overwriting --input-path in place.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        help=(
            "DeepSeek model name. A provider prefix such as "
            "'deepseek:deepseek-v4-flash' is accepted and stripped."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help="DeepSeek OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY"),
        help="DeepSeek API key. Defaults to DEEPSEEK_API_KEY from the environment or .env.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("DEEPSEEK_MAX_TOKENS", "10000")),
        help="Maximum output tokens for each completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted by default.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("DEEPSEEK_TIMEOUT", "600")),
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("DEEPSEEK_MAX_WORKERS", "1")),
        help="Number of prompts to process concurrently.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
        help="Retries per prompt after transient request failures.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=float(os.getenv("DEEPSEEK_RETRY_SLEEP", "5")),
        help="Base sleep seconds between retries; multiplied by attempt number.",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt. Empty by default so the prompt is sent directly.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows, useful for smoke tests. 0 means all rows.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based row offset to start processing from.",
    )
    parser.add_argument(
        "--skip-nonempty-article",
        action="store_true",
        help="Do not call DeepSeek for rows whose article field is already non-empty.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak timestamped copy when overwriting in place.",
    )
    parser.add_argument(
        "--keep-thinking",
        action="store_true",
        help="Keep provider-injected <think>...</think> blocks if they appear in content.",
    )
    return parser.parse_args()


def normalize_model_name(model: str) -> str:
    provider, sep, model_id = model.partition(":")
    if sep and provider.lower() == "deepseek" and model_id:
        return model_id
    return model


def strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            if "prompt" not in item:
                raise ValueError(f"Line {line_number} has no prompt field")
            rows.append(item)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]], create_backup: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if create_backup and path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
        shutil.copy2(path, backup_path)
        print(f"Backup written to {backup_path}")

    tmp_path.replace(path)


def call_deepseek(
    *,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
    system_prompt: str,
    strip_thinking: bool,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = chat_completions_url(base_url)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code} from DeepSeek: {response.text[:500]}"
                )
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
            return strip_thinking_tags(content) if strip_thinking else content
        except Exception as exc:
            last_error = exc
            if attempt > max_retries:
                break
            sleep_seconds = retry_sleep * attempt
            print(f"Request failed on attempt {attempt}; retrying in {sleep_seconds:.1f}s: {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"DeepSeek request failed after {max_retries + 1} attempts: {last_error}")


def selected_indexes(
    rows: list[dict[str, Any]],
    *,
    start_index: int,
    limit: int,
    skip_nonempty_article: bool,
) -> list[int]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    end_index = len(rows) if limit <= 0 else min(len(rows), start_index + limit)
    indexes = list(range(start_index, end_index))
    if skip_nonempty_article:
        indexes = [index for index in indexes if not rows[index].get("article")]
    return indexes


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in .env or pass --api-key.")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be >= 1")

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else input_path
    in_place = input_path == output_path

    rows = read_jsonl(input_path)
    indexes = selected_indexes(
        rows,
        start_index=args.start_index,
        limit=args.limit,
        skip_nonempty_article=args.skip_nonempty_article,
    )
    model = normalize_model_name(args.model)
    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Processing {len(indexes)} rows with model={model}, max_workers={args.max_workers}")

    def run_one(index: int) -> tuple[int, str]:
        prompt = str(rows[index]["prompt"])
        article = call_deepseek(
            prompt=prompt,
            api_key=args.api_key,
            base_url=args.base_url,
            model=model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            system_prompt=args.system_prompt,
            strip_thinking=not args.keep_thinking,
        )
        return index, article

    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_one, index): index for index in indexes}
        for future in as_completed(futures):
            index = futures[future]
            try:
                row_index, article = future.result()
            except Exception as exc:
                print(f"Failed on row {index + 1}: {exc}", file=sys.stderr)
                raise
            rows[row_index]["article"] = article
            completed += 1
            row_id = rows[row_index].get("id", row_index + 1)
            print(f"[{completed}/{len(indexes)}] updated row {row_index + 1} (id={row_id})")

    write_jsonl_atomic(output_path, rows, create_backup=in_place and not args.no_backup)
    print(f"Wrote updated JSONL to {output_path}")


if __name__ == "__main__":
    main()
