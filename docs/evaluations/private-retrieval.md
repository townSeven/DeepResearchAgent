# Private Paper Retrieval Evaluation

Status: Awaiting a locally labeled dataset with at least 15 retrieval questions.

Run:

```bash
uv run python tests/evaluate_private_retrieval.py path/to/retrieval_eval.jsonl \
  --output docs/evaluations/private-retrieval.md
```

Do not claim Recall@K improvements until the gold chunks have been manually verified against the fixed index version.
