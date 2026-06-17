"""SQLite persistence for user-visible research state."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}
SENSITIVE_CONFIG_SUFFIXES = ("_api_key", "_key", "_secret", "_token", "_password")


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return configuration without credentials or other sensitive values."""
    return {
        key: _safe_value(value)
        for key, value in config.items()
        if key.lower() not in SENSITIVE_CONFIG_KEYS
        and not key.lower().endswith(SENSITIVE_CONFIG_SUFFIXES)
    }


def _safe_value(value: Any) -> Any:
    """Recursively remove sensitive keys from nested configuration values."""
    if isinstance(value, dict):
        return _safe_config(value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    return value


class SQLiteResearchStore:
    """Persist research run snapshots in a local SQLite database."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the database and create its schema when needed."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, version)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_research_runs_updated_at
                ON research_runs(updated_at DESC)
                """
            )

    def save_run(self, snapshot: dict[str, Any], config: dict[str, Any]) -> None:
        """Insert or replace one complete user-visible run snapshot."""
        safe_config = _safe_config(config)
        stored_snapshot = {**snapshot, "config": safe_config}
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_runs (
                    id, title, status, snapshot_json, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    status = excluded.status,
                    snapshot_json = excluded.snapshot_json,
                    config_json = excluded.config_json,
                    updated_at = excluded.updated_at
                """,
                (
                    snapshot["id"],
                    snapshot.get("title") or "Untitled research",
                    snapshot["status"],
                    json.dumps(stored_snapshot, ensure_ascii=False),
                    json.dumps(safe_config, ensure_ascii=False),
                    snapshot["created_at"],
                    snapshot["updated_at"],
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one complete run snapshot."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM research_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return json.loads(row["snapshot_json"]) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return compact run summaries ordered by recent activity."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, status, created_at, updated_at
                FROM research_runs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        summaries = []
        for row in rows:
            summaries.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return summaries

    def delete_run(self, run_id: str) -> bool:
        """Delete one research run and all report versions attached to it."""
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM research_reports WHERE run_id = ?", (run_id,))
            cursor = connection.execute("DELETE FROM research_runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0

    def mark_active_runs_interrupted(self) -> list[str]:
        """Mark runs left active by a previous process as interrupted."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, snapshot_json FROM research_runs WHERE status IN ('queued', 'running')"
            ).fetchall()
            interrupted = []
            for row in rows:
                snapshot = json.loads(row["snapshot_json"])
                snapshot["status"] = "interrupted"
                connection.execute(
                    """
                    UPDATE research_runs
                    SET status = 'interrupted', snapshot_json = ?
                    WHERE id = ?
                    """,
                    (json.dumps(snapshot, ensure_ascii=False), row["id"]),
                )
                interrupted.append(row["id"])
        return interrupted

    def add_report(
        self,
        run_id: str,
        title: str,
        content: str,
        created_at: str,
    ) -> dict[str, Any]:
        """Persist a new final-report version for a research thread."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS version FROM research_reports WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            version = int(row["version"])
            cursor = connection.execute(
                """
                INSERT INTO research_reports(run_id, version, title, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, version, title, content, created_at),
            )
        return {
            "id": cursor.lastrowid,
            "run_id": run_id,
            "version": version,
            "title": title,
            "content": content,
            "created_at": created_at,
        }

    def list_reports(self, run_id: str) -> list[dict[str, Any]]:
        """Return all final-report versions for one research thread."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, run_id, version, title, content, created_at
                FROM research_reports
                WHERE run_id = ?
                ORDER BY version DESC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_reports(
        self,
        query: str,
        limit: int = 5,
        exclude_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return locally relevant final reports using a rebuildable lexical index."""
        query_terms = _terms(query)
        if not query_terms:
            return []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, run_id, version, title, content, created_at
                FROM research_reports
                WHERE (? IS NULL OR run_id != ?)
                """,
                (exclude_run_id, exclude_run_id),
            ).fetchall()
        scored = []
        for row in rows:
            score = len(query_terms & _terms(f"{row['title']} {row['content']}"))
            if score:
                scored.append((score, row["created_at"], dict(row)))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]


def _terms(text: str) -> set[str]:
    """Tokenize English words and Chinese characters for local fallback search."""
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()))
