"""SQLite snapshot store: the last good regime payload, served stale on outage."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path("data") / "regime_cache.db"


class SnapshotStore:
    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS snapshots ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " created_at TEXT NOT NULL,"
            " payload TEXT NOT NULL)"
        )
        return conn

    def _save(self, payload_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots (created_at, payload) VALUES (?, ?)",
                (datetime.now(timezone.utc).isoformat(), payload_json),
            )

    def _latest(self) -> tuple[str, str] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT created_at, payload FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        return None if row is None else (str(row[0]), str(row[1]))

    async def save(self, payload_json: str) -> None:
        await asyncio.to_thread(self._save, payload_json)

    async def latest(self) -> tuple[str, str] | None:
        return await asyncio.to_thread(self._latest)
