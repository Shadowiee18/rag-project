"""Saved student profiles: one row per user (study year + the academic year it was given in), in SQLite.

SQLite is part of Python and fine for one server. For several servers, replace this file with a shared
database (e.g. Postgres); the API only uses get() and save().
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SavedProfile:
    year: str      # "Year2"
    given_in: str  # academic year the answer was given in, "2026-27"


class ProfileStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._run("PRAGMA journal_mode=WAL")  # readers don't wait for a writer
        self._run("CREATE TABLE IF NOT EXISTS profiles ("
                  "user_id TEXT PRIMARY KEY, year TEXT NOT NULL, given_in TEXT NOT NULL, updated_at TEXT NOT NULL)")

    def get(self, user_id: str) -> SavedProfile | None:
        row = self._run("SELECT year, given_in FROM profiles WHERE user_id = ?", (user_id,))
        return SavedProfile(*row) if row else None

    def save(self, user_id: str, year: str, given_in: str) -> None:
        self._run(
            "INSERT INTO profiles (user_id, year, given_in, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET year = excluded.year, given_in = excluded.given_in, "
            "updated_at = excluded.updated_at",
            (user_id, year, given_in, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )

    def _run(self, sql: str, params: tuple = ()) -> tuple | None:
        """One connection per call, so requests on different threads never share one; commits on success."""
        with closing(sqlite3.connect(self.path, timeout=5)) as conn, conn:
            return conn.execute(sql, params).fetchone()
