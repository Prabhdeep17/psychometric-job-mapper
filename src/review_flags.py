"""Local notes that a mapping should be reviewed by a person.

Stored in SQLite under the project data directory. Flags never change
the reference scores. Job-description text is not stored.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config import DATA_DIR


def _db_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "review_flags.sqlite"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_db_path())
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS review_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            role TEXT NOT NULL,
            domain_id TEXT NOT NULL,
            note TEXT NOT NULL
        )
        """
    )
    return connection


@dataclass
class ReviewFlag:
    id: int
    created_at: str
    role: str
    domain_id: str
    note: str


def add_flag(role: str, domain_id: str, note: str) -> None:
    """Save a short review note. Empty notes are ignored."""
    cleaned = " ".join(note.split())
    if not cleaned:
        return
    if len(cleaned) > 500:
        cleaned = cleaned[:500]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with _connect() as connection:
        connection.execute(
            "INSERT INTO review_flags (created_at, role, domain_id, note) VALUES (?, ?, ?, ?)",
            (stamp, role, domain_id, cleaned),
        )


def list_flags(limit: int = 50) -> List[ReviewFlag]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, role, domain_id, note
            FROM review_flags
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [ReviewFlag(*row) for row in rows]


def clear_flags() -> None:
    """Delete local review notes. Does not touch the reference dataset."""
    with _connect() as connection:
        connection.execute("DELETE FROM review_flags")
