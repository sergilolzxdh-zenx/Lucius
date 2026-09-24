"""SQLite access layer.

One connection per thread (SQLite connections are not shareable across threads), WAL mode
so background processors can read while the recorder writes, and foreign keys enforced.
Schema changes are applied as numbered migrations recorded in ``schema_migrations``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from lucius.errors import StorageError
from lucius.timeutil import now

JSON_COLUMNS_HINT = "Columns holding JSON are decoded by the repositories, never here."


def _migrations() -> list[tuple[int, str]]:
    schema = resources.files("lucius.storage").joinpath("schema.sql").read_text()
    return [(1, schema)]


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=_json_default)


def loads(value: str | bytes | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._migrate()

    # -- connections -------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- migrations --------------------------------------------------------------------
    def _migrate(self) -> None:
        conn = self.conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
        )
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for version, script in _migrations():
            if version in applied:
                continue
            try:
                conn.execute("BEGIN")
                for statement in _split_sql(script):
                    conn.execute(statement)
                conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (version, now()))
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise StorageError(f"migration {version} failed: {exc}", version=version) from exc

    # -- statements --------------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialise writers within the process and wrap work in BEGIN IMMEDIATE/COMMIT."""
        with self._write_lock:
            conn = self.conn
            if conn.in_transaction:
                # Nested use joins the outer transaction.
                yield conn
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self.transaction() as conn:
            return conn.execute(sql, params)

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            return
        with self.transaction() as conn:
            conn.executemany(sql, rows)

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return None if row is None else row[0]

    def insert(self, table: str, values: dict[str, Any], *, or_replace: bool = False) -> None:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        verb = "INSERT OR REPLACE" if or_replace else "INSERT"
        self.execute(f"{verb} INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))

    def update(self, table: str, key: str, key_value: Any, values: dict[str, Any]) -> int:
        if not values:
            return 0
        assignments = ", ".join(f"{col} = ?" for col in values)
        cur = self.execute(
            f"UPDATE {table} SET {assignments} WHERE {key} = ?", (*values.values(), key_value)
        )
        return cur.rowcount


def _split_sql(script: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(buffer).rstrip(";"))
            buffer = []
    if buffer:
        statements.append("\n".join(buffer))
    return statements
