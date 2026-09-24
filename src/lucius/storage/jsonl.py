"""JSON Lines helpers used for journals, exports and datasets."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

from lucius.storage.db import dumps


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps(row))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield rows; a truncated final line (crash mid-write) is skipped, not fatal."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class JsonlAppender:
    """Append-only, fsync-on-flush journal (crash recovery for in-flight recordings)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = self.path.open("a", encoding="utf-8")

    def append(self, row: Any) -> None:
        if self._handle is None:
            raise ValueError("journal closed")
        self._handle.write(dumps(row))
        self._handle.write("\n")

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self.flush()
            self._handle.close()
            self._handle = None
