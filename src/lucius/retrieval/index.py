"""Embedding index with caching (section 30).

Only compact textual representations are embedded (skill descriptions, failure rules, semantic
statements, episode summaries) -- never raw events. Vectors are cached per (owner, provider)
with a hash of the embedded text, so unchanged items are never re-embedded and a failed
embedding only needs retrying for the items it failed on.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable

import numpy as np

from lucius.logging_setup import get_logger
from lucius.providers.base import EmbeddingProvider
from lucius.storage.db import Database
from lucius.timeutil import now

log = get_logger("retrieval.index")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class VectorIndex:
    def __init__(self, db: Database, provider: EmbeddingProvider) -> None:
        self.db = db
        self.provider = provider
        self._cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._lock = threading.Lock()

    def sync(self, kind: str, items: Iterable[tuple[str, str]]) -> dict[str, int]:
        """Ensure every ``(owner_id, text)`` is embedded with the current text. Returns counts."""
        items = list(items)
        existing = {r["owner_id"]: r["text_hash"] for r in self.db.query(
            "SELECT owner_id, text_hash FROM embeddings WHERE owner_kind = ? AND provider = ?",
            (kind, self.provider.name))}
        wanted = {owner: text for owner, text in items}
        stale = [(o, t) for o, t in wanted.items() if existing.get(o) != _hash(t)]
        removed = [o for o in existing if o not in wanted]
        embedded = failed = 0
        for start in range(0, len(stale), 64):
            batch = stale[start:start + 64]
            try:
                vectors = self.provider.embed([t for _o, t in batch])
            except Exception as exc:  # keep the rest of the index usable; retried on next sync
                log.warning("embedding batch failed (%d items): %s", len(batch), exc)
                failed += len(batch)
                continue
            rows = [(kind, owner, self.provider.name, _hash(text), int(vec.shape[0]),
                     vec.astype(np.float32).tobytes(), now()) for (owner, text), vec in zip(batch, vectors)]
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings (owner_kind, owner_id, provider, text_hash, dim, vector, created_at)"
                " VALUES (?,?,?,?,?,?,?)", rows)
            embedded += len(rows)
        for owner in removed:
            self.db.execute("DELETE FROM embeddings WHERE owner_kind = ? AND owner_id = ? AND provider = ?",
                            (kind, owner, self.provider.name))
        if embedded or removed:
            with self._lock:
                self._cache.pop(kind, None)
        return {"embedded": embedded, "removed": len(removed), "failed": failed, "cached": len(wanted) - len(stale)}

    def _matrix(self, kind: str) -> tuple[list[str], np.ndarray]:
        with self._lock:
            if kind in self._cache:
                return self._cache[kind]
        rows = self.db.query("SELECT owner_id, vector FROM embeddings WHERE owner_kind = ? AND provider = ?",
                             (kind, self.provider.name))
        ids = [r["owner_id"] for r in rows]
        matrix = (np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
                  if rows else np.zeros((0, self.provider.dim), dtype=np.float32))
        with self._lock:
            self._cache[kind] = (ids, matrix)
        return ids, matrix

    def embed_query(self, text: str) -> np.ndarray:
        return self.provider.embed([text])[0]

    def scores(self, kind: str, query: np.ndarray) -> dict[str, float]:
        ids, matrix = self._matrix(kind)
        if not ids:
            return {}
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        sims = matrix @ query / np.where(norms == 0, 1.0, norms)
        return {i: float(s) for i, s in zip(ids, sims)}
