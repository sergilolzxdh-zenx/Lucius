"""Segment model and persistence."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from lucius.storage.db import Database, dumps, loads


class BoundaryReason(BaseModel):
    code: str                  # pause, mode_change, undo, navigation_start, actor_change, ...
    strength: float
    detail: str = ""


class LabelEvidence(BaseModel):
    reason_code: str
    detail: str
    weight: float = 1.0


class Segment(BaseModel):
    id: str
    session_id: str
    idx: int
    t_start: float
    t_end: float
    step_start: int
    step_end: int
    label: str
    title: str | None = None
    label_confidence: float
    origin: str = "deterministic"      # deterministic | model | human
    locked: bool = False               # human-edited segments survive re-segmentation
    outcome: str = "unknown"           # unknown | success | failure | corrected
    boundary_reasons: list[BoundaryReason] = Field(default_factory=list)
    label_evidence: list[LabelEvidence] = Field(default_factory=list)
    representative_frame_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)


def _decode(row: Any) -> Segment:
    return Segment(
        id=row["id"], session_id=row["session_id"], idx=row["idx"], t_start=row["t_start"], t_end=row["t_end"],
        step_start=row["step_start"], step_end=row["step_end"], label=row["label"], title=row["title"],
        label_confidence=row["label_confidence"], origin=row["origin"], locked=bool(row["locked"]),
        outcome=row["outcome"],
        boundary_reasons=[BoundaryReason.model_validate(b) for b in loads(row["boundary_reasons"], [])],
        label_evidence=[LabelEvidence.model_validate(e) for e in loads(row["label_evidence"], [])],
        representative_frame_ids=loads(row["representative_frame_ids"], []), summary=row["summary"],
        meta=loads(row["meta"], {}),
    )


class SegmentStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def for_session(self, session_id: str) -> list[Segment]:
        rows = self.db.query("SELECT * FROM segments WHERE session_id = ? ORDER BY step_start", (session_id,))
        return [_decode(r) for r in rows]

    def get(self, segment_id: str) -> Segment | None:
        row = self.db.query_one("SELECT * FROM segments WHERE id = ?", (segment_id,))
        return None if row is None else _decode(row)

    def replace_unlocked(self, session_id: str, segments: list[Segment]) -> None:
        """Replace machine-produced segments, keeping every human-locked one."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM segments WHERE session_id = ? AND locked = 0", (session_id,))
            for seg in segments:
                if seg.locked:
                    continue
                conn.execute(
                    "INSERT INTO segments (id, session_id, idx, t_start, t_end, step_start, step_end, label, title,"
                    " label_confidence, origin, locked, outcome, boundary_reasons, label_evidence,"
                    " representative_frame_ids, summary, meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    self._encode(seg))
            self._reindex(conn, session_id)

    def save(self, seg: Segment) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM segments WHERE id = ?", (seg.id,))
            conn.execute(
                "INSERT INTO segments (id, session_id, idx, t_start, t_end, step_start, step_end, label, title,"
                " label_confidence, origin, locked, outcome, boundary_reasons, label_evidence,"
                " representative_frame_ids, summary, meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                self._encode(seg))
            self._reindex(conn, seg.session_id)

    def delete(self, segment_id: str) -> None:
        self.db.execute("DELETE FROM segments WHERE id = ?", (segment_id,))

    @staticmethod
    def _reindex(conn: Any, session_id: str) -> None:
        rows = conn.execute("SELECT id FROM segments WHERE session_id = ? ORDER BY step_start", (session_id,)).fetchall()
        for i, row in enumerate(rows):
            conn.execute("UPDATE segments SET idx = ? WHERE id = ?", (i, row["id"]))

    @staticmethod
    def _encode(seg: Segment) -> tuple[Any, ...]:
        return (seg.id, seg.session_id, seg.idx, seg.t_start, seg.t_end, seg.step_start, seg.step_end, seg.label,
                seg.title, seg.label_confidence, seg.origin, int(seg.locked), seg.outcome,
                dumps([b.model_dump() for b in seg.boundary_reasons]),
                dumps([e.model_dump() for e in seg.label_evidence]), dumps(seg.representative_frame_ids),
                seg.summary, dumps(seg.meta))
