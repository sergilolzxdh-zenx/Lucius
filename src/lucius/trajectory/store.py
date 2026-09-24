"""Persistence for trajectory steps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lucius.provenance import ActionSource, EvidenceKind
from lucius.storage.db import Database, dumps, loads
from lucius.trajectory.model import CandidateAction, TrajectoryStep

_COLUMNS = (
    "id", "session_id", "idx", "t_start", "t_end", "frame_before_id", "frame_after_id", "action_type",
    "action_payload", "action_source", "evidence_kind", "action_confidence", "evidence", "candidate_actions",
    "window_title", "window_bounds", "mode_label", "tool_label", "selection_hint", "undo_redo_flag", "actor",
    "segment_id", "event_seq_start", "event_seq_end", "media_timestamp", "state_before", "state_after", "meta",
)


def _encode(step: TrajectoryStep) -> tuple[Any, ...]:
    return (
        step.id, step.session_id, step.idx, step.t_start, step.t_end, step.frame_before_id, step.frame_after_id,
        step.action_type, dumps(step.action_payload), step.action_source.value, step.evidence_kind.value,
        step.action_confidence, dumps(step.evidence), dumps([c.model_dump() for c in step.candidate_actions]),
        step.window_title, dumps(step.window_bounds) if step.window_bounds else None, step.mode_label,
        step.tool_label, step.selection_hint, step.undo_redo_flag, step.actor, step.segment_id,
        step.event_seq_start, step.event_seq_end, step.media_timestamp,
        dumps(step.state_before) if step.state_before is not None else None,
        dumps(step.state_after) if step.state_after is not None else None, dumps(step.meta),
    )


def _decode(row: Any) -> TrajectoryStep:
    return TrajectoryStep(
        id=row["id"], session_id=row["session_id"], idx=row["idx"], t_start=row["t_start"], t_end=row["t_end"],
        frame_before_id=row["frame_before_id"], frame_after_id=row["frame_after_id"], action_type=row["action_type"],
        action_payload=loads(row["action_payload"], {}), action_source=ActionSource(row["action_source"]),
        evidence_kind=EvidenceKind(row["evidence_kind"]), action_confidence=row["action_confidence"],
        evidence=loads(row["evidence"], []),
        candidate_actions=[CandidateAction.model_validate(c) for c in loads(row["candidate_actions"], [])],
        window_title=row["window_title"], window_bounds=loads(row["window_bounds"]), mode_label=row["mode_label"],
        tool_label=row["tool_label"], selection_hint=row["selection_hint"], undo_redo_flag=row["undo_redo_flag"],
        actor=row["actor"], segment_id=row["segment_id"], event_seq_start=row["event_seq_start"],
        event_seq_end=row["event_seq_end"], media_timestamp=row["media_timestamp"],
        state_before=loads(row["state_before"]), state_after=loads(row["state_after"]), meta=loads(row["meta"], {}),
    )


class TrajectoryStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def replace(self, session_id: str, steps: Sequence[TrajectoryStep]) -> None:
        """Atomically replace a session's trajectory (processing is re-runnable)."""
        marks = ", ".join("?" for _ in _COLUMNS)
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM trajectory_steps WHERE session_id = ?", (session_id,))
            conn.executemany(f"INSERT INTO trajectory_steps ({', '.join(_COLUMNS)}) VALUES ({marks})",
                             [_encode(s) for s in steps])

    def for_session(self, session_id: str) -> list[TrajectoryStep]:
        rows = self.db.query("SELECT * FROM trajectory_steps WHERE session_id = ? ORDER BY idx", (session_id,))
        return [_decode(r) for r in rows]

    def get(self, step_id: str) -> TrajectoryStep | None:
        row = self.db.query_one("SELECT * FROM trajectory_steps WHERE id = ?", (step_id,))
        return None if row is None else _decode(row)

    def assign_segments(self, session_id: str, ranges: Sequence[tuple[str, int, int]]) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE trajectory_steps SET segment_id = NULL WHERE session_id = ?", (session_id,))
            for segment_id, start, end in ranges:
                conn.execute(
                    "UPDATE trajectory_steps SET segment_id = ? WHERE session_id = ? AND idx BETWEEN ? AND ?",
                    (segment_id, session_id, start, end),
                )

    def confirm_action(self, step_id: str, action_type: str, params: dict[str, Any] | None = None) -> None:
        """Human correction of an action interpretation (10R): becomes human_confirmed evidence."""
        step = self.get(step_id)
        if step is None:
            return
        payload = dict(step.action_payload)
        if params is not None:
            payload["params"] = params
        payload.setdefault("original_interpretation", {"action_type": step.action_type,
                                                       "source": step.action_source.value,
                                                       "confidence": step.action_confidence})
        self.db.execute(
            "UPDATE trajectory_steps SET action_type = ?, action_payload = ?, action_source = ?, evidence_kind = ?,"
            " action_confidence = ? WHERE id = ?",
            (action_type, dumps(payload), ActionSource.HUMAN_CONFIRMED.value, EvidenceKind.HUMAN_ANNOTATION.value,
             0.97, step_id),
        )

    def count(self, session_id: str) -> int:
        return self.db.scalar("SELECT COUNT(*) FROM trajectory_steps WHERE session_id = ?", (session_id,))
