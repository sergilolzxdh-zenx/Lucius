"""Persistence for sessions, raw events and frames."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

from PIL import Image

from lucius.errors import NotFoundError
from lucius.ids import new_id
from lucius.provenance import DataPolicy, SourceClass
from lucius.sessions.models import (
    CapturedEvent,
    EnvironmentInfo,
    FrameRecord,
    Outcome,
    Session,
    SessionKind,
    SessionStatus,
)
from lucius.storage.db import Database, dumps, loads
import numpy as np

from lucius.storage.frames import FrameStore, thumbnail, visual_change
from lucius.timeutil import now


def _row_to_session(row: Any) -> Session:
    return Session(
        id=row["id"],
        user_id=row["user_id"],
        kind=SessionKind(row["kind"]),
        source=SourceClass(row["source"]),
        status=SessionStatus(row["status"]),
        task_id=row["task_id"],
        task_text=row["task_text"],
        task_class=row["task_class"],
        reference_ids=loads(row["reference_ids"], []),
        environment=EnvironmentInfo(
            blender_version=row["blender_version"],
            os_version=row["os_version"],
            resolution=row["resolution"],
            dpi_scale=row["dpi_scale"],
            monitor_layout=loads(row["monitor_layout"], []) or [],
        ),
        start_time=row["start_time"],
        end_time=row["end_time"],
        outcome=Outcome(row["outcome"]) if row["outcome"] else None,
        recording_config=loads(row["recording_config"], {}),
        policy=DataPolicy.model_validate(loads(row["policy"], {})),
        content_hash=row["content_hash"],
        processing=loads(row["processing"], {}),
        meta=loads(row["meta"], {}),
    )


def _row_to_frame(row: Any) -> FrameRecord:
    return FrameRecord(
        id=row["id"], session_id=row["session_id"], seq=row["seq"], ts=row["ts"], path=row["path"],
        sha256=row["sha256"], width=row["width"], height=row["height"], source=row["source"],
        media_asset_id=row["media_asset_id"], media_timestamp=row["media_timestamp"],
        window_bounds=loads(row["window_bounds"]), active_area=row["active_area"], dhash=row["dhash"],
        change_score=row["change_score"], meta=loads(row["meta"], {}),
    )


class SessionStore:
    def __init__(self, db: Database, frames: FrameStore) -> None:
        self.db = db
        self.frames = frames

    # -- users -------------------------------------------------------------------------
    def ensure_user(self, user_id: str, name: str | None = None) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO users(id, name, created_at) VALUES (?, ?, ?)", (user_id, name or user_id, now())
        )

    # -- sessions ----------------------------------------------------------------------
    def create(
        self,
        *,
        user_id: str,
        kind: SessionKind,
        policy: DataPolicy,
        task_text: str | None = None,
        task_id: str | None = None,
        task_class: str | None = None,
        environment: EnvironmentInfo | None = None,
        recording_config: dict[str, Any] | None = None,
        reference_ids: Sequence[str] = (),
        status: SessionStatus = SessionStatus.RECORDING,
        start_time: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Session:
        self.ensure_user(user_id)
        env = environment or EnvironmentInfo()
        session = Session(
            id=new_id("session"), user_id=user_id, kind=kind, source=policy.source, status=status,
            task_id=task_id, task_text=task_text, task_class=task_class, reference_ids=list(reference_ids),
            environment=env, start_time=start_time if start_time is not None else now(),
            recording_config=recording_config or {}, policy=policy, meta=meta or {},
        )
        self.db.insert("sessions", {
            "id": session.id, "user_id": user_id, "kind": kind.value, "source": policy.source.value,
            "status": status.value, "task_id": task_id, "task_text": task_text, "task_class": task_class,
            "reference_ids": dumps(list(reference_ids)), "blender_version": env.blender_version,
            "os_version": env.os_version, "resolution": env.resolution, "dpi_scale": env.dpi_scale,
            "monitor_layout": dumps(env.monitor_layout), "start_time": session.start_time,
            "recording_config": dumps(session.recording_config), "policy": dumps(policy),
            "meta": dumps(session.meta),
        })
        return session

    def get(self, session_id: str) -> Session:
        row = self.db.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            raise NotFoundError(f"session {session_id} not found", session_id=session_id)
        return _row_to_session(row)

    def list(self, *, kind: SessionKind | None = None, status: SessionStatus | None = None,
             limit: int = 100, offset: int = 0) -> list[Session]:
        clauses, params = [], []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(
            f"SELECT * FROM sessions {where} ORDER BY start_time DESC LIMIT ? OFFSET ?", (*params, limit, offset)
        )
        return [_row_to_session(r) for r in rows]

    def update(self, session_id: str, **fields: Any) -> None:
        encoded: dict[str, Any] = {}
        for key, value in fields.items():
            if key == "environment" and isinstance(value, EnvironmentInfo):
                encoded.update({
                    "blender_version": value.blender_version, "os_version": value.os_version,
                    "resolution": value.resolution, "dpi_scale": value.dpi_scale,
                    "monitor_layout": dumps(value.monitor_layout),
                })
            elif key in {"reference_ids", "recording_config", "processing", "meta", "policy"}:
                encoded[key] = dumps(value)
            elif hasattr(value, "value"):
                encoded[key] = value.value
            else:
                encoded[key] = value
        if self.db.update("sessions", "id", session_id, encoded) == 0:
            raise NotFoundError(f"session {session_id} not found", session_id=session_id)

    def set_processing_stage(self, session_id: str, stage: str, status: str, **info: Any) -> None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT processing FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"session {session_id} not found", session_id=session_id)
            processing = loads(row["processing"], {})
            processing[stage] = {"status": status, "ts": now(), **info}
            conn.execute("UPDATE sessions SET processing = ? WHERE id = ?", (dumps(processing), session_id))

    def finalize(self, session_id: str, *, end_time: float | None = None, outcome: Outcome | None = None,
                 status: SessionStatus = SessionStatus.FINALIZED) -> Session:
        end = end_time if end_time is not None else now()
        fields: dict[str, Any] = {"end_time": end, "status": status, "content_hash": self.content_hash(session_id)}
        if outcome is not None:
            fields["outcome"] = outcome
        self.update(session_id, **fields)
        return self.get(session_id)

    def content_hash(self, session_id: str) -> str:
        """Hash of the raw evidence (events + frame digests): used for duplicate detection."""
        digest = hashlib.sha256()
        for row in self.db.conn.execute(
            "SELECT kind, payload FROM events WHERE session_id = ? ORDER BY seq", (session_id,)
        ):
            digest.update(row["kind"].encode())
            digest.update(row["payload"].encode())
        for row in self.db.conn.execute("SELECT sha256 FROM frames WHERE session_id = ? ORDER BY seq", (session_id,)):
            digest.update(row["sha256"].encode())
        return digest.hexdigest()

    def delete(self, session_id: str) -> dict[str, int]:
        """Delete a session and every frame file no longer referenced by another session."""
        self.get(session_id)
        shas = {r["path"] for r in self.db.query("SELECT DISTINCT path FROM frames WHERE session_id = ?", (session_id,))}
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM skill_examples WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        removed = 0
        for rel in shas:
            still_used = self.db.scalar("SELECT 1 FROM frames WHERE path = ? LIMIT 1", (rel,))
            if not still_used:
                path = self.frames.path(rel)
                if path.exists():
                    path.unlink()
                    removed += 1
        return {"frames_removed": removed}

    # -- events ------------------------------------------------------------------------
    def next_seq(self, session_id: str) -> int:
        e = self.db.scalar("SELECT COALESCE(MAX(seq), -1) FROM events WHERE session_id = ?", (session_id,))
        f = self.db.scalar("SELECT COALESCE(MAX(seq), -1) FROM frames WHERE session_id = ?", (session_id,))
        return max(e, f) + 1

    def append_events(self, session_id: str, events: Iterable[CapturedEvent]) -> int:
        rows = [
            (session_id, e.seq, e.ts, e.kind.value, e.actor.value,
             None if e.blender_active is None else int(e.blender_active), dumps(e.payload))
            for e in events
        ]
        self.db.executemany(
            "INSERT OR IGNORE INTO events(session_id, seq, ts, kind, actor, blender_active, payload)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def events(self, session_id: str, *, kinds: Sequence[str] | None = None) -> list[CapturedEvent]:
        if kinds:
            marks = ",".join("?" for _ in kinds)
            rows = self.db.query(
                f"SELECT * FROM events WHERE session_id = ? AND kind IN ({marks}) ORDER BY seq", (session_id, *kinds)
            )
        else:
            rows = self.db.query("SELECT * FROM events WHERE session_id = ? ORDER BY seq", (session_id,))
        return [
            CapturedEvent(
                seq=r["seq"], ts=r["ts"], kind=r["kind"], actor=r["actor"],
                blender_active=None if r["blender_active"] is None else bool(r["blender_active"]),
                payload=loads(r["payload"], {}),
            )
            for r in rows
        ]

    def event_count(self, session_id: str) -> int:
        return self.db.scalar("SELECT COUNT(*) FROM events WHERE session_id = ?", (session_id,))

    def last_activity(self, session_id: str) -> float | None:
        return self.db.scalar(
            "SELECT MAX(ts) FROM (SELECT MAX(ts) AS ts FROM events WHERE session_id = ?"
            " UNION ALL SELECT MAX(ts) FROM frames WHERE session_id = ?)",
            (session_id, session_id),
        )

    # -- frames ------------------------------------------------------------------------
    def add_frame(
        self,
        session_id: str,
        image: Image.Image,
        *,
        seq: int,
        ts: float,
        source: str,
        window_bounds: dict[str, int] | None = None,
        active_area: str | None = None,
        media_asset_id: str | None = None,
        media_timestamp: float | None = None,
        previous_thumb: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
    ) -> FrameRecord:
        """Store a frame. ``previous_thumb`` (see :func:`thumbnail`) yields its visual change score."""
        stored = self.frames.put_image(image)
        change = visual_change(previous_thumb, thumbnail(image))
        frame = FrameRecord(
            id=new_id("frame"), session_id=session_id, seq=seq, ts=ts, path=stored.rel_path,
            sha256=stored.sha256, width=stored.width, height=stored.height, source=source,
            media_asset_id=media_asset_id, media_timestamp=media_timestamp, window_bounds=window_bounds,
            active_area=active_area, dhash=stored.dhash, change_score=change, meta=meta or {},
        )
        self.db.insert("frames", {
            "id": frame.id, "session_id": session_id, "seq": seq, "ts": ts, "path": frame.path,
            "sha256": frame.sha256, "width": frame.width, "height": frame.height, "source": source,
            "media_asset_id": media_asset_id, "media_timestamp": media_timestamp,
            "window_bounds": dumps(window_bounds) if window_bounds else None, "active_area": active_area,
            "dhash": frame.dhash, "change_score": change, "meta": dumps(frame.meta),
        })
        return frame

    def frames_for(self, session_id: str) -> list[FrameRecord]:
        rows = self.db.query("SELECT * FROM frames WHERE session_id = ? ORDER BY ts, seq", (session_id,))
        return [_row_to_frame(r) for r in rows]

    def frame(self, frame_id: str) -> FrameRecord:
        row = self.db.query_one("SELECT * FROM frames WHERE id = ?", (frame_id,))
        if row is None:
            raise NotFoundError(f"frame {frame_id} not found", frame_id=frame_id)
        return _row_to_frame(row)

    def frame_count(self, session_id: str) -> int:
        return self.db.scalar("SELECT COUNT(*) FROM frames WHERE session_id = ?", (session_id,))
