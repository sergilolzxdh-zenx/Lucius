"""In-process event bus.

Subsystems announce what they did (``SEGMENT_CREATED``, ``SKILL_PROMOTED``...) instead of
calling each other directly. Handlers run synchronously but are isolated: a failing handler
is logged and never breaks the publisher (capture must not fail because the UI stream did).

Low-frequency, decision-bearing events are persisted to ``event_log`` so that every learning
decision stays inspectable. High-frequency capture events are only streamed live; their
durable record is the ``frames``/``events`` tables themselves.
"""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from lucius.logging_setup import get_logger
from lucius.storage.db import Database, dumps
from lucius.timeutil import now

log = get_logger("events")


class EventType(StrEnum):
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"
    SESSION_RECOVERED = "SESSION_RECOVERED"
    FRAME_CAPTURED = "FRAME_CAPTURED"
    ACTION_CAPTURED = "ACTION_CAPTURED"
    BLENDER_STATE = "BLENDER_STATE"
    TRAJECTORY_BUILT = "TRAJECTORY_BUILT"
    SEGMENT_CREATED = "SEGMENT_CREATED"
    SEGMENT_EDITED = "SEGMENT_EDITED"
    INTENT_INFERRED = "INTENT_INFERRED"
    SKILL_CANDIDATE_CREATED = "SKILL_CANDIDATE_CREATED"
    SKILL_EVIDENCE_ADDED = "SKILL_EVIDENCE_ADDED"
    SKILL_VERSIONED = "SKILL_VERSIONED"
    SKILL_PROMOTED = "SKILL_PROMOTED"
    SKILL_DEMOTED = "SKILL_DEMOTED"
    MEMORY_CREATED = "MEMORY_CREATED"
    RETRIEVAL_COMPLETED = "RETRIEVAL_COMPLETED"
    PLAN_CREATED = "PLAN_CREATED"
    STATE_TRANSITION = "STATE_TRANSITION"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    ACTION_REJECTED = "ACTION_REJECTED"
    CHECKPOINT_PASSED = "CHECKPOINT_PASSED"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    HUMAN_TAKEOVER = "HUMAN_TAKEOVER"
    HUMAN_RESUME = "HUMAN_RESUME"
    CORRECTION_RECORDED = "CORRECTION_RECORDED"
    FAILURE_RECORDED = "FAILURE_RECORDED"
    FAILURE_PROMOTED = "FAILURE_PROMOTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    PROCESSING_STAGE = "PROCESSING_STAGE"
    MEDIA_STATUS = "MEDIA_STATUS"
    MASTERY_UPDATED = "MASTERY_UPDATED"
    DATASET_BUILT = "DATASET_BUILT"
    MODEL_CALL = "MODEL_CALL"


# Streamed live but not written to event_log (their durable record lives elsewhere).
EPHEMERAL = {EventType.FRAME_CAPTURED, EventType.ACTION_CAPTURED, EventType.BLENDER_STATE}


class BusEvent(BaseModel):
    type: EventType
    ts: float = Field(default_factory=now)
    subject_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


Handler = Callable[[BusEvent], None]


class EventBus:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._streams: list[queue.Queue[BusEvent]] = []

    def subscribe(self, event_type: EventType | str, handler: Handler) -> Callable[[], None]:
        key = str(event_type)
        with self._lock:
            self._handlers[key].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers[key]:
                    self._handlers[key].remove(handler)

        return unsubscribe

    def open_stream(self, maxsize: int = 1000) -> queue.Queue[BusEvent]:
        """A bounded queue receiving every event (used by the UI's server-sent events)."""
        stream: queue.Queue[BusEvent] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._streams.append(stream)
        return stream

    def close_stream(self, stream: queue.Queue[BusEvent]) -> None:
        with self._lock:
            if stream in self._streams:
                self._streams.remove(stream)

    def publish(self, event_type: EventType, subject_id: str | None = None, **payload: Any) -> BusEvent:
        event = BusEvent(type=event_type, subject_id=subject_id, payload=payload)
        if self._db is not None and event_type not in EPHEMERAL:
            try:
                self._db.execute(
                    "INSERT INTO event_log(type, ts, subject_id, payload) VALUES (?, ?, ?, ?)",
                    (event.type.value, event.ts, subject_id, dumps(payload)),
                )
            except Exception:  # observability must never break the publisher
                log.exception("failed to persist bus event %s", event_type)
        with self._lock:
            handlers = list(self._handlers.get(event.type.value, ())) + list(self._handlers.get("*", ()))
            streams = list(self._streams)
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                log.exception("event handler failed for %s", event_type)
        for stream in streams:
            try:
                stream.put_nowait(event)
            except queue.Full:
                pass  # slow consumer: drop rather than block capture
        return event

    def recent(self, limit: int = 100, types: list[str] | None = None) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        from lucius.storage.db import loads

        if types:
            marks = ",".join("?" for _ in types)
            rows = self._db.query(
                f"SELECT * FROM event_log WHERE type IN ({marks}) ORDER BY id DESC LIMIT ?", (*types, limit)
            )
        else:
            rows = self._db.query("SELECT * FROM event_log ORDER BY id DESC LIMIT ?", (limit,))
        return [
            {"id": r["id"], "type": r["type"], "ts": r["ts"], "subject_id": r["subject_id"],
             "payload": loads(r["payload"], {})}
            for r in rows
        ]
