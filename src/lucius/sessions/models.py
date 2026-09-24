"""Session and raw-event models (the V0 'clean demonstration session')."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from lucius.provenance import DataPolicy, SourceClass


class SessionKind(StrEnum):
    LIVE_DEMO = "live_demo"
    AGENT_EXECUTION = "agent_execution"
    PRACTICE = "practice"
    EXTERNAL_MEDIA = "external_media"
    BENCHMARK = "benchmark"
    VALIDATION = "validation"


class SessionStatus(StrEnum):
    RECORDING = "recording"
    FINALIZED = "finalized"
    INTERRUPTED = "interrupted"   # recorder died; data kept and recoverable
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    EXECUTED_UNVERIFIED = "executed_unverified"
    UNKNOWN = "unknown"


class EventKind(StrEnum):
    MOUSE_MOVE = "mouse_move"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    SCROLL = "scroll"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    WINDOW_CONTEXT = "window_context"
    BLENDER_STATE = "blender_state"
    BLENDER_OPERATOR = "blender_operator"
    UNDO = "undo"
    REDO = "redo"
    TAKEOVER_START = "takeover_start"
    TAKEOVER_END = "takeover_end"
    AGENT_ACTION = "agent_action"
    ANNOTATION = "annotation"
    MARKER = "marker"
    FOCUS_LOST = "focus_lost"
    PAUSED = "recording_paused"
    RESUMED = "recording_resumed"
    MEDIA_OBSERVATION = "media_observation"


class Actor(StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


class EnvironmentInfo(BaseModel):
    blender_version: str | None = None
    os_version: str | None = None
    resolution: str | None = None
    dpi_scale: float | None = None
    monitor_layout: list[dict[str, Any]] = Field(default_factory=list)


class Session(BaseModel):
    id: str
    user_id: str
    kind: SessionKind
    source: SourceClass
    status: SessionStatus
    task_id: str | None = None
    task_text: str | None = None
    task_class: str | None = None
    reference_ids: list[str] = Field(default_factory=list)
    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    start_time: float
    end_time: float | None = None
    outcome: Outcome | None = None
    recording_config: dict[str, Any] = Field(default_factory=dict)
    policy: DataPolicy
    content_hash: str | None = None
    processing: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        return None if self.end_time is None else self.end_time - self.start_time


class CapturedEvent(BaseModel):
    seq: int
    ts: float
    kind: EventKind
    actor: Actor = Actor.HUMAN
    blender_active: bool | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class FrameRecord(BaseModel):
    id: str
    session_id: str
    seq: int
    ts: float
    path: str
    sha256: str
    width: int
    height: int
    source: str
    media_asset_id: str | None = None
    media_timestamp: float | None = None
    window_bounds: dict[str, int] | None = None
    active_area: str | None = None
    dhash: str | None = None
    change_score: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
