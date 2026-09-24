"""WATCH ME: the demonstration recorder.

Capture first, interpret later. The recorder never waits on a model or on downstream
processing: it writes frames and events incrementally (SQLite + an fsync'ed JSONL journal) so
a crash of Blender, the agent or the machine loses at most the last flush interval.

Threads:
    capture  -- polls the active window at the configured frame rate and stores frames while an
                allowed (Blender) window is focused
    input    -- the OS input backend pushes raw events into the recorder's queue
    bridge   -- the Blender add-on pushes observed operators / undo / state changes
    writer   -- drains the queue into SQLite and the journal every ``flush_interval_s``
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lucius.blender.bridge import BlenderBridge
from lucius.blender.state import BlenderState
from lucius.config import LuciusConfig
from lucius.errors import BlenderBridgeError, RecorderError
from lucius.events.bus import EventBus, EventType
from lucius.logging_setup import get_logger
from lucius.provenance import DataPolicy
from lucius.recorder.compression import MouseMoveCompressor
from lucius.recorder.keys import MODIFIERS
from lucius.recorder.ledger import AgentActionLedger
from lucius.recorder.privacy import PrivacyDecision, PrivacyFilter
from lucius.recorder.sources import FrameGrabber, InputSource, RawInput
from lucius.recorder.window import WindowContextProvider, WindowInfo, os_version
from lucius.sessions.models import (
    Actor,
    CapturedEvent,
    EnvironmentInfo,
    EventKind,
    Outcome,
    Session,
    SessionKind,
    SessionStatus,
)
from lucius.sessions.store import SessionStore
from lucius.storage.frames import thumbnail, visual_change
from lucius.storage.jsonl import JsonlAppender
from lucius.timeutil import now

log = get_logger("recorder")

MAX_STATIC_FRAME_INTERVAL_S = 5.0
IDENTICAL_CHANGE = 0.002  # mean thumbnail difference below which a frame adds nothing


@dataclass
class RecorderSources:
    window: WindowContextProvider | None = None
    grabber_factory: Callable[[], FrameGrabber] | None = None
    input: InputSource | None = None
    bridge: BlenderBridge | None = None
    notes: dict[str, str] = field(default_factory=dict)  # why a source is unavailable


@dataclass
class _Stats:
    frames: int = 0
    frames_skipped_identical: int = 0
    frames_skipped_privacy: int = 0
    events: int = 0
    input_dropped_privacy: int = 0
    moves_compressed: int = 0
    bridge_events: int = 0
    errors: list[str] = field(default_factory=list)


class DemonstrationRecorder:
    def __init__(self, config: LuciusConfig, sessions: SessionStore, bus: EventBus, sources: RecorderSources,
                 ledger: AgentActionLedger | None = None,
                 on_human_interference: Callable[[CapturedEvent], None] | None = None) -> None:
        self.config = config
        self.sessions = sessions
        self.bus = bus
        self.sources = sources
        self.ledger = ledger
        self.on_human_interference = on_human_interference
        self.privacy = PrivacyFilter(config.privacy, capture_only_blender=config.recording.capture_only_blender)
        self._session: Session | None = None
        self._seq = itertools.count()
        self._seq_lock = threading.Lock()
        self._queue: queue.Queue[CapturedEvent] = queue.Queue()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._threads: list[threading.Thread] = []
        self._window: WindowInfo | None = None
        self._window_lock = threading.Lock()
        self._window_key: tuple[Any, ...] | None = None
        self._decision = PrivacyDecision(False, False, "not_started")
        self._modifiers: set[str] = set()
        self._compressor = MouseMoveCompressor(config.recording.mouse_move_hz, config.recording.mouse_min_distance_px)
        self._journal: JsonlAppender | None = None
        self._stats = _Stats()
        self._last_thumb = None
        self._last_frame_ts = 0.0
        self._focus_lost_emitted = False

    # -- lifecycle ---------------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._session is not None and not self._stop.is_set()

    @property
    def session(self) -> Session | None:
        return self._session

    def start(self, *, task_text: str | None = None, task_id: str | None = None,
              kind: SessionKind = SessionKind.LIVE_DEMO, policy: DataPolicy | None = None,
              reference_ids: list[str] | None = None, meta: dict[str, Any] | None = None) -> Session:
        if self.recording:
            raise RecorderError("a recording is already in progress", session_id=self._session.id)
        if not any((self.sources.grabber_factory, self.sources.input, self.sources.bridge)):
            raise RecorderError("no capture source available", notes=self.sources.notes)
        self._stop.clear()
        self._paused.clear()
        self._stats = _Stats()
        self._seq = itertools.count()
        env = self._environment()
        session = self.sessions.create(
            user_id=self.config.user_id, kind=kind, policy=policy or DataPolicy.for_live_demo(),
            task_text=task_text, task_id=task_id, environment=env,
            recording_config=self.config.recording.model_dump(), reference_ids=reference_ids or [],
            meta={**(meta or {}), "capture_sources": self._source_summary()},
        )
        self._session = session
        journal_dir = self.config.journal_dir
        self._journal = JsonlAppender(journal_dir / f"{session.id}.jsonl")
        _write_lock(journal_dir / f"{session.id}.lock")
        self._spawn(self._writer_loop, "lucius-recorder-writer")
        if self.sources.bridge is not None:
            self._attach_bridge(self.sources.bridge)
        if self.sources.input is not None:
            try:
                self.sources.input.start(self._on_input)
            except Exception as exc:
                self._note_error(f"input source failed: {exc}")
        if self.sources.grabber_factory is not None or self.sources.window is not None:
            self._spawn(self._capture_loop, "lucius-recorder-capture")
        self.bus.publish(EventType.SESSION_STARTED, session.id, kind=kind.value, task_text=task_text,
                         sources=self._source_summary())
        log.info("recording started: %s", session.id)
        return session

    def stop(self, outcome: Outcome | None = None) -> Session:
        session = self._require_session()
        if self.sources.bridge is not None:
            self._detach_bridge(self.sources.bridge)
        if self.sources.input is not None:
            try:
                self.sources.input.stop()
            except Exception as exc:
                self._note_error(f"input stop failed: {exc}")
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=10)
        self._threads.clear()
        self._drain()
        if self._journal is not None:
            self._journal.close()
            self._journal = None
        finalized = self.sessions.finalize(session.id, outcome=outcome)
        self.sessions.update(session.id, meta={**finalized.meta, "capture_stats": self.stats()})
        _remove_lock(self.config.journal_dir / f"{session.id}.lock")
        self._session = None
        self.bus.publish(EventType.SESSION_ENDED, session.id, kind=finalized.kind.value, stats=self.stats(),
                         duration=finalized.duration)
        log.info("recording stopped: %s (%d frames, %d events)", session.id, self._stats.frames, self._stats.events)
        return self.sessions.get(session.id)

    def pause(self) -> None:
        self._require_session()
        self._paused.set()
        self._emit(EventKind.PAUSED, Actor.HUMAN, {}, blender_active=None)

    def resume(self) -> None:
        self._require_session()
        self._paused.clear()
        self._emit(EventKind.RESUMED, Actor.HUMAN, {}, blender_active=None)

    # -- markers used by other subsystems ------------------------------------------------------
    def annotate(self, text: str, *, label: str | None = None) -> None:
        """A human-provided note (e.g. why they are correcting something)."""
        self._emit(EventKind.ANNOTATION, Actor.HUMAN, {"text": text, "label": label}, blender_active=None)

    def mark(self, label: str, **payload: Any) -> None:
        self._emit(EventKind.MARKER, Actor.SYSTEM, {"label": label, **payload}, blender_active=None)

    def record_agent_action(self, action: dict[str, Any], result: dict[str, Any] | None = None) -> None:
        self._emit(EventKind.AGENT_ACTION, Actor.AGENT, {"action": action, "result": result}, blender_active=None)

    def takeover_start(self, *, reason: str | None, agent_context: dict[str, Any]) -> float:
        ts = now()
        self._emit(EventKind.TAKEOVER_START, Actor.HUMAN,
                   {"reason": reason, "agent_context": agent_context, "state": self._state_snapshot()},
                   blender_active=None, ts=ts)
        return ts

    def takeover_end(self, *, reason: str | None = None) -> float:
        ts = now()
        self._emit(EventKind.TAKEOVER_END, Actor.HUMAN, {"reason": reason, "state": self._state_snapshot()},
                   blender_active=None, ts=ts)
        return ts

    def flush(self) -> None:
        self._drain()

    # -- status ----------------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        s = self._stats
        return {"frames": s.frames, "frames_skipped_identical": s.frames_skipped_identical,
                "frames_skipped_privacy": s.frames_skipped_privacy, "events": s.events,
                "input_dropped_privacy": s.input_dropped_privacy, "moves_compressed": self._compressor.dropped,
                "bridge_events": s.bridge_events, "errors": s.errors[-10:]}

    def status(self) -> dict[str, Any]:
        session = self._session
        return {
            "recording": self.recording,
            "paused": self._paused.is_set(),
            "session_id": session.id if session else None,
            "task_text": session.task_text if session else None,
            "started_at": session.start_time if session else None,
            "blender_focused": self._decision.is_blender,
            "capture_allowed": self._decision.allowed,
            "privacy_reason": self._decision.reason,
            "sources": self._source_summary(),
            "stats": self.stats(),
        }

    # -- internals: environment ----------------------------------------------------------------
    def _source_summary(self) -> dict[str, Any]:
        return {
            "screen": self.sources.grabber_factory is not None,
            "window": getattr(self.sources.window, "name", None),
            "input": self.sources.input is not None,
            "blender_bridge": self.sources.bridge is not None and self.sources.bridge.connected,
            "unavailable": dict(self.sources.notes),
        }

    def _environment(self) -> EnvironmentInfo:
        env = EnvironmentInfo(os_version=os_version())
        provider = self.sources.window
        if provider is not None:
            try:
                with self._window_lock:
                    monitors = provider.monitors()
                    env.dpi_scale = provider.dpi_scale()
                env.monitor_layout = [m.model_dump() for m in monitors]
                if monitors:
                    env.resolution = f"{monitors[0].width}x{monitors[0].height}"
            except Exception as exc:
                self._note_error(f"monitor query failed: {exc}")
        bridge = self.sources.bridge
        if bridge is not None and bridge.connected:
            env.blender_version = bridge.info.get("blender_version")
        return env

    # -- internals: events -----------------------------------------------------------------------
    def _next_seq(self) -> int:
        with self._seq_lock:
            return next(self._seq)

    def _emit(self, kind: EventKind, actor: Actor, payload: dict[str, Any], *, blender_active: bool | None,
              ts: float | None = None) -> CapturedEvent | None:
        if self._session is None:
            return None
        event = CapturedEvent(seq=self._next_seq(), ts=ts if ts is not None else now(), kind=kind, actor=actor,
                              blender_active=blender_active, payload=payload)
        self._queue.put(event)
        return event

    def _drain(self) -> None:
        batch: list[CapturedEvent] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not batch or self._session is None:
            return
        batch.sort(key=lambda e: e.seq)
        if self._journal is not None:
            for event in batch:
                self._journal.append({"session_id": self._session.id, **event.model_dump(mode="json")})
            self._journal.flush()
        self.sessions.append_events(self._session.id, batch)
        self._stats.events += len(batch)
        actions = [e for e in batch if e.kind in (EventKind.MOUSE_DOWN, EventKind.KEY_DOWN, EventKind.BLENDER_OPERATOR,
                                                   EventKind.UNDO, EventKind.REDO, EventKind.SCROLL)]
        if actions:
            self.bus.publish(EventType.ACTION_CAPTURED, self._session.id, count=len(actions),
                             last={"kind": actions[-1].kind.value, "payload": actions[-1].payload})

    def _writer_loop(self) -> None:
        interval = self.config.recording.flush_interval_s
        while not self._stop.wait(interval):
            try:
                self._drain()
            except Exception as exc:  # keep capturing; the journal still holds the events
                self._note_error(f"flush failed: {exc}")
                log.exception("recorder flush failed")

    # -- internals: window & frames --------------------------------------------------------------
    def _refresh_window(self) -> tuple[WindowInfo | None, PrivacyDecision]:
        provider = self.sources.window
        if provider is None:
            # Without window information we cannot scope capture to Blender, so we only allow it
            # when the operator explicitly disabled Blender-only capture.
            decision = PrivacyDecision(not self.config.recording.capture_only_blender, False, "no_window_provider")
            self._decision = decision
            return None, decision
        try:
            with self._window_lock:
                window = provider.active_window()
        except Exception as exc:
            self._note_error(f"window query failed: {exc}")
            window = None
        decision = self.privacy.decide(window)
        self._window, self._decision = window, decision
        key = (window.window_id, window.title, window.bounds) if window and decision.allowed else (decision.reason,)
        if key != self._window_key:
            self._window_key = key
            if decision.allowed and window is not None:
                payload = {"allowed": True, "is_blender": decision.is_blender, "title": window.title,
                           "process": window.process, "pid": window.pid,
                           "bounds": window.bounds.model_dump() if window.bounds else None,
                           "monitor": window.monitor}
            else:
                # Never store the identity of windows we are not allowed to capture.
                payload = {"allowed": False, "reason": decision.reason}
            self._emit(EventKind.WINDOW_CONTEXT, Actor.SYSTEM, payload, blender_active=decision.is_blender)
        return window, decision

    def _capture_loop(self) -> None:
        grabber: FrameGrabber | None = None
        if self.sources.grabber_factory is not None:
            try:
                grabber = self.sources.grabber_factory()
            except Exception as exc:
                self._note_error(f"screen capture unavailable: {exc}")
        interval = 1.0 / self.config.recording.fps
        try:
            while not self._stop.is_set():
                started = now()
                window, decision = self._refresh_window()
                if grabber is not None and not self._paused.is_set():
                    if decision.allowed:
                        self._capture_frame(grabber, window)
                    else:
                        self._stats.frames_skipped_privacy += 1
                self._stop.wait(max(0.0, interval - (now() - started)))
        finally:
            if grabber is not None:
                grabber.close()

    def _capture_frame(self, grabber: FrameGrabber, window: WindowInfo | None) -> None:
        session = self._session
        if session is None:
            return
        region = window.bounds if window is not None else None
        try:
            image = grabber.grab(region)
        except Exception as exc:
            self._note_error(f"grab failed: {exc}")
            return
        ts = now()
        thumb = thumbnail(image)
        change = visual_change(self._last_thumb, thumb)
        if (self.config.recording.skip_identical_frames and change is not None and change < IDENTICAL_CHANGE
                and ts - self._last_frame_ts < MAX_STATIC_FRAME_INTERVAL_S):
            self._stats.frames_skipped_identical += 1
            return
        try:
            frame = self.sessions.add_frame(
                session.id, image, seq=self._next_seq(), ts=ts, source="window" if region else "screen",
                window_bounds=region.model_dump() if region else None, previous_thumb=self._last_thumb,
            )
        except Exception as exc:
            self._note_error(f"frame store failed: {exc}")
            return
        self._last_thumb = thumb
        self._last_frame_ts = ts
        self._stats.frames += 1
        self.bus.publish(EventType.FRAME_CAPTURED, session.id, frame_id=frame.id, ts=ts,
                         change_score=frame.change_score)

    # -- internals: input ------------------------------------------------------------------------
    def _on_input(self, raw: RawInput) -> None:
        if not self.recording or self._paused.is_set():
            return
        if raw.kind in ("mouse_down", "key_down"):
            window, decision = self._refresh_window()  # focus often changes on click
        else:
            window, decision = self._window, self._decision
        if not decision.allowed and self.config.privacy.drop_input_outside_allowed:
            self._stats.input_dropped_privacy += 1
            if not self._focus_lost_emitted:
                self._focus_lost_emitted = True
                self._emit(EventKind.FOCUS_LOST, Actor.SYSTEM, {"reason": decision.reason}, blender_active=False)
            return
        self._focus_lost_emitted = False
        payload = dict(raw.payload)
        if raw.kind == "mouse_move":
            if not self._compressor.offer(raw.ts, payload["x"], payload["y"]):
                return
        elif raw.kind in ("mouse_down", "mouse_up"):
            pending = self._compressor.flush_pending()
            if pending is not None:
                self._record_input("mouse_move", pending[0], {"x": pending[1], "y": pending[2]}, window, decision,
                                   raw.injected)
        elif raw.kind in ("key_down", "key_up"):
            key = payload.get("key")
            if key in MODIFIERS:
                (self._modifiers.add if raw.kind == "key_down" else self._modifiers.discard)(key)
            payload["modifiers"] = sorted(self._modifiers - {key})
        self._record_input(raw.kind, raw.ts, payload, window, decision, raw.injected)

    def _record_input(self, kind: str, ts: float, payload: dict[str, Any], window: WindowInfo | None,
                      decision: PrivacyDecision, injected: bool | None) -> None:
        if window is not None and window.bounds is not None and "x" in payload:
            payload["wx"] = payload["x"] - window.bounds.x
            payload["wy"] = payload["y"] - window.bounds.y
        actor = Actor.HUMAN
        if injected:
            actor = Actor.AGENT
        elif self.ledger is not None and self.ledger.attribute(ts, kind, payload) == "agent":
            actor = Actor.AGENT
        event = self._emit(EventKind(kind), actor, payload, blender_active=decision.is_blender, ts=ts)
        if (event is not None and actor == Actor.HUMAN and self.ledger is not None and kind != "mouse_move"
                and self.ledger.agent_active(ts) and self.on_human_interference is not None):
            self.on_human_interference(event)

    # -- internals: Blender bridge ---------------------------------------------------------------
    def _attach_bridge(self, bridge: BlenderBridge) -> None:
        try:
            bridge.subscribe(self._on_bridge_push)
            bridge.set_recording_indicator(True)
            initial = bridge.get_state()
            self._emit(EventKind.BLENDER_STATE, Actor.SYSTEM, {**initial.model_dump(), "reason": "initial"},
                       blender_active=True)
        except BlenderBridgeError as exc:
            self._note_error(f"blender bridge unavailable: {exc}")

    def _detach_bridge(self, bridge: BlenderBridge) -> None:
        if not bridge.connected:
            return
        try:
            final = bridge.get_state()
            self._emit(EventKind.BLENDER_STATE, Actor.SYSTEM, {**final.model_dump(), "reason": "final"},
                       blender_active=True)
            bridge.set_recording_indicator(False)
            bridge.unsubscribe(self._on_bridge_push)
        except BlenderBridgeError as exc:
            self._note_error(f"blender bridge detach failed: {exc}")

    def _on_bridge_push(self, kind: str, ts: float, data: dict[str, Any]) -> None:
        if not self.recording or self._paused.is_set():
            return
        self._stats.bridge_events += 1
        actor = Actor.AGENT if self.ledger is not None and self.ledger.agent_active(ts) else Actor.HUMAN
        if kind == "blender_state":
            state = BlenderState.from_capture(data)
            payload = {**(state.model_dump() if state else {}), "reason": data.get("reason")}
            self._emit(EventKind.BLENDER_STATE, Actor.SYSTEM, payload, blender_active=True, ts=ts)
        elif kind in ("blender_operator", "blender_operator_adjusted"):
            payload = {"operator": data.get("operator", {}), "adjusted": kind.endswith("adjusted"),
                       "geometry_changed": data.get("geometry_changed", [])}
            self._emit(EventKind.BLENDER_OPERATOR, actor, payload, blender_active=True, ts=ts)
        elif kind in ("undo", "redo"):
            self._emit(EventKind.UNDO if kind == "undo" else EventKind.REDO, actor, {}, blender_active=True, ts=ts)

    def _state_snapshot(self) -> dict[str, Any] | None:
        bridge = self.sources.bridge
        if bridge is None or not bridge.connected:
            return None
        try:
            return bridge.get_state().compact()
        except BlenderBridgeError:
            return None

    # -- misc ----------------------------------------------------------------------------------------
    def _spawn(self, target: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _require_session(self) -> Session:
        if self._session is None:
            raise RecorderError("no recording in progress")
        return self._session

    def _note_error(self, message: str) -> None:
        self._stats.errors.append(message)
        log.warning(message)


# -- crash recovery -----------------------------------------------------------------------------

def _write_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "ts": now()}))


def _remove_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_interrupted_sessions(sessions: SessionStore, journal_dir: Path, bus: EventBus | None = None,
                                 *, active_session_ids: set[str] | None = None) -> list[str]:
    """Finalize recordings whose recorder died, replaying any journaled events not yet in SQLite.

    Raw evidence is never discarded: the session is kept with status ``interrupted`` and can be
    processed like any other.
    """
    recovered = []
    active = active_session_ids or set()
    for session in sessions.list(status=SessionStatus.RECORDING, limit=10000):
        if session.id in active:
            continue
        lock = journal_dir / f"{session.id}.lock"
        if lock.exists():
            try:
                info = json.loads(lock.read_text())
                if info.get("host") == socket.gethostname() and _pid_alive(int(info.get("pid", 0))) \
                        and int(info.get("pid", 0)) != os.getpid():
                    continue  # another live process is still recording it
            except (ValueError, OSError):
                pass
        journal = journal_dir / f"{session.id}.jsonl"
        replayed = 0
        if journal.exists():
            from lucius.storage.jsonl import read_jsonl

            events = [CapturedEvent.model_validate({k: v for k, v in row.items() if k != "session_id"})
                      for row in read_jsonl(journal)]
            before = sessions.event_count(session.id)
            sessions.append_events(session.id, events)
            replayed = sessions.event_count(session.id) - before
        end = sessions.last_activity(session.id) or session.start_time
        sessions.finalize(session.id, end_time=end, status=SessionStatus.INTERRUPTED)
        sessions.update(session.id, meta={**session.meta, "recovered": {"at": now(), "replayed_events": replayed}})
        _remove_lock(lock)
        recovered.append(session.id)
        if bus is not None:
            bus.publish(EventType.SESSION_RECOVERED, session.id, replayed_events=replayed)
    return recovered
