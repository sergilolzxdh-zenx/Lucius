"""Local HTTP API and control-center host (sections 55-63, 10W).

The server binds to loopback. Two protections apply to every request:

* **Host check** -- only loopback host names are accepted (defeats DNS rebinding);
* **Token** -- every mutating request must carry ``X-Lucius-Token``. The token is written to
  ``<data_dir>/api_token`` (owner-only) and injected into the control-center page, which a
  cross-origin page cannot read.

Long operations (agent runs, practice, experiments) run as background jobs; the UI follows
them through ``/api/events/stream`` (server-sent events from the event bus).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import secrets
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from lucius.api.jobs import JobManager
from lucius.app import Lucius
from lucius.config import LuciusConfig
from lucius.events.bus import EventType
from lucius.errors import (
    ActionRejected,
    BlenderUnavailable,
    CaptureUnavailable,
    ConflictError,
    LuciusError,
    NotFoundError,
    PolicyViolation,
    ProviderUnavailable,
    RecorderError,
    ValidationError,
)
from lucius.ids import new_id
from lucius.logging_setup import get_logger
from lucius.planner.model import PlanAction
from lucius.provenance import ConsentStatus, DataPolicy
from lucius.retrieval import RetrievalQuery
from lucius.sessions import Outcome, SessionKind
from lucius.skills import SkillStatus
from lucius.storage.db import loads
from lucius.timeutil import now

log = get_logger(__name__)
STATIC = Path(__file__).parent / "static"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
MAX_UPLOAD_BYTES = 4 * 1024 ** 3
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

STATUS_CODES: list[tuple[type[LuciusError], int]] = [
    (NotFoundError, 404), (ConflictError, 409), (PolicyViolation, 403), (ValidationError, 422),
    (ActionRejected, 422), (CaptureUnavailable, 503), (RecorderError, 409), (BlenderUnavailable, 503),
    (ProviderUnavailable, 503),
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def load_token(config: LuciusConfig) -> str:
    """The API token for this data directory (created once, owner-only permissions)."""
    path = config.data_dir / "api_token"
    if path.exists():
        token = path.read_text().strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    path.write_text(token)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


# -- request bodies -----------------------------------------------------------------------------------

class WatchStart(BaseModel):
    task_text: str | None = None
    training_consent: bool = False
    reference_ids: list[str] = Field(default_factory=list)


class WatchStop(BaseModel):
    outcome: Outcome | None = None


class Note(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    label: str | None = None


class TakeoverResume(BaseModel):
    reason: str | None = None
    note: str | None = None


class HumanAction(BaseModel):
    action: dict[str, Any]


class PolicyUpdate(BaseModel):
    training_consent: bool | None = None
    export_allowed: bool | None = None
    memory_allowed: bool | None = None
    learning_allowed: bool | None = None
    license: str | None = None


class Relabel(BaseModel):
    label: str
    title: str | None = None


class MergeSegments(BaseModel):
    segment_ids: list[str]
    label: str | None = None


class SplitSegment(BaseModel):
    at_step: int


class SegmentOutcome(BaseModel):
    outcome: str


class IntentEdit(BaseModel):
    category: str
    target: str | None = None
    scope: str | None = None


class ConfirmAction(BaseModel):
    action_type: str
    params: dict[str, Any] | None = None


class SkillEdit(BaseModel):
    changes: dict[str, Any]


class Version(BaseModel):
    version: int


class Accept(BaseModel):
    accept: bool
    note: str = ""


class Disable(BaseModel):
    disabled: bool


class MergeInto(BaseModel):
    into: str


class SplitSkill(BaseModel):
    phases: list[str]
    new_name: str


class FailureEdit(BaseModel):
    changes: dict[str, Any]


class PreferenceValue(BaseModel):
    value: Any


class RetrievalDebug(BaseModel):
    text: str
    task_class: str | None = None
    categories: list[str] = Field(default_factory=list)
    strategy: str = "hybrid"
    top_k: int = 8


class RunRequest(BaseModel):
    task_text: str = Field(min_length=1, max_length=2000)
    backend: str = "live"                       # live | headless
    reference_ids: list[str] = Field(default_factory=list)
    task_params: dict[str, Any] = Field(default_factory=dict)
    allow_takeover: bool = True


class Review(BaseModel):
    passed: bool
    rating: int | None = Field(default=None, ge=1, le=5)
    feedback: str | None = None


class TrainRequest(BaseModel):
    stage: int | None = None
    attempts: int = Field(default=3, ge=1, le=50)
    seed: int | None = None
    backend: str = "headless"
    allow_takeover: bool = False


class Override(BaseModel):
    status: str | None = None                  # mastered | locked | None (clear)


class DatasetRequest(BaseModel):
    name: str
    purpose: str = "training"
    kinds: list[str] | None = None
    sources: list[str] | None = None
    task_class: str | None = None
    min_quality: float = 0.5
    session_ids: list[str] | None = None
    include_media: bool = False


class ExperimentRequest(BaseModel):
    name: str
    arms: list[str]
    benchmarks: list[str]
    repeats: int = Field(default=1, ge=1, le=20)
    backend: str = "headless"


class DemoPolicy(BaseModel):
    training_allowed: bool | None = None
    export_allowed: bool | None = None
    reference_only: bool | None = None
    license: str | None = None
    consent: bool | None = None


class SettingsUpdate(BaseModel):
    recording: dict[str, Any] | None = None
    privacy: dict[str, Any] | None = None
    providers: dict[str, Any] | None = None
    processing: dict[str, Any] | None = None
    safety: dict[str, Any] | None = None
    blender: dict[str, Any] | None = None


# -- application ----------------------------------------------------------------------------------------

def create_app(lucius: Lucius, *, token: str | None = None) -> FastAPI:
    token = token or load_token(lucius.config)
    jobs = JobManager(lucius.bus)
    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        jobs.shutdown()

    api = FastAPI(title="Lucius", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json",
                  lifespan=lifespan)
    api.state.lucius = lucius
    api.state.jobs = jobs
    api.state.token = token

    @api.middleware("http")
    async def guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        if (request.url.hostname or "") not in LOOPBACK_HOSTS:
            return JSONResponse({"code": "forbidden_host", "message": "Lucius only serves loopback hosts"}, 403)
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/"):
            if not secrets.compare_digest(request.headers.get("x-lucius-token", ""), token):
                return JSONResponse({"code": "unauthorized", "message": "missing or invalid X-Lucius-Token"}, 401)
        return await call_next(request)

    @api.exception_handler(LuciusError)
    async def lucius_error(_request: Request, exc: LuciusError) -> JSONResponse:
        status = next((code for cls, code in STATUS_CODES if isinstance(exc, cls)), 500)
        return JSONResponse(exc.to_dict(), status_code=status)

    @api.exception_handler(PydanticValidationError)
    async def invalid(_request: Request, exc: PydanticValidationError) -> JSONResponse:
        errors = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse({"code": "invalid", "message": "; ".join(f"{e['field']}: {e['message']}" for e in errors),
                             "details": {"errors": errors}}, status_code=422)

    r = APIRouter(prefix="/api")

    def backend_factory(name: str):  # type: ignore[no-untyped-def]
        if name == "live":
            lucius.live_backend()  # fail fast (503) if no Blender with the add-on is running
            return lucius.live_backend
        if name == "headless":
            return lucius.headless_backend
        raise ValidationError(f"unknown backend {name!r}; use 'live' or 'headless'")

    # -- status & observability --------------------------------------------------------------------------
    @r.get("/status")
    def status() -> dict[str, Any]:
        db = lucius.db
        rec = lucius._recorder
        skills: dict[str, int] = {}
        for row in db.query("SELECT status, COUNT(*) AS n FROM skills GROUP BY status"):
            skills[row["status"]] = row["n"]
        runs = {row["status"]: row["n"] for row in db.query("SELECT status, COUNT(*) AS n FROM runs GROUP BY status")}
        pending = lucius.human.pending
        return {
            "recording": rec.status() if rec is not None else {"recording": False, "sources": None},
            "takeover": pending.model_dump(mode="json") if pending else None,
            "jobs": [j.model_dump(mode="json") for j in jobs.active()],
            "counts": {
                "sessions": db.scalar("SELECT COUNT(*) FROM sessions"),
                "demonstrations": db.scalar("SELECT COUNT(*) FROM sessions WHERE kind IN ('live_demo','external_media')"),
                "skills": skills, "runs": runs,
                "learned_skills": db.scalar("SELECT COUNT(*) FROM skills WHERE source_class != 'system_seeded' "
                                            "AND status NOT IN ('disabled','merged')"),
                "validated_skills": db.scalar("SELECT COUNT(*) FROM skills WHERE source_class != 'system_seeded' "
                                              "AND status IN ('validated','high_confidence')"),
                "failures": db.scalar("SELECT COUNT(*) FROM failure_records"),
                "episodes": db.scalar("SELECT COUNT(*) FROM memory_episodes"),
                "needs_review": db.scalar("SELECT COUNT(*) FROM runs WHERE status = 'needs_human'"),
            },
            "providers": {**lucius.providers.available(), "notes": lucius.provider_notes},
            "live_blender": lucius._backend is not None and lucius._backend.bridge.connected,
            "time": now(),
        }

    @r.get("/metrics")
    def metrics() -> dict[str, Any]:
        db = lucius.db
        calls = db.query("SELECT provider, purpose, status, COUNT(*) AS n, AVG(latency_s) AS latency_s, "
                         "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens FROM model_calls "
                         "GROUP BY provider, purpose, status")
        stages = db.query("SELECT stage, status, COUNT(*) AS n FROM processing_jobs GROUP BY stage, status")
        verdicts = db.query("SELECT mode, status, COUNT(*) AS n FROM runs GROUP BY mode, status")
        rate = db.query_one("SELECT AVG(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END) AS rate, COUNT(*) AS n "
                            "FROM runs WHERE mode = 'execute'")
        return {"model_calls": [dict(c) for c in calls], "processing": [dict(s) for s in stages],
                "runs": [dict(v) for v in verdicts],
                "execution_success_rate": rate["rate"] if rate and rate["n"] else None,
                "takeovers": db.scalar("SELECT COUNT(*) FROM corrections WHERE kind = 'takeover'"),
                "events": lucius.bus.recent(limit=30)}

    @r.get("/events/recent")
    def events_recent(limit: int = 100, types: str | None = None) -> list[dict[str, Any]]:
        return lucius.bus.recent(limit=min(limit, 1000), types=types.split(",") if types else None)

    @r.get("/events/stream")
    async def events_stream(request: Request) -> StreamingResponse:
        stream = lucius.bus.open_stream()

        async def gen():  # type: ignore[no-untyped-def]
            try:
                yield "retry: 3000\n\n"
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.to_thread(stream.get, True, 1.0)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {event.model_dump_json()}\n\n"
            finally:
                lucius.bus.close_stream(stream)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @r.get("/jobs")
    def jobs_list() -> list[dict[str, Any]]:
        return [j.model_dump(mode="json") for j in jobs.list()]

    @r.get("/jobs/{job_id}")
    def job_get(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return _jsonable(job.model_dump(mode="json"))

    # -- WATCH ME ------------------------------------------------------------------------------------------
    @r.get("/watch")
    def watch_status() -> dict[str, Any]:
        rec = lucius._recorder
        return rec.status() if rec is not None else {"recording": False, "sources": None}

    @r.post("/watch/sources")
    def watch_detect() -> dict[str, Any]:
        rec = lucius._recorder
        if rec is not None and rec.recording:
            raise ConflictError("cannot re-detect capture sources while recording")
        from lucius.recorder import detect_sources

        return lucius.recorder(detect_sources(lucius.config)).status()

    @r.post("/watch/start")
    def watch_start(body: WatchStart) -> dict[str, Any]:
        recorder = lucius.recorder()
        session = recorder.start(task_text=body.task_text,
                                 policy=DataPolicy.for_live_demo(training_consent=body.training_consent),
                                 reference_ids=body.reference_ids)
        return {"session_id": session.id, **recorder.status()}

    @r.post("/watch/stop")
    def watch_stop(body: WatchStop) -> dict[str, Any]:
        rec = lucius._recorder
        if rec is None or not rec.recording:
            raise ConflictError("no recording in progress")
        session = rec.stop(outcome=body.outcome)
        return {"session": session.model_dump(mode="json")}

    @r.post("/watch/pause")
    def watch_pause() -> dict[str, Any]:
        rec = _active_recorder()
        rec.pause()
        return rec.status()

    @r.post("/watch/resume")
    def watch_resume() -> dict[str, Any]:
        rec = _active_recorder()
        rec.resume()
        return rec.status()

    @r.post("/watch/annotate")
    def watch_annotate(body: Note) -> dict[str, Any]:
        rec = _active_recorder()
        rec.annotate(body.text, label=body.label)
        return {"ok": True}

    def _active_recorder():  # type: ignore[no-untyped-def]
        rec = lucius._recorder
        if rec is None or not rec.recording:
            raise ConflictError("no recording in progress")
        return rec

    # -- human takeover --------------------------------------------------------------------------------------
    @r.get("/takeover")
    def takeover() -> dict[str, Any] | None:
        pending = lucius.human.pending
        return {"pending": pending.model_dump(mode="json") if pending else None,
                "can_act": jobs.active_backend is not None}

    @r.post("/takeover/resume")
    def takeover_resume(body: TakeoverResume) -> dict[str, Any]:
        if not lucius.human.resume(reason=body.reason, note=body.note):
            raise ConflictError("no takeover is pending")
        return {"ok": True}

    @r.post("/takeover/abort")
    def takeover_abort(body: TakeoverResume) -> dict[str, Any]:
        if not lucius.human.abort(note=body.note):
            raise ConflictError("no takeover is pending")
        return {"ok": True}

    @r.post("/takeover/action")
    def takeover_action(body: HumanAction) -> dict[str, Any]:
        """A human correction applied through the bridge (same validation as agent actions)."""
        pending = lucius.human.pending
        if pending is None:
            raise ConflictError("no takeover is pending")
        backend = jobs.active_backend
        if backend is None:
            raise ConflictError("the run's Blender backend is not available to the API")
        payload = {"id": new_id("step"), "layer": "blender_api", **body.action}
        payload.setdefault("name", payload.get("action_type", "human_action"))
        action = PlanAction.model_validate(payload)
        result = lucius.engine.record_human_action(pending.run_id, backend, action)
        return _jsonable(result)

    # -- sessions & timeline ------------------------------------------------------------------------------------
    @r.get("/sessions")
    def sessions(kind: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        items = lucius.sessions.list(kind=SessionKind(kind) if kind else None, limit=min(limit, 500), offset=offset)
        return [{**s.model_dump(mode="json", exclude={"recording_config"}),
                 "event_count": lucius.sessions.event_count(s.id), "frame_count": lucius.sessions.frame_count(s.id)}
                for s in items]

    @r.get("/sessions/{session_id}")
    def session_get(session_id: str) -> dict[str, Any]:
        s = lucius.sessions.get(session_id)
        return {"session": s.model_dump(mode="json"), "processing": lucius.pipeline.status(session_id),
                "episode": _jsonable(lucius.episodes.get(session_id)),
                "runs": [dict(r) for r in lucius.db.query("SELECT id, task_text, status, mode, started_at FROM runs "
                                                          "WHERE session_id = ?", (session_id,))]}

    @r.delete("/sessions/{session_id}")
    def session_delete(session_id: str) -> dict[str, Any]:
        rec = lucius._recorder
        if rec is not None and rec.recording and rec.session and rec.session.id == session_id:
            raise ConflictError("stop the recording before deleting it")
        return lucius.sessions.delete(session_id)

    @r.get("/sessions/{session_id}/timeline")
    def timeline(session_id: str) -> dict[str, Any]:
        s = lucius.sessions.get(session_id)
        steps = lucius.trajectories.for_session(session_id)
        segments = lucius.segments.for_session(session_id)
        frames = lucius.sessions.frames_for(session_id)
        markers = [e.model_dump(mode="json") for e in lucius.sessions.events(
            session_id, kinds=["takeover_start", "takeover_end", "undo", "redo", "annotation", "marker", "focus_lost"])]
        return {
            "session": s.model_dump(mode="json", exclude={"recording_config"}),
            "steps": [st.model_dump(mode="json") for st in steps],
            "segments": [{**seg.model_dump(mode="json"),
                          "intent": lucius.intents.summary(lucius.intents.primary(seg.id))} for seg in segments],
            "frames": [{"id": f.id, "seq": f.seq, "ts": f.ts, "source": f.source,
                        "media_timestamp": f.media_timestamp} for f in frames],
            "markers": markers,
        }

    @r.get("/sessions/{session_id}/events")
    def session_events(session_id: str, kinds: str | None = None, limit: int = 2000) -> list[dict[str, Any]]:
        events = lucius.sessions.events(session_id, kinds=kinds.split(",") if kinds else None)
        return [e.model_dump(mode="json") for e in events[:limit]]

    @r.post("/sessions/{session_id}/process")
    def session_process(session_id: str, force: bool = False) -> dict[str, Any]:
        lucius.sessions.get(session_id)
        job = jobs.submit("process", f"process {session_id}",
                          lambda: lucius.pipeline.process(session_id, force=force))
        return job.model_dump(mode="json")

    @r.patch("/sessions/{session_id}/policy")
    def session_policy(session_id: str, body: PolicyUpdate) -> dict[str, Any]:
        """Dataset opt-in/opt-out and other policy changes, always explicit and recorded."""
        s = lucius.sessions.get(session_id)
        updates: dict[str, Any] = {}
        if body.training_consent is not None:
            updates["training_allowed"] = body.training_consent
            updates["consent_status"] = ConsentStatus.GRANTED if body.training_consent else ConsentStatus.DENIED
        for key in ("export_allowed", "memory_allowed", "learning_allowed", "license"):
            if getattr(body, key) is not None:
                updates[key] = getattr(body, key)
        policy = s.policy.model_copy(update=updates)
        lucius.sessions.update(session_id, policy=policy)
        lucius.bus.publish(EventType.POLICY_CHANGED, session_id, changes=_jsonable(updates))
        return {"policy": policy.model_dump(mode="json")}

    @r.get("/sessions/{session_id}/export")
    def session_export(session_id: str, include_frames: bool = True) -> FileResponse:
        path = lucius.datasets.export_session(session_id, include_frames=include_frames)
        return FileResponse(path, filename=path.name, media_type="application/zip")

    @r.post("/sessions/import")
    def session_import(file: UploadFile = File(...)) -> dict[str, Any]:
        path = _save_upload(file, "imports")
        session = lucius.datasets.import_session(path)
        return {"session": session.model_dump(mode="json")}

    @r.get("/frames/{frame_id}/image")
    def frame_image(frame_id: str) -> FileResponse:
        frame = lucius.sessions.frame(frame_id)
        path = lucius.sessions.frames.path(frame.path)
        if not path.exists():
            raise NotFoundError(f"frame file for {frame_id} is missing")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})

    # -- segments, intents, steps ------------------------------------------------------------------------------
    @r.get("/taxonomy/{kind}")
    def taxonomy(kind: str) -> dict[str, Any]:
        return lucius.taxonomy.terms(kind)

    @r.post("/segments/{segment_id}/relabel")
    def seg_relabel(segment_id: str, body: Relabel) -> dict[str, Any]:
        return lucius.segment_editor.relabel(segment_id, body.label, body.title).model_dump(mode="json")

    @r.post("/segments/merge")
    def seg_merge(body: MergeSegments) -> dict[str, Any]:
        return lucius.segment_editor.merge(body.segment_ids, body.label).model_dump(mode="json")

    @r.post("/segments/{segment_id}/split")
    def seg_split(segment_id: str, body: SplitSegment) -> list[dict[str, Any]]:
        return [s.model_dump(mode="json") for s in lucius.segment_editor.split(segment_id, body.at_step)]

    @r.post("/segments/{segment_id}/outcome")
    def seg_outcome(segment_id: str, body: SegmentOutcome) -> dict[str, Any]:
        return lucius.segment_editor.set_outcome(segment_id, body.outcome).model_dump(mode="json")

    @r.get("/segments/{segment_id}/intents")
    def seg_intents(segment_id: str) -> list[dict[str, Any]]:
        return [i.model_dump(mode="json") for i in lucius.intents.for_segment(segment_id)]

    @r.post("/segments/{segment_id}/intent")
    def seg_intent(segment_id: str, body: IntentEdit) -> dict[str, Any]:
        return lucius.intents.set_human(segment_id, category=body.category, target=body.target, scope=body.scope,
                                        user_id=lucius.config.user_id).model_dump(mode="json")

    @r.post("/steps/{step_id}/confirm")
    def step_confirm(step_id: str, body: ConfirmAction) -> dict[str, Any]:
        lucius.ingestion.confirm_action(step_id, body.action_type, body.params, user_id=lucius.config.user_id)
        step = lucius.trajectories.get(step_id)
        return step.model_dump(mode="json") if step else {"ok": True}

    # -- skills ---------------------------------------------------------------------------------------------------
    @r.get("/skills")
    def skills(status: str | None = None, search: str | None = None,
               include_inactive: bool = False) -> list[dict[str, Any]]:
        statuses = {SkillStatus(s) for s in status.split(",")} if status else None
        return [_skill_summary(s) for s in lucius.library.list(statuses=statuses, search=search,
                                                               include_inactive=include_inactive)]

    def _skill_summary(s: Any) -> dict[str, Any]:
        d = s.definition
        return {"id": s.id, "name": d.name, "purpose": d.purpose, "status": s.status.value,
                "confidence": s.confidence, "version": s.current_version, "source_class": s.source_class,
                "object_class": d.object_class, "object_role": d.object_role, "categories": d.categories,
                "usage_count": s.usage_count, "success_count": s.success_count, "success_rate": s.success_rate,
                "origin_sources": s.origin_sources, "updated_at": s.updated_at}

    @r.get("/skills/{skill_id}")
    def skill_get(skill_id: str) -> dict[str, Any]:
        s = lucius.library.get(skill_id)
        failures = lucius.failures.relevant(task_class=s.definition.object_class, skill_ids={skill_id})
        return {
            "skill": s.model_dump(mode="json"), "summary": _skill_summary(s),
            "versions": _jsonable(lucius.library.versions(skill_id)),
            "evidence": _jsonable(lucius.library.evidence(skill_id)),
            "examples": [e.model_dump(mode="json") for e in lucius.library.examples(skill_id)],
            "provenance": _jsonable(lucius.library.provenance(skill_id)),
            "failures": [f.model_dump(mode="json") for f in failures],
            "graph": lucius.graph.neighbourhood("skill", skill_id, depth=1),
        }

    @r.get("/skills/{skill_id}/versions/{version}")
    def skill_version(skill_id: str, version: int) -> dict[str, Any]:
        return lucius.library.definition(skill_id, version).model_dump(mode="json")

    @r.patch("/skills/{skill_id}")
    def skill_edit(skill_id: str, body: SkillEdit) -> dict[str, Any]:
        return _skill_summary(lucius.library.human_edit(skill_id, body.changes, user_id=lucius.config.user_id))

    @r.post("/skills/{skill_id}/rollback")
    def skill_rollback(skill_id: str, body: Version) -> dict[str, Any]:
        return _skill_summary(lucius.library.rollback(skill_id, body.version, user_id=lucius.config.user_id))

    @r.post("/skills/{skill_id}/review")
    def skill_review(skill_id: str, body: Accept) -> dict[str, Any]:
        return _skill_summary(lucius.library.review(skill_id, accept=body.accept, user_id=lucius.config.user_id))

    @r.post("/skills/{skill_id}/disable")
    def skill_disable(skill_id: str, body: Disable) -> dict[str, Any]:
        return _skill_summary(lucius.library.set_disabled(skill_id, body.disabled, user_id=lucius.config.user_id))

    @r.post("/skills/{skill_id}/merge")
    def skill_merge(skill_id: str, body: MergeInto) -> dict[str, Any]:
        return _skill_summary(lucius.library.merge(skill_id, body.into, user_id=lucius.config.user_id))

    @r.post("/skills/{skill_id}/split")
    def skill_split(skill_id: str, body: SplitSkill) -> dict[str, Any]:
        return _skill_summary(lucius.library.split(skill_id, body.phases, new_name=body.new_name,
                                                   user_id=lucius.config.user_id))

    @r.get("/skills/{skill_id}/export")
    def skill_export(skill_id: str) -> FileResponse:
        path = lucius.datasets.export_skill(skill_id)
        return FileResponse(path, filename=path.name, media_type="application/json")

    @r.post("/skills/import")
    def skill_import(file: UploadFile = File(...), as_new: bool = Form(False)) -> dict[str, Any]:
        path = _save_upload(file, "imports")
        return {"skill_id": lucius.datasets.import_skill(path, as_new=as_new)}

    # -- memory, graph, retrieval -----------------------------------------------------------------------------
    @r.get("/memory/episodes")
    def episodes(task_class: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return [e.model_dump(mode="json") for e in lucius.episodes.list(task_class=task_class, limit=limit)]

    @r.get("/memory/semantic")
    def semantic(include_rejected: bool = False) -> list[dict[str, Any]]:
        return [s.model_dump(mode="json") for s in lucius.semantic.list(include_rejected=include_rejected)]

    @r.post("/memory/semantic/{statement_id}/review")
    def semantic_review(statement_id: str, body: Accept) -> dict[str, Any]:
        return lucius.semantic.review(statement_id, accept=body.accept).model_dump(mode="json")

    @r.get("/memory/preferences")
    def preferences() -> dict[str, Any]:
        return {k: v.model_dump(mode="json") for k, v in lucius.preferences.get_all().items()}

    @r.put("/memory/preferences/{key}")
    def preference_set(key: str, body: PreferenceValue) -> dict[str, Any]:
        return lucius.preferences.set(key, body.value).model_dump(mode="json")

    @r.delete("/memory/preferences/{key}")
    def preference_clear(key: str) -> dict[str, Any]:
        lucius.preferences.clear(key)
        return {"ok": True}

    @r.get("/graph")
    def graph(kind: str | None = None, node_id: str | None = None, depth: int = 2,
              limit: int = 400) -> dict[str, Any]:
        """A node's neighbourhood, or (without a node) the most recent part of the learning graph."""
        if kind and node_id:
            return lucius.graph.neighbourhood(kind, node_id, depth=min(depth, 4), limit=min(limit, 1000))
        rows = lucius.db.query("SELECT * FROM graph_edges ORDER BY created_at DESC LIMIT ?", (min(limit, 2000),))
        nodes = {(row["src_kind"], row["src_id"]) for row in rows} | {(row["dst_kind"], row["dst_id"]) for row in rows}
        return {"nodes": [{"kind": k, "id": n} for k, n in nodes],
                "edges": [{**dict(row), "meta": loads(row["meta"], {})} for row in rows]}

    @r.post("/retrieval/debug")
    def retrieval_debug(body: RetrievalDebug) -> dict[str, Any]:
        query = RetrievalQuery(text=body.text, task_class=body.task_class, categories=body.categories,
                               strategy=body.strategy, top_k=body.top_k)  # type: ignore[arg-type]
        return lucius.retriever.retrieve(query).model_dump(mode="json")

    @r.get("/retrieval/records")
    def retrieval_records(limit: int = 30) -> list[dict[str, Any]]:
        rows = lucius.db.query("SELECT * FROM retrieval_records ORDER BY created_at DESC LIMIT ?", (min(limit, 500),))
        return [{**dict(row), "query_meta": loads(row["query_meta"], {}), "results": loads(row["results"], {}),
                 "feedback": loads(row["feedback"], {})} for row in rows]

    # -- failures & corrections --------------------------------------------------------------------------------
    @r.get("/failures")
    def failures(task_class: str | None = None, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return [f.model_dump(mode="json") for f in lucius.failures.list(task_class=task_class, status=status,
                                                                          limit=limit)]

    @r.get("/failures/{failure_id}")
    def failure_get(failure_id: str) -> dict[str, Any]:
        f = lucius.failures.get(failure_id)
        return {"failure": f.model_dump(mode="json"),
                "corrections": [c.model_dump(mode="json") for c in lucius.corrections.list(failure_id=failure_id)],
                "graph": lucius.graph.neighbourhood("failure", failure_id, depth=1)}

    @r.patch("/failures/{failure_id}")
    def failure_edit(failure_id: str, body: FailureEdit) -> dict[str, Any]:
        return lucius.failures.edit(failure_id, body.changes, user_id=lucius.config.user_id).model_dump(mode="json")

    @r.post("/failures/{failure_id}/review")
    def failure_review(failure_id: str, body: Accept) -> dict[str, Any]:
        return lucius.failures.review_rule(failure_id, accept=body.accept, user_id=lucius.config.user_id,
                                           note=body.note).model_dump(mode="json")

    @r.get("/corrections")
    def corrections(run_id: str | None = None, failure_id: str | None = None) -> list[dict[str, Any]]:
        return [c.model_dump(mode="json") for c in lucius.corrections.list(run_id=run_id, failure_id=failure_id)]

    # -- agent runs ---------------------------------------------------------------------------------------------
    @r.post("/runs")
    def run_start(body: RunRequest) -> dict[str, Any]:
        factory = backend_factory(body.backend)
        references = lucius.ingestion.references.silhouettes(body.reference_ids) if body.reference_ids else []
        job = jobs.submit(
            "run", body.task_text,
            lambda backend: lucius.engine.run(body.task_text, backend, mode="execute", task_params=body.task_params,
                                              references=references, reference_ids=body.reference_ids,
                                              human=lucius.human if body.allow_takeover else None),
            uses_blender=True, backend_factory=factory, backend_name=body.backend)
        return job.model_dump(mode="json")

    @r.get("/runs")
    def runs(mode: str | None = None, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        clauses, params = [], []
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = lucius.db.query(f"SELECT id, session_id, task_text, task_class, mode, state, status, backend, "
                               f"environment, arm, metrics, started_at, ended_at FROM runs {where} "
                               f"ORDER BY started_at DESC LIMIT ?", [*params, min(limit, 500)])
        return [{**dict(row), "metrics": loads(row["metrics"], {})} for row in rows]

    @r.get("/runs/{run_id}")
    def run_get(run_id: str) -> dict[str, Any]:
        row = lucius.db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if row is None:
            raise NotFoundError(f"run {run_id} not found")
        transitions = lucius.db.query("SELECT * FROM run_transitions WHERE run_id = ? ORDER BY seq", (run_id,))
        return {
            "run": {**dict(row), "plan": loads(row["plan"], None), "params": loads(row["params"], {}),
                    "metrics": loads(row["metrics"], {})},
            "transitions": [{**dict(t), "evidence": loads(t["evidence"], []), "context": loads(t["context"], {})}
                            for t in transitions],
            "evaluations": lucius.evaluator.for_subject("run", run_id),
            "corrections": [c.model_dump(mode="json") for c in lucius.corrections.list(run_id=run_id)],
        }

    @r.post("/runs/{run_id}/review")
    def run_review(run_id: str, body: Review) -> dict[str, Any]:
        return lucius.engine.resolve_human_review(run_id, passed=body.passed, rating=body.rating,
                                                  feedback=body.feedback, user_id=lucius.config.user_id)

    # -- practice -------------------------------------------------------------------------------------------------
    @r.get("/practice")
    def practice_list() -> list[dict[str, Any]]:
        return [{"name": c.name, "title": c.title, "description": c.description,
                 "overview": lucius.practice.overview(c.name)} for c in lucius.practice.curricula()]

    @r.get("/practice/{name}")
    def practice_get(name: str) -> dict[str, Any]:
        cur = lucius.practice.curriculum(name)
        return {"curriculum": cur.model_dump(mode="json"), "overview": lucius.practice.overview(cur.name)}

    @r.post("/practice/{name}/train")
    def practice_train(name: str, body: TrainRequest) -> dict[str, Any]:
        cur = lucius.practice.curriculum(name)
        factory = backend_factory(body.backend)
        job = jobs.submit(
            "practice", f"practice {cur.title}",
            lambda backend: lucius.practice.train(cur.name, stage=body.stage, attempts=body.attempts, backend=backend,
                                                  seed=body.seed,
                                                  human=lucius.human if body.allow_takeover else None),
            uses_blender=True, backend_factory=factory, backend_name=body.backend)
        return job.model_dump(mode="json")

    @r.post("/practice/{name}/stages/{index}/override")
    def practice_override(name: str, index: int, body: Override) -> dict[str, Any]:
        lucius.practice.set_override(name, index, body.status, user_id=lucius.config.user_id)
        return lucius.practice.overview(lucius.practice.curriculum(name).name)

    # -- external demonstrations ------------------------------------------------------------------------------------
    @r.post("/demonstrations")
    def demo_create(files: list[UploadFile] = File(default=[]), roles: str = Form("[]"), title: str = Form(...),
                    task_text: str | None = Form(None), instructions: str | None = Form(None),
                    license: str = Form("unknown"), training_consent: bool = Form(False),
                    reference_only: bool = Form(False)) -> dict[str, Any]:
        """Upload media (video, images, before/after pairs, references, .blend, text) as one demonstration.

        ``roles`` is a JSON list aligned with ``files``: ``{"role", "view", "target"}`` per file.
        """
        from lucius.ingestion import MediaInput
        from lucius.ingestion.media import MediaRole

        meta = json.loads(roles or "[]")
        inputs = []
        for i, upload in enumerate(files):
            info = meta[i] if i < len(meta) and isinstance(meta[i], dict) else {}
            path = _save_upload(upload, "uploads")
            inputs.append(MediaInput(path=str(path), filename=upload.filename,
                                     role=MediaRole(info.get("role", "demonstration")),
                                     view=info.get("view", "front"), target=info.get("target") or None))
        lines = [ln.strip() for ln in (instructions or "").splitlines() if ln.strip()]
        if not inputs and not lines:
            raise ValidationError("upload at least one file or provide written instructions")
        demo = lucius.ingestion.create(title=title, task_text=task_text, inputs=inputs, instructions=lines)
        policy = demo.policy.model_copy(update={"license": license, "reference_only": reference_only or
                                                demo.policy.reference_only})
        if training_consent:
            # Explicit, recorded user choice; still subject to the licence the user asserted.
            policy = policy.model_copy(update={"training_allowed": True, "consent_status": ConsentStatus.GRANTED})
        lucius.ingestion.set_policy(demo.id, policy)
        lucius.ingestion.start(demo.id, background=True)
        return lucius.ingestion.get(demo.id).model_dump(mode="json")

    @r.get("/demonstrations")
    def demo_list() -> list[dict[str, Any]]:
        return [d.model_dump(mode="json") for d in lucius.ingestion.list()]

    @r.get("/demonstrations/{demo_id}")
    def demo_get(demo_id: str) -> dict[str, Any]:
        demo = lucius.ingestion.get(demo_id)
        out: dict[str, Any] = {"demonstration": demo.model_dump(mode="json")}
        if demo.session_id:
            out["timeline"] = timeline(demo.session_id)
        media_ids = [m for m in (lucius.sessions.get(demo.session_id).meta.get("media", []) if demo.session_id else [])]
        out["constraints"] = [c.model_dump(mode="json") for c in lucius.ingestion.references.for_media(media_ids)]
        return out

    @r.post("/demonstrations/{demo_id}/policy")
    def demo_policy(demo_id: str, body: DemoPolicy) -> dict[str, Any]:
        demo = lucius.ingestion.get(demo_id)
        updates = {k: v for k, v in body.model_dump().items() if v is not None and k != "consent"}
        if body.consent is not None:
            updates["consent_status"] = ConsentStatus.GRANTED if body.consent else ConsentStatus.DENIED
            updates.setdefault("training_allowed", body.consent)
        policy = demo.policy.model_copy(update=updates)
        lucius.ingestion.set_policy(demo_id, policy)
        return {"policy": policy.model_dump(mode="json")}

    @r.post("/demonstrations/{demo_id}/reprocess")
    def demo_reprocess(demo_id: str) -> dict[str, Any]:
        lucius.ingestion.get(demo_id)
        lucius.ingestion.start(demo_id, background=True)
        return {"ok": True}

    @r.get("/media/{media_id}")
    def media_file(media_id: str) -> FileResponse:
        asset = lucius.ingestion.media.get(media_id)
        return FileResponse(lucius.ingestion.media.path(asset))

    # -- datasets & training --------------------------------------------------------------------------------------------
    @r.get("/datasets")
    def datasets() -> list[dict[str, Any]]:
        return lucius.datasets.list()

    @r.post("/datasets")
    def dataset_build(body: DatasetRequest) -> dict[str, Any]:
        from lucius.dataset.service import DatasetFilters

        if body.purpose not in ("training", "export"):
            raise ValidationError("purpose must be 'training' or 'export'")
        filters = DatasetFilters(min_quality=body.min_quality, sources=body.sources, task_class=body.task_class,
                                 session_ids=body.session_ids, **({"kinds": body.kinds} if body.kinds else {}))
        return lucius.datasets.build(body.name, purpose=body.purpose, filters=filters,  # type: ignore[arg-type]
                                     include_media=body.include_media).model_dump(mode="json")

    @r.get("/datasets/validate")
    def dataset_validate(session_ids: str | None = None) -> dict[str, Any]:
        report = lucius.datasets.validate(session_ids.split(",") if session_ids else None)
        from lucius.dataset.validation import DatasetValidator

        return {"report": report.model_dump(mode="json"), "summary": DatasetValidator.summary(report)}

    @r.get("/datasets/{dataset_id}/download")
    def dataset_download(dataset_id: str) -> FileResponse:
        row = lucius.db.query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if row is None:
            raise NotFoundError(f"dataset {dataset_id} not found")
        folder = Path(row["path"])
        archive = folder.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(folder))
        return FileResponse(archive, filename=f"{row['name']}.zip", media_type="application/zip")

    @r.post("/datasets/{dataset_id}/training-files")
    def dataset_training_files(dataset_id: str) -> dict[str, Any]:
        from lucius.training import write_training_files

        row = lucius.db.query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if row is None:
            raise NotFoundError(f"dataset {dataset_id} not found")
        out = Path(row["path"]) / "training"
        return {"dir": str(out), "files": write_training_files(Path(row["path"]), out)}

    @r.get("/training")
    def training() -> dict[str, Any]:
        from lucius.training import TrainerRegistry, TrainingAdvisor

        return {"advisor": TrainingAdvisor(lucius.db).assess(), "backends": TrainerRegistry().available(),
                "note": "Training is an interface: no trainer backend ships with Lucius and none runs automatically."}

    # -- benchmarks & experiments ------------------------------------------------------------------------------------------
    @r.get("/benchmarks")
    def benchmarks() -> dict[str, Any]:
        from lucius.benchmarks import ARMS, UNAVAILABLE_ARMS

        return {"benchmarks": lucius.benchmarks.list(), "arms": list(ARMS), "unavailable_arms": UNAVAILABLE_ARMS,
                "summary": lucius.benchmarks.summary()}

    @r.get("/experiments")
    def experiments() -> list[dict[str, Any]]:
        rows = lucius.db.query("SELECT * FROM experiments ORDER BY created_at DESC")
        return [{**dict(row), "arms": loads(row["arms"], []), "benchmark_ids": loads(row["benchmark_ids"], []),
                 "summary": loads(row["summary"], {})} for row in rows]

    @r.post("/experiments")
    def experiment_start(body: ExperimentRequest) -> dict[str, Any]:
        for name in body.benchmarks:
            lucius.benchmarks.get(name)
        factory = backend_factory(body.backend)
        job = jobs.submit(
            "experiment", f"experiment {body.name}",
            lambda backend: lucius.benchmarks.experiment(body.name, arms=body.arms, benchmarks=body.benchmarks,
                                                         backend=backend, repeats=body.repeats),
            uses_blender=True, backend_factory=factory, backend_name=body.backend)
        return job.model_dump(mode="json")

    # -- settings -------------------------------------------------------------------------------------------------------------
    @r.get("/settings")
    def settings() -> dict[str, Any]:
        cfg = lucius.config.model_dump(mode="json")
        cfg["blender"]["bridge_token"] = "set" if lucius.config.blender.bridge_token else None
        return {"config": cfg, "data_dir": str(lucius.config.data_dir),
                "credentials": {"ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY"))},
                "provider_notes": lucius.provider_notes}

    @r.put("/settings")
    def settings_update(body: SettingsUpdate) -> dict[str, Any]:
        current = lucius.config.model_dump(mode="python")
        changed = []
        for section, values in body.model_dump(exclude_none=True).items():
            values = {k: v for k, v in values.items() if not (section == "blender" and k == "bridge_token")}
            current[section] = {**current[section], **values}
            changed.append(section)
        new = LuciusConfig.model_validate(current)
        for section in changed:
            setattr(lucius.config, section, getattr(new, section))
        path = lucius.config.save()
        return {"saved": str(path), "changed": changed,
                "restart_required": sorted(set(changed) & {"providers", "blender", "processing"})}

    api.include_router(r)

    # -- control center -------------------------------------------------------------------------------------------------------
    index_html = (STATIC / "index.html").read_text()

    @api.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = index_html.replace("__LUCIUS_TOKEN__", token)
        return HTMLResponse(page, headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY",
                                           "Referrer-Policy": "no-referrer"})

    @api.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    api.mount("/static", StaticFiles(directory=STATIC), name="static")

    def _save_upload(upload: UploadFile, folder: str) -> Path:
        target_dir = lucius.config.media_dir / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        name = _SAFE_NAME.sub("_", Path(upload.filename or "upload").name)[-120:] or "upload"
        path = target_dir / f"{new_id('media')}_{name}"
        written = 0
        with path.open("wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    out.close()
                    path.unlink(missing_ok=True)
                    raise ValidationError("upload exceeds the size limit")
                out.write(chunk)
        return path

    return api


def serve(lucius: Lucius, *, host: str = "127.0.0.1", port: int = 8765) -> None:  # pragma: no cover - manual
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValidationError("the Lucius API only binds to loopback addresses")
    app = create_app(lucius)
    log.info("Lucius control center on http://%s:%d/", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["create_app", "load_token", "serve"]
