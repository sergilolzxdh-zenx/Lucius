"""Asynchronous, resumable post-capture processing (sections 83-85).

Capture never waits for processing. After a session ends, its stages run in the background:

    trajectory -> segmentation -> refinement -> intents -> skills|corrections -> episode
    -> semantic -> preferences -> index

Each stage is a row in ``processing_jobs``. A failed stage records a structured error and
stops the stages that depend on it; the raw session is untouched, and ``process()`` resumes at
the first stage that is not done. Data-use policy gates learning and memory stages.
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from lucius.errors import LuciusError
from lucius.events.bus import EventType
from lucius.ids import new_id
from lucius.logging_setup import get_logger
from lucius.sessions.models import EventKind, SessionKind, SessionStatus
from lucius.storage.db import dumps, loads
from lucius.timeutil import now
from lucius.trajectory.builder import BuildResult, TrajectoryBuilder

if TYPE_CHECKING:
    from lucius.app import Lucius

log = get_logger("processing")

STAGES = ["trajectory", "segmentation", "refinement", "intents", "skills", "episode", "semantic", "preferences",
          "index"]
LEARNING_KINDS = {SessionKind.LIVE_DEMO, SessionKind.EXTERNAL_MEDIA}


class Skipped(Exception):
    """A stage that does not apply to this session (recorded as 'skipped' with the reason)."""


class ProcessingPipeline:
    def __init__(self, app: Lucius) -> None:
        self.app = app
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._cache: dict[str, BuildResult] = {}

    # -- job bookkeeping ----------------------------------------------------------------------------
    def _job(self, session_id: str, stage: str) -> dict[str, Any] | None:
        row = self.app.db.query_one(
            "SELECT * FROM processing_jobs WHERE subject_kind = 'session' AND subject_id = ? AND stage = ?",
            (session_id, stage))
        return None if row is None else dict(row)

    def _set(self, session_id: str, stage: str, status: str, *, error: dict | None = None,
             result: dict | None = None) -> None:
        t = now()
        job = self._job(session_id, stage)
        if job is None:
            self.app.db.insert("processing_jobs", {
                "id": new_id("job"), "subject_kind": "session", "subject_id": session_id, "stage": stage,
                "status": status, "attempts": int(status == "running"), "error": dumps(error) if error else None,
                "result": dumps(result or {}), "created_at": t, "updated_at": t})
        else:
            self.app.db.update("processing_jobs", "id", job["id"], {
                "status": status, "attempts": job["attempts"] + int(status == "running"),
                "error": dumps(error) if error else None, "result": dumps(result or {}), "updated_at": t})
        self.app.sessions.set_processing_stage(session_id, stage, status, **({"error": error} if error else {}))
        self.app.bus.publish(EventType.PROCESSING_STAGE, session_id, stage=stage, status=status,
                             error=error, result=result)

    def status(self, session_id: str) -> dict[str, Any]:
        rows = self.app.db.query("SELECT stage, status, attempts, error, result, updated_at FROM processing_jobs"
                                 " WHERE subject_kind = 'session' AND subject_id = ?", (session_id,))
        by_stage = {r["stage"]: {"status": r["status"], "attempts": r["attempts"], "error": loads(r["error"]),
                                 "result": loads(r["result"], {}), "updated_at": r["updated_at"]} for r in rows}
        return {stage: by_stage.get(stage, {"status": "pending"}) for stage in STAGES}

    # -- execution ------------------------------------------------------------------------------------
    def process(self, session_id: str, *, force: bool = False, stages: list[str] | None = None) -> dict[str, Any]:
        session = self.app.sessions.get(session_id)
        if session.status == SessionStatus.RECORDING:
            raise LuciusError("session is still recording", code="still_recording", session_id=session_id)
        self.app.sessions.update(session_id, status=SessionStatus.PROCESSING)
        wanted = stages or STAGES
        failed = False
        for stage in STAGES:
            if stage not in wanted:
                continue
            job = self._job(session_id, stage)
            if job and job["status"] in ("done", "skipped") and not force:
                continue
            if failed:
                self._set(session_id, stage, "pending", error={"code": "blocked", "message": "an earlier stage failed"})
                continue
            self._set(session_id, stage, "running")
            try:
                result = getattr(self, f"_stage_{stage}")(session_id) or {}
            except Skipped as exc:
                self._set(session_id, stage, "skipped", result={"reason": str(exc)})
                continue
            except Exception as exc:  # recorded, not swallowed: the job row keeps the structured error
                failed = True
                error = exc.to_dict() if isinstance(exc, LuciusError) else {
                    "code": type(exc).__name__, "message": str(exc), "details": {"traceback": traceback.format_exc(limit=6)}}
                log.exception("stage %s failed for %s", stage, session_id)
                self._set(session_id, stage, "failed", error=error)
                continue
            self._set(session_id, stage, "done", result=result)
        self.app.sessions.update(session_id, status=SessionStatus.FAILED if failed else SessionStatus.PROCESSED)
        self._cache.pop(session_id, None)
        return self.status(session_id)

    def submit(self, session_id: str) -> None:
        """Queue a session for background processing."""
        self._ensure_workers()
        self._queue.put(session_id)

    def _ensure_workers(self) -> None:
        if self._workers:
            return
        for i in range(self.app.config.processing.workers):
            thread = threading.Thread(target=self._worker, name=f"lucius-processing-{i}", daemon=True)
            thread.start()
            self._workers.append(thread)

    def _worker(self) -> None:
        while True:
            session_id = self._queue.get()
            if session_id is None:
                return
            try:
                self.process(session_id)
            except Exception:
                log.exception("background processing failed for %s", session_id)
            finally:
                self._queue.task_done()

    def wait_idle(self, timeout: float = 120.0) -> bool:
        done = threading.Event()

        def waiter() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=waiter, daemon=True).start()
        return done.wait(timeout)

    def shutdown(self) -> None:
        for _ in self._workers:
            self._queue.put(None)
        self._workers.clear()

    # -- stages ---------------------------------------------------------------------------------------
    def _build(self, session_id: str) -> BuildResult:
        if session_id not in self._cache:
            session = self.app.sessions.get(session_id)
            events = self.app.sessions.events(session_id)
            frames = self.app.sessions.frames_for(session_id)
            sources = session.meta.get("capture_sources", {})
            operator_log = bool(sources.get("blender_bridge")) and any(
                e.kind == EventKind.BLENDER_OPERATOR for e in events)
            self._cache[session_id] = TrajectoryBuilder().build(session_id, events, frames,
                                                                operator_log_available=operator_log)
        return self._cache[session_id]

    def _stage_trajectory(self, session_id: str) -> dict[str, Any]:
        if self.app.sessions.get(session_id).kind == SessionKind.EXTERNAL_MEDIA:
            steps = self.app.trajectories.for_session(session_id)
            if not steps:
                raise Skipped("media trajectory is written by the ingestion pipeline")
            return {"steps": len(steps), "source": "media_ingestion"}
        built = self._build(session_id)
        self.app.trajectories.replace(session_id, built.steps)
        self.app.bus.publish(EventType.TRAJECTORY_BUILT, session_id, steps=len(built.steps), **built.stats)
        return {"steps": len(built.steps), **built.stats}

    def _annotations(self, session_id: str) -> list[dict[str, Any]]:
        return [{"ts": e.ts, "seq": e.seq, **e.payload}
                for e in self.app.sessions.events(session_id, kinds=[EventKind.ANNOTATION.value])]

    def _stage_segmentation(self, session_id: str) -> dict[str, Any]:
        steps = self.app.trajectories.for_session(session_id)
        frames = self.app.sessions.frames_for(session_id)
        segments = self.app.segmenter.segment(session_id, steps, frames, self._annotations(session_id))
        return {"segments": len(segments), "labels": [s.label for s in segments]}

    def _stage_refinement(self, session_id: str) -> dict[str, Any]:
        if not self.app.config.processing.model_refinement:
            raise Skipped("model refinement disabled in configuration")
        if not self.app.providers.has("llm"):
            raise Skipped("no LLM provider configured")
        session = self.app.sessions.get(session_id)
        steps = self.app.trajectories.for_session(session_id)
        segments = self.app.segments.for_session(session_id)
        frame_paths = {f.id: f.path for f in self.app.sessions.frames_for(session_id)}
        report = self.app.refiner.refine(session, segments, steps, frame_paths)
        if report.status == "skipped":
            raise Skipped(report.reason)
        if report.status == "failed":
            # Model trouble must not block learning: deterministic labels remain in place.
            return {"status": "failed_degraded", "reason": report.reason}
        return {"changed": report.changed, "model": report.model}

    def _stage_intents(self, session_id: str) -> dict[str, Any]:
        session = self.app.sessions.get(session_id)
        steps = self.app.trajectories.for_session(session_id)
        counts: dict[str, int] = {}
        for seg in self.app.segments.for_session(session_id):
            hyps = self.app.intents.hypotheses(seg, steps[seg.step_start:seg.step_end + 1], session.task_class)
            self.app.intents.store(seg.id, hyps)
            if hyps:
                counts[hyps[0].category] = counts.get(hyps[0].category, 0) + 1
        return {"primary_categories": counts}

    def _stage_skills(self, session_id: str) -> dict[str, Any]:
        session = self.app.sessions.get(session_id)
        if not session.policy.learning_allowed:
            raise Skipped("learning not allowed by the session's data policy")
        steps = self.app.trajectories.for_session(session_id)
        if session.kind in LEARNING_KINDS:
            segments = self.app.segments.for_session(session_id)
            result = self.app.extractor.extract(session, steps, segments)
            return {"skills": result.skill_ids, "created": result.created, "updated": result.updated,
                    "failures": result.failure_ids}
        # Agent runs: the agent's own actions are not new procedural knowledge, but the human's
        # corrections inside takeovers are.
        from lucius.skills.recovery import learn_from_takeovers

        takeovers = self._build(session_id).takeovers
        learned = learn_from_takeovers(session, steps, takeovers, self.app.corrections, self.app.library)
        return {"corrections": learned.corrections_updated, "skills_updated": learned.skills_updated}

    def _stage_episode(self, session_id: str) -> dict[str, Any]:
        session = self.app.sessions.get(session_id)
        if not session.policy.memory_allowed:
            raise Skipped("memory not allowed by the session's data policy")
        steps = self.app.trajectories.for_session(session_id)
        segments = self.app.segments.for_session(session_id)
        intents = {seg.id: self.app.intents.primary(seg.id) for seg in segments}
        skills_job = self._job(session_id, "skills")
        skill_result = loads(skills_job["result"], {}) if skills_job else {}
        failure_ids = skill_result.get("failures", []) + [
            r["id"] for r in self.app.db.query(
                "SELECT DISTINCT f.id FROM failure_records f WHERE f.evidence LIKE ?", (f'%"{session_id}"%',))]
        skill_ids = list(skill_result.get("skills") or [])
        run = self.app.db.query_one("SELECT plan FROM runs WHERE session_id = ?", (session_id,))
        if run is not None:  # an agent run: the skills it planned with
            plan_steps = (loads(run["plan"], {}) or {}).get("steps", [])
            skill_ids += [s["skill_id"] for s in plan_steps if s.get("skill_id")]
        skill_ids = list(dict.fromkeys(skill_ids))
        takeovers = self._build(session_id).takeovers if session.kind != SessionKind.EXTERNAL_MEDIA else []
        episode = self.app.episodes.build(session, steps, segments, intents, failure_ids=sorted(set(failure_ids)),
                                          skill_ids=skill_ids, annotations=self._annotations(session_id),
                                          takeovers=takeovers)
        self.app.graph.link(("episode", episode.id), "derived_from", ("session", session_id))
        return {"episode_id": episode.id}

    def _stage_semantic(self, session_id: str) -> dict[str, Any]:
        statements = self.app.semantic.mine(self.app.episodes.list(limit=5000), self.app.failures.list(limit=5000))
        return {"statements": len(statements)}

    def _stage_preferences(self, session_id: str) -> dict[str, Any]:
        learned = self.app.preferences.learn(self.app.episodes.list(limit=5000))
        return {k: {"value": p.value, "confidence": p.confidence, "source": p.source} for k, p in learned.items()}

    def _stage_index(self, session_id: str) -> dict[str, Any]:
        return self.app.retriever.refresh()


def on_session_end(pipeline: ProcessingPipeline, *, background: bool = True) -> Callable[[Any], None]:
    """Bus handler: queue every finished recording for processing."""

    def handler(event: Any) -> None:
        if event.subject_id:
            if background:
                pipeline.submit(event.subject_id)
            else:
                pipeline.process(event.subject_id)

    return handler
