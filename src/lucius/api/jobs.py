"""Background jobs for the API (agent runs, practice, experiments, reprocessing).

Long operations never block a request. Jobs that drive Blender are serialised through one
worker: two runs sharing a scene would corrupt each other's evidence. While such a job runs,
its backend is exposed so a human takeover can act on the same scene through the API.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from lucius.events.bus import EventBus, EventType
from lucius.ids import new_id
from lucius.logging_setup import get_logger
from lucius.timeutil import now

log = get_logger(__name__)


class Job(BaseModel):
    id: str
    kind: str                      # run, practice, experiment, process, ...
    title: str
    status: str = "queued"         # queued, running, done, error
    uses_blender: bool = False
    backend: str | None = None     # live, headless
    result: Any = None
    error: str | None = None
    created_at: float = Field(default_factory=now)
    started_at: float | None = None
    ended_at: float | None = None


class JobManager:
    def __init__(self, bus: EventBus | None = None, max_general: int = 2) -> None:
        self.bus = bus
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._blender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lucius-blender-job")
        self._general = ThreadPoolExecutor(max_workers=max_general, thread_name_prefix="lucius-job")
        self.active_backend: Any = None

    def submit(self, kind: str, title: str, fn: Callable[..., Any], *, uses_blender: bool = False,
               backend_factory: Callable[[], Any] | None = None, backend_name: str | None = None) -> Job:
        """Queue ``fn``. With ``backend_factory`` the backend is created on the worker and passed to ``fn``."""
        job = Job(id=new_id("job"), kind=kind, title=title, uses_blender=uses_blender, backend=backend_name)
        with self._lock:
            self._jobs[job.id] = job
        pool = self._blender if uses_blender else self._general
        pool.submit(self._run, job, fn, backend_factory)
        self._publish(job)
        return job

    def _run(self, job: Job, fn: Callable[..., Any], backend_factory: Callable[[], Any] | None) -> None:
        job.status, job.started_at = "running", now()
        self._publish(job)
        try:
            if backend_factory is not None:
                backend = backend_factory()
                self.active_backend = backend
                try:
                    job.result = fn(backend)
                finally:
                    self.active_backend = None
            else:
                job.result = fn()
            job.status = "done"
        except Exception as exc:  # the job's error is reported, never swallowed
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            log.error("job %s (%s) failed: %s\n%s", job.id, job.kind, exc, traceback.format_exc())
        finally:
            job.ended_at = now()
            self._publish(job)

    def _publish(self, job: Job) -> None:
        if self.bus is not None:
            self.bus.publish(EventType.JOB_STATUS, job.id, kind=job.kind, status=job.status, title=job.title,
                             error=job.error)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def active(self) -> list[Job]:
        return [j for j in self.list(500) if j.status in ("queued", "running")]

    def shutdown(self) -> None:
        self._general.shutdown(wait=False, cancel_futures=True)
        self._blender.shutdown(wait=False, cancel_futures=True)
