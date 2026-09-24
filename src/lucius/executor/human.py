"""Human-in-the-loop channel for takeovers (sections 10, 43).

When the agent reaches a state it cannot verify or recover, it asks for a human takeover and
waits. The human corrects the scene in Blender (the recorder or the run session captures the
correction) and resumes. The reason is stored only if the human gives one.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from pydantic import BaseModel, Field

from lucius.ids import new_id
from lucius.timeutil import now


class TakeoverRequest(BaseModel):
    id: str = Field(default_factory=lambda: new_id("correction"))
    run_id: str
    reason_code: str
    message: str
    phase: str | None = None
    skill_id: str | None = None
    failed_checkpoints: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)


class TakeoverOutcome(BaseModel):
    resumed: bool
    reason: str | None = None       # only what the human said
    note: str | None = None


class HumanChannel(Protocol):
    def available(self) -> bool: ...

    def request(self, request: TakeoverRequest) -> TakeoverOutcome: ...


class InteractiveHumanChannel:
    """Blocks the run until the UI/API resumes or aborts (or the timeout expires)."""

    def __init__(self, timeout_s: float = 1800.0) -> None:
        self.timeout_s = timeout_s
        self.pending: TakeoverRequest | None = None
        self._event = threading.Event()
        self._outcome: TakeoverOutcome | None = None
        self._lock = threading.Lock()
        self.enabled = True

    def available(self) -> bool:
        return self.enabled

    def request(self, request: TakeoverRequest) -> TakeoverOutcome:
        with self._lock:
            self.pending = request
            self._event.clear()
            self._outcome = None
        if not self._event.wait(self.timeout_s):
            outcome = TakeoverOutcome(resumed=False, note="takeover timed out")
        else:
            outcome = self._outcome or TakeoverOutcome(resumed=False)
        with self._lock:
            self.pending = None
        return outcome

    def resume(self, reason: str | None = None, note: str | None = None) -> bool:
        with self._lock:
            if self.pending is None:
                return False
            self._outcome = TakeoverOutcome(resumed=True, reason=reason or None, note=note)
            self._event.set()
            return True

    def abort(self, note: str | None = None) -> bool:
        with self._lock:
            if self.pending is None:
                return False
            self._outcome = TakeoverOutcome(resumed=False, note=note or "aborted by user")
            self._event.set()
            return True
