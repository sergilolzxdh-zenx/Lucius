"""Explicit, persisted execution state machine (section 36).

Every transition is validated against the transition table and stored in ``run_transitions``
with a reason code, evidence and confidence -- concise structured reasoning metadata, never
free-form model reasoning.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from lucius.errors import InvalidTransition
from lucius.events.bus import EventBus, EventType
from lucius.storage.db import Database, dumps
from lucius.timeutil import now


class ExecState(StrEnum):
    IDLE = "IDLE"
    OBSERVE = "OBSERVE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"
    HUMAN_TAKEOVER = "HUMAN_TAKEOVER"
    RESUME = "RESUME"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


TRANSITIONS: dict[ExecState, set[ExecState]] = {
    ExecState.IDLE: {ExecState.OBSERVE, ExecState.FAILURE},
    ExecState.OBSERVE: {ExecState.PLAN, ExecState.EXECUTE, ExecState.VERIFY, ExecState.FAILURE},
    ExecState.PLAN: {ExecState.EXECUTE, ExecState.HUMAN_TAKEOVER, ExecState.FAILURE},
    ExecState.EXECUTE: {ExecState.EXECUTE, ExecState.VERIFY, ExecState.RECOVER, ExecState.HUMAN_TAKEOVER,
                        ExecState.FAILURE},
    ExecState.VERIFY: {ExecState.VERIFY, ExecState.EXECUTE, ExecState.RECOVER, ExecState.HUMAN_TAKEOVER,
                       ExecState.SUCCESS, ExecState.FAILURE},
    ExecState.RECOVER: {ExecState.EXECUTE, ExecState.VERIFY, ExecState.HUMAN_TAKEOVER, ExecState.FAILURE},
    ExecState.HUMAN_TAKEOVER: {ExecState.RESUME, ExecState.FAILURE},
    ExecState.RESUME: {ExecState.OBSERVE, ExecState.VERIFY, ExecState.EXECUTE},
    ExecState.SUCCESS: set(),
    ExecState.FAILURE: set(),
}


class RunStateMachine:
    def __init__(self, db: Database, run_id: str, bus: EventBus | None = None) -> None:
        self.db = db
        self.run_id = run_id
        self.bus = bus
        self.state = ExecState.IDLE
        self._seq = 0
        self.history: list[dict[str, Any]] = []

    @property
    def terminal(self) -> bool:
        return not TRANSITIONS[self.state]

    def transition(self, to: ExecState, reason_code: str, *, evidence: list[str] | None = None,
                   confidence: float | None = None, **context: Any) -> None:
        if to not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.state} -> {to} is not allowed", from_state=self.state.value,
                                    to_state=to.value, reason_code=reason_code)
        record = {"from": self.state.value, "to": to.value, "reason_code": reason_code, "evidence": evidence or [],
                  "confidence": confidence, "context": context, "ts": now()}
        self.db.execute(
            "INSERT INTO run_transitions (run_id, seq, from_state, to_state, reason_code, evidence, confidence, context, ts)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (self.run_id, self._seq, self.state.value, to.value, reason_code, dumps(evidence or []), confidence,
             dumps(context), record["ts"]))
        self.db.execute("UPDATE runs SET state = ? WHERE id = ?", (to.value, self.run_id))
        self._seq += 1
        self.history.append(record)
        self.state = to
        if self.bus is not None:
            self.bus.publish(EventType.STATE_TRANSITION, self.run_id, from_state=record["from"], to_state=to.value,
                             reason_code=reason_code, evidence=evidence or [], confidence=confidence)
