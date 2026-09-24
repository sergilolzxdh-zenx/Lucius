"""Correction records: the DAgger-style dataset of human fixes to agent states (section 43).

Each record ties the agent's context (plan step, skill, failed checkpoints, state before) to the
human's correction (the steps performed, state after) and to the failure record it resolves.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from lucius.events.bus import EventBus, EventType
from lucius.graph import Graph
from lucius.ids import new_id
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now


class Correction(BaseModel):
    id: str
    session_id: str | None
    run_id: str | None
    failure_id: str | None
    kind: str
    reason: str | None
    start_time: float
    end_time: float | None
    before_state: dict[str, Any] | None
    after_state: dict[str, Any] | None
    before_frame_id: str | None = None
    after_frame_id: str | None = None
    agent_context: dict[str, Any] = Field(default_factory=dict)
    correction_steps: list[str] = Field(default_factory=list)
    outcome: str | None = None
    created_at: float


def _decode(row: Any) -> Correction:
    return Correction(
        id=row["id"], session_id=row["session_id"], run_id=row["run_id"], failure_id=row["failure_id"],
        kind=row["kind"], reason=row["reason"], start_time=row["start_time"], end_time=row["end_time"],
        before_state=loads(row["before_state"]), after_state=loads(row["after_state"]),
        before_frame_id=row["before_frame_id"], after_frame_id=row["after_frame_id"],
        agent_context=loads(row["agent_context"], {}), correction_steps=loads(row["correction_steps"], []),
        outcome=row["outcome"], created_at=row["created_at"])


class CorrectionStore:
    def __init__(self, db: Database, graph: Graph, bus: EventBus | None = None) -> None:
        self.db = db
        self.graph = graph
        self.bus = bus

    def record(self, *, kind: str, session_id: str | None, run_id: str | None, failure_id: str | None,
               start_time: float, end_time: float | None, before_state: dict[str, Any] | None,
               after_state: dict[str, Any] | None, agent_context: dict[str, Any], reason: str | None = None,
               correction_steps: list[str] | None = None, outcome: str | None = None,
               before_frame_id: str | None = None, after_frame_id: str | None = None,
               correction_id: str | None = None) -> Correction:
        correction = Correction(
            id=correction_id or new_id("correction"), session_id=session_id, run_id=run_id, failure_id=failure_id,
            kind=kind, reason=reason, start_time=start_time, end_time=end_time, before_state=before_state,
            after_state=after_state, before_frame_id=before_frame_id, after_frame_id=after_frame_id,
            agent_context=agent_context, correction_steps=correction_steps or [], outcome=outcome, created_at=now())
        self.db.insert("corrections", {
            "id": correction.id, "session_id": session_id, "run_id": run_id, "failure_id": failure_id, "kind": kind,
            "reason": reason, "start_time": start_time, "end_time": end_time,
            "before_state": dumps(before_state) if before_state is not None else None,
            "after_state": dumps(after_state) if after_state is not None else None,
            "before_frame_id": before_frame_id, "after_frame_id": after_frame_id,
            "agent_context": dumps(agent_context), "correction_steps": dumps(correction.correction_steps),
            "outcome": outcome, "created_at": correction.created_at})
        if failure_id:
            self.graph.link(("failure", failure_id), "corrected_by", ("correction", correction.id))
        if session_id:
            self.graph.link(("correction", correction.id), "demonstrated_in", ("session", session_id))
        if self.bus is not None:
            self.bus.publish(EventType.CORRECTION_RECORDED, correction.id, kind=kind, run_id=run_id,
                             failure_id=failure_id, outcome=outcome, reason_given=reason is not None)
        return correction

    def attach_steps(self, correction_id: str, step_ids: list[str]) -> None:
        self.db.execute("UPDATE corrections SET correction_steps = ? WHERE id = ?", (dumps(step_ids), correction_id))

    def get(self, correction_id: str) -> Correction | None:
        row = self.db.query_one("SELECT * FROM corrections WHERE id = ?", (correction_id,))
        return None if row is None else _decode(row)

    def list(self, *, failure_id: str | None = None, run_id: str | None = None, limit: int = 500) -> list[Correction]:
        clauses, params = [], []
        if failure_id:
            clauses.append("failure_id = ?")
            params.append(failure_id)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(f"SELECT * FROM corrections {where} ORDER BY start_time DESC LIMIT ?", (*params, limit))
        return [_decode(r) for r in rows]
