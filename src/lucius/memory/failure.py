"""Failure / recovery memory (sections 27-28).

A failure record captures ``bad state -> diagnosis -> correction -> rule``. Repeated
occurrences of the same failure (same signature) increase its retrieval priority; corrections
that keep working promote the future rule. Confidence always comes from counted evidence.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from lucius.confidence import Evidence, score
from lucius.errors import NotFoundError, ValidationError
from lucius.events.bus import EventBus, EventType
from lucius.ids import new_id
from lucius.provenance import SOURCE_WEIGHT, SourceClass
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now

PROMOTION_MIN_SUCCESSES = 2
PROMOTION_MIN_RATIO = 0.66


class FailureEvidence(BaseModel):
    kind: str                         # demo_undo, checkpoint_failure, takeover, human_report, validation
    source_class: str
    session_id: str | None = None
    run_id: str | None = None
    segment_id: str | None = None
    step_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    detail: str = ""
    weight: float = 1.0
    ts: float = Field(default_factory=now)


class FailureObservation(BaseModel):
    task_class: str | None
    phase: str | None
    observed_problem: str
    symptoms: list[str] = Field(default_factory=list)
    likely_cause: str | None = None
    correction: str | None = None
    future_rule: str | None = None
    trigger_action: str | None = None
    guard_checkpoint: str | None = None
    problem_code: str | None = None
    skill_id: str | None = None
    evidence: FailureEvidence


class FailureRecord(BaseModel):
    id: str
    signature: str
    task_class: str | None
    phase: str | None
    observed_problem: str
    symptoms: list[str]
    likely_cause: str | None
    correction: str | None
    future_rule: str | None
    rule_status: str
    trigger_action: str | None
    guard_checkpoint: str | None
    evidence: list[FailureEvidence]
    confidence: float
    confidence_breakdown: dict[str, Any]
    occurrence_count: int
    correction_attempts: int
    correction_successes: int
    retrieval_priority: float
    skill_id: str | None
    source_class: str
    created_at: float
    updated_at: float
    last_seen: float

    def text(self) -> str:
        return " \n".join(x for x in [self.observed_problem, " ".join(self.symptoms), self.likely_cause or "",
                                      self.correction or "", self.future_rule or "", self.task_class or "",
                                      self.phase or "", self.trigger_action or ""] if x)


def signature_for(obs: FailureObservation) -> str:
    key = obs.trigger_action or obs.problem_code or obs.observed_problem.lower()[:60]
    return f"{obs.task_class or '*'}|{obs.phase or '*'}|{key}"


def _decode(row: Any) -> FailureRecord:
    return FailureRecord(
        id=row["id"], signature=row["signature"], task_class=row["task_class"], phase=row["phase"],
        observed_problem=row["observed_problem"], symptoms=loads(row["symptoms"], []),
        likely_cause=row["likely_cause"], correction=row["correction"], future_rule=row["future_rule"],
        rule_status=row["rule_status"], trigger_action=row["trigger_action"], guard_checkpoint=row["guard_checkpoint"],
        evidence=[FailureEvidence.model_validate(e) for e in loads(row["evidence"], [])],
        confidence=row["confidence"], confidence_breakdown=loads(row["confidence_breakdown"], {}),
        occurrence_count=row["occurrence_count"], correction_attempts=row["correction_attempts"],
        correction_successes=row["correction_successes"], retrieval_priority=row["retrieval_priority"],
        skill_id=row["skill_id"], source_class=row["source_class"], created_at=row["created_at"],
        updated_at=row["updated_at"], last_seen=row["last_seen"],
    )


class FailureMemory:
    def __init__(self, db: Database, bus: EventBus | None = None) -> None:
        self.db = db
        self.bus = bus

    # -- evidence-based scores -----------------------------------------------------------------
    @staticmethod
    def _rescore(rec: FailureRecord) -> None:
        confirmations = sum(1 for e in rec.evidence if e.kind == "human_confirmation")
        rejections = sum(1 for e in rec.evidence if e.kind == "human_rejection")
        occurrences = [e for e in rec.evidence if e.kind not in ("human_confirmation", "human_rejection",
                                                                   "correction_outcome")]
        ev = Evidence(
            demonstrations=sum(e.weight for e in occurrences), demonstration_count=len(occurrences),
            distinct_sources=len({e.source_class for e in occurrences}),
            distinct_instances=len({e.session_id or e.run_id for e in occurrences}),
            successes=rec.correction_successes, failures=rec.correction_attempts - rec.correction_successes,
            human_confirmations=confirmations, human_rejections=rejections,
        )
        rec.confidence, rec.confidence_breakdown = score(ev)
        rec.retrieval_priority = round(1.0 + math.log2(max(1, rec.occurrence_count))
                                       + (0.5 if rec.rule_status in ("promoted", "confirmed") else 0.0)
                                       - (1.0 if rec.rule_status == "rejected" else 0.0), 3)

    def _save(self, rec: FailureRecord, *, insert: bool) -> None:
        values = {
            "id": rec.id, "signature": rec.signature, "task_class": rec.task_class, "phase": rec.phase,
            "observed_problem": rec.observed_problem, "symptoms": dumps(rec.symptoms), "likely_cause": rec.likely_cause,
            "correction": rec.correction, "future_rule": rec.future_rule, "rule_status": rec.rule_status,
            "trigger_action": rec.trigger_action, "guard_checkpoint": rec.guard_checkpoint,
            "evidence": dumps([e.model_dump() for e in rec.evidence]), "confidence": rec.confidence,
            "confidence_breakdown": dumps(rec.confidence_breakdown), "occurrence_count": rec.occurrence_count,
            "correction_attempts": rec.correction_attempts, "correction_successes": rec.correction_successes,
            "retrieval_priority": rec.retrieval_priority, "skill_id": rec.skill_id, "source_class": rec.source_class,
            "created_at": rec.created_at, "updated_at": rec.updated_at, "last_seen": rec.last_seen,
        }
        if insert:
            self.db.insert("failure_records", values)
        else:
            self.db.update("failure_records", "id", rec.id, {k: v for k, v in values.items() if k != "id"})

    # -- API ---------------------------------------------------------------------------------------
    def record(self, obs: FailureObservation) -> FailureRecord:
        """Record an occurrence. The same failure seen again strengthens the existing record."""
        signature = signature_for(obs)
        t = now()
        row = self.db.query_one("SELECT * FROM failure_records WHERE signature = ?", (signature,))
        weight = SOURCE_WEIGHT.get(SourceClass(obs.evidence.source_class), 0.5) * obs.evidence.weight
        obs.evidence.weight = round(weight, 3)
        if row is None:
            rec = FailureRecord(
                id=new_id("failure"), signature=signature, task_class=obs.task_class, phase=obs.phase,
                observed_problem=obs.observed_problem, symptoms=list(dict.fromkeys(obs.symptoms)),
                likely_cause=obs.likely_cause, correction=obs.correction, future_rule=obs.future_rule,
                rule_status="candidate", trigger_action=obs.trigger_action, guard_checkpoint=obs.guard_checkpoint,
                evidence=[obs.evidence], confidence=0.0, confidence_breakdown={}, occurrence_count=1,
                correction_attempts=0, correction_successes=0, retrieval_priority=1.0, skill_id=obs.skill_id,
                source_class=obs.evidence.source_class, created_at=t, updated_at=t, last_seen=t,
            )
            self._rescore(rec)
            self._save(rec, insert=True)
            event = EventType.FAILURE_RECORDED
        else:
            rec = _decode(row)
            already = any(e.kind == obs.evidence.kind and e.session_id == obs.evidence.session_id
                          and e.run_id == obs.evidence.run_id and e.step_ids == obs.evidence.step_ids
                          for e in rec.evidence)
            if already:
                return rec  # re-processing the same session must not inflate counts
            rec.occurrence_count += 1
            rec.evidence.append(obs.evidence)
            rec.symptoms = list(dict.fromkeys(rec.symptoms + obs.symptoms))
            if rec.rule_status == "candidate":
                # Later observations may carry a better diagnosis (e.g. a human annotation).
                rec.likely_cause = rec.likely_cause or obs.likely_cause
                rec.correction = rec.correction or obs.correction
                rec.future_rule = rec.future_rule or obs.future_rule
                rec.guard_checkpoint = rec.guard_checkpoint or obs.guard_checkpoint
            rec.skill_id = rec.skill_id or obs.skill_id
            rec.updated_at = rec.last_seen = t
            self._rescore(rec)
            self._save(rec, insert=False)
            event = EventType.FAILURE_RECORDED
        if self.bus is not None:
            self.bus.publish(event, rec.id, signature=signature, occurrence_count=rec.occurrence_count,
                             source=obs.evidence.source_class, confidence=rec.confidence)
        return rec

    def record_correction_outcome(self, failure_id: str, *, success: bool, evidence: FailureEvidence) -> FailureRecord:
        rec = self.get(failure_id)
        rec.correction_attempts += 1
        rec.correction_successes += int(success)
        evidence.kind = "correction_outcome"
        evidence.detail = ("success: " if success else "failure: ") + evidence.detail
        rec.evidence.append(evidence)
        promoted = False
        if (rec.rule_status == "candidate" and rec.correction_successes >= PROMOTION_MIN_SUCCESSES
                and rec.correction_successes / rec.correction_attempts >= PROMOTION_MIN_RATIO):
            rec.rule_status = "promoted"
            promoted = True
        rec.updated_at = now()
        self._rescore(rec)
        self._save(rec, insert=False)
        if promoted and self.bus is not None:
            self.bus.publish(EventType.FAILURE_PROMOTED, rec.id, future_rule=rec.future_rule,
                             successes=rec.correction_successes, attempts=rec.correction_attempts)
        return rec

    def review_rule(self, failure_id: str, *, accept: bool, user_id: str = "local", note: str = "") -> FailureRecord:
        rec = self.get(failure_id)
        rec.rule_status = "confirmed" if accept else "rejected"
        rec.evidence.append(FailureEvidence(kind="human_confirmation" if accept else "human_rejection",
                                            source_class=SourceClass.HUMAN_CORRECTION.value, detail=note or user_id))
        rec.updated_at = now()
        self._rescore(rec)
        self._save(rec, insert=False)
        if accept and self.bus is not None:
            self.bus.publish(EventType.FAILURE_PROMOTED, rec.id, future_rule=rec.future_rule, by="human")
        return rec

    EDITABLE = {"observed_problem", "symptoms", "likely_cause", "correction", "future_rule", "phase",
                "guard_checkpoint", "trigger_action"}

    def edit(self, failure_id: str, changes: dict[str, Any], user_id: str = "local") -> FailureRecord:
        unknown = set(changes) - self.EDITABLE
        if unknown:
            raise ValidationError(f"fields not editable: {sorted(unknown)}")
        rec = self.get(failure_id)
        before = {k: getattr(rec, k) for k in changes}
        for key, value in changes.items():
            setattr(rec, key, value)
        rec.updated_at = now()
        self._save(rec, insert=False)
        self.db.insert("human_edits", {"id": new_id("edit"), "subject_kind": "failure", "subject_id": failure_id,
                                       "op": "edit", "before": dumps(before), "after": dumps(changes),
                                       "user_id": user_id, "created_at": now()})
        return rec

    def get(self, failure_id: str) -> FailureRecord:
        row = self.db.query_one("SELECT * FROM failure_records WHERE id = ?", (failure_id,))
        if row is None:
            raise NotFoundError(f"failure {failure_id} not found", failure_id=failure_id)
        return _decode(row)

    def list(self, *, task_class: str | None = None, status: str | None = None, limit: int = 200) -> list[FailureRecord]:
        clauses, params = [], []
        if task_class:
            clauses.append("(task_class = ? OR task_class IS NULL)")
            params.append(task_class)
        if status:
            clauses.append("rule_status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(f"SELECT * FROM failure_records {where} ORDER BY retrieval_priority DESC, last_seen DESC"
                             f" LIMIT ?", (*params, limit))
        return [_decode(r) for r in rows]

    def relevant(self, *, task_class: str | None, action_types: set[str] | None = None,
                 phases: set[str] | None = None, skill_ids: set[str] | None = None) -> list[FailureRecord]:
        """Failures worth guarding against for a planned task (rejected rules excluded)."""
        out = []
        for rec in self.list(task_class=task_class):
            if rec.rule_status == "rejected":
                continue
            relevant = (
                (action_types and rec.trigger_action in action_types)
                or (phases and rec.phase in phases)
                or (skill_ids and rec.skill_id in skill_ids)
            )
            if relevant or (action_types is None and phases is None and skill_ids is None):
                out.append(rec)
        out.sort(key=lambda r: r.retrieval_priority * r.confidence, reverse=True)
        return out
