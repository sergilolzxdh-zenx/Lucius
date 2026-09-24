"""Checkpoint evaluation and verdicts (sections 37, 38, 92).

Levels: 1 execution, 2 structural, 3 visual, 4 human, 5 generalisation (benchmarks).

False-success prevention: a run only succeeds when every *required* checkpoint was actually
evaluated and passed, and at least one of them is objective (structural or measured visual).
If only a model's visual judgement supports success, the verdict is ``subjective_pass``; if a
required checkpoint could not be evaluated, the verdict is ``needs_human``.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from lucius.errors import ProviderError
from lucius.evaluation.checks import CHECKS, STRUCTURE_VIEW, CheckOutcome, ReferenceSilhouette
from lucius.evaluation.silhouette import mask_image, rasterize
from lucius.events.bus import EventBus, EventType
from lucius.ids import new_id
from lucius.logging_setup import get_logger
from lucius.providers.base import ImageInput, Providers
from lucius.skills.schema import Checkpoint
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now

log = get_logger("evaluation")

_REF = re.compile(r"^\{([a-zA-Z0-9_]+)\}$")


def resolve(value: Any, params: dict[str, Any]) -> Any:
    """Substitute ``"{param}"`` references (recursively) with parameter values."""
    if isinstance(value, str):
        match = _REF.match(value)
        if match:
            return params.get(match.group(1), value)
        return value
    if isinstance(value, list):
        return [resolve(v, params) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, params) for k, v in value.items()}
    return value


class CheckpointResult(BaseModel):
    checkpoint_id: str
    description: str
    level: int
    method: str
    required: bool
    passed: bool | None
    score: float | None = None
    subjective: bool = False
    reason_code: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    evaluator: str = "structural"


class EvaluationReport(BaseModel):
    id: str
    subject_kind: str
    subject_id: str
    run_id: str | None = None
    execution_ok: bool
    results: list[CheckpointResult] = Field(default_factory=list)
    verdict: str
    objective_passes: int = 0
    required_failed: list[str] = Field(default_factory=list)
    required_unevaluated: list[str] = Field(default_factory=list)
    summary: str = ""
    created_at: float = Field(default_factory=now)


def verdict_for(execution_ok: bool, results: list[CheckpointResult]) -> tuple[str, list[str], list[str], int]:
    required = [r for r in results if r.required]
    failed = [r.checkpoint_id for r in required if r.passed is False]
    unevaluated = [r.checkpoint_id for r in required if r.passed is None]
    objective = sum(1 for r in results if r.passed and not r.subjective and r.level in (2, 3))
    if not execution_ok or failed:
        return "failure", failed, unevaluated, objective
    if unevaluated:
        return "needs_human", failed, unevaluated, objective
    if objective:
        return "success", failed, unevaluated, objective
    if any(r.passed and r.subjective for r in results):
        return "subjective_pass", failed, unevaluated, objective
    return "executed_unverified", failed, unevaluated, objective


class Evaluator:
    def __init__(self, db: Database, providers: Providers | None = None, bus: EventBus | None = None) -> None:
        self.db = db
        self.providers = providers
        self.bus = bus

    def evaluate_checkpoint(self, checkpoint: Checkpoint, params: dict[str, Any], structure: dict[str, Any],
                            references: list[ReferenceSilhouette], *, use_model: bool = True) -> CheckpointResult:
        check = resolve(checkpoint.check, params)
        kind = check.get("type")
        if checkpoint.method == "human" or kind == "human":
            return CheckpointResult(checkpoint_id=checkpoint.id, description=checkpoint.description, level=4,
                                    method="human", required=checkpoint.required, passed=None,
                                    reason_code="awaiting_human", evidence={"prompt": check.get("prompt")},
                                    evaluator="human")
        fn = CHECKS.get(kind)
        if fn is None:
            outcome = CheckOutcome(None, None, f"unknown_check:{kind}")
        else:
            try:
                outcome = fn(check, structure, references)
            except (KeyError, ValueError, TypeError) as exc:
                outcome = CheckOutcome(None, None, "check_error", {"error": f"{type(exc).__name__}: {exc}"})
        result = CheckpointResult(
            checkpoint_id=checkpoint.id, description=resolve(checkpoint.description, params), level=checkpoint.level,
            method="visual_measured" if kind == "silhouette" else "structural", required=checkpoint.required,
            passed=outcome.passed, score=outcome.score, reason_code=outcome.reason_code, evidence=outcome.evidence,
            evaluator="measured")
        if kind == "silhouette" and outcome.passed is None and use_model:
            result = self._model_judgement(checkpoint, check, structure, references, result)
        return result

    def _model_judgement(self, checkpoint: Checkpoint, check: dict[str, Any], structure: dict[str, Any],
                         references: list[ReferenceSilhouette], fallback: CheckpointResult) -> CheckpointResult:
        if self.providers is None or not self.providers.has("evaluation"):
            return fallback
        obj = next((o for o in structure.get("objects", []) if o.get("name") == check.get("object")), None)
        if obj is None:
            return fallback
        images = []
        for view in check.get("views", []):
            tris = (obj.get("silhouettes") or {}).get(STRUCTURE_VIEW.get(view, view))
            if tris:
                images.append(ImageInput.from_image(mask_image(rasterize(tris, size=256)[0]),
                                                    label=f"Model silhouette, {view} view:"))
        if not images:
            return fallback
        try:
            judged = self.providers.evaluation.judge(criterion=checkpoint.description, images=images,
                                                     context=f"object {check.get('object')}")
        except ProviderError as exc:
            log.warning("visual judge unavailable: %s", exc.message)
            return fallback
        return CheckpointResult(
            checkpoint_id=checkpoint.id, description=checkpoint.description, level=3, method="visual_model",
            required=checkpoint.required, passed=judged.passed, score=judged.score, subjective=True,
            reason_code=(judged.reason_codes or ["model_judgement"])[0],
            evidence={"observations": judged.observations, "confidence": judged.confidence, "model": judged.model},
            evaluator=f"{judged.provider}:{judged.model}")

    def evaluate(self, *, subject_kind: str, subject_id: str, checkpoints: list[Checkpoint], params: dict[str, Any],
                 structure: dict[str, Any], references: list[ReferenceSilhouette] | None = None,
                 execution_ok: bool = True, run_id: str | None = None, persist: bool = True,
                 use_model: bool = True) -> EvaluationReport:
        results = [self.evaluate_checkpoint(cp, params, structure, references or [], use_model=use_model)
                   for cp in checkpoints]
        verdict, failed, unevaluated, objective = verdict_for(execution_ok, results)
        report = EvaluationReport(
            id=new_id("evaluation"), subject_kind=subject_kind, subject_id=subject_id, run_id=run_id,
            execution_ok=execution_ok, results=results, verdict=verdict, objective_passes=objective,
            required_failed=failed, required_unevaluated=unevaluated,
            summary=f"{verdict}: {sum(1 for r in results if r.passed)}/{len(results)} checkpoints passed"
                    + (f"; failed {failed}" if failed else "") + (f"; not evaluable {unevaluated}" if unevaluated else ""))
        if persist:
            self._persist(report)
        return report

    def _persist(self, report: EvaluationReport) -> None:
        rows = [(new_id("evaluation"), report.subject_kind, report.subject_id, report.run_id, r.level, r.checkpoint_id,
                 None if r.passed is None else int(r.passed), r.score, r.method, int(r.subjective), r.evaluator,
                 dumps({"reason_code": r.reason_code, **r.evidence}), now()) for r in report.results]
        rows.append((report.id, report.subject_kind, report.subject_id, report.run_id, 1, None,
                     int(report.execution_ok), None, "execution", 0, "executor",
                     dumps({"verdict": report.verdict, "summary": report.summary}), now()))
        self.db.executemany(
            "INSERT INTO evaluations (id, subject_kind, subject_id, run_id, level, checkpoint_id, passed, score, method,"
            " subjective, evaluator, evidence, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        if self.bus is not None:
            for r in report.results:
                if r.passed is not None:
                    self.bus.publish(EventType.CHECKPOINT_PASSED if r.passed else EventType.CHECKPOINT_FAILED,
                                     report.subject_id, checkpoint=r.checkpoint_id, method=r.method, score=r.score,
                                     reason_code=r.reason_code, run_id=report.run_id)

    def record_human(self, *, subject_kind: str, subject_id: str, run_id: str | None, passed: bool | None,
                     rating: int | None = None, feedback: str | None = None, checkpoint_id: str | None = None,
                     user_id: str = "local") -> str:
        evaluation_id = new_id("evaluation")
        self.db.insert("evaluations", {
            "id": evaluation_id, "subject_kind": subject_kind, "subject_id": subject_id, "run_id": run_id, "level": 4,
            "checkpoint_id": checkpoint_id, "passed": None if passed is None else int(passed),
            "score": None if rating is None else rating / 5.0, "method": "human", "subjective": 1,
            "evaluator": f"human:{user_id}", "evidence": dumps({"rating": rating, "feedback": feedback}),
            "created_at": now()})
        return evaluation_id

    def for_subject(self, subject_kind: str, subject_id: str) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM evaluations WHERE subject_kind = ? AND subject_id = ? ORDER BY created_at",
                             (subject_kind, subject_id))
        return [{**dict(r), "evidence": loads(r["evidence"], {})} for r in rows]
