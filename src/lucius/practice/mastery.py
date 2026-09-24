"""Mastery metrics and gates (section 41).

Mastery is never a single opaque number: the underlying metrics are stored and shown, and the
gate checks each of them. ``mastery_score`` is only a convenience summary of the same metrics.
"""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from lucius.practice.curriculum import MasteryGate
from lucius.storage.db import Database, loads


class MasteryMetrics(BaseModel):
    attempts: int = 0
    completion_rate: float | None = None
    checkpoint_pass_rate: float | None = None
    recovery_rate: float | None = None
    takeover_rate: float | None = None
    human_rating: float | None = None
    consistency: float | None = None
    false_success_rate: float | None = None
    generalization: float | None = None
    needs_demonstration: int = 0
    last_verdicts: list[str] = Field(default_factory=list)


class GateResult(BaseModel):
    mastered: bool
    unmet: list[str]
    mastery_score: float | None


def compute_metrics(db: Database, practice_task_ids: list[str]) -> MasteryMetrics:
    if not practice_task_ids:
        return MasteryMetrics()
    marks = ",".join("?" for _ in practice_task_ids)
    runs = db.query(f"SELECT id, status, metrics, params, plan FROM runs WHERE practice_task_id IN ({marks})"
                    f" ORDER BY started_at", practice_task_ids)
    if not runs:
        return MasteryMetrics()
    metrics = [loads(r["metrics"], {}) for r in runs]
    ratios = [m["checkpoints_passed"] / m["checkpoints_total"] for m in metrics if m.get("checkpoints_total")]
    had_failure = [m for m in metrics if m.get("recoveries") or m.get("takeovers") or m.get("actions_failed")]
    successes = [r for r in runs if r["status"] == "success"]
    human = {row["run_id"]: row for row in db.query(
        f"SELECT run_id, passed, score FROM evaluations WHERE method = 'human' AND run_id IN"
        f" (SELECT id FROM runs WHERE practice_task_id IN ({marks}))", practice_task_ids)}
    judged_success = [r for r in successes if r["id"] in human and human[r["id"]]["passed"] is not None]
    false_success = [r for r in judged_success if human[r["id"]]["passed"] == 0]
    ratings = [row["score"] * 5 for row in human.values() if row["score"] is not None]
    needs_demo = sum(1 for r in runs if "needs_demonstration" in (loads(r["plan"], {}) or {}).get("reason_codes", []))
    return MasteryMetrics(
        attempts=len(runs),
        completion_rate=round(len(successes) / len(runs), 4),
        checkpoint_pass_rate=round(sum(ratios) / len(ratios), 4) if ratios else None,
        recovery_rate=round(sum(1 for m in had_failure if m.get("recoveries")) / len(had_failure), 4) if had_failure else None,
        takeover_rate=round(sum(1 for m in metrics if m.get("takeovers")) / len(runs), 4),
        human_rating=round(sum(ratings) / len(ratings), 3) if ratings else None,
        consistency=round(max(0.0, 1.0 - statistics.pstdev(ratios)), 4) if len(ratios) >= 2 else None,
        false_success_rate=round(len(false_success) / len(judged_success), 4) if judged_success else None,
        generalization=round(len({r["params"] for r in successes}) / len(successes), 4) if successes else None,
        needs_demonstration=needs_demo,
        last_verdicts=[r["status"] for r in runs[-5:]],
    )


def check_gate(m: MasteryMetrics, gate: MasteryGate) -> GateResult:
    unmet = []

    def need(ok: bool, label: str) -> None:
        if not ok:
            unmet.append(label)

    need(m.attempts >= gate.min_attempts, f"attempts {m.attempts}/{gate.min_attempts}")
    need((m.completion_rate or 0) >= gate.completion_rate, f"completion rate < {gate.completion_rate}")
    need((m.checkpoint_pass_rate or 0) >= gate.checkpoint_pass_rate, f"checkpoint pass rate < {gate.checkpoint_pass_rate}")
    need((m.takeover_rate if m.takeover_rate is not None else 1) <= gate.max_takeover_rate,
         f"takeover rate > {gate.max_takeover_rate}")
    if m.false_success_rate is not None:
        need(m.false_success_rate <= gate.max_false_success_rate, f"false-success rate > {gate.max_false_success_rate}")
    need((m.consistency or 0) >= gate.min_consistency, f"consistency < {gate.min_consistency}")
    need((m.generalization or 0) >= gate.min_generalization, f"generalisation < {gate.min_generalization}")
    if gate.min_human_rating is not None:
        need((m.human_rating or 0) >= gate.min_human_rating, f"human rating < {gate.min_human_rating}")
    parts = [v for v in (m.completion_rate, m.checkpoint_pass_rate, m.consistency, m.generalization,
                         None if m.takeover_rate is None else 1 - m.takeover_rate) if v is not None]
    score = round(sum(parts) / len(parts), 4) if parts else None
    return GateResult(mastered=not unmet, unmet=unmet, mastery_score=score)


def summarize(m: MasteryMetrics) -> dict[str, Any]:
    return m.model_dump()
