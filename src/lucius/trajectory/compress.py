"""Compressed semantic trajectory (observation compression, section 90).

Long raw trajectories are summarised into spans such as *viewport inspection (front, right,
top)* while every span keeps the index range of the raw steps it covers, so nothing is lost:
``skill -> segment -> span -> step -> frame`` stays traceable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from lucius.trajectory import vocabulary as vocab
from lucius.trajectory.model import TrajectoryStep


class SemanticSpan(BaseModel):
    kind: str                 # action type, or an aggregate: viewport_inspection, selection, text_entry
    step_start: int
    step_end: int
    t_start: float
    t_end: float
    count: int
    summary: str
    params: dict[str, Any] = Field(default_factory=dict)
    actor: str = "human"
    min_confidence: float = 1.0


def _aggregate_kind(step: TrajectoryStep) -> str | None:
    family = vocab.spec(step.action_type).family
    if family == "navigation":
        return "viewport_inspection"
    if family == "selection":
        return "selection"
    return None


def compress(steps: list[TrajectoryStep], *, include_cancelled: bool = False) -> list[SemanticSpan]:
    spans: list[SemanticSpan] = []
    for step in steps:
        if step.meta.get("cancelled") and not include_cancelled:
            continue
        kind = _aggregate_kind(step)
        last = spans[-1] if spans else None
        if kind is not None and last is not None and last.kind == kind and last.actor == step.actor:
            last.step_end = step.idx
            last.t_end = step.t_end
            last.count += 1
            last.min_confidence = min(last.min_confidence, step.action_confidence)
            if kind == "viewport_inspection":
                view = step.params.get("view") or (step.state_after or {}).get("view")
                if view and view not in last.params.setdefault("views", []):
                    last.params["views"].append(view)
                last.params.setdefault("moves", []).append(step.action_type)
            last.summary = _summarise(last)
            continue
        span = SemanticSpan(kind=kind or step.action_type, step_start=step.idx, step_end=step.idx,
                            t_start=step.t_start, t_end=step.t_end, count=1, summary="", actor=step.actor,
                            min_confidence=step.action_confidence)
        if kind == "viewport_inspection":
            view = step.params.get("view") or (step.state_after or {}).get("view")
            span.params = {"views": [view] if view else [], "moves": [step.action_type]}
        elif kind is None:
            span.params = dict(step.params)
            if step.undo_redo_flag:
                span.params["undo_redo"] = step.undo_redo_flag
        span.summary = _summarise(span, step)
        spans.append(span)
    return spans


def _summarise(span: SemanticSpan, step: TrajectoryStep | None = None) -> str:
    if span.kind == "viewport_inspection":
        views = ", ".join(v for v in span.params.get("views", []) if v) or "free orbit"
        return f"viewport inspection ({span.count} moves; views: {views})"
    if span.kind == "selection":
        return f"selection ({span.count} actions)"
    if step is not None:
        return step.describe()
    return span.kind


def render_spans(spans: list[SemanticSpan], t0: float) -> str:
    """Compact text rendering for model prompts and logs."""
    from lucius.timeutil import fmt_offset

    return "\n".join(f"{fmt_offset(s.t_start - t0)} [{s.step_start}-{s.step_end}] {s.actor}: {s.summary}" for s in spans)
