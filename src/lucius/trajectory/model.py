"""Trajectory step model shared by live demonstrations, agent runs and media reconstructions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from lucius.provenance import EVIDENCE_WEIGHT, ActionSource, EvidenceKind


class CandidateAction(BaseModel):
    """One hypothesis for an action that was not directly observed (10G, 10T)."""

    action_type: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    id: str
    session_id: str
    idx: int
    t_start: float
    t_end: float
    frame_before_id: str | None = None
    frame_after_id: str | None = None
    action_type: str
    action_payload: dict[str, Any] = Field(default_factory=dict)
    action_source: ActionSource
    evidence_kind: EvidenceKind
    action_confidence: float
    evidence: list[str] = Field(default_factory=list)
    candidate_actions: list[CandidateAction] = Field(default_factory=list)
    window_title: str | None = None
    window_bounds: dict[str, int] | None = None
    mode_label: str | None = None
    tool_label: str | None = None
    selection_hint: str | None = None
    undo_redo_flag: str | None = None
    actor: str = "human"
    segment_id: str | None = None
    event_seq_start: int | None = None
    event_seq_end: int | None = None
    media_timestamp: float | None = None
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def params(self) -> dict[str, Any]:
        return self.action_payload.get("params", {})

    @property
    def evidence_weight(self) -> float:
        return EVIDENCE_WEIGHT[self.evidence_kind] * self.action_confidence

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params.items() if not isinstance(v, (dict, list)) or k == "axis")
        suffix = f"({params})" if params else ""
        source = "" if self.action_source == ActionSource.OBSERVED else f" [{self.action_source.value}]"
        return f"{self.action_type}{suffix}{source}"
