"""Plan data model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from lucius.skills.schema import Checkpoint

Layer = Literal["blender_api", "gui", "internal", "observation"]


class TaskSpec(BaseModel):
    text: str
    object_class: str | None = None
    categories: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)          # explicit values from text / references
    qualifiers: dict[str, float] = Field(default_factory=dict)    # e.g. {"length": 1.3} from "long"
    reference_ids: list[str] = Field(default_factory=list)
    param_sources: dict[str, str] = Field(default_factory=dict)   # param -> task | reference | qualifier


class PlanAction(BaseModel):
    id: str
    layer: Layer
    name: str                             # bridge action or internal macro
    args: dict[str, Any] = Field(default_factory=dict)
    action_type: str                      # semantic vocabulary
    description: str = ""
    optional: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    source: str | None = None             # skill_id/phase/index


class Guard(BaseModel):
    failure_id: str
    trigger_action: str
    checkpoint_id: str
    rule: str
    confidence: float


class RecoveryPlan(BaseModel):
    id: str
    description: str
    when_failure_ids: list[str] = Field(default_factory=list)
    when_checkpoints: list[str] = Field(default_factory=list)
    actions: list[PlanAction] = Field(default_factory=list)
    source: str


class PlanStep(BaseModel):
    id: str
    phase: str
    skill_id: str | None
    skill_version: int | None = None
    skill_name: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    actions: list[PlanAction] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    guards: list[Guard] = Field(default_factory=list)
    recovery: list[RecoveryPlan] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class Plan(BaseModel):
    id: str
    task: TaskSpec
    steps: list[PlanStep] = Field(default_factory=list)
    retrieval_id: str | None = None
    strategy: str = "hybrid"
    unresolved: list[dict[str, Any]] = Field(default_factory=list)
    preferences_applied: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    created_at: float = 0.0

    @property
    def skill_ids(self) -> list[str]:
        return list(dict.fromkeys(s.skill_id for s in self.steps if s.skill_id))

    def outline(self) -> list[str]:
        return [f"{s.skill_name or s.skill_id}: {s.phase} ({len(s.actions)} actions, {len(s.checkpoints)} checkpoints"
                f"{', guarded' if s.guards else ''})" for s in self.steps]
