"""Procedural skill schema (sections 18-23, 10N).

A skill is a reusable capability, not a macro: its actions are templates over named
parameters and resolution-independent selections, it declares preconditions, evaluable
checkpoints, known failure conditions with recovery procedures, alternative implementations
(variants) and the provenance of every piece of evidence it was built from.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SkillStatus(StrEnum):
    CANDIDATE_PATTERN = "candidate_pattern"   # seen once
    CANDIDATE_SKILL = "candidate_skill"       # repeated / human-confirmed
    VALIDATED = "validated"                   # reproduced with objective checkpoints passing
    HIGH_CONFIDENCE = "high_confidence"       # repeated success across distinct instances
    DISABLED = "disabled"
    MERGED = "merged"

    @property
    def rank(self) -> int:
        return {"candidate_pattern": 0, "candidate_skill": 1, "validated": 2, "high_confidence": 3}.get(self.value, -1)


class ParamSpec(BaseModel):
    name: str
    kind: Literal["float", "int", "enum", "bool", "str", "vec3"] = "float"
    description: str = ""
    default: Any = None
    range: tuple[float, float] | None = None
    choices: list[Any] | None = None
    unit: str | None = None
    source: str = "observed"            # observed, task, reference, derived, human
    invariant: bool = False             # identical in every example seen so far
    observed_values: list[Any] = Field(default_factory=list)


class Selection(BaseModel):
    """Resolution-independent edit-mode selection (normalised bounding-box predicate)."""

    kind: Literal["all", "region", "normal", "unknown"] = "all"
    axis: str | None = None
    min: float | None = None
    max: float | None = None
    direction: list[float] | None = None


class ActionTemplate(BaseModel):
    action_type: str
    description: str = ""
    args: dict[str, Any] = Field(default_factory=dict)     # values or "{param}" references
    selection: Selection | None = None                    # edit-mode selection this action applies to
    requires_mode: str | None = None                       # OBJECT / EDIT
    object_ref: str | None = "{object_name}"
    gui_hint: list[str] = Field(default_factory=list)      # observed hotkey path (for GUI execution)
    optional: bool = False
    frequency: float = 1.0                                 # share of examples containing this step
    evidence: list[str] = Field(default_factory=list)      # example ids / step ids


class Checkpoint(BaseModel):
    id: str
    description: str
    level: int = 2                        # 2 structural, 3 visual, 4 human
    method: Literal["structural", "visual_measured", "visual_model", "human"] = "structural"
    check: dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    after_phase: str | None = None
    derived_from: list[str] = Field(default_factory=list)


class FailureCondition(BaseModel):
    id: str
    description: str
    phase: str | None = None
    trigger_action: str | None = None
    guard_checkpoint: str | None = None
    failure_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class RecoveryAction(BaseModel):
    id: str
    description: str
    when: str | None = None               # failure condition id
    when_checkpoints: list[str] = Field(default_factory=list)  # checkpoints whose failure this recovery addresses
    actions: list[ActionTemplate] = Field(default_factory=list)
    source: str = "human_correction"
    evidence: list[str] = Field(default_factory=list)
    attempts: int = 0
    successes: int = 0


class Condition(BaseModel):
    kind: Literal["mode", "object_exists", "object_absent", "object_type", "modifier_present", "skill_completed",
                  "scene_empty"]
    value: Any = None
    description: str = ""


class SkillPhase(BaseModel):
    name: str                             # setup, primary_form, secondary_form, detail, inspection, verification
    description: str = ""
    actions: list[ActionTemplate] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)
    preconditions: list[Condition] = Field(default_factory=list)


class SkillVariant(BaseModel):
    variant_id: str
    name: str
    description: str = ""
    phase: str
    actions: list[ActionTemplate] = Field(default_factory=list)
    context_conditions: list[str] = Field(default_factory=list)
    source_classes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    preference_key: str | None = None


class SkillDefinition(BaseModel):
    skill_id: str
    name: str
    purpose: str
    categories: list[str] = Field(default_factory=list)
    object_class: str | None = None
    object_role: str | None = None        # e.g. "blade", "guard"
    task: str | None = None               # normalised task a tutorial chapter showed, when the object alone
                                          # does not tell skills apart (see extract.task_key)
    applicable_contexts: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    parameters: list[ParamSpec] = Field(default_factory=list)
    phases: list[SkillPhase] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    failure_conditions: list[FailureCondition] = Field(default_factory=list)
    recovery_actions: list[RecoveryAction] = Field(default_factory=list)
    variants: list[SkillVariant] = Field(default_factory=list)
    source_class: str = "user_demo"
    notes: list[str] = Field(default_factory=list)

    def param(self, name: str) -> ParamSpec | None:
        return next((p for p in self.parameters if p.name == name), None)

    def checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        return next((c for c in self.checkpoints if c.id == checkpoint_id), None)

    @property
    def action_schema(self) -> list[dict[str, Any]]:
        """Flattened phase/action view (the 'action_schema' of the spec)."""
        return [{"phase": phase.name, **action.model_dump(exclude_defaults=True)}
                for phase in self.phases for action in phase.actions]

    def signature(self) -> list[str]:
        return [a.action_type for phase in self.phases if phase.name not in ("inspection", "verification")
                for a in phase.actions if not a.optional]

    def text(self) -> str:
        """Text used for embedding and lexical retrieval."""
        parts = [self.name, self.purpose, " ".join(self.categories), self.object_class or "", self.object_role or "",
                 self.task or "", " ".join(self.triggers), " ".join(p.name for p in self.parameters),
                 " ".join(f"{ph.name}: " + " ".join(a.action_type for a in ph.actions) for ph in self.phases),
                 " ".join(c.description for c in self.checkpoints),
                 " ".join(f.description for f in self.failure_conditions)]
        return " \n".join(p for p in parts if p)


class Skill(BaseModel):
    """A skill row plus its current definition."""

    id: str
    name: str
    status: SkillStatus
    current_version: int
    source_class: str
    origin_sources: list[str]
    categories: list[str]
    object_class: str | None
    confidence: float
    confidence_breakdown: dict[str, Any]
    usage_count: int
    success_count: int
    failure_count: int
    human_confirmations: int
    human_rejections: int
    last_used: float | None
    parent_skill_id: str | None
    created_at: float
    updated_at: float
    definition: SkillDefinition

    @property
    def success_rate(self) -> float | None:
        return None if self.usage_count == 0 else self.success_count / self.usage_count


class SkillExample(BaseModel):
    id: str
    skill_id: str
    skill_version: int
    role: str
    source_class: str
    session_id: str | None = None
    segment_ids: list[str] = Field(default_factory=list)
    media_asset_id: str | None = None
    run_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    frame_ids: list[str] = Field(default_factory=list)
    outcome: str | None = None
    evidence_weight: float
    instance_signature: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)   # the per-instance extracted definition
    created_at: float
