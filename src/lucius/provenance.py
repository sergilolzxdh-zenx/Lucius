"""Shared provenance vocabulary: where knowledge came from and how strong the evidence is.

These distinctions must survive every transformation (trajectory -> skill -> retrieval ->
planning) so the agent can say *"this workflow was extracted from tutorial_042"* instead of
*"you taught me this"* when the user never demonstrated it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SourceClass(StrEnum):
    SYSTEM_SEEDED = "system_seeded"
    USER_DEMO = "user_demo"
    EXTERNAL_VIDEO = "external_video"
    EXTERNAL_IMAGE = "external_image"
    EXTERNAL_DOCUMENTATION = "external_documentation"
    EXTERNAL_PROJECT = "external_project"
    AGENT_SUCCESS = "agent_success"
    AGENT_FAILURE = "agent_failure"
    HUMAN_CORRECTION = "human_correction"
    DERIVED = "derived"
    PROMOTED = "promoted"

    @property
    def is_external(self) -> bool:
        return self.value.startswith("external_")

    @property
    def is_user(self) -> bool:
        return self in (SourceClass.USER_DEMO, SourceClass.HUMAN_CORRECTION)


class ActionSource(StrEnum):
    """How an action in a trajectory is known."""

    OBSERVED = "observed"              # captured directly (input event, Blender operator report)
    INFERRED = "inferred"              # deterministic inference from observed signals
    MODEL_INFERRED = "model_inferred"  # a VLM/LLM proposed it
    HUMAN_CONFIRMED = "human_confirmed"


class EvidenceKind(StrEnum):
    DIRECT_BLENDER_EVENT = "direct_blender_event"
    PROJECT_FILE_STATE = "project_file_state"
    HUMAN_ANNOTATION = "human_annotation"
    DIRECT_INPUT = "direct_input"
    VISIBLE_SHORTCUT = "visible_shortcut"
    VISIBLE_UI_OPERATION = "visible_ui_operation"
    VISUAL_STATE_TRANSITION = "visual_state_transition"
    TEXT_INSTRUCTION = "text_instruction"
    VLM_INFERENCE = "vlm_inference"
    WEAK_VISUAL_GUESS = "weak_visual_guess"
    AGENT_EXECUTION = "agent_execution"


# Conceptual ordering of evidence quality (10K). Used as multiplicative weights when
# aggregating evidence into confidence; the ordering matters more than the exact values.
EVIDENCE_WEIGHT: dict[EvidenceKind, float] = {
    EvidenceKind.DIRECT_BLENDER_EVENT: 1.0,
    EvidenceKind.HUMAN_ANNOTATION: 0.95,
    EvidenceKind.PROJECT_FILE_STATE: 0.9,
    EvidenceKind.AGENT_EXECUTION: 0.9,
    EvidenceKind.DIRECT_INPUT: 0.8,
    EvidenceKind.VISIBLE_SHORTCUT: 0.75,
    EvidenceKind.VISIBLE_UI_OPERATION: 0.7,
    EvidenceKind.VISUAL_STATE_TRANSITION: 0.55,
    EvidenceKind.TEXT_INSTRUCTION: 0.5,
    EvidenceKind.VLM_INFERENCE: 0.45,
    EvidenceKind.WEAK_VISUAL_GUESS: 0.25,
}

SOURCE_WEIGHT: dict[SourceClass, float] = {
    SourceClass.SYSTEM_SEEDED: 0.6,
    SourceClass.USER_DEMO: 1.0,
    SourceClass.HUMAN_CORRECTION: 1.0,
    SourceClass.EXTERNAL_PROJECT: 0.8,
    SourceClass.EXTERNAL_VIDEO: 0.6,
    SourceClass.EXTERNAL_IMAGE: 0.4,
    SourceClass.EXTERNAL_DOCUMENTATION: 0.5,
    SourceClass.AGENT_SUCCESS: 0.9,
    SourceClass.AGENT_FAILURE: 0.9,
    SourceClass.DERIVED: 0.7,
    SourceClass.PROMOTED: 0.8,
}


class ConsentStatus(StrEnum):
    GRANTED = "granted"
    DENIED = "denied"
    UNKNOWN = "unknown"


class DataPolicy(BaseModel):
    """Data-use permissions attached to every recording and media asset (45, 10Q).

    Training eligibility is opt-in: nothing enters a training dataset unless
    ``training_allowed`` is true *and* consent is granted.
    """

    source: SourceClass
    learning_allowed: bool = True
    memory_allowed: bool = True
    execution_allowed: bool = True
    training_allowed: bool = False
    export_allowed: bool = False
    reference_only: bool = False
    private: bool = True
    license: str = "unknown"
    consent_status: ConsentStatus = ConsentStatus.UNKNOWN
    notes: str | None = None

    @property
    def training_eligible(self) -> bool:
        return (
            self.training_allowed
            and self.consent_status == ConsentStatus.GRANTED
            and not self.reference_only
        )

    @classmethod
    def for_live_demo(cls, *, training_consent: bool = False) -> DataPolicy:
        return cls(
            source=SourceClass.USER_DEMO,
            training_allowed=training_consent,
            export_allowed=True,
            license="user_owned",
            consent_status=ConsentStatus.GRANTED if training_consent else ConsentStatus.UNKNOWN,
        )

    @classmethod
    def for_external(cls, source: SourceClass, *, license: str = "unknown",
                     reference_only: bool = False) -> DataPolicy:
        # External material: usable for learning and reference, never redistributed or
        # trained on unless the user explicitly changes the policy.
        return cls(source=source, license=license, reference_only=reference_only,
                   training_allowed=False, export_allowed=False)


class Provenance(BaseModel):
    """A pointer from a learned artifact back to its evidence."""

    source_class: SourceClass
    session_id: str | None = None
    segment_ids: list[str] = Field(default_factory=list)
    media_asset_id: str | None = None
    source_filename: str | None = None
    source_hash: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    frame_ids: list[str] = Field(default_factory=list)
    run_id: str | None = None
    extraction_method: str | None = None
    model_used: str | None = None
    evidence_weight: float = 1.0
