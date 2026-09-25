"""Stage 2: model-assisted segment relabelling.

Only compact structure is sent -- compressed action spans, deterministic labels with their
evidence codes and at most a few representative frames -- never every frame. Human-locked
segments are never sent or changed. The model's self-reported confidence is discounted, and
the deterministic label is kept alongside for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucius.errors import ProviderError, ProviderUnavailable
from lucius.logging_setup import get_logger
from lucius.providers.base import ImageInput, Providers
from lucius.segmentation.model import LabelEvidence, Segment, SegmentStore
from lucius.sessions.models import Session
from lucius.storage.frames import FrameStore
from lucius.taxonomy import Taxonomy
from lucius.trajectory.compress import compress, render_spans
from lucius.trajectory.model import TrajectoryStep

log = get_logger("segmentation.refine")

MODEL_CONFIDENCE_DISCOUNT = 0.85
BATCH_SEGMENTS = 40

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "label": {"type": "string"},
                    "title": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason_codes": {"type": "array", "items": {"type": "string"}},
                    "new_label_description": {"type": ["string", "null"]},
                },
                "required": ["idx", "label", "title", "confidence", "reason_codes", "new_label_description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}

SYSTEM = (
    "You label phases of a recorded Blender modelling session. Each segment lists the structured actions a "
    "human performed. Choose the best label from the provided taxonomy; propose a new snake_case label only if "
    "none fits and describe it in new_label_description. Base every decision on the listed actions and "
    "images; reason_codes are short snake_case evidence codes. confidence is 0..1."
)


@dataclass
class RefineReport:
    status: str                       # applied, skipped, failed
    reason: str = ""
    changed: list[str] = field(default_factory=list)
    model: str | None = None


class SegmentRefiner:
    def __init__(self, providers: Providers, segments: SegmentStore, frames: FrameStore, taxonomy: Taxonomy,
                 max_images: int = 6) -> None:
        self.providers = providers
        self.segments = segments
        self.frames = frames
        self.taxonomy = taxonomy
        self.max_images = max_images

    def refine(self, session: Session, segments: list[Segment], steps: list[TrajectoryStep],
               frame_paths: dict[str, str]) -> RefineReport:
        if not self.providers.has("llm"):
            return RefineReport("skipped", "no LLM provider configured")
        candidates = [s for s in segments if not s.locked]
        if not candidates:
            return RefineReport("skipped", "all segments are human-locked")
        changed: list[str] = []
        models: list[str] = []
        errors: list[str] = []
        # Long tutorial chapters have hundreds of segments: one call's answer would exceed its output limit
        # (seen live on a 641-step chapter), so segments are labelled in batches.
        for first in range(0, len(candidates), BATCH_SEGMENTS):
            try:
                batch_changed, model = self._refine_batch(session, candidates[first:first + BATCH_SEGMENTS], steps,
                                                          frame_paths)
            except ProviderError as exc:
                log.warning("segment refinement failed: %s", exc.message)
                errors.append(f"{exc.code}: {exc.message}")
                if isinstance(exc, ProviderUnavailable) or "quota" in exc.message:
                    break  # later batches would fail the same way
                continue
            changed += batch_changed
            models.append(model)
        if not models:
            return RefineReport("failed", errors[0] if errors else "no batch was labelled")
        return RefineReport("applied", "; ".join(errors), changed=changed, model=models[0])

    def _refine_batch(self, session: Session, candidates: list[Segment], steps: list[TrajectoryStep],
                      frame_paths: dict[str, str]) -> tuple[list[str], str]:
        labels = self.taxonomy.terms("segment_label")  # grows when a batch proposes a new label
        lines = [f"Task: {session.task_text or 'unspecified'}", "Taxonomy:"]
        lines += [f"- {term}: {info['description'] or ''}" for term, info in labels.items()]
        t0 = steps[0].t_start if steps else session.start_time
        for seg in candidates:
            spans = compress(steps[seg.step_start:seg.step_end + 1])
            lines.append(f"\nSegment {seg.idx} [{seg.t_start - t0:.1f}s-{seg.t_end - t0:.1f}s] current label "
                         f"{seg.label} ({seg.label_confidence:.2f}); evidence: "
                         + ", ".join(e.reason_code for e in seg.label_evidence))
            lines.append(render_spans(spans[:40], t0))
            if seg.meta.get("annotations"):
                # Human notes and tutorial narration; long tutorial segments are capped.
                lines.append("Notes: " + " | ".join(seg.meta["annotations"])[:1500])
        images: list[ImageInput] = []
        use_vision = self.providers.has("vlm")
        if use_vision:
            for seg in sorted(candidates, key=lambda s: s.label_confidence)[: self.max_images]:
                if seg.representative_frame_ids:
                    rel = frame_paths.get(seg.representative_frame_ids[0])
                    if rel:
                        images.append(ImageInput.from_path(self.frames.path(rel), label=f"Segment {seg.idx} frame:"))
        llm = self.providers.vlm if images else self.providers.llm
        result = llm.complete_json(purpose="segment_labeling", system=SYSTEM, prompt="\n".join(lines),
                                   schema=SCHEMA, images=images, max_tokens=8000)
        by_idx = {s.idx: s for s in candidates}
        changed = []
        for item in result.data.get("segments", []):
            seg = by_idx.get(item.get("idx"))
            if seg is None:
                continue
            label = str(item.get("label", "")).strip().lower().replace(" ", "_")
            if not label:
                continue
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0)))) * MODEL_CONFIDENCE_DISCOUNT
            if label not in labels:
                self.taxonomy.ensure("segment_label", label, source="model",
                                     description=item.get("new_label_description"))
            if label == seg.label:
                seg.label_evidence.append(LabelEvidence(reason_code="model_agrees", detail=result.model,
                                                        weight=confidence))
                seg.label_confidence = min(0.95, max(seg.label_confidence, (seg.label_confidence + confidence) / 2 + 0.05))
            elif confidence > seg.label_confidence + 0.05:
                seg.meta["deterministic_label"] = {"label": seg.label, "confidence": seg.label_confidence}
                seg.label = label
                seg.title = str(item.get("title") or seg.title)
                seg.label_confidence = confidence
                seg.origin = "model"
                seg.label_evidence.append(LabelEvidence(
                    reason_code="model_relabel", detail=f"{result.model}: " + ", ".join(item.get("reason_codes", [])[:5]),
                    weight=confidence))
                changed.append(seg.id)
            else:
                continue
            self.segments.save(seg)
        return changed, result.model
