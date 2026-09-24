"""Stage 3: human segment corrections (merge, split, relabel, outcome).

Every edit locks the resulting segments against re-segmentation and is itself recorded in
``human_edits`` -- human corrections are learning data, not just UI state.
"""

from __future__ import annotations

from lucius.errors import NotFoundError, ValidationError
from lucius.events.bus import EventBus, EventType
from lucius.ids import new_id
from lucius.segmentation.model import LabelEvidence, Segment, SegmentStore
from lucius.storage.db import Database, dumps
from lucius.taxonomy import Taxonomy
from lucius.timeutil import now
from lucius.trajectory.store import TrajectoryStore

OUTCOMES = {"unknown", "success", "failure", "corrected"}


class SegmentEditor:
    def __init__(self, db: Database, segments: SegmentStore, trajectories: TrajectoryStore, taxonomy: Taxonomy,
                 bus: EventBus | None = None, user_id: str = "local") -> None:
        self.db = db
        self.segments = segments
        self.trajectories = trajectories
        self.taxonomy = taxonomy
        self.bus = bus
        self.user_id = user_id

    def _get(self, segment_id: str) -> Segment:
        seg = self.segments.get(segment_id)
        if seg is None:
            raise NotFoundError(f"segment {segment_id} not found", segment_id=segment_id)
        return seg

    def _log(self, subject_id: str, op: str, before: object, after: object) -> None:
        self.db.insert("human_edits", {"id": new_id("edit"), "subject_kind": "segment", "subject_id": subject_id,
                                       "op": op, "before": dumps(before), "after": dumps(after),
                                       "user_id": self.user_id, "created_at": now()})
        if self.bus is not None:
            self.bus.publish(EventType.SEGMENT_EDITED, subject_id, op=op)

    def _resync(self, session_id: str) -> None:
        stored = self.segments.for_session(session_id)
        self.trajectories.assign_segments(session_id, [(s.id, s.step_start, s.step_end) for s in stored])

    def relabel(self, segment_id: str, label: str, title: str | None = None) -> Segment:
        seg = self._get(segment_id)
        before = {"label": seg.label, "title": seg.title, "origin": seg.origin, "confidence": seg.label_confidence}
        label = label.strip().lower().replace(" ", "_")
        self.taxonomy.ensure("segment_label", label, source="human")
        seg.meta.setdefault("previous_labels", []).append(before)
        seg.label = label
        seg.title = title or seg.title
        seg.origin = "human"
        seg.locked = True
        seg.label_confidence = 1.0
        seg.label_evidence.append(LabelEvidence(reason_code="human_label", detail="set by user"))
        self.segments.save(seg)
        self._log(segment_id, "relabel", before, {"label": seg.label, "title": seg.title})
        return seg

    def set_outcome(self, segment_id: str, outcome: str) -> Segment:
        if outcome not in OUTCOMES:
            raise ValidationError(f"outcome must be one of {sorted(OUTCOMES)}")
        seg = self._get(segment_id)
        before = seg.outcome
        seg.outcome = outcome
        seg.locked = True
        seg.meta["outcome_source"] = "human"
        self.segments.save(seg)
        self._log(segment_id, "set_outcome", before, outcome)
        return seg

    def merge(self, segment_ids: list[str], label: str | None = None) -> Segment:
        segs = sorted((self._get(s) for s in segment_ids), key=lambda s: s.step_start)
        if len(segs) < 2:
            raise ValidationError("merge needs at least two segments")
        if len({s.session_id for s in segs}) != 1:
            raise ValidationError("segments belong to different sessions")
        for a, b in zip(segs, segs[1:]):
            if b.step_start != a.step_end + 1:
                raise ValidationError("only adjacent segments can be merged")
        first = segs[0]
        merged = Segment(
            id=new_id("segment"), session_id=first.session_id, idx=first.idx, t_start=first.t_start,
            t_end=segs[-1].t_end, step_start=first.step_start, step_end=segs[-1].step_end,
            label=(label or first.label).strip().lower().replace(" ", "_"), title=first.title,
            label_confidence=1.0 if label else first.label_confidence, origin="human", locked=True,
            outcome=first.outcome, boundary_reasons=first.boundary_reasons,
            label_evidence=[LabelEvidence(reason_code="human_merge", detail=f"merged {len(segs)} segments")],
            representative_frame_ids=[f for s in segs for f in s.representative_frame_ids][:6],
            summary="; ".join(s.summary or "" for s in segs)[:2000],
            meta={"merged_from": [s.id for s in segs]},
        )
        if label:
            self.taxonomy.ensure("segment_label", label, source="human")
        for s in segs:
            self.segments.delete(s.id)
        self.segments.save(merged)
        self._resync(first.session_id)
        self._log(merged.id, "merge", [s.id for s in segs], merged.id)
        return merged

    def split(self, segment_id: str, at_step: int) -> tuple[Segment, Segment]:
        seg = self._get(segment_id)
        if not seg.step_start < at_step <= seg.step_end:
            raise ValidationError("split point must be inside the segment (and not its first step)")
        steps = {s.idx: s for s in self.trajectories.for_session(seg.session_id)}
        left = seg.model_copy(deep=True)
        right = seg.model_copy(deep=True)
        left.id, right.id = new_id("segment"), new_id("segment")
        left.step_end, right.step_start = at_step - 1, at_step
        left.t_end, right.t_start = steps[at_step - 1].t_end, steps[at_step].t_start
        for part in (left, right):
            part.locked = True
            part.origin = "human"
            part.meta = {**seg.meta, "split_from": seg.id}
            part.label_evidence = [*seg.label_evidence, LabelEvidence(reason_code="human_split", detail=f"at step {at_step}")]
        self.segments.delete(seg.id)
        self.segments.save(left)
        self.segments.save(right)
        self._resync(seg.session_id)
        self._log(segment_id, "split", {"segment": seg.id}, {"left": left.id, "right": right.id, "at_step": at_step})
        return left, right
