"""Hybrid segmentation orchestration (stage 1 + labels; stages 2/3 in refine.py and edits.py)."""

from __future__ import annotations

from lucius.events.bus import EventBus, EventType
from lucius.ids import new_id
from lucius.segmentation.labeler import features, label_segment, title_for
from lucius.segmentation.model import Segment, SegmentStore
from lucius.segmentation.signals import SignalConfig, boundary_signals, select_boundaries
from lucius.sessions.models import FrameRecord
from lucius.trajectory.compress import compress
from lucius.trajectory.model import TrajectoryStep
from lucius.trajectory.store import TrajectoryStore


# Segments whose edits mean an object has received its primary shaping (scene setup does not).
SHAPING_LABELS = {"primary_blockout", "secondary_forms", "detail_pass", "corrective_pass"}


class Segmenter:
    def __init__(self, segments: SegmentStore, trajectories: TrajectoryStore, bus: EventBus | None = None,
                 config: SignalConfig | None = None, representative_frames: int = 3) -> None:
        self.segments = segments
        self.trajectories = trajectories
        self.bus = bus
        self.config = config or SignalConfig()
        self.representative_frames = representative_frames

    def segment(self, session_id: str, steps: list[TrajectoryStep], frames: list[FrameRecord],
                annotations: list[dict] | None = None) -> list[Segment]:
        if not steps:
            self.segments.replace_unlocked(session_id, [])
            return self.segments.for_session(session_id)
        locked = [s for s in self.segments.for_session(session_id) if s.locked]
        covered = set()
        for seg in locked:
            covered.update(range(seg.step_start, seg.step_end + 1))
        change = {f.id: f.change_score or 0.0 for f in frames}
        reasons = boundary_signals(steps, self.config, change)
        drafts: list[Segment] = []
        for start, end in _free_ranges(len(steps), covered):
            sub = steps[start:end + 1]
            sub_reasons = {i - start: r for i, r in reasons.items() if start < i <= end}
            starts = select_boundaries(sub, sub_reasons, self.config)
            bounds = starts + [len(sub)]
            for a, b in zip(bounds, bounds[1:]):
                first, last = sub[a], sub[b - 1]
                drafts.append(Segment(
                    id=new_id("segment"), session_id=session_id, idx=0, t_start=first.t_start, t_end=last.t_end,
                    step_start=first.idx, step_end=last.idx, label="unlabeled", label_confidence=0.0,
                    boundary_reasons=reasons.get(first.idx, []),
                ))
        ordered = sorted(drafts + locked, key=lambda s: s.step_start)
        shaped: set[str] = set()
        for i, seg in enumerate(ordered):
            seg_steps = steps[seg.step_start:seg.step_end + 1]
            f = features(steps, seg_steps, shaped_objects=shaped, annotations=annotations)
            if not seg.locked:
                nxt = ordered[i + 1] if i + 1 < len(ordered) else None
                next_alt = nxt is not None and any(
                    "alternative_to" in s.meta for s in steps[nxt.step_start:nxt.step_end + 1])
                decision = label_segment(f, next_has_alternative=next_alt)
                seg.label = decision.label
                seg.label_confidence = decision.confidence
                seg.label_evidence = decision.evidence
                seg.outcome = decision.outcome
                seg.title = title_for(decision.label, f)
                seg.representative_frame_ids = self._representative(seg, seg_steps, frames)
                seg.summary = "; ".join(s.summary for s in compress(seg_steps))[:2000]
                seg.meta = {"annotations": f.annotations, "objects": sorted(f.objects), "views": sorted(f.views),
                            "action_counts": dict(f.actions)}
            if f.mutating and seg.label in SHAPING_LABELS:
                shaped |= f.objects
        self.segments.replace_unlocked(session_id, ordered)
        stored = self.segments.for_session(session_id)
        self.trajectories.assign_segments(session_id, [(s.id, s.step_start, s.step_end) for s in stored])
        if self.bus is not None:
            for seg in stored:
                if not seg.locked:
                    self.bus.publish(EventType.SEGMENT_CREATED, seg.id, session_id=session_id, label=seg.label,
                                     confidence=seg.label_confidence,
                                     reasons=[b.code for b in seg.boundary_reasons])
        return stored

    def _representative(self, seg: Segment, seg_steps: list[TrajectoryStep], frames: list[FrameRecord]) -> list[str]:
        inside = [f for f in frames if seg.t_start - 0.5 <= f.ts <= seg.t_end + 0.5]
        chosen: list[str] = []
        if inside:
            picks = [inside[0], max(inside, key=lambda f: f.change_score or 0.0), inside[-1]]
            for frame in picks:
                if frame.id not in chosen:
                    chosen.append(frame.id)
        else:
            for fid in (seg_steps[0].frame_before_id, seg_steps[-1].frame_after_id):
                if fid and fid not in chosen:
                    chosen.append(fid)
        return chosen[: self.representative_frames]


def _free_ranges(n: int, covered: set[int]) -> list[tuple[int, int]]:
    ranges, start = [], None
    for i in range(n):
        if i in covered:
            if start is not None:
                ranges.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        ranges.append((start, n - 1))
    return ranges
