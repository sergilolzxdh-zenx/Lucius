"""Deterministic segment labelling with explicit evidence.

Each rule records *why* it fired (reason code + detail) and a confidence that reflects how
specific the evidence is. Model refinement and human edits can override labels later; the
evidence trail of the deterministic label is kept either way.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from lucius.segmentation.model import LabelEvidence
from lucius.trajectory import vocabulary as vocab
from lucius.trajectory.model import TrajectoryStep

ORTHO_VIEWS = {"front", "back", "left", "right", "top", "bottom"}


@dataclass
class SegmentFeatures:
    n_steps: int
    duration: float
    nav_steps: int
    mutating: list[TrajectoryStep]
    actions: Counter
    detail_levels: Counter
    undo_count: int
    undone: list[TrajectoryStep]
    alternatives: list[TrajectoryStep]
    views: set[str]
    objects: set[str]
    position: float
    after_last_mutation: bool
    first_shaping_for_object: bool
    modifiers_added: list[str]
    cancelled: int
    annotations: list[str] = field(default_factory=list)


def features(steps: list[TrajectoryStep], seg_steps: list[TrajectoryStep], *, shaped_objects: set[str],
             annotations: list[dict] | None = None) -> SegmentFeatures:
    live = [s for s in seg_steps if not s.meta.get("cancelled")]
    mutating = [s for s in live if vocab.mutates(s.action_type) and s.action_type not in ("undo", "redo")]
    t0 = steps[0].t_start if steps else 0.0
    total = max(1e-6, (steps[-1].t_end - t0) if steps else 1.0)
    last_mut_idx = max((s.idx for s in steps if vocab.mutates(s.action_type) and not s.meta.get("cancelled")),
                       default=-1)
    views = set()
    for s in live:
        view = s.params.get("view") or (s.state_after or {}).get("view")
        if view in ORTHO_VIEWS and vocab.is_navigation(s.action_type):
            views.add(view)
    objects = set()
    for s in mutating:
        # A created object becomes active afterwards; other edits act on the active object before.
        state = s.state_after if s.action_type in ("add_primitive", "duplicate") else s.state_before
        name = s.params.get("name") or (state or {}).get("active_object")
        if name and not (s.action_type == "delete"):
            objects.add(name)
    start, end = seg_steps[0].t_start, seg_steps[-1].t_end
    notes = [a.get("text", "") for a in (annotations or []) if start - 2.0 <= a.get("ts", 0) <= end + 2.0]
    return SegmentFeatures(
        n_steps=len(seg_steps), duration=end - start, nav_steps=sum(1 for s in live if vocab.is_navigation(s.action_type)),
        mutating=mutating, actions=Counter(s.action_type for s in live),
        detail_levels=Counter(vocab.detail_level(s.action_type, s.params) for s in mutating),
        undo_count=sum(1 for s in live if s.action_type == "undo"),
        undone=[s for s in mutating if "undone_by" in s.meta and "redone_by" not in s.meta],
        alternatives=[s for s in mutating if "alternative_to" in s.meta],
        views=views, objects=objects, position=(start - t0) / total,
        after_last_mutation=seg_steps[0].idx > last_mut_idx,
        first_shaping_for_object=bool(objects - shaped_objects) or (not objects and not shaped_objects),
        modifiers_added=[str(s.params.get("type")) for s in mutating if s.action_type == "add_modifier"],
        cancelled=sum(1 for s in seg_steps if s.meta.get("cancelled")), annotations=notes,
    )


@dataclass
class LabelDecision:
    label: str
    confidence: float
    evidence: list[LabelEvidence]
    outcome: str = "unknown"


def label_segment(f: SegmentFeatures, *, next_has_alternative: bool = False) -> LabelDecision:
    ev: list[LabelEvidence] = []

    def e(code: str, detail: str, weight: float = 1.0) -> None:
        ev.append(LabelEvidence(reason_code=code, detail=detail, weight=weight))

    outcome = "unknown"
    if f.mutating and f.undone and len(f.undone) >= max(1, len(f.mutating) // 2):
        outcome = "failure"
        e("mutations_undone", f"{len(f.undone)}/{len(f.mutating)} edits later undone")

    if f.actions.get("load_reference"):
        e("reference_loaded", "reference image added to the scene")
        return LabelDecision("reference_alignment", 0.8, ev, outcome)

    if f.undo_count:
        e("undo", f"{f.undo_count} undo step(s)")
        if f.alternatives or next_has_alternative:
            e("alternative_after_undo", "a different edit followed the undo")
            return LabelDecision("corrective_pass", 0.8, ev, "corrected")
        return LabelDecision("recovery", 0.7, ev, outcome)

    if not f.mutating:
        if f.nav_steps and len(f.views) >= 2:
            e("multi_view", f"orthographic views: {sorted(f.views)}")
            e("no_geometry_mutation", "no edits in segment")
            if f.after_last_mutation:
                e("after_last_edit", "no edits follow in the session")
                return LabelDecision("verification", 0.8, ev, outcome)
            return LabelDecision("inspection", 0.8, ev, outcome)
        if f.nav_steps:
            e("navigation_only", f"{f.nav_steps} navigation step(s)")
            if f.after_last_mutation:
                return LabelDecision("verification", 0.6, ev, outcome)
            label = "inspection" if f.nav_steps >= 3 else "navigation"
            return LabelDecision(label, 0.6 if label == "inspection" else 0.55, ev, outcome)
        if f.position < 0.15:
            e("early_non_edit", "non-editing actions at session start")
            return LabelDecision("scene_setup", 0.5, ev, outcome)
        e("no_signal", "no editing or navigation evidence")
        return LabelDecision("unlabeled", 0.3, ev, outcome)

    n = len(f.mutating)
    detail = f.detail_levels.get(2, 0)
    level0 = f.detail_levels.get(0, 0)
    level1 = f.detail_levels.get(1, 0)
    shaping = sum(f.actions.get(a, 0) for a in ("scale", "extrude", "loop_cut", "taper", "translate", "rotate"))
    setup_ops = (f.actions.get("delete", 0) + f.actions.get("add_primitive", 0) + f.actions.get("set_symmetry", 0)
                 + sum(1 for m in f.modifiers_added if m == "MIRROR"))
    if detail / n >= 0.5:
        e("detail_operations", f"{detail}/{n} edits are detail-level (bevel, subdivision, shading)")
        return LabelDecision("detail_pass", 0.75, ev, outcome)
    if setup_ops and shaping <= 1 and f.position <= 0.3:
        e("setup_operations", f"{setup_ops} setup operation(s) early in the session")
        return LabelDecision("scene_setup", 0.7, ev, outcome)
    if level0 >= level1:
        objects = ", ".join(sorted(f.objects)) or "active object"
        e("primary_operations", f"{level0}/{n} edits shape primary volume of {objects}")
        if f.first_shaping_for_object:
            e("first_shaping", "first shaping pass for this object")
            return LabelDecision("primary_blockout", 0.75, ev, outcome)
        return LabelDecision("secondary_forms", 0.55, ev, outcome)
    e("secondary_operations", f"{level1}/{n} edits are secondary-form operations")
    return LabelDecision("secondary_forms", 0.65, ev, outcome)


def title_for(label: str, f: SegmentFeatures) -> str:
    top = ", ".join(f"{a}×{c}" if c > 1 else a for a, c in f.actions.most_common(3))
    target = f" · {', '.join(sorted(f.objects))}" if f.objects else ""
    return f"{label.replace('_', ' ')}{target} ({top})" if top else label.replace("_", " ")
