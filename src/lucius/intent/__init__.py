"""Structured intent inference (V3).

Intent is stored as ``category / target / scope / confidence / evidence`` -- never as free-form
reasoning. Rules turn action-state evidence into hypotheses; human annotations and edits are
stronger evidence; a model can refine, but its output is recorded as ``model`` evidence.
Inferred intent is always a hypothesis with a confidence, never a fact.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from lucius.errors import ValidationError
from lucius.events.bus import EventBus, EventType
from lucius.ids import new_id
from lucius.segmentation.model import Segment
from lucius.storage.db import Database, dumps, loads
from lucius.taxonomy import Taxonomy
from lucius.timeutil import now
from lucius.trajectory import vocabulary as vocab
from lucius.trajectory.model import TrajectoryStep


class Intent(BaseModel):
    id: str
    segment_id: str
    category: str
    target: str | None = None
    scope: str | None = None
    confidence: float
    evidence_source: str            # rules, annotation (parsed human note), model, human (explicit edit)
    evidence: list[str] = Field(default_factory=list)
    reason_code: str
    is_current: bool = True
    created_at: float = Field(default_factory=now)


# Dimension names per object class and axis (for turning "scale along x" into "blade_width").
AXIS_TARGETS: dict[str, dict[str, str]] = {
    "blade_weapon": {"x": "blade_width", "y": "thickness", "z": "blade_length"},
}
UNDONE_TARGET = {"bevel": "edge_detail", "scale": "proportions", "extrude": "primary_form", "loop_cut": "edge_flow",
                 "inset": "secondary_form", "taper": "tip_profile", "subdivide": "topology",
                 "add_modifier": "edge_detail", "translate": "proportions"}
ANNOTATION_TARGETS = [
    (r"\btip\b", "tip_profile"), (r"\bwidth|wide|narrow|broad\b", "proportions"), (r"silhouette|outline", "silhouette"),
    (r"proportion", "proportions"), (r"symmetr|mirror", "symmetry"), (r"topolog|edge flow|loop", "topology"),
    (r"bevel|edge", "edge_detail"), (r"reference|ref\b", "reference_alignment"),
]
ANNOTATION_CATEGORIES = [
    (r"\b(fix|fixing|correct|wrong|mistake|too early|too big|too small|too broad|undo)\b", "correct"),
    (r"\b(check|look|inspect|compare)\b", "inspect"), (r"\b(verify|final|done)\b", "verify"),
]


def _axis_of(step: TrajectoryStep) -> str | None:
    axis = step.params.get("axis")
    if isinstance(axis, str) and len(axis) == 1:
        return axis
    value = step.params.get("value")
    if isinstance(value, list) and len(value) == 3 and step.action_type == "scale":
        changed = [a for a, v in zip("xyz", value) if abs(float(v) - 1.0) > 1e-6]
        return changed[0] if len(changed) == 1 else None
    return None


def _scope(segment: Segment) -> str | None:
    objects = segment.meta.get("objects") or []
    return ",".join(objects) if objects else None


class IntentEngine:
    def __init__(self, db: Database, taxonomy: Taxonomy, bus: EventBus | None = None) -> None:
        self.db = db
        self.taxonomy = taxonomy
        self.bus = bus

    # -- inference -------------------------------------------------------------------------------
    def hypotheses(self, segment: Segment, seg_steps: list[TrajectoryStep], task_class: str | None) -> list[Intent]:
        out: list[Intent] = []
        scope = _scope(segment)
        base = segment.label_confidence

        def add(category: str, target: str | None, confidence: float, reason: str, evidence: list[str],
                source: str = "rules") -> None:
            out.append(Intent(id=new_id("intent"), segment_id=segment.id, category=category, target=target,
                              scope=scope, confidence=round(max(0.05, min(0.97, confidence)), 3),
                              evidence_source=source, evidence=evidence, reason_code=reason))

        live = [s for s in seg_steps if not s.meta.get("cancelled")]
        mutations = [s for s in live if vocab.mutates(s.action_type) and s.action_type not in ("undo", "redo")]
        views = segment.meta.get("views") or []
        label = segment.label
        if label in ("inspection", "verification") or (not mutations and len(views) >= 2):
            category = "verify" if label == "verification" else "inspect"
            target = "silhouette" if {"front", "right", "left", "side"} & set(views) else "primary_form"
            add(category, target, base * 0.9, "multi_view_no_mutation",
                [f"views: {', '.join(views) or 'free orbit'}", "no geometry mutation"])
        elif label == "navigation":
            add("navigate", None, 0.6, "navigation_only", ["viewport movement without inspection pattern"])
        if label in ("corrective_pass", "recovery") or any(s.action_type == "undo" for s in live):
            undone = [s for s in seg_steps if "undone_by" in s.meta]
            alternatives = [s for s in seg_steps if "alternative_to" in s.meta]
            target = UNDONE_TARGET.get(undone[0].action_type) if undone else None
            evidence = [f"undo reverted {u.describe()}" for u in undone] + \
                       [f"alternative: {a.describe()}" for a in alternatives]
            if alternatives:
                alt_target = self._target_for(alternatives[0], task_class)
                add("correct", alt_target or target, 0.75, "undo_then_alternative", evidence)
            else:
                add("recover", target, 0.65, "undo_without_alternative", evidence)
        if label == "scene_setup":
            symmetric = any(s.action_type == "add_modifier" and s.params.get("type") == "MIRROR" for s in mutations)
            add("setup", "symmetry" if symmetric else "scene", base * 0.9,
                "setup_operations", [s.describe() for s in mutations[:5]])
        if label == "reference_alignment":
            add("setup", "reference_alignment", base * 0.9, "reference_loaded", ["reference image added"])
        if label in ("primary_blockout", "secondary_forms"):
            form = "primary_form" if label == "primary_blockout" else "secondary_form"
            creates = any(s.action_type in ("add_primitive", "extrude") for s in mutations)
            targets = Counter(t for t in (self._target_for(s, task_class) for s in mutations) if t)
            if creates or not targets:
                add("create" if creates else "refine", form, base * 0.9, "shaping_operations",
                    [s.describe() for s in mutations[:6]])
            total = sum(targets.values())
            for target, count in targets.most_common(3):
                add("refine", target, base * (0.5 + 0.4 * count / total), "dimension_adjustment",
                    [s.describe() for s in mutations if self._target_for(s, task_class) == target][:4])
        if label == "detail_pass":
            add("refine", "edge_detail", base * 0.9, "detail_operations", [s.describe() for s in mutations[:5]])
        for note in segment.meta.get("annotations") or []:
            category = next((c for pattern, c in ANNOTATION_CATEGORIES if re.search(pattern, note, re.I)), None)
            target = next((t for pattern, t in ANNOTATION_TARGETS if re.search(pattern, note, re.I)), None)
            if category or target:
                add(category or (out[0].category if out else "refine"), target, 0.9, "human_annotation",
                    [f"annotation: {note}"], source="annotation")
        if not out:
            add("navigate" if not mutations else "refine", None, 0.25, "weak_evidence", ["no specific pattern"])
        out.sort(key=lambda i: i.confidence, reverse=True)
        return out

    @staticmethod
    def _target_for(step: TrajectoryStep, task_class: str | None) -> str | None:
        if step.action_type in ("scale", "set_dimensions", "translate"):
            axis = _axis_of(step)
            if axis and task_class in AXIS_TARGETS:
                return AXIS_TARGETS[task_class][axis]
            return "proportions" if axis else None
        if step.action_type == "taper":
            return "tip_profile" if task_class == "blade_weapon" else "proportions"
        if step.action_type in ("loop_cut", "subdivide"):
            return "edge_flow"
        if step.action_type in ("bevel", "inset"):
            return "edge_detail" if step.action_type == "bevel" else "secondary_form"
        return None

    # -- persistence ---------------------------------------------------------------------------------
    def store(self, segment_id: str, intents: list[Intent]) -> None:
        """Replace machine intents for a segment. Human intents are kept and stay current."""
        with self.db.transaction() as conn:
            human = conn.execute(
                "SELECT COUNT(*) FROM intents WHERE segment_id = ? AND evidence_source = 'human' AND is_current = 1",
                (segment_id,)).fetchone()[0]
            conn.execute("DELETE FROM intents WHERE segment_id = ? AND evidence_source != 'human'", (segment_id,))
            for rank, intent in enumerate(intents):
                current = 1 if rank == 0 and not human else 0
                conn.execute(
                    "INSERT INTO intents (id, segment_id, category, target, scope, confidence, evidence_source,"
                    " evidence, reason_code, is_current, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (intent.id, segment_id, intent.category, intent.target, intent.scope, intent.confidence,
                     intent.evidence_source, dumps(intent.evidence), intent.reason_code, current, intent.created_at))
        if self.bus is not None and intents:
            top = intents[0]
            self.bus.publish(EventType.INTENT_INFERRED, segment_id, category=top.category, target=top.target,
                             confidence=top.confidence, reason_code=top.reason_code)

    def for_segment(self, segment_id: str, *, current_only: bool = False) -> list[Intent]:
        sql = "SELECT * FROM intents WHERE segment_id = ?" + (" AND is_current = 1" if current_only else "")
        rows = self.db.query(sql + " ORDER BY is_current DESC, confidence DESC", (segment_id,))
        return [Intent(id=r["id"], segment_id=r["segment_id"], category=r["category"], target=r["target"],
                       scope=r["scope"], confidence=r["confidence"], evidence_source=r["evidence_source"],
                       evidence=loads(r["evidence"], []), reason_code=r["reason_code"], is_current=bool(r["is_current"]),
                       created_at=r["created_at"]) for r in rows]

    def primary(self, segment_id: str) -> Intent | None:
        current = self.for_segment(segment_id, current_only=True)
        return current[0] if current else None

    def set_human(self, segment_id: str, *, category: str, target: str | None, scope: str | None,
                  user_id: str = "local") -> Intent:
        category = category.strip().lower()
        if not category:
            raise ValidationError("intent category is required")
        self.taxonomy.ensure("intent_category", category, source="human")
        if target:
            self.taxonomy.ensure("intent_target", target, source="human")
        before = [i.model_dump() for i in self.for_segment(segment_id, current_only=True)]
        intent = Intent(id=new_id("intent"), segment_id=segment_id, category=category, target=target, scope=scope,
                        confidence=1.0, evidence_source="human", evidence=["set by user"], reason_code="human_edit")
        with self.db.transaction() as conn:
            conn.execute("UPDATE intents SET is_current = 0 WHERE segment_id = ?", (segment_id,))
            conn.execute(
                "INSERT INTO intents (id, segment_id, category, target, scope, confidence, evidence_source, evidence,"
                " reason_code, is_current, created_at) VALUES (?,?,?,?,?,?,?,?,?,1,?)",
                (intent.id, segment_id, category, target, scope, 1.0, "human", dumps(intent.evidence), "human_edit",
                 intent.created_at))
            conn.execute(
                "INSERT INTO human_edits (id, subject_kind, subject_id, op, before, after, user_id, created_at)"
                " VALUES (?, 'segment', ?, 'set_intent', ?, ?, ?, ?)",
                (new_id("edit"), segment_id, dumps(before), dumps(intent.model_dump()), user_id, now()))
        return intent

    def summary(self, intent: Intent | None) -> dict[str, Any] | None:
        if intent is None:
            return None
        return {"category": intent.category, "target": intent.target, "scope": intent.scope,
                "confidence": intent.confidence, "source": intent.evidence_source, "reason_code": intent.reason_code}
