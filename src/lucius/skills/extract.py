"""Skill extraction from processed demonstrations (V4).

A processed session (trajectory + segments + intents) is split into *units* -- the work on one
object from setup through its checks -- and each unit becomes a per-instance candidate
definition:

* actions become templates over named parameters (a scale is parameterised by the dimension
  it produced, not by the typed factor; selections by normalised regions, never coordinates),
* inspection segments become visual checkpoints, final state becomes structural checkpoints,
* undo -> alternative sequences become failure records (failure memory) plus recovery actions.

The candidate is then matched against the library: a known skill gains a new example and is
re-generalised; an unknown one enters as ``candidate_pattern``. Nothing is promoted here.
"""

from __future__ import annotations

import difflib
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from lucius.events.bus import EventBus
from lucius.graph import Graph
from lucius.ids import new_id
from lucius.intent import AXIS_TARGETS
from lucius.memory.failure import FailureEvidence, FailureMemory, FailureObservation
from lucius.provenance import EVIDENCE_WEIGHT, SOURCE_WEIGHT, SourceClass
from lucius.segmentation.model import Segment
from lucius.sessions.models import Session
from lucius.skills.generalize import generalize, human_edited_fields
from lucius.skills.library import SkillLibrary, slugify
from lucius.skills.schema import (
    ActionTemplate,
    Checkpoint,
    Condition,
    FailureCondition,
    ParamSpec,
    RecoveryAction,
    Selection,
    SkillDefinition,
    SkillExample,
    SkillPhase,
    SkillStatus,
)
from lucius.storage.db import loads
from lucius.taxonomy import OBJECT_CLASSES, classify_task
from lucius.timeutil import now
from lucius.trajectory import vocabulary as vocab
from lucius.trajectory.model import TrajectoryStep

PHASE_OF_LABEL = {
    "scene_setup": "setup", "reference_alignment": "reference", "primary_blockout": "primary_form",
    "secondary_forms": "secondary_form", "detail_pass": "detail", "inspection": "inspection",
    "verification": "verification",
}
SHAPING_LABELS = {"primary_blockout", "secondary_forms", "detail_pass"}
ATTACHED_LABELS = {"inspection", "verification", "corrective_pass", "recovery", "navigation"}
DEFAULT_OBJECT_NAMES = {"cube", "plane", "cylinder", "cone", "sphere", "icosphere", "torus", "suzanne", "camera",
                        "light"}
SKIPPED_ACTIONS = {"undo", "redo", "save", "text_entry", "add_menu", "search_menu", "context_menu", "ui_click",
                   "select_click", "select_box", "select_all", "deselect_all", "select_mode", "tool_change",
                   "workspace_change", "rename", "snapshot", "restore", "unknown_action"}
SIMILARITY_THRESHOLD = 0.7


@dataclass
class Unit:
    segments: list[Segment] = field(default_factory=list)
    setup: list[Segment] = field(default_factory=list)

    @property
    def all_segments(self) -> list[Segment]:
        return sorted(self.setup + self.segments, key=lambda s: s.step_start)

    def objects(self) -> set[str]:
        return {o for s in self.segments if s.label in SHAPING_LABELS for o in s.meta.get("objects", [])}


@dataclass
class ExtractionResult:
    skill_ids: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    failure_ids: list[str] = field(default_factory=list)
    skipped_units: list[str] = field(default_factory=list)


def _dims(state: dict[str, Any] | None) -> list[float] | None:
    dims = (state or {}).get("dimensions")
    return [float(d) for d in dims] if isinstance(dims, list) and len(dims) == 3 else None


def _selection(state: dict[str, Any] | None) -> Selection | None:
    sel = (state or {}).get("edit_selection")
    if not isinstance(sel, dict):
        return None
    bbox = sel.get("normalized_bbox")
    if sel.get("verts") is not None and sel.get("verts") == sel.get("total_verts"):
        return Selection(kind="all")
    if not isinstance(bbox, dict):
        return Selection(kind="unknown")
    spans = {a: (b[1] - b[0]) for a, b in bbox.items() if isinstance(b, list) and len(b) == 2}
    if not spans:
        return Selection(kind="unknown")
    axis = min(spans, key=spans.get)
    if spans[axis] >= 0.999:
        return Selection(kind="all")
    lo, hi = bbox[axis]
    if hi - lo < 0.02:  # a selected edge loop has no extent along its axis: keep a small margin
        lo, hi = max(0.0, lo - 0.02), min(1.0, hi + 0.02)
    return Selection(kind="region", axis=axis, min=round(lo, 4), max=round(hi, 4))


def _region_name(sel: Selection | None) -> str:
    """Band of the normalised bounding box a region selection sits in (bottom/lower/upper/top)."""
    if sel is None or sel.kind != "region" or sel.min is None or sel.max is None:
        return ""
    centre = (sel.min + sel.max) / 2
    return "bottom" if centre < 0.25 else "lower" if centre < 0.5 else "upper" if centre < 0.75 else "top"


def _vec_axis(value: Any, neutral: float) -> str | None:
    """The single axis a 3-vector changes (None if zero or several axes change)."""
    if isinstance(value, list) and len(value) == 3:
        changed = [a for a, v in zip("xyz", value) if abs(float(v) - neutral) > 1e-6]
        return changed[0] if len(changed) == 1 else None
    return None


def _vector(step: TrajectoryStep) -> Any:
    """The step's 3-vector amount: operator ``value``, or bridge-action ``factor``/``offset``."""
    for key in ("value", "factor", "offset"):
        if step.params.get(key) is not None:
            return step.params[key]
    return None


def _axis_of(step: TrajectoryStep) -> str | None:
    axis = step.params.get("axis")
    if isinstance(axis, str) and len(axis) == 1:
        return axis
    return _vec_axis(_vector(step), 1.0 if step.action_type == "scale" else 0.0)


def _axis_value(step: TrajectoryStep, axis: str) -> float | None:
    value = _vector(step)
    if isinstance(value, list) and len(value) == 3:
        return float(value["xyz".index(axis)])
    if isinstance(value, (int, float)):
        return float(value)
    return None


class _CandidateBuilder:
    """Builds one per-instance definition for a unit."""

    def __init__(self, session: Session, unit: Unit, steps: list[TrajectoryStep], role: str | None,
                 object_name: str | None, object_class: str | None, categories: tuple[str, ...]) -> None:
        self.session = session
        self.unit = unit
        self.steps = steps
        self.role = role
        self.object_name = object_name
        self.object_class = object_class
        self.categories = categories
        self.params: dict[str, ParamSpec] = {}

    # -- parameters --------------------------------------------------------------------------------
    def param(self, name: str, value: Any, *, kind: str = "float", unit: str | None = None,
              description: str = "") -> str:
        name = slugify(name)
        spec = self.params.get(name)
        if spec is None:
            spec = ParamSpec(name=name, kind=kind, unit=unit, description=description, default=value)
            self.params[name] = spec
        spec.observed_values.append(value)
        spec.default = value
        return "{" + name + "}"

    def dimension_param(self, axis: str) -> str:
        role = self.role or "object"
        if self.object_class in AXIS_TARGETS and self.role and self.role in OBJECT_CLASSES.get(
                self.object_class, ((), ()))[0]:
            return AXIS_TARGETS[self.object_class][axis]
        return f"{role}_size_{axis}"

    # -- actions -----------------------------------------------------------------------------------
    def template(self, step: TrajectoryStep) -> ActionTemplate | None:
        at = step.action_type
        hint = [i["hotkey"] for i in step.action_payload.get("input", []) if isinstance(i, dict) and "hotkey" in i]
        base = {"description": step.describe(), "gui_hint": hint, "evidence": [step.id]}
        mode = (step.mode_label or "").split("_")[0] or None
        role = self.role or "object"
        if at == "add_primitive":
            # An unknown kind stays unknown (None): a video often shows an object appear without saying what it
            # is, and guessing "cube" turned "add an area light" into a validated "add a cube".
            return ActionTemplate(action_type=at, args={"kind": step.params.get("kind"), "name": "{object_name}"},
                                  requires_mode="OBJECT", **base)
        if at == "delete" and (step.state_before or {}).get("active_object", "").lower() in DEFAULT_OBJECT_NAMES:
            return ActionTemplate(action_type="reset_scene", args={"keep_camera_light": True}, object_ref=None,
                                  description="start from an empty scene (default objects removed)",
                                  gui_hint=hint, evidence=[step.id])
        if at == "mode_change":
            target = step.params.get("mode") or (step.state_after or {}).get("mode")
            if not target:
                return None
            return ActionTemplate(action_type=at, args={"mode": "EDIT" if str(target).startswith("EDIT") else "OBJECT"},
                                  **base)
        if at == "add_modifier":
            return ActionTemplate(action_type=at, args={"type": step.params.get("type")}, **base)
        if at in ("scale", "translate", "rotate"):
            axis = _axis_of(step)
            sel = _selection(step.state_before) if mode == "EDIT" else None
            if axis is None:
                return ActionTemplate(action_type=at, args={"value": step.params.get("value")}, selection=sel,
                                      requires_mode=mode, optional=True, **base)
            if at == "scale" and (sel is None or sel.kind == "all"):
                before, after = _dims(step.state_before), _dims(step.state_after)
                if after is not None and (before is None or abs(after["xyz".index(axis)] - before["xyz".index(axis)]) > 1e-6):
                    size = round(after["xyz".index(axis)], 4)
                    ref = self.param(self.dimension_param(axis), size, unit="m",
                                     description=f"{role} size along {axis} after this step")
                    return ActionTemplate(action_type=at, args={"axis": axis, "target_size": ref}, selection=sel,
                                          requires_mode=mode, **base)
            value = _axis_value(step, axis)
            region = _region_name(sel)
            name = f"{role}_{region + '_' if region else ''}{at}_{axis}"
            ref = self.param(name, round(value, 4) if value is not None else None,
                             unit="ratio" if at == "scale" else "m",
                             description=f"{at} factor along {axis}" + (f" on the {region} region" if region else ""))
            args = {"axis": axis, "factor" if at == "scale" else "amount": ref}
            return ActionTemplate(action_type=at, args=args, selection=sel, requires_mode=mode, **base)
        if at == "extrude":
            offset = step.params.get("offset")
            axis = _vec_axis(offset, 0.0)
            if axis is None or not isinstance(offset, list):
                return ActionTemplate(action_type=at, args={"offset": offset}, requires_mode="EDIT",
                                      selection=_selection(step.state_before), optional=True, **base)
            ref = self.param(f"{role}_extrude_{axis}", round(float(offset["xyz".index(axis)]), 4), unit="m",
                             description=f"extrusion distance along {axis}")
            return ActionTemplate(action_type=at, args={"axis": axis, "distance": ref}, requires_mode="EDIT",
                                  selection=_selection(step.state_before), **base)
        if at == "loop_cut":
            dims = _dims(step.state_before)
            axis = "xyz"[dims.index(max(dims))] if dims else "z"
            ref = self.param(f"{role}_loop_cuts", int(step.params.get("cuts") or 1), kind="int", unit="count",
                             description="edge loops across the long axis")
            return ActionTemplate(action_type=at, args={"axis": axis, "cuts": ref, "axis_source": "longest_dimension"},
                                  requires_mode="EDIT", **base)
        if at == "bevel":
            width = self.param(f"{role}_bevel_width", step.params.get("offset", step.params.get("value")), unit="m")
            segments = self.param(f"{role}_bevel_segments", int(step.params.get("segments") or 1), kind="int",
                                  unit="count")
            return ActionTemplate(action_type=at, args={"offset": width, "segments": segments}, requires_mode="EDIT",
                                  selection=_selection(step.state_before), **base)
        if at == "inset":
            ref = self.param(f"{role}_inset", step.params.get("thickness", step.params.get("value")), unit="m")
            return ActionTemplate(action_type=at, args={"thickness": ref}, requires_mode="EDIT",
                                  selection=_selection(step.state_before), **base)
        if at == "taper":
            ref = self.param(f"{role}_taper", step.params.get("amount"), unit="ratio")
            return ActionTemplate(action_type=at, args={"along": step.params.get("along", "z"), "amount": ref,
                                                        "affect": step.params.get("affect", "x")},
                                  requires_mode="EDIT", **base)
        if at in ("set_dimensions",):
            dims = step.params.get("dimensions")
            refs = [self.param(self.dimension_param(a), v, unit="m") for a, v in zip("xyz", dims or [])]
            return ActionTemplate(action_type=at, args={"dimensions": refs}, requires_mode="OBJECT", **base)
        if at in ("apply_transform", "set_symmetry", "shade_smooth", "shade_flat", "merge", "subdivide", "fill",
                  "bridge_loops", "duplicate", "delete", "geometry_edit", "load_reference", "sculpt_stroke", "knife"):
            return ActionTemplate(action_type=at, args=dict(step.params), requires_mode=mode,
                                  optional=at in ("geometry_edit", "knife", "sculpt_stroke"), **base)
        if at.startswith("op.") or at.startswith("api."):
            return ActionTemplate(action_type=at, args=dict(step.params), requires_mode=mode, optional=True, **base)
        return None

    def live_steps(self, segment: Segment) -> list[TrajectoryStep]:
        out = []
        for step in self.steps[segment.step_start:segment.step_end + 1]:
            if step.meta.get("cancelled") or step.action_type in SKIPPED_ACTIONS:
                continue
            if "undone_by" in step.meta and "redone_by" not in step.meta:
                continue  # undone work is failure evidence, not procedure
            if vocab.is_navigation(step.action_type):
                continue
            out.append(step)
        return out


class SkillExtractor:
    def __init__(self, library: SkillLibrary, failures: FailureMemory, graph: Graph,
                 bus: EventBus | None = None) -> None:
        self.library = library
        self.failures = failures
        self.graph = graph
        self.bus = bus

    # -- units -------------------------------------------------------------------------------------
    @staticmethod
    def units(segments: list[Segment]) -> list[Unit]:
        units: list[Unit] = []
        pending_setup: list[Segment] = []
        current: Unit | None = None
        for seg in sorted(segments, key=lambda s: s.step_start):
            if seg.label in ("scene_setup", "reference_alignment"):
                pending_setup.append(seg)
                continue
            if seg.label in SHAPING_LABELS or seg.label not in ATTACHED_LABELS:
                objects = set(seg.meta.get("objects") or [])
                if current is None or (objects and not objects & current.objects()):
                    current = Unit(segments=[seg], setup=pending_setup)
                    pending_setup = []
                    units.append(current)
                else:
                    current.segments.append(seg)
                continue
            if current is not None:
                current.segments.append(seg)
        return [u for u in units if any(s.label in SHAPING_LABELS for s in u.segments)]

    # -- extraction --------------------------------------------------------------------------------
    def extract(self, session: Session, steps: list[TrajectoryStep], segments: list[Segment]) -> ExtractionResult:
        result = ExtractionResult()
        object_class, categories = classify_task(session.task_text)
        previous_skill: str | None = None
        for unit in self.units(segments):
            object_name = self._main_object(unit, steps)
            role = object_name.lower() if object_name and object_name.lower() not in DEFAULT_OBJECT_NAMES else None
            unit_class = object_class
            if role and not unit_class:
                unit_class, categories = classify_task(role)
            builder = _CandidateBuilder(session, unit, steps, role, object_name, unit_class, categories)
            candidate = self._candidate(builder)
            if candidate is None:
                result.skipped_units.append(",".join(s.id for s in unit.segments))
                continue
            skill_id, created = self._resolve_target(session, candidate)
            candidate = candidate.model_copy(update={"skill_id": skill_id})
            failure_ids = self._failures(session, unit, steps, builder, candidate)
            result.failure_ids += failure_ids
            self._add_evidence(session, unit, steps, candidate)
            result.skill_ids.append(skill_id)
            (result.created if created else result.updated).append(skill_id)
            for fid in failure_ids:
                self.graph.link(("skill", skill_id), "causes_failure", ("failure", fid))
                self.graph.link(("failure", fid), "recovers_with", ("skill", skill_id),
                                meta={"recovery": "inline recovery action"})
            if previous_skill and previous_skill != skill_id:
                self.graph.link(("skill", previous_skill), "follows", ("skill", skill_id))
            previous_skill = skill_id
        for a in set(result.skill_ids):
            for b in set(result.skill_ids):
                if a < b:
                    self.graph.link(("skill", a), "co_used", ("skill", b))
        return result

    @staticmethod
    def _main_object(unit: Unit, steps: list[TrajectoryStep]) -> str | None:
        counts: Counter[str] = Counter()
        for seg in unit.segments:
            if seg.label in SHAPING_LABELS:
                counts.update(seg.meta.get("objects") or [])
        if not counts:
            return None
        named = [(o, c) for o, c in counts.most_common() if o.lower() not in DEFAULT_OBJECT_NAMES]
        return (named or counts.most_common())[0][0]

    def _candidate(self, b: _CandidateBuilder) -> SkillDefinition | None:
        phases: list[SkillPhase] = []
        checkpoints: list[Checkpoint] = []
        views_after: dict[str, set[str]] = {}
        last_shaping: str | None = None
        for seg in b.unit.all_segments:
            name = PHASE_OF_LABEL.get(seg.label)
            if seg.label in ("corrective_pass", "recovery"):
                # The surviving correction is procedure (the failure is recorded separately). File it
                # the way the labeller files ordinary work: a first pass on the object is primary form,
                # later work on an already-shaped object is secondary form, detail-level work is detail.
                kept = b.live_steps(seg)
                levels = [vocab.detail_level(s.action_type, s.params) for s in kept if vocab.mutates(s.action_type)]
                if levels and min(levels) >= 2:
                    name = "detail"
                else:
                    name = "secondary_form" if last_shaping else "primary_form"
            if name is None:
                continue
            if name in ("inspection", "verification"):
                views = set(seg.meta.get("views") or [])
                if last_shaping:
                    views_after.setdefault(last_shaping, set()).update(views)
                continue
            actions = [t for t in (b.template(s) for s in b.live_steps(seg)) if t is not None]
            if not actions:
                continue
            if phases and phases[-1].name == name:
                phases[-1].actions += actions
            else:
                phases.append(SkillPhase(name=name, actions=actions,
                                         description=f"{name.replace('_', ' ')} ({seg.label})"))
            if name != "setup":
                last_shaping = name
        if not any(p.name in ("primary_form", "secondary_form", "detail") for p in phases):
            return None
        obj = "{object_name}"
        role = b.role or "object"
        first_shaping = next(p for p in phases if p.name != "setup")
        final_dims = self._final_dims(b)
        created_in_skill = any(a.action_type == "add_primitive" for p in phases for a in p.actions)
        first_shaping.checkpoints.append(f"{role}_exists")
        checkpoints.append(Checkpoint(id=f"{role}_exists", description=f"{role} object exists", level=2,
                                      check={"type": "object_exists", "object": obj}, after_phase=first_shaping.name))
        dimension_params = {a: n for a, n in ((ax, b.dimension_param(ax)) for ax in "xyz") if n in b.params}
        if dimension_params:
            cp_id = f"{role}_dimensions"
            checkpoints.append(Checkpoint(
                id=cp_id, description=f"{role} dimensions match the requested parameters "
                                       + ", ".join(f"{a}={n}" for a, n in dimension_params.items()),
                level=2, check={"type": "dimensions_match", "object": obj,
                                "axes": {a: "{" + n + "}" for a, n in dimension_params.items()}, "tolerance": 0.08},
                after_phase=first_shaping.name))
            first_shaping.checkpoints.append(cp_id)
        if final_dims and min(final_dims) > 0:
            order = sorted(range(3), key=lambda i: final_dims[i], reverse=True)
            ratio = final_dims[order[0]] / final_dims[order[1]]
            if ratio >= 1.5:
                cp_id = f"{role}_aspect"
                checkpoints.append(Checkpoint(
                    id=cp_id, description=f"{role} is elongated along {'xyz'[order[0]]} "
                                          f"(observed {ratio:.1f}:1 against {'xyz'[order[1]]})",
                    level=2, required=False,
                    check={"type": "dimension_ratio", "object": obj, "numerator": "xyz"[order[0]],
                           "denominator": "xyz"[order[1]], "min": round(ratio * 0.6, 3), "max": round(ratio * 1.6, 3)},
                    after_phase=first_shaping.name))
        if any(a.action_type == "add_modifier" and a.args.get("type") == "MIRROR" for p in phases for a in p.actions):
            checkpoints.append(Checkpoint(id=f"{role}_symmetry", description=f"{role} is mirror-symmetric along x",
                                          level=2, check={"type": "symmetry", "object": obj, "axis": "x",
                                                          "max_error": 0.02}, after_phase=first_shaping.name))
            first_shaping.checkpoints.append(f"{role}_symmetry")
        tip = [a for p in phases for a in p.actions if a.selection is not None and a.selection.kind == "region"
               and a.action_type == "scale"]
        if tip and final_dims:
            along = tip[0].selection.axis or "z"
            width_axis = tip[0].args.get("axis", "x")
            cp_id = f"{role}_taper"
            checkpoints.append(Checkpoint(
                id=cp_id, description=f"{role} narrows towards the {_region_name(tip[0].selection) or 'end'} along {along}",
                level=2, check={"type": "taper", "object": obj, "along": along, "width_axis": width_axis,
                                "region": [tip[0].selection.min, tip[0].selection.max], "max_ratio": 0.8},
                after_phase=last_shaping))
            for p in phases:
                if p.name == last_shaping:
                    p.checkpoints.append(cp_id)
        for phase_name, views in views_after.items():
            ortho = sorted(v for v in views if v in ("front", "back", "left", "right", "top", "bottom"))
            if not ortho:
                continue
            cp_id = f"{role}_{phase_name}_silhouette"
            final = phase_name == last_shaping
            # Only the last shaping phase must match the final target; earlier inspections checked an
            # unfinished form, so they are advisory with a relaxed threshold.
            check = {"type": "silhouette", "object": obj, "views": ortho}
            if not final:
                check["stage"] = "intermediate"
            checkpoints.append(Checkpoint(
                id=cp_id, description=f"{role} silhouette reads correctly from {', '.join(ortho)} views"
                                      + ("" if final else " (intermediate form)"),
                level=3, method="visual_measured", check=check, required=final, after_phase=phase_name,
                derived_from=[s.id for s in b.unit.segments if s.label in ("inspection", "verification")]))
            for p in phases:
                if p.name == phase_name:
                    p.checkpoints.append(cp_id)
        for p in phases:
            if p.name != "setup":
                p.preconditions = [Condition(kind="object_exists", value=obj, description=f"{role} exists")]
                break
        if not created_in_skill:
            b.param("object_name", b.object_name, kind="str", description="object the skill operates on")
        else:
            b.param("object_name", (b.object_name or role).title(), kind="str", description="name of the created object")
        what = "blockout" if first_shaping.name == "primary_form" else first_shaping.name.replace("_", " ")
        base_category = b.categories[0] if b.categories else "general"
        skill_id = slugify(f"{base_category} {role} {what}")
        triggers = sorted({role, what, *(b.categories or ()), *(b.object_class.split("_") if b.object_class else ()),
                           *(k for k in OBJECT_CLASSES.get(b.object_class or "", ((), ()))[0][:4])} - {"object"})
        return SkillDefinition(
            skill_id=skill_id, name=f"{role.replace('_', ' ').title()} {what}",
            purpose=f"Establish the {role} {'primary form and proportions' if what == 'blockout' else what}"
                    + (f" for a {b.object_class.replace('_', ' ')}" if b.object_class else ""),
            categories=list(b.categories) + ([what] if what not in b.categories else []), object_class=b.object_class,
            object_role=role, applicable_contexts=[c for c in (
                f"object_class:{b.object_class}" if b.object_class else None,
                f"blender:{b.session.environment.blender_version}" if b.session.environment.blender_version else None,
            ) if c], triggers=triggers, parameters=list(b.params.values()), phases=phases, checkpoints=checkpoints,
            source_class=b.session.source.value,
        )

    @staticmethod
    def _final_dims(b: _CandidateBuilder) -> list[float] | None:
        dims = None
        for seg in b.unit.segments:
            for step in b.steps[seg.step_start:seg.step_end + 1]:
                state = step.state_after or {}
                if state.get("active_object") == b.object_name and _dims(state):
                    dims = _dims(state)
        return dims

    # -- failures & recovery -----------------------------------------------------------------------
    def _failures(self, session: Session, unit: Unit, steps: list[TrajectoryStep], b: _CandidateBuilder,
                  candidate: SkillDefinition) -> list[str]:
        ids = []
        role = b.role or "object"
        silhouette = next((c.id for c in candidate.checkpoints if c.id.endswith("_silhouette")), None)
        for seg in unit.segments:
            for step in steps[seg.step_start:seg.step_end + 1]:
                if "undone_by" not in step.meta or "redone_by" in step.meta:
                    continue
                undo = steps[step.meta["undone_by"]]
                alt_idx = step.meta.get("replaced_by")
                alternatives = []
                if alt_idx is not None:
                    # the alternative and its immediate follow-ups within the same segment
                    alt_seg_end = next((s.step_end for s in unit.segments if s.step_start <= alt_idx <= s.step_end),
                                       alt_idx)
                    alternatives = [s for s in steps[alt_idx:alt_seg_end + 1]
                                    if vocab.mutates(s.action_type) and s.action_type not in ("undo", "redo")
                                    and not s.meta.get("cancelled")]
                notes = [a for a in seg.meta.get("annotations", [])]
                level_bad = vocab.detail_level(step.action_type, step.params)
                level_alt = min((vocab.detail_level(a.action_type, a.params) for a in alternatives), default=level_bad)
                phase = {0: "primary_form", 1: "secondary_form", 2: "detail"}.get(level_alt, "primary_form")
                premature = level_bad > level_alt
                symptoms = [n.split(" - ", 1)[-1].strip() for n in notes] or [f"{step.action_type} reverted by the user"]
                if premature:
                    cause = (f"{step.action_type} (detail-level edit) applied before the primary {role} form was "
                             f"finalised")
                    view_text = "front and side" if silhouette else "multiple"
                    rule = (f"Do not add {step.action_type} until the primary {role} silhouette passes "
                            f"{view_text} checks")
                elif alternatives and alternatives[0].action_type == step.action_type:
                    cause = f"wrong parameters for {step.action_type}"
                    rule = f"Use {alternatives[0].describe()} rather than {step.describe()} for the {role}"
                else:
                    cause = f"{step.action_type} was the wrong operation at this point" + (
                        f"; replaced by {alternatives[0].action_type}" if alternatives else "")
                    rule = (f"Prefer {alternatives[0].action_type} over {step.action_type} during {phase.replace('_', ' ')}"
                            if alternatives else f"Avoid {step.action_type} during {phase.replace('_', ' ')}")
                correction = f"undo {step.describe()}" + (
                    "; then " + ", ".join(a.describe() for a in alternatives) if alternatives else "")
                weight = EVIDENCE_WEIGHT[step.evidence_kind] * step.action_confidence * (1.15 if notes else 1.0)
                record = self.failures.record(FailureObservation(
                    task_class=b.object_class, phase=phase,
                    observed_problem=f"{step.action_type} was undone during {phase.replace('_', ' ')}"
                                     + (f": {notes[0]}" if notes else ""),
                    symptoms=symptoms, likely_cause=cause, correction=correction, future_rule=rule,
                    trigger_action=step.action_type, guard_checkpoint=silhouette if premature else None,
                    skill_id=candidate.skill_id,
                    evidence=FailureEvidence(
                        kind="demo_undo", source_class=session.source.value, session_id=session.id, segment_id=seg.id,
                        step_ids=[step.id, undo.id] + [a.id for a in alternatives],
                        frame_ids=[f for f in (step.frame_before_id, step.frame_after_id,
                                               alternatives[-1].frame_after_id if alternatives else None) if f],
                        detail="; ".join(notes), weight=min(1.0, weight)),
                ))
                ids.append(record.id)
                fc_id = f"fc_{step.action_type}_{phase}"
                if not any(fc.id == fc_id for fc in candidate.failure_conditions):
                    candidate.failure_conditions.append(FailureCondition(
                        id=fc_id, description=rule, phase=phase, trigger_action=step.action_type,
                        guard_checkpoint=record.guard_checkpoint, failure_ids=[record.id], confidence=record.confidence))
                    templates = [t for t in (b.template(a) for a in alternatives) if t is not None]
                    candidate.recovery_actions.append(RecoveryAction(
                        id=f"rc_{step.action_type}_{phase}", description=correction, when=fc_id,
                        actions=[ActionTemplate(action_type="restore", args={"to": "before_trigger"},
                                                description=f"revert the {step.action_type}", object_ref=None),
                                 *templates],
                        source=SourceClass.HUMAN_CORRECTION.value if session.source == SourceClass.USER_DEMO
                        else session.source.value, evidence=[step.id, undo.id] + [a.id for a in alternatives]))
        return ids

    # -- library integration -----------------------------------------------------------------------
    def _resolve_target(self, session: Session, candidate: SkillDefinition) -> tuple[str, bool]:
        target = self._match(candidate)
        if target is not None:
            return target, False
        self.library.create(candidate, created_by="extractor", status=SkillStatus.CANDIDATE_PATTERN,
                            change_note=f"extracted from session {session.id}")
        return candidate.skill_id, True

    def _add_evidence(self, session: Session, unit: Unit, steps: list[TrajectoryStep],
                      candidate: SkillDefinition) -> None:
        skill_id = candidate.skill_id
        instance = hashlib.sha1(repr(sorted(
            (p.name, round(p.default, 2) if isinstance(p.default, float) else p.default)
            for p in candidate.parameters if p.name != "object_name")).encode()).hexdigest()[:12]
        segs = unit.all_segments
        unit_steps = [s for seg in segs for s in steps[seg.step_start:seg.step_end + 1]]
        quality = sum(s.evidence_weight for s in unit_steps) / max(1, len(unit_steps))
        weight = round(SOURCE_WEIGHT.get(session.source, 0.5) * quality, 3)
        outcome = "corrected" if any(s.outcome == "corrected" for s in segs) else (
            "failure" if any(s.outcome == "failure" for s in segs) else "success")
        self.library.add_example(SkillExample(
            id=new_id("example"), skill_id=skill_id, skill_version=self.library.get(skill_id).current_version,
            role="demonstration", source_class=session.source.value, session_id=session.id,
            segment_ids=[s.id for s in segs], t_start=segs[0].t_start, t_end=segs[-1].t_end,
            frame_ids=[f for s in segs for f in s.representative_frame_ids][:12], outcome=outcome,
            evidence_weight=weight, instance_signature=instance,
            summary={"definition": candidate.model_dump(mode="json"), "task_text": session.task_text},
            created_at=now(),
        ))
        self.regeneralize(skill_id, note=f"added demonstration from session {session.id}")

    def _match(self, candidate: SkillDefinition) -> str | None:
        if self.library.exists(candidate.skill_id):
            skill = self.library.get(candidate.skill_id)
            if skill.status not in (SkillStatus.DISABLED, SkillStatus.MERGED):
                return candidate.skill_id
        best, best_score = None, 0.0
        signature = candidate.signature()
        for skill in self.library.list():
            d = skill.definition
            if d.object_class != candidate.object_class or d.object_role != candidate.object_role:
                continue
            ratio = difflib.SequenceMatcher(a=signature, b=d.signature(), autojunk=False).ratio()
            if ratio > best_score:
                best, best_score = skill.id, ratio
        return best if best_score >= SIMILARITY_THRESHOLD else None

    def regeneralize(self, skill_id: str, note: str) -> None:
        examples = self.library.examples(skill_id, role="demonstration")
        instances = [(ex.id, ex.source_class, ex.evidence_weight,
                      SkillDefinition.model_validate(ex.summary["definition"]))
                     for ex in examples if ex.summary.get("definition")]
        if not instances:
            return
        for _i, _s, _w, inst in instances:
            inst.skill_id = skill_id
        current = self.library.get(skill_id).definition
        merged = generalize(instances)
        merged.skill_id = skill_id
        edits = [{"after": loads(r["after"], {})} for r in self.library.db.query(
            "SELECT after FROM human_edits WHERE subject_kind = 'skill' AND subject_id = ? AND op = 'edit'", (skill_id,))]
        for key in human_edited_fields(edits):
            if hasattr(merged, key):
                setattr(merged, key, getattr(current, key))
        self.library.new_version(skill_id, merged, change_note=f"{note}; generalised from {len(instances)} examples",
                                 created_by="generalizer")
