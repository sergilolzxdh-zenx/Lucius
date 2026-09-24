"""Structural and measured-visual checkpoint implementations (evaluation levels 2 and 3).

Each check reads a structure snapshot from the bridge (``inspect_structure``) and returns a
:class:`CheckOutcome`. ``passed=None`` means *not evaluable* (e.g. missing data) -- it is never
coerced into a pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lucius.evaluation.silhouette import iou, rasterize, width_profile

VIEW_AXES = {"front": ("x", "z"), "side": ("y", "z"), "right": ("y", "z"), "left": ("y", "z"), "top": ("x", "y"),
             "back": ("x", "z")}
STRUCTURE_VIEW = {"front": "front", "back": "front", "right": "side", "left": "side", "side": "side", "top": "top"}


@dataclass
class CheckOutcome:
    passed: bool | None
    score: float | None
    reason_code: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceSilhouette:
    """A measured target silhouette for one view (from a reference image)."""

    view: str
    mask: np.ndarray
    media_asset_id: str | None = None
    min_iou: float = 0.75
    target: str | None = None      # object role/name it depicts; None = the whole scene


def _object(structure: dict[str, Any], name: str | None) -> dict[str, Any] | None:
    for obj in structure.get("objects", []):
        if obj.get("name") == name:
            return obj
    return None


def _size(obj: dict[str, Any]) -> list[float]:
    evaluated = obj.get("evaluated") or {}
    return [float(v) for v in (evaluated.get("size") or obj.get("dimensions") or [0, 0, 0])]


def _axis(a: str) -> int:
    return "xyz".index(a)


def check_object_exists(check: dict, structure: dict, _refs: list[ReferenceSilhouette]) -> CheckOutcome:
    obj = _object(structure, check.get("object"))
    return CheckOutcome(obj is not None, 1.0 if obj else 0.0, "object_found" if obj else "object_missing",
                        {"object": check.get("object")})


def check_object_count(check: dict, structure: dict, _refs: list) -> CheckOutcome:
    count = structure.get("mesh_object_count", 0)
    ok = check.get("min", 0) <= count <= check.get("max", 10 ** 6)
    return CheckOutcome(ok, None, "object_count", {"count": count})


def check_dimensions_match(check: dict, structure: dict, _refs: list) -> CheckOutcome:
    obj = _object(structure, check.get("object"))
    if obj is None:
        return CheckOutcome(False, 0.0, "object_missing", {"object": check.get("object")})
    size = _size(obj)
    targets: dict[str, float] = {}
    if isinstance(check.get("axes"), dict):
        targets = {a: float(v) for a, v in check["axes"].items() if isinstance(v, (int, float))}
    elif isinstance(check.get("vector"), list):
        targets = {a: float(v) for a, v in zip("xyz", check["vector"])}
    if not targets:
        return CheckOutcome(None, None, "unresolved_target", {"check": check})
    tol = float(check.get("tolerance", 0.05))
    errors = {a: abs(size[_axis(a)] - t) / max(abs(t), 1e-6) for a, t in targets.items()}
    worst = max(errors.values())
    return CheckOutcome(worst <= tol, round(max(0.0, 1.0 - worst), 4),
                        "dimensions_within_tolerance" if worst <= tol else "dimension_mismatch",
                        {"actual": dict(zip("xyz", [round(s, 4) for s in size])), "target": targets,
                         "relative_error": {a: round(e, 4) for a, e in errors.items()}, "tolerance": tol})


def check_dimension_ratio(check: dict, structure: dict, _refs: list) -> CheckOutcome:
    obj = _object(structure, check.get("object"))
    if obj is None:
        return CheckOutcome(False, 0.0, "object_missing", {})
    size = _size(obj)
    denominator = size[_axis(check["denominator"])]
    if denominator <= 0:
        return CheckOutcome(None, None, "degenerate_dimension", {"size": size})
    ratio = size[_axis(check["numerator"])] / denominator
    ok = check.get("min", 0.0) <= ratio <= check.get("max", float("inf"))
    return CheckOutcome(ok, None, "ratio_in_range" if ok else "ratio_out_of_range",
                        {"ratio": round(ratio, 4), "min": check.get("min"), "max": check.get("max")})


def check_symmetry(check: dict, structure: dict, _refs: list) -> CheckOutcome:
    obj = _object(structure, check.get("object"))
    if obj is None:
        return CheckOutcome(False, 0.0, "object_missing", {})
    errors = (obj.get("evaluated") or {}).get("symmetry_error")
    if not errors:
        return CheckOutcome(None, None, "symmetry_unmeasured", {})
    error = float(errors.get(check.get("axis", "x"), 1.0))
    limit = float(check.get("max_error", 0.02))
    return CheckOutcome(error <= limit, round(max(0.0, 1.0 - error / max(limit, 1e-9) / 2), 4),
                        "symmetric" if error <= limit else "asymmetric", {"error": error, "limit": limit})


def check_modifier(check: dict, structure: dict, _refs: list, present: bool = True) -> CheckOutcome:
    obj = _object(structure, check.get("object"))
    if obj is None:
        return CheckOutcome(False, 0.0, "object_missing", {})
    types = [m.get("type") for m in obj.get("modifiers", [])]
    found = check.get("modifier") in types
    ok = found if present else not found
    return CheckOutcome(ok, None, "modifier_present" if found else "modifier_absent", {"modifiers": types})


def check_mode(check: dict, structure: dict, _refs: list) -> CheckOutcome:
    mode = structure.get("mode")
    return CheckOutcome(mode == check.get("mode"), None, "mode", {"mode": mode})


def check_vertex_count(check: dict, structure: dict, _refs: list) -> CheckOutcome:
    obj = _object(structure, check.get("object"))
    if obj is None:
        return CheckOutcome(False, 0.0, "object_missing", {})
    verts = (obj.get("evaluated") or {}).get("verts") or (obj.get("mesh") or {}).get("verts", 0)
    ok = check.get("min", 0) <= verts <= check.get("max", 10 ** 9)
    return CheckOutcome(ok, None, "vertex_count", {"verts": verts})


def check_manifold(check: dict, structure: dict, _refs: list) -> CheckOutcome:
    obj = _object(structure, check.get("object"))
    if obj is None:
        return CheckOutcome(False, 0.0, "object_missing", {})
    bad = (obj.get("evaluated") or {}).get("non_manifold_edges")
    if bad is None:
        return CheckOutcome(None, None, "manifold_unmeasured", {})
    return CheckOutcome(bad <= check.get("max_non_manifold", 0), None, "manifold", {"non_manifold_edges": bad})


def _silhouette_for(obj: dict[str, Any], view: str) -> np.ndarray | None:
    triangles = (obj.get("silhouettes") or {}).get(STRUCTURE_VIEW.get(view, view))
    if not triangles:
        return None
    return rasterize(triangles)[0]


def check_taper(check: dict, structure: dict, _refs: list) -> CheckOutcome:
    obj = _object(structure, check.get("object"))
    if obj is None:
        return CheckOutcome(False, 0.0, "object_missing", {})
    along, width_axis = check.get("along", "z"), check.get("width_axis", "x")
    view = next((v for v, axes in VIEW_AXES.items() if axes == (width_axis, along)), None)
    if view is None:
        return CheckOutcome(None, None, "no_view_for_axes", {"along": along, "width_axis": width_axis})
    mask = _silhouette_for(obj, view)
    if mask is None:
        return CheckOutcome(None, None, "silhouette_unavailable", {"view": view})
    profile = width_profile(mask, bins=20)
    lo, hi = check.get("region", [0.8, 1.0])
    region = [w for i, w in enumerate(profile) if lo <= (i + 0.5) / 20 <= hi]
    base = max(w for i, w in enumerate(profile) if (i + 0.5) / 20 < lo) if lo > 0.05 else 1.0
    if not region or base <= 0:
        return CheckOutcome(None, None, "profile_empty", {"profile": profile})
    ratio = float(np.mean(region)) / base
    limit = float(check.get("max_ratio", 0.8))
    return CheckOutcome(ratio <= limit, round(max(0.0, 1.0 - ratio), 4), "tapered" if ratio <= limit else "not_tapered",
                        {"width_ratio": round(ratio, 4), "limit": limit, "view": view,
                         "profile": [round(w, 3) for w in profile]})


def _scene_silhouette(structure: dict[str, Any], view: str) -> np.ndarray | None:
    triangles = [t for o in structure.get("objects", []) for t in (o.get("silhouettes") or {}).get(
        STRUCTURE_VIEW.get(view, view), [])]
    return rasterize(triangles)[0] if triangles else None


def _matches(ref: ReferenceSilhouette, name: str | None) -> bool:
    if ref.target is None or name is None:
        return False
    return ref.target.lower() == name.lower() or name.lower().startswith(ref.target.lower())


def check_silhouette(check: dict, structure: dict, refs: list[ReferenceSilhouette]) -> CheckOutcome:
    """Measured comparison with reference silhouettes.

    Object-scoped checks use references that depict that object; ``scope: scene`` checks compare the
    whole scene with untargeted references. ``stage: intermediate`` (a check during blockout, before the
    final shape exists) uses a relaxed threshold. Without a matching reference the check is not
    evaluable here -- the evaluator may ask a vision model, recorded as subjective.
    """
    scene = check.get("scope") == "scene"
    obj = None if scene else _object(structure, check.get("object"))
    if not scene and obj is None:
        return CheckOutcome(False, 0.0, "object_missing", {})
    views = check.get("views") or []
    side = {"side", "right", "left"}
    comparable = [r for r in refs if (r.view in views or (r.view in side and set(views) & side))
                  and ((scene and r.target is None) or (not scene and _matches(r, check.get("object"))))]
    if not comparable:
        return CheckOutcome(None, None, "no_reference_silhouette", {"views": views, "scope": "scene" if scene else
                                                                    check.get("object")})
    relax = 0.8 if check.get("stage") == "intermediate" else 1.0
    scores = {}
    for ref in comparable:
        mask = _scene_silhouette(structure, ref.view) if scene else _silhouette_for(obj, ref.view)
        if mask is None:
            return CheckOutcome(None, None, "silhouette_unavailable", {"view": ref.view})
        scores[ref.view] = (iou(mask, ref.mask), round(ref.min_iou * relax, 4))
    worst = min(s for s, _m in scores.values())
    ok = all(s >= m for s, m in scores.values())
    return CheckOutcome(ok, round(worst, 4), "silhouette_matches_reference" if ok else "silhouette_mismatch",
                        {"iou": {v: round(s, 4) for v, (s, _m) in scores.items()},
                         "thresholds": {v: m for v, (_s, m) in scores.items()}})


CHECKS: dict[str, Callable[[dict, dict, list[ReferenceSilhouette]], CheckOutcome]] = {
    "object_exists": check_object_exists,
    "object_count": check_object_count,
    "dimensions_match": check_dimensions_match,
    "dimension_ratio": check_dimension_ratio,
    "symmetry": check_symmetry,
    "modifier_present": check_modifier,
    "modifier_absent": lambda c, s, r: check_modifier(c, s, r, present=False),
    "mode_is": check_mode,
    "vertex_count": check_vertex_count,
    "manifold": check_manifold,
    "taper": check_taper,
    "silhouette": check_silhouette,
}
