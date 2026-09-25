"""Compile skill action templates into concrete, allowlisted actions.

Templates are resolved against parameters and compiled for the *Blender API* layer (bridge
actions) plus a few internal macros that need live state (``scale_to_size`` reads the current
dimension before scaling). Edit-mode requirements insert mode switches automatically.
Anything that cannot be compiled faithfully raises :class:`Uncompilable` instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucius.evaluation.evaluator import resolve
from lucius.ids import new_id
from lucius.lessons.catalogue import RECIPE_ACTIONS, normalize_args
from lucius.planner.model import PlanAction
from lucius.skills.schema import ActionTemplate, Selection
from lucius.trajectory import vocabulary as vocab

AXIS_VECTOR = {"x": 0, "y": 1, "z": 2}


class Uncompilable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class CompileContext:
    mode: str = "OBJECT"
    object_name: str | None = None
    gui_available: bool = False
    notes: list[str] = field(default_factory=list)


def _action(name: str, action_type: str, args: dict[str, Any], description: str, *, layer: str = "blender_api",
            optional: bool = False, reasons: list[str] | None = None, source: str | None = None) -> PlanAction:
    return PlanAction(id=new_id("step"), layer=layer, name=name, args=args, action_type=action_type,
                      description=description, optional=optional, reason_codes=reasons or [], source=source)


def _vec(axis: str, value: float, neutral: float) -> list[float]:
    vec = [neutral, neutral, neutral]
    vec[AXIS_VECTOR[axis]] = float(value)
    return vec


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Uncompilable(f"parameter {name} unresolved ({value!r})")
    return float(value)


def _selection(sel: Selection | None, obj: str, source: str | None) -> list[PlanAction]:
    if sel is None or sel.kind == "all":
        return [_action("select_all", "select_all", {"object": obj, "action": "SELECT"}, "select all", source=source)]
    if sel.kind == "region":
        return [_action("select_elements", "select_elements",
                        {"object": obj, "axis": sel.axis, "min": sel.min, "max": sel.max, "space": "normalized"},
                        f"select the {sel.axis} range {sel.min:.2f}-{sel.max:.2f} of {obj}", source=source)]
    if sel.kind == "normal" and sel.direction:
        return [_action("select_faces_by_normal", "select_elements", {"object": obj, "direction": sel.direction},
                        "select faces by normal", source=source)]
    return [_action("select_all", "select_all", {"object": obj, "action": "SELECT"},
                    "select all (demonstrated selection unknown)", reasons=["selection_unknown_assumed_all"],
                    source=source)]


def _mode_args(mode: str, obj: str | None) -> dict[str, Any]:
    # Leaving edit mode acts on whatever is active (the target object may not exist yet);
    # entering edit mode targets the skill's object.
    return {"mode": mode} if mode == "OBJECT" or obj is None else {"object": obj, "mode": mode}


def _ensure_mode(ctx: CompileContext, mode: str | None, obj: str | None, source: str | None) -> list[PlanAction]:
    if mode is None or ctx.mode == mode or (mode == "EDIT" and obj is None):
        return []
    ctx.mode = mode
    return [_action("set_mode", "mode_change", _mode_args(mode, obj), f"switch to {mode.lower()} mode",
                    reasons=["auto_mode_switch"], source=source)]


def compile_template(t: ActionTemplate, params: dict[str, Any], ctx: CompileContext,
                     source: str | None = None) -> list[PlanAction]:
    args = resolve(t.args, params)
    obj = resolve(t.object_ref, params) if t.object_ref else None
    if isinstance(obj, str) and obj.startswith("{"):
        raise Uncompilable(f"object reference {obj} unresolved")
    at = t.action_type
    out: list[PlanAction] = []

    def mode(m: str | None) -> None:
        out.extend(_ensure_mode(ctx, m, obj, source))

    if at == "reset_scene":
        ctx.mode = "OBJECT"
        return [_action("reset_scene", at, {"keep_camera_light": bool(args.get("keep_camera_light", True))},
                        "clear the scene", source=source)]
    if at == "add_primitive":
        kind = args.get("kind")
        if kind not in vocab.PRIMITIVE_KINDS:
            raise Uncompilable(f"primitive kind {kind!r} is unknown or not a mesh primitive")
        mode("OBJECT")
        prim = {"kind": kind, "name": args.get("name") or obj}
        if isinstance(args.get("size"), (int, float)):
            prim["size"] = float(args["size"])
        ctx.object_name = prim["name"]
        return out + [_action("add_primitive", at, prim, f"add {prim['kind']} '{prim['name']}'", source=source)]
    if at == "mode_change":
        target = str(args.get("mode", "OBJECT")).upper()
        if ctx.mode == target:
            return []
        ctx.mode = target
        return [_action("set_mode", at, _mode_args(target, obj), f"{target.lower()} mode", source=source)]
    if at == "add_modifier":
        return [_action("add_modifier", at, {"object": obj, "type": args.get("type")}, f"add {args.get('type')} modifier",
                        source=source)]
    if at in ("scale", "translate", "rotate"):
        axis = args.get("axis")
        in_edit = (t.requires_mode or ctx.mode) == "EDIT"
        if axis not in AXIS_VECTOR:
            raise Uncompilable(f"{at} without a single axis")
        if at == "scale" and "target_size" in args:
            size = _num(args["target_size"], "target_size")
            mode("EDIT" if in_edit else "OBJECT")
            sel = _selection(t.selection, obj, source) if in_edit else []
            return out + sel + [_action("scale_to_size", at, {"object": obj, "axis": axis, "size": size,
                                                               "space": "edit" if in_edit else "object"},
                                        f"resize {obj} to {size:g} along {axis}", layer="internal", source=source)]
        if at == "scale":
            factor = _num(args.get("factor"), "factor")
            if in_edit:
                mode("EDIT")
                return out + _selection(t.selection, obj, source) + [
                    _action("scale_selection", at, {"object": obj, "factor": _vec(axis, factor, 1.0),
                                                    "pivot": "median"}, f"scale {axis} x{factor:g}", source=source),
                    *_selection(Selection(kind="all"), obj, source)]
            mode("OBJECT")
            return out + [_action("transform_object", at, {"object": obj, "scale": _vec(axis, factor, 1.0),
                                                           "relative": True}, f"scale {axis} x{factor:g}", source=source)]
        amount = _num(args.get("amount"), "amount")
        if at == "translate" and in_edit:
            mode("EDIT")
            return out + _selection(t.selection, obj, source) + [
                _action("translate_selection", at, {"object": obj, "offset": _vec(axis, amount, 0.0)},
                        f"move selection {amount:g} along {axis}", source=source)]
        mode("OBJECT")
        key = "location" if at == "translate" else "rotation"
        return out + [_action("transform_object", at, {"object": obj, key: _vec(axis, amount, 0.0), "relative": True},
                              f"{at} {axis} {amount:g}", source=source)]
    if at == "extrude":
        axis = args.get("axis")
        if axis not in AXIS_VECTOR:
            raise Uncompilable("extrude without a single axis")
        distance = _num(args.get("distance"), "distance")
        mode("EDIT")
        sel = t.selection if t.selection is not None and t.selection.kind != "unknown" else Selection(
            kind="normal", direction=_vec(axis, 1.0 if distance >= 0 else -1.0, 0.0))
        return out + _selection(sel, obj, source) + [
            _action("extrude", at, {"object": obj, "offset": _vec(axis, distance, 0.0)},
                    f"extrude {distance:g} along {axis}", source=source)]
    if at == "loop_cut":
        cuts = int(_num(args.get("cuts", 1), "cuts"))
        axis = args.get("axis", "z")
        mode("EDIT")
        positions = [round((i + 1) / (cuts + 1), 4) for i in range(cuts)]
        return out + [_action("loop_cut_axis", at, {"object": obj, "axis": axis, "positions": positions},
                              f"{cuts} loop cut(s) across {axis}", source=source)]
    if at in ("bevel", "inset", "taper"):
        mode("EDIT")
        sel = _selection(t.selection, obj, source)
        if at == "bevel":
            call = _action("bevel", at, {"object": obj, "offset": _num(args.get("offset"), "offset"),
                                         "segments": int(_num(args.get("segments", 1), "segments"))},
                           "bevel edges", source=source)
        elif at == "inset":
            call = _action("inset", at, {"object": obj, "thickness": _num(args.get("thickness"), "thickness")},
                           "inset faces", source=source)
        else:
            call = _action("taper_selection", at, {"object": obj, "along": args.get("along", "z"),
                                                   "affect": args.get("affect", "x"),
                                                   "amount": _num(args.get("amount"), "amount")}, "taper", source=source)
        return out + sel + [call]
    if at == "set_dimensions":
        dims = args.get("dimensions")
        if not isinstance(dims, list) or len(dims) != 3:
            raise Uncompilable("set_dimensions needs three values")
        mode("OBJECT")
        return out + [_action("set_dimensions", at, {"object": obj, "dimensions": [_num(d, "dimension") for d in dims]},
                              "set dimensions", source=source),
                      _action("apply_transform", "apply_transform", {"object": obj}, "apply scale", source=source)]
    if at == "apply_transform":
        mode("OBJECT")
        return out + [_action("apply_transform", at, {"object": obj}, "apply transform", source=source)]
    if at == "set_symmetry":
        return [_action("set_symmetry", at, {"object": obj, "axes": args.get("axes", [True, False, False])},
                        "enable mirror editing", source=source)]
    if at in ("shade_smooth", "shade_flat"):
        return [_action("shade", at, {"object": obj, "smooth": at == "shade_smooth"}, at.replace("_", " "),
                        source=source)]
    if at == "merge":
        mode("EDIT")
        return out + [_action("merge_by_distance", at, {"object": obj}, "merge by distance", source=source)]
    if at == "select_objects":
        return [_action("select_objects", at, {"names": [n for n in args.get("names", []) if isinstance(n, str)]},
                        "select objects", source=source)]
    if at == "rename":
        return [_action("rename_object", at, {"object": obj, "new_name": args.get("new_name")}, "rename",
                        source=source)]
    if at == "save":
        if not args.get("path"):
            raise Uncompilable("save requires a path")
        return [_action("save_file", at, {"path": args["path"]}, "save file", source=source)]
    if at == "restore":
        return [_action("restore_snapshot", at, {"tag": args.get("to", "before_trigger")}, "restore snapshot",
                        layer="internal", source=source)]
    if at in ("view_preset", "viewport_orbit"):
        if not ctx.gui_available:
            return [_action("observe_views", at, {"views": args.get("views") or [args.get("view")]},
                            "inspect views (measured silhouettes in headless mode)", layer="observation",
                            source=source)]
        if at == "view_preset":
            return [_action("set_view", at, {"view": str(v).upper(), "ortho": True}, f"{v} view", source=source)
                    for v in (args.get("views") or [args.get("view", "FRONT")])]
        return [_action("orbit_view", at, {"direction": args.get("direction", "ORBITLEFT"),
                                           "angle": float(args.get("angle", 0.2618))}, "orbit", source=source)]
    if at in RECIPE_ACTIONS:
        # A learned recipe step names its Blender action directly, with concrete values.
        recipe_args, _notes = normalize_args(at, args)
        ctx.mode = "UNKNOWN"   # recipe actions switch modes themselves; the next mode requirement re-asserts it
        return [_action(at, vocab.BRIDGE_ACTION_TYPES.get(at, at), recipe_args, t.description or at, source=source)]
    raise Uncompilable(f"no compiler for action type {at}")
