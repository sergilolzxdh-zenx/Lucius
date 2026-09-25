"""Semantic action vocabulary shared by every demonstration source.

Blender operators (observed), default-keymap hotkeys (inferred from input), bridge actions
(agent executions) and video-inferred operations all map onto the same ``action_type`` names,
which is what lets demonstrations from different sources be compared and merged.

The vocabulary is extensible at runtime (``register_action_type``) and persisted in the
``taxonomy_terms`` table by :mod:`lucius.taxonomy`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    name: str
    family: str          # navigation, selection, creation, edit, transform, modifier, mode, history, file, ui, view, unknown
    mutates: bool        # changes scene/geometry state
    detail_level: int    # 0 setup/blockout, 1 secondary form, 2 detail; -1 not a modelling operation


ACTION_TYPES: dict[str, ActionSpec] = {}


def register_action_type(name: str, family: str, mutates: bool, detail_level: int = -1) -> ActionSpec:
    spec = ActionSpec(name, family, mutates, detail_level)
    ACTION_TYPES[name] = spec
    return spec


for _name, _family, _mut, _lvl in [
    ("viewport_orbit", "navigation", False, -1), ("viewport_pan", "navigation", False, -1),
    ("viewport_zoom", "navigation", False, -1), ("view_preset", "navigation", False, -1),
    ("toggle_ortho", "navigation", False, -1), ("frame_selected", "navigation", False, -1),
    ("select_click", "selection", False, -1), ("select_box", "selection", False, -1),
    ("select_all", "selection", False, -1), ("deselect_all", "selection", False, -1),
    ("select_elements", "selection", False, -1), ("select_mode", "selection", False, -1),
    ("select_objects", "selection", False, -1),
    ("add_primitive", "creation", True, 0), ("duplicate", "creation", True, 1),
    ("extrude", "edit", True, 0), ("inset", "edit", True, 1), ("loop_cut", "edit", True, 0),
    ("bevel", "edit", True, 2), ("subdivide", "edit", True, 1), ("knife", "edit", True, 1),
    ("merge", "edit", True, 1), ("delete", "edit", True, 0), ("fill", "edit", True, 1),
    ("bridge_loops", "edit", True, 1), ("sculpt_stroke", "edit", True, 1),
    ("translate", "transform", True, 0), ("rotate", "transform", True, 0), ("scale", "transform", True, 0),
    ("taper", "transform", True, 0), ("set_dimensions", "transform", True, 0),
    ("apply_transform", "transform", True, 0),
    ("add_modifier", "modifier", True, 1), ("apply_modifier", "modifier", True, 1),
    ("remove_modifier", "modifier", True, 1), ("set_symmetry", "modifier", True, 0),
    ("shade_smooth", "modifier", True, 2), ("shade_flat", "modifier", True, 2),
    ("mode_change", "mode", False, -1), ("tool_change", "mode", False, -1),
    ("workspace_change", "mode", False, -1),
    ("undo", "history", False, -1), ("redo", "history", False, -1),
    ("save", "file", False, -1), ("load_reference", "file", True, 0),
    ("add_menu", "ui", False, -1), ("search_menu", "ui", False, -1), ("context_menu", "ui", False, -1),
    ("ui_click", "ui", False, -1), ("text_entry", "ui", False, -1), ("rename", "ui", True, -1),
    ("snapshot", "file", False, -1), ("restore", "history", True, -1), ("reset_scene", "file", True, -1),
    ("geometry_edit", "edit", True, 1), ("unknown_action", "unknown", False, -1),
    ("separate", "edit", True, 1), ("set_material", "shading", True, 2), ("scatter", "shading", True, 2),
    ("add_light", "lighting", True, -1), ("set_world", "lighting", True, -1), ("add_camera", "camera", True, -1),
    ("render", "render", False, -1),
]:
    register_action_type(_name, _family, _mut, _lvl)

# Bridge action -> semantic action type (for actions whose names differ).
BRIDGE_ACTION_TYPES = {
    "set_mode": "mode_change", "transform_object": "translate", "select_box": "select_elements",
    "select_faces_by_normal": "select_elements", "translate_selection": "translate", "scale_selection": "scale",
    "rotate_selection": "rotate", "taper_selection": "taper", "loop_cut_axis": "loop_cut", "merge_by_distance": "merge",
    "delete_elements": "delete", "delete_objects": "delete", "bridge_edge_loops": "bridge_loops",
    "separate_selection": "separate", "duplicate_object": "duplicate", "shade": "shade_smooth", "add_scatter": "scatter",
    "render_image": "render", "set_render": "render", "rename_object": "rename", "save_file": "save",
}

# Mesh primitives the Blender bridge can add (``add_primitive`` kinds).
PRIMITIVE_KINDS = ("cube", "plane", "cylinder", "cone", "uv_sphere", "ico_sphere", "torus", "circle", "monkey")

# Modifier types whose addition is setup (symmetry) rather than detail.
MODIFIER_DETAIL = {"MIRROR": 0, "SOLIDIFY": 1, "ARRAY": 1, "SIMPLE_DEFORM": 0, "BEVEL": 2, "SUBSURF": 2,
                   "WEIGHTED_NORMAL": 2, "TRIANGULATE": 2, "DECIMATE": 2, "BOOLEAN": 1}


def spec(action_type: str) -> ActionSpec:
    return ACTION_TYPES.get(action_type) or ACTION_TYPES["unknown_action"]


def detail_level(action_type: str, params: dict[str, Any] | None = None) -> int:
    if action_type == "add_modifier" and params and params.get("type") in MODIFIER_DETAIL:
        return MODIFIER_DETAIL[params["type"]]
    return spec(action_type).detail_level


def is_navigation(action_type: str) -> bool:
    return spec(action_type).family == "navigation"


def mutates(action_type: str) -> bool:
    return spec(action_type).mutates


# -- Blender operators (observed via window_manager.operators) -----------------------------------

def _p(props: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: props[k] for k in keys if k in props}


def _macro_props(op: dict[str, Any], idname: str) -> dict[str, Any]:
    for macro in op.get("macros") or []:
        if macro.get("idname") == idname:
            return macro.get("properties") or {}
    return {}


def _primitive(kind: str) -> Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]:
    return lambda op: ("add_primitive", {"kind": kind, **_p(op.get("properties", {}), "size", "radius", "depth",
                                                           "vertices", "location")})


OperatorMapper = Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]

OPERATOR_MAP: dict[str, OperatorMapper] = {
    "MESH_OT_primitive_cube_add": _primitive("cube"),
    "MESH_OT_primitive_plane_add": _primitive("plane"),
    "MESH_OT_primitive_cylinder_add": _primitive("cylinder"),
    "MESH_OT_primitive_cone_add": _primitive("cone"),
    "MESH_OT_primitive_uv_sphere_add": _primitive("uv_sphere"),
    "MESH_OT_primitive_ico_sphere_add": _primitive("ico_sphere"),
    "MESH_OT_primitive_torus_add": _primitive("torus"),
    "MESH_OT_primitive_circle_add": _primitive("circle"),
    "MESH_OT_primitive_monkey_add": _primitive("monkey"),
    "MESH_OT_extrude_region_move": lambda op: ("extrude", {"offset": _macro_props(op, "TRANSFORM_OT_translate").get("value")}),
    "MESH_OT_extrude_context_move": lambda op: ("extrude", {"offset": _macro_props(op, "TRANSFORM_OT_translate").get("value")}),
    "MESH_OT_extrude_region_shrink_fatten": lambda op: ("extrude", {"shrink_fatten": _macro_props(op, "TRANSFORM_OT_shrink_fatten").get("value")}),
    "MESH_OT_inset": lambda op: ("inset", _p(op.get("properties", {}), "thickness", "depth")),
    "MESH_OT_bevel": lambda op: ("bevel", _p(op.get("properties", {}), "offset", "segments", "affect")),
    "MESH_OT_loopcut_slide": lambda op: ("loop_cut", {"cuts": _macro_props(op, "MESH_OT_loopcut").get("number_cuts")}),
    "MESH_OT_subdivide": lambda op: ("subdivide", _p(op.get("properties", {}), "number_cuts")),
    "MESH_OT_knife_tool": lambda op: ("knife", {}),
    "MESH_OT_merge": lambda op: ("merge", _p(op.get("properties", {}), "type")),
    "MESH_OT_remove_doubles": lambda op: ("merge", _p(op.get("properties", {}), "threshold")),
    "MESH_OT_delete": lambda op: ("delete", _p(op.get("properties", {}), "type")),
    "MESH_OT_fill": lambda op: ("fill", {}),
    "MESH_OT_bridge_edge_loops": lambda op: ("bridge_loops", {}),
    "MESH_OT_select_all": lambda op: ("select_all", _p(op.get("properties", {}), "action")),
    "MESH_OT_select_mode": lambda op: ("select_mode", _p(op.get("properties", {}), "type")),
    "OBJECT_OT_select_all": lambda op: ("select_all", _p(op.get("properties", {}), "action")),
    "VIEW3D_OT_select": lambda op: ("select_click", _p(op.get("properties", {}), "extend", "deselect", "toggle")),
    "VIEW3D_OT_select_box": lambda op: ("select_box", _p(op.get("properties", {}), "mode")),
    "TRANSFORM_OT_translate": lambda op: ("translate", _transform_params(op)),
    "TRANSFORM_OT_rotate": lambda op: ("rotate", _transform_params(op)),
    "TRANSFORM_OT_resize": lambda op: ("scale", _transform_params(op)),
    "TRANSFORM_OT_shrink_fatten": lambda op: ("translate", {"shrink_fatten": op.get("properties", {}).get("value")}),
    "OBJECT_OT_duplicate_move": lambda op: ("duplicate", {}),
    "MESH_OT_duplicate_move": lambda op: ("duplicate", {}),
    "OBJECT_OT_delete": lambda op: ("delete", {"target": "object"}),
    "OBJECT_OT_modifier_add": lambda op: ("add_modifier", _p(op.get("properties", {}), "type")),
    "OBJECT_OT_modifier_apply": lambda op: ("apply_modifier", _p(op.get("properties", {}), "modifier")),
    "OBJECT_OT_modifier_remove": lambda op: ("remove_modifier", _p(op.get("properties", {}), "modifier")),
    "OBJECT_OT_transform_apply": lambda op: ("apply_transform", _p(op.get("properties", {}), "location", "rotation", "scale")),
    "OBJECT_OT_shade_smooth": lambda op: ("shade_smooth", {}),
    "OBJECT_OT_shade_flat": lambda op: ("shade_flat", {}),
    "OBJECT_OT_editmode_toggle": lambda op: ("mode_change", {"toggle": "edit"}),
    "OBJECT_OT_mode_set": lambda op: ("mode_change", _p(op.get("properties", {}), "mode")),
    "SCULPT_OT_brush_stroke": lambda op: ("sculpt_stroke", _p(op.get("properties", {}), "mode")),
    "WM_OT_save_mainfile": lambda op: ("save", {}),
    "WM_OT_save_as_mainfile": lambda op: ("save", {}),
    "VIEW3D_OT_view_axis": lambda op: ("view_preset", {"view": op.get("properties", {}).get("type", "").lower()}),
    "VIEW3D_OT_view_selected": lambda op: ("frame_selected", {}),
    "OBJECT_OT_empty_image_add": lambda op: ("load_reference", {}),
}


def _transform_params(op: dict[str, Any]) -> dict[str, Any]:
    props = op.get("properties", {})
    params: dict[str, Any] = {}
    if "value" in props:
        params["value"] = props["value"]
    axes = props.get("constraint_axis")
    if isinstance(axes, list) and any(axes):
        params["axis"] = "".join(a for a, on in zip("xyz", axes) if on)
    if props.get("use_proportional_edit"):
        params["proportional"] = True
    return params


def map_operator(op: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    mapper = OPERATOR_MAP.get(op.get("idname", ""))
    if mapper is None:
        return None
    action_type, params = mapper(op)
    return action_type, {k: v for k, v in params.items() if v is not None}


# -- Blender default keymap (inference from direct input) ----------------------------------------

# (key, modifiers, mode_group) -> action_type ; mode_group: "any", "object", "edit", "sculpt"
HOTKEYS: dict[tuple[str, frozenset[str], str], str] = {}


def _hk(key: str, action: str, mods: tuple[str, ...] = (), mode: str = "any") -> None:
    HOTKEYS[(key, frozenset(mods), mode)] = action


for _key, _action, _mods, _mode in [
    ("G", "translate", (), "any"), ("R", "rotate", (), "any"), ("S", "scale", (), "any"),
    ("E", "extrude", (), "edit"), ("I", "inset", (), "edit"), ("R", "loop_cut", ("CTRL",), "edit"),
    ("B", "bevel", ("CTRL",), "edit"), ("K", "knife", (), "edit"), ("M", "merge", (), "edit"),
    ("F", "fill", (), "edit"), ("X", "delete", (), "any"), ("DEL", "delete", (), "any"),
    ("D", "duplicate", ("SHIFT",), "any"), ("A", "add_menu", ("SHIFT",), "any"),
    ("A", "select_all", (), "any"), ("A", "deselect_all", ("ALT",), "any"),
    ("ONE", "select_mode", (), "edit"), ("TWO", "select_mode", (), "edit"), ("THREE", "select_mode", (), "edit"),
    ("TAB", "mode_change", (), "any"), ("Z", "undo", ("CTRL",), "any"), ("Z", "redo", ("CTRL", "SHIFT"), "any"),
    ("S", "save", ("CTRL",), "any"), ("F3", "search_menu", (), "any"),
    ("NUMPAD_1", "view_preset", (), "any"), ("NUMPAD_3", "view_preset", (), "any"),
    ("NUMPAD_7", "view_preset", (), "any"), ("NUMPAD_1", "view_preset", ("CTRL",), "any"),
    ("NUMPAD_3", "view_preset", ("CTRL",), "any"), ("NUMPAD_7", "view_preset", ("CTRL",), "any"),
    ("NUMPAD_9", "view_preset", (), "any"), ("NUMPAD_5", "toggle_ortho", (), "any"),
    ("NUMPAD_2", "viewport_orbit", (), "any"), ("NUMPAD_4", "viewport_orbit", (), "any"),
    ("NUMPAD_6", "viewport_orbit", (), "any"), ("NUMPAD_8", "viewport_orbit", (), "any"),
    ("NUMPAD_PERIOD", "frame_selected", (), "any"), ("NUMPAD_PLUS", "viewport_zoom", (), "any"),
    ("NUMPAD_MINUS", "viewport_zoom", (), "any"), ("A", "apply_transform", ("CTRL",), "object"),
    ("F2", "rename", (), "any"),
]:
    _hk(_key, _action, _mods, _mode)

VIEW_KEYS = {("NUMPAD_1", False): "front", ("NUMPAD_1", True): "back", ("NUMPAD_3", False): "right",
             ("NUMPAD_3", True): "left", ("NUMPAD_7", False): "top", ("NUMPAD_7", True): "bottom",
             ("NUMPAD_9", False): "opposite"}

# Operations that enter a modal state in which axis keys / typed numbers / clicks refine them.
MODAL_ACTIONS = {"translate", "rotate", "scale", "extrude", "inset", "bevel", "loop_cut", "duplicate"}


def mode_group(mode: str | None) -> str:
    if not mode:
        return "any"
    if mode.startswith("EDIT"):
        return "edit"
    if mode.startswith("SCULPT"):
        return "sculpt"
    return "object"


def map_hotkey(key: str, modifiers: frozenset[str], mode: str | None) -> str | None:
    group = mode_group(mode)
    return HOTKEYS.get((key, modifiers, group)) or HOTKEYS.get((key, modifiers, "any"))


# -- bridge actions (agent executions) ---------------------------------------------------------

BRIDGE_ACTION_MAP = {
    "add_primitive": "add_primitive", "select_objects": "select_objects", "delete_objects": "delete",
    "set_mode": "mode_change", "transform_object": "translate", "set_dimensions": "set_dimensions",
    "apply_transform": "apply_transform", "rename_object": "rename", "select_elements": "select_elements",
    "select_faces_by_normal": "select_elements", "select_all": "select_all", "extrude": "extrude",
    "translate_selection": "translate", "scale_selection": "scale", "taper_selection": "taper",
    "inset": "inset", "bevel": "bevel", "loop_cut_axis": "loop_cut", "merge_by_distance": "merge",
    "add_modifier": "add_modifier", "remove_modifier": "remove_modifier", "apply_modifier": "apply_modifier",
    "set_symmetry": "set_symmetry", "shade": "shade_smooth", "reset_scene": "reset_scene",
    "set_view": "view_preset", "orbit_view": "viewport_orbit", "frame_selected": "frame_selected",
    "undo": "undo", "redo": "redo", "save_file": "save", "load_reference_image": "load_reference",
    "snapshot": "snapshot", "restore": "restore",
}
