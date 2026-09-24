"""Blender state capture.

Every field is either captured from Blender or listed in ``unavailable`` with the reason.
Nothing is guessed: in background mode there is no interactive viewport, so viewport fields
are reported unavailable rather than invented.
"""

import math
import os
import time

import bpy

NAMED_VIEWS = {
    # Region-3D view rotation quaternions (w, x, y, z) for the numpad views.
    "top": (1.0, 0.0, 0.0, 0.0),
    "bottom": (0.0, 1.0, 0.0, 0.0),
    "front": (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
    "back": (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
    "right": (0.5, 0.5, 0.5, 0.5),
    "left": (0.5, 0.5, -0.5, -0.5),
}

MAX_LISTED_OBJECTS = 200


def _named_view(rotation):
    q = tuple(float(v) for v in rotation)
    for name, ref in NAMED_VIEWS.items():
        dot = abs(sum(a * b for a, b in zip(q, ref)))
        if dot > 0.9998:  # within ~2 degrees
            return name
    return "user"


class _Collector:
    def __init__(self):
        self.values = {}
        self.unavailable = {}

    def field(self, name, getter):
        try:
            value = getter()
        except Exception as exc:  # any bpy access can fail depending on context
            self.unavailable[name] = f"{type(exc).__name__}: {exc}"
            return None
        if value is None:
            self.unavailable.setdefault(name, "not set")
        else:
            self.values[name] = value
        return value


def find_view3d(context=None):
    context = context or bpy.context
    wm = context.window_manager
    for window in getattr(wm, "windows", []):
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                return window, area
    return None, None


def _viewport(background):
    if background:
        raise RuntimeError("no interactive viewport in background mode")
    _window, area = find_view3d()
    if area is None:
        return None
    r3d = area.spaces.active.region_3d
    return {
        "perspective": r3d.view_perspective,
        "named_view": _named_view(r3d.view_rotation),
        "rotation": [round(float(v), 5) for v in r3d.view_rotation],
        "distance": round(float(r3d.view_distance), 5),
        "location": [round(float(v), 5) for v in r3d.view_location],
    }


def _workspace(context):
    workspace = getattr(context, "workspace", None)
    if workspace is None:
        windows = context.window_manager.windows
        workspace = windows[0].workspace if len(windows) else None
    return workspace


def _active_tool(context):
    workspace = _workspace(context)
    if workspace is None:
        return None
    tool = workspace.tools.from_space_view3d_mode(context.mode, create=False)
    return None if tool is None else tool.idname


def _edit_selection(obj):
    import bmesh

    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return None
    bm = bmesh.from_edit_mesh(obj.data)
    selected = [v.co for v in bm.verts if v.select]
    info = {
        "verts": len(selected),
        "total_verts": len(bm.verts),
        "edges": sum(1 for e in bm.edges if e.select),
        "faces": sum(1 for f in bm.faces if f.select),
        "select_mode": sorted(bm.select_mode),
    }
    if selected and len(bm.verts):
        # Selection extent in the object's normalised bounding box: a resolution-independent
        # description ("the top 20% along z") that a skill can re-apply to another object.
        lo = [min(v.co[i] for v in bm.verts) for i in range(3)]
        hi = [max(v.co[i] for v in bm.verts) for i in range(3)]
        info["normalized_bbox"] = {
            axis: [round((min(c[i] for c in selected) - lo[i]) / ((hi[i] - lo[i]) or 1.0), 3),
                   round((max(c[i] for c in selected) - lo[i]) / ((hi[i] - lo[i]) or 1.0), 3)]
            for i, axis in enumerate("xyz")
        }
    return info


def object_summary(obj):
    data = {
        "name": obj.name,
        "type": obj.type,
        "mode": obj.mode,
        "location": [round(float(v), 5) for v in obj.location],
        "rotation_euler": [round(float(v), 5) for v in obj.rotation_euler],
        "scale": [round(float(v), 5) for v in obj.scale],
        "dimensions": [round(float(v), 5) for v in obj.dimensions],
        "modifiers": [{"name": m.name, "type": m.type} for m in obj.modifiers],
    }
    if obj.type == "MESH" and obj.data is not None:
        mesh = obj.data
        data["mesh"] = {
            "verts": len(mesh.vertices), "edges": len(mesh.edges), "faces": len(mesh.polygons),
            "use_mirror_x": bool(mesh.use_mirror_x), "use_mirror_y": bool(mesh.use_mirror_y),
            "use_mirror_z": bool(mesh.use_mirror_z),
        }
    return data


def capture_state(include_objects=True):
    context = bpy.context
    background = bool(bpy.app.background)
    c = _Collector()
    c.field("blender_version", lambda: bpy.app.version_string)
    c.field("background", lambda: background)
    c.field("file", lambda: os.path.basename(bpy.data.filepath) or None)
    c.field("file_dirty", lambda: bool(bpy.data.is_dirty))
    c.field("scene", lambda: context.scene.name)
    c.field("mode", lambda: context.mode)
    c.field("workspace", lambda: _workspace(context).name if _workspace(context) else None)
    c.field("active_tool", lambda: _active_tool(context))
    c.field("camera", lambda: context.scene.camera.name if context.scene.camera else None)
    active = c.field("active_object", lambda: context.view_layer.objects.active.name
                     if context.view_layer.objects.active else None)
    c.field("selected_objects", lambda: [o.name for o in context.view_layer.objects if o.select_get()][:MAX_LISTED_OBJECTS])
    c.field("visible_objects", lambda: [o.name for o in context.view_layer.objects if o.visible_get()][:MAX_LISTED_OBJECTS])
    c.field("object_count", lambda: len(context.scene.objects))
    c.field("viewport", lambda: _viewport(background))
    active_obj = context.view_layer.objects.active
    c.field("edit_selection", lambda: _edit_selection(active_obj))
    if include_objects and active is not None:
        c.field("active_object_summary", lambda: object_summary(active_obj))
    return {"schema": 1, "ts": time.time(), "values": c.values, "unavailable": c.unavailable}


def light_state_key(state):
    """Fields whose change is worth pushing as a new state event."""
    v = state.get("values", {})
    view = v.get("viewport") or {}
    summary = v.get("active_object_summary") or {}
    return (
        v.get("mode"), v.get("workspace"), v.get("active_tool"), v.get("active_object"),
        tuple(v.get("selected_objects") or ()), view.get("named_view"), view.get("perspective"),
        tuple(m["type"] for m in summary.get("modifiers", ())), v.get("object_count"),
        tuple(sorted((v.get("edit_selection") or {}).items())) if isinstance(v.get("edit_selection"), dict) else None,
    )
