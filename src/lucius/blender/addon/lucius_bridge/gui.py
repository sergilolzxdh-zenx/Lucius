"""What an input actuator needs to know about Blender's screen (read-only; interactive sessions only).

Keyboard shortcuts act on the editor under the mouse pointer, so the actuator must know where the 3D
viewport is. Area and region rectangles are in window pixels with the origin at the bottom left,
as Blender reports them. The operator log lets the actuator confirm that a keystroke sequence ran
the operator it meant, with the values it typed.
"""

import os

import bpy

from .protocol import BridgeCommandError
from .watch import sanitize_properties


def _require_gui():
    if bpy.app.background:
        raise BridgeCommandError("gui_unavailable", "no user interface in background mode")


def gui_layout(_args):
    _require_gui()
    context = bpy.context
    windows = []
    for window in context.window_manager.windows:
        areas = []
        for area in window.screen.areas:
            regions = [{"type": r.type, "x": r.x, "y": r.y, "width": r.width, "height": r.height}
                       for r in area.regions if r.width > 1 and r.height > 1]
            areas.append({"type": area.type, "x": area.x, "y": area.y, "width": area.width, "height": area.height,
                          "regions": regions})
        modal = [op.bl_idname for op in getattr(window, "modal_operators", [])]  # Blender 4.2+
        windows.append({"x": window.x, "y": window.y, "width": window.width, "height": window.height,
                        "areas": areas, "modal_operators": modal})
    prefs = context.preferences
    obj = context.view_layer.objects.active
    return {
        "pid": os.getpid(),
        "windows": windows,
        "mode": context.mode,
        "active_object": obj.name if obj is not None else None,
        "pivot_point": context.scene.tool_settings.transform_pivot_point,
        "ui_scale": round(float(prefs.view.ui_scale), 3),
        "pixel_size": round(float(prefs.system.pixel_size), 3),
        "auto_perspective": bool(getattr(prefs.inputs, "use_auto_perspective", True)),
        "orbit_step_deg": float(getattr(prefs.view, "rotation_angle", 15.0)),
        "operator_count": len(context.window_manager.operators),
    }


def operator_log(args):
    """The most recent operators (newest last) with their final property values."""
    _require_gui()
    limit = max(1, min(int(args.get("limit", 8)), 64))
    entries = []
    for op in list(bpy.context.window_manager.operators)[-limit:]:
        entry = {"id": str(op.as_pointer()), "idname": op.bl_idname, "properties": sanitize_properties(op.properties)}
        macros = getattr(op, "macros", None)
        if macros:
            entry["macros"] = [{"idname": m.bl_idname, "properties": sanitize_properties(m.properties)} for m in macros]
        entries.append(entry)
    return {"operators": entries}


def project(args):
    """Window coordinates of an object's bounding-box centre in the first 3D viewport."""
    _require_gui()
    from bpy_extras import view3d_utils
    from mathutils import Vector

    name = args.get("object")
    obj = bpy.data.objects.get(name) if isinstance(name, str) else None
    if obj is None:
        raise BridgeCommandError("object_not_found", f"object {name!r} not found", object=name)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is None:
                continue
            rv3d = area.spaces.active.region_3d
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            centre = sum(corners, Vector()) / 8.0
            point = view3d_utils.location_3d_to_region_2d(region, rv3d, centre)
            if point is None:
                raise BridgeCommandError("not_visible", f"{name} is behind the view", object=name)
            inside = 0 <= point.x < region.width and 0 <= point.y < region.height
            return {"object": obj.name, "x": region.x + float(point.x), "y": region.y + float(point.y),
                    "inside_region": inside, "window_height": window.height}
    raise BridgeCommandError("gui_unavailable", "no 3D viewport is open")
