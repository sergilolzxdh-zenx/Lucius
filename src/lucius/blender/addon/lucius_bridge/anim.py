"""Animation: the frame range, keyframes (I), a camera shake and rendering an animation to a video.

Keyframes are set the way a tutorial does it: go to a frame, put the object where it should be, press I
(or have auto keying on). Only the values a step gives are keyed. Blender interpolates between keys.
"""

import os
import re

import bpy

from .actions import OBJ, REQUIRED, _check_path, _leave_edit_mode, _obj
from .protocol import BridgeCommandError
from .scene import _look_at

INTERPOLATIONS = ("BEZIER", "LINEAR", "CONSTANT")
HANDLES = ("AUTO_CLAMPED", "AUTO", "VECTOR", "ALIGNED", "FREE")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
AXES = ("x", "y", "z", "xy", "xz", "yz", "xyz")
VIDEO_EXTENSIONS = (".mp4",)


def _fcurves(id_block):
    """The F-curves animating a datablock (an object, a camera's data), whatever the action layout."""
    anim = id_block.animation_data
    if anim is None or anim.action is None:
        return []
    action = anim.action
    if hasattr(action, "fcurves"):   # Blender < 4.4
        return list(action.fcurves)
    from bpy_extras import anim_utils

    bag = anim_utils.action_get_channelbag_for_slot(action, anim.action_slot)
    return list(bag.fcurves) if bag is not None else []


def set_frames(p):
    """Timeline and output properties: the first and last frame, frames per second, the current frame."""
    scene = bpy.context.scene
    if p["start"] is not None:
        scene.frame_start = p["start"]
    if p["end"] is not None:
        scene.frame_end = p["end"]
    if scene.frame_end < scene.frame_start:
        raise BridgeCommandError("invalid_param", "end must not come before start", param="end")
    if p["fps"] is not None:
        if not 1 <= p["fps"] <= 240:
            raise BridgeCommandError("invalid_param", "fps must be 1..240", param="fps")
        scene.render.fps = p["fps"]
    if p["current"] is not None:
        scene.frame_set(p["current"])
    return {"start": scene.frame_start, "end": scene.frame_end, "fps": scene.render.fps,
            "current": scene.frame_current}


def insert_keyframe(p):
    """At a frame, set what is given (location, rotation or look_at, scale; a camera's lens, focus distance,
    f-stop) and key it (I). Values not given are neither changed nor keyed."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    scene = bpy.context.scene
    frame = p["frame"]
    scene.frame_set(frame)
    keyed = []
    axes = [AXIS_INDEX[a] for a in p["axes"]]
    if p["location"] is not None:
        for i in axes:
            obj.location[i] = p["location"][i]
            obj.keyframe_insert("location", index=i, frame=frame)
        keyed.append("location")
    if p["look_at"] is not None:
        _look_at(obj, p["look_at"])
    elif p["rotation"] is not None:
        obj.rotation_euler = p["rotation"]
    if p["look_at"] is not None or p["rotation"] is not None:
        for i in (axes if p["look_at"] is None else range(3)):
            obj.keyframe_insert("rotation_euler", index=i, frame=frame)
        keyed.append("rotation")
    if p["scale"] is not None:
        for i in axes:
            obj.scale[i] = p["scale"][i]
            obj.keyframe_insert("scale", index=i, frame=frame)
        keyed.append("scale")
    camera_values = [p["lens"], p["focus_distance"], p["fstop"]]
    if any(v is not None for v in camera_values):
        if obj.type != "CAMERA":
            raise BridgeCommandError("invalid_param", "lens, focus_distance and fstop belong to a camera", param="object")
        data = obj.data
        if p["lens"] is not None:
            data.lens = p["lens"]
            data.keyframe_insert("lens", frame=frame)
            keyed.append("lens")
        if p["focus_distance"] is not None or p["fstop"] is not None:
            data.dof.use_dof = True
        if p["focus_distance"] is not None:
            data.dof.focus_object = None   # an animated distance, not an object, decides the focus
            data.dof.focus_distance = p["focus_distance"]
            data.dof.keyframe_insert("focus_distance", frame=frame)
            keyed.append("focus_distance")
        if p["fstop"] is not None:
            data.dof.aperture_fstop = p["fstop"]
            data.dof.keyframe_insert("aperture_fstop", frame=frame)
            keyed.append("fstop")
    if not keyed:
        raise BridgeCommandError("invalid_param", "give at least one value to key", param="location")
    for block in (obj, obj.data if obj.type == "CAMERA" else None):
        if block is None:
            continue
        for curve in _fcurves(block):
            for point in curve.keyframe_points:
                if int(round(point.co.x)) == frame:
                    point.interpolation = p["interpolation"]
                    if p["handle"] is not None:
                        point.handle_left_type = point.handle_right_type = p["handle"]
            curve.update()
    return {"object": obj.name, "frame": frame, "keyed": keyed}


def clear_animation(p):
    """Delete an object's keyframes (all of them: Alt+I / Clear Keyframes), leaving it where it is now."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    removed = len(_fcurves(obj))
    obj.animation_data_clear()
    if obj.data is not None and getattr(obj.data, "animation_data", None) is not None:
        removed += len(_fcurves(obj.data))
        obj.data.animation_data_clear()
    return {"object": obj.name, "removed_curves": removed}


def add_shake(p):
    """Handheld camera shake (a camera-shake add-on's job): noise added on top of the animation of the object's
    location and rotation, with a strength, a speed (scale in frames) and an influence 0..1."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    scene = bpy.context.scene
    if not any(c.data_path in ("location", "rotation_euler") for c in _fcurves(obj)):
        obj.keyframe_insert("location", frame=scene.frame_start)
        obj.keyframe_insert("rotation_euler", frame=scene.frame_start)
    added = 0
    for curve in _fcurves(obj):
        if curve.data_path not in ("location", "rotation_euler"):
            continue
        for old in [m for m in curve.modifiers if m.type == "NOISE"]:
            curve.modifiers.remove(old)
        noise = curve.modifiers.new("NOISE")
        noise.strength = p["strength"] if curve.data_path == "location" else p["rotation_strength"]
        noise.scale = p["scale"]
        noise.phase = 13.0 * (curve.array_index + 1) + (7.0 if curve.data_path == "rotation_euler" else 0.0)
        noise.use_influence = True
        noise.influence = p["influence"]
        added += 1
    return {"object": obj.name, "noisy_curves": added}


def render_animation(p):
    """Render the frame range with the scene camera into an MP4 (H.264) inside the allowed directories;
    ``step`` renders every n-th frame (a quick preview plays at the same speed)."""
    path = _check_path(p["path"], "LUCIUS_ALLOWED_SAVE_DIRS", VIDEO_EXTENSIONS)
    scene = bpy.context.scene
    if scene.camera is None:
        raise BridgeCommandError("invalid_param", "the scene has no camera to render from")
    _leave_edit_mode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    render = scene.render
    settings = render.image_settings
    saved = {"path": render.filepath, "format": settings.file_format, "step": scene.frame_step,
             "media": getattr(settings, "media_type", None),
             "pct": render.resolution_percentage, "fps": render.fps, "engine": render.engine,
             "x": render.resolution_x, "y": render.resolution_y,
             "samples": scene.cycles.samples if hasattr(scene, "cycles") else None}
    try:
        if bpy.app.background and render.engine != "CYCLES":
            render.engine = "CYCLES"   # Eevee needs a GPU context headless Blender does not have
        if render.engine == "CYCLES" and bpy.app.background:
            scene.cycles.device = "CPU"
        if p["samples"] is not None and hasattr(scene, "cycles"):
            scene.cycles.samples = p["samples"]
        # The video's size at that percentage, rounded to even pixels (H.264 needs them).
        render.resolution_x = max(16, round(saved["x"] * saved["pct"] / 100 * p["percentage"] / 100 / 2) * 2)
        render.resolution_y = max(16, round(saved["y"] * saved["pct"] / 100 * p["percentage"] / 100 / 2) * 2)
        render.resolution_percentage = 100
        scene.frame_step = p["step"]
        render.fps = max(1, round(saved["fps"] / p["step"]))
        if saved["media"] is not None:
            settings.media_type = "VIDEO"   # Blender 5: video formats are a media type of their own
        settings.file_format = "FFMPEG"
        render.ffmpeg.format = "MPEG4"
        render.ffmpeg.codec = "H264"
        render.ffmpeg.constant_rate_factor = "HIGH"
        render.filepath = path
        bpy.ops.render.render(animation=True)
    finally:
        render.filepath = saved["path"]
        if saved["media"] is not None:
            settings.media_type = saved["media"]
        settings.file_format = saved["format"]
        scene.frame_step = saved["step"]
        render.resolution_x, render.resolution_y = saved["x"], saved["y"]
        render.resolution_percentage = saved["pct"]
        render.fps = saved["fps"]
        render.engine = saved["engine"]
        if saved["samples"] is not None:
            scene.cycles.samples = saved["samples"]
    if not os.path.exists(path):
        raise BridgeCommandError("operator_failed", "the animation was not written")
    frames = len(range(scene.frame_start, scene.frame_end + 1, p["step"]))
    return {"path": path, "frames": frames, "bytes": os.path.getsize(path)}


CHANNELS = ("location", "rotation", "scale", "influence")
_BONE_PATH = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)*)"\]\.(.+)$')


def _channel(prop):
    if prop.startswith("constraints["):
        return "influence" if prop.endswith(".influence") else None
    return {"location": "location", "rotation_euler": "rotation", "rotation_quaternion": "rotation",
            "scale": "scale"}.get(prop)


def _select_curves(obj, bones, channels, axes):
    """The curves a graph-editor selection covers: optionally only some bones' channels, only location, rotation,
    scale or constraint influence, only some axes."""
    known = set(obj.pose.bones.keys()) if obj.type == "ARMATURE" else set()
    for bone in bones or ():
        if bone not in known:
            raise BridgeCommandError("invalid_param", f"bone {bone!r} not found on {obj.name}", param="bones")
    for channel in channels or ():
        if channel not in CHANNELS:
            raise BridgeCommandError("invalid_param", f"channel must be one of {CHANNELS}", param="channels")
    indexes = {AXIS_INDEX[a] for a in axes}
    chosen = []
    for curve in _fcurves(obj):
        match = _BONE_PATH.match(curve.data_path)
        bone, prop = (match.group(1), match.group(2)) if match else (None, curve.data_path)
        if bones and bone not in bones:
            continue
        channel = _channel(prop)
        if channels and channel not in channels:
            continue
        if channel in ("location", "scale") or prop == "rotation_euler":
            if curve.array_index not in indexes:
                continue
        chosen.append(curve)
    if not chosen:
        raise BridgeCommandError("invalid_param", f"{obj.name} has no keyframes matching that selection",
                                 param="object")
    return chosen


def _in_range(x, p):
    return (p["start"] is None or x >= p["start"] - 1e-4) and (p["end"] is None or x <= p["end"] + 1e-4)


def set_interpolation(p):
    """Graph editor: select keyframes (all of an object's, or some bones' or channels' within a frame range),
    then T (interpolation), V (handle type), or flat handles for an ease in / ease out given as a percentage of
    the gap to the neighbouring key (what an ease add-on such as Graph Pilot applies)."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    if p["interpolation"] is None and p["handle"] is None and p["ease_in"] is None and p["ease_out"] is None:
        raise BridgeCommandError("invalid_param", "give an interpolation, a handle type or an ease", param="interpolation")
    for key in ("ease_in", "ease_out"):
        if p[key] is not None and not 0 <= p[key] <= 100:
            raise BridgeCommandError("invalid_param", f"{key} is a percentage 0..100", param=key)
    changed = 0
    for curve in _select_curves(obj, p["bones"], p["channels"], p["axes"]):
        points = list(curve.keyframe_points)
        for i, point in enumerate(points):
            if not _in_range(point.co.x, p):
                continue
            changed += 1
            if p["interpolation"] is not None:
                point.interpolation = p["interpolation"]
            if p["handle"] is not None:
                point.handle_left_type = point.handle_right_type = p["handle"]
            if p["ease_in"] is not None or p["ease_out"] is not None:
                point.interpolation = "BEZIER"
                point.handle_left_type = point.handle_right_type = "FREE"
                x, y = point.co.x, point.co.y
                if p["ease_in"] is not None and i > 0:
                    point.handle_left = (x - (x - points[i - 1].co.x) * max(p["ease_in"], 1) / 100, y)
                if p["ease_out"] is not None and i + 1 < len(points):
                    point.handle_right = (x + (points[i + 1].co.x - x) * max(p["ease_out"], 1) / 100, y)
        curve.update()
    if not changed:
        raise BridgeCommandError("invalid_param", "no keyframes in that frame range", param="start")
    return {"object": obj.name, "keys": changed}


def retime_keys(p):
    """Dope sheet / graph editor timing: select the keyframes in a frame range (optionally only some bones or
    channels) and scale them in time around a pivot frame (S X with the 2D cursor as the pivot: below 1 is faster)
    and/or move them by some frames (G X). Keys snap to whole frames, as the editors do by default."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    if p["scale"] <= 0:
        raise BridgeCommandError("invalid_param", "scale must be positive", param="scale")
    if p["scale"] == 1 and p["offset"] == 0:
        raise BridgeCommandError("invalid_param", "give a scale other than 1 or an offset", param="scale")
    pivot = p["pivot"] if p["pivot"] is not None else (p["start"] if p["start"] is not None else 0)
    moved = 0
    curves = _select_curves(obj, p["bones"], p["channels"], p["axes"])
    plans = []
    for curve in curves:
        new_x = []
        for point in curve.keyframe_points:
            x = point.co.x
            if _in_range(x, p):
                target = pivot + (x - pivot) * p["scale"] + p["offset"]
                new_x.append(round(target) if p["snap"] else target)
            else:
                new_x.append(x)
        ordered = sorted(new_x)
        if any(abs(b - a) < 1e-3 for a, b in zip(ordered, ordered[1:])):
            raise BridgeCommandError("invalid_param", f"keys of {curve.data_path}[{curve.array_index}] would land "
                                     "on the same frame; move the neighbouring keys first", param="offset")
        plans.append((curve, new_x))
    for curve, new_x in plans:
        for point, x in zip(list(curve.keyframe_points), new_x):
            shift = x - point.co.x
            if abs(shift) < 1e-9:
                continue
            old = point.co.x
            factor = p["scale"] if _in_range(old, p) else 1.0
            left, right = point.handle_left.x - old, point.handle_right.x - old
            point.co.x = x
            point.handle_left.x = x + left * factor
            point.handle_right.x = x + right * factor
            moved += 1
        curve.update()
    frames = sorted({round(pt.co.x, 3) for c in curves for pt in c.keyframe_points})
    return {"object": obj.name, "moved": moved, "first": frames[0], "last": frames[-1]}


def animation_summary():
    scene = bpy.context.scene
    animated = []
    for obj in scene.objects:
        blocks = [obj, obj.data] if obj.type == "CAMERA" else [obj]
        if any(_fcurves(b) for b in blocks if b is not None):
            animated.append(obj.name)
    return {"start": scene.frame_start, "end": scene.frame_end, "fps": scene.render.fps, "animated": animated,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y]}


ACTIONS = {
    "set_frames": (set_frames, {"start": ("int", None), "end": ("int", None), "fps": ("int", None),
                                "current": ("int", None)}),
    "insert_keyframe": (insert_keyframe, {
        "object": OBJ, "frame": ("int", REQUIRED), "location": ("vec3", None), "rotation": ("vec3", None),
        "look_at": ("vec3", None), "scale": ("vec3", None), "lens": ("float", None),
        "focus_distance": ("float", None), "fstop": ("float", None),
        "interpolation": (INTERPOLATIONS, "BEZIER"), "handle": (HANDLES, None),
        "axes": (AXES, "xyz")}),
    "clear_animation": (clear_animation, {"object": OBJ}),
    "add_shake": (add_shake, {"object": OBJ, "strength": ("float", 0.05), "rotation_strength": ("float", 0.01),
                              "scale": ("float", 20.0), "influence": ("float", 0.5)}),
    "render_animation": (render_animation, {"path": ("path", REQUIRED), "step": ("int", 1),
                                            "samples": ("int", None), "percentage": ("int", 100)}),
    "set_interpolation": (set_interpolation, {
        "object": OBJ, "bones": ("names", None), "channels": ("names", None), "axes": (AXES, "xyz"),
        "start": ("float", None), "end": ("float", None), "interpolation": (INTERPOLATIONS, None),
        "handle": (HANDLES, None), "ease_in": ("float", None), "ease_out": ("float", None)}),
    "retime_keys": (retime_keys, {
        "object": OBJ, "bones": ("names", None), "channels": ("names", None), "axes": (AXES, "xyz"),
        "start": ("float", None), "end": ("float", None), "scale": ("float", 1.0), "pivot": ("float", None),
        "offset": ("float", 0.0), "snap": ("bool", True)}),
}

