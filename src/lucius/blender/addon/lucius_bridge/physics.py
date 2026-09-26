"""Physics: cloth / soft body caches baked over the timeline, and fluid simulations the tutorial way -- Quick Liquid, the flows' and domain's settings (set_property on
``modifiers["Fluid"]...``), and baking the result into a cache inside the allowed save folders."""

import os
import time
import uuid

import bpy

from .actions import REQUIRED, _allowed_dirs, _context_override, _leave_edit_mode, _obj, _op_result
from .protocol import BridgeCommandError


def quick_liquid(p):
    """F3 > Quick Liquid: the selected meshes become liquid emitters (flows, geometry behaviour, hidden in renders)
    inside a new domain box that encloses them with room below; the domain is where the simulation happens (liquid
    can't leave it). Done step by step as the operator does (it needs a window headless Blender doesn't have)."""
    from mathutils import Vector

    _leave_edit_mode()
    objects = [_obj(n) for n in p["objects"]]
    if not objects or any(o.type != "MESH" for o in objects):
        raise BridgeCommandError("invalid_param", "give one or more mesh objects to emit the liquid", param="objects")
    lo = Vector((1e9, 1e9, 1e9))
    hi = -lo
    for obj in objects:
        modifier = obj.modifiers.get("Fluid") or obj.modifiers.new("Fluid", "FLUID")
        modifier.fluid_type = "FLOW"
        flow = modifier.flow_settings
        flow.flow_type = "LIQUID"
        flow.flow_behavior = "GEOMETRY"
        flow.surface_distance = 0.0
        obj.display_type = "WIRE"
        obj.hide_render = True
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, world))
            hi = Vector(map(max, hi, world))
    mesh = bpy.data.meshes.new(p["domain"] or "Liquid Domain")
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(mesh)
    bm.free()
    domain = bpy.data.objects.new(p["domain"] or "Liquid Domain", mesh)
    bpy.context.collection.objects.link(domain)
    domain.location = 0.5 * (hi + lo) + Vector((0.0, 0.0, -1.0))
    domain.scale = 0.5 * (hi - lo) + Vector((1.0, 1.0, 2.0))
    modifier = domain.modifiers.new("Fluid", "FLUID")
    modifier.fluid_type = "DOMAIN"
    settings = modifier.domain_settings
    for side in ("front", "back", "right", "left", "top", "bottom"):
        setattr(settings, f"use_collision_border_{side}", True)
    if bpy.app.build_options.openvdb:
        settings.cache_data_format = "OPENVDB"
    settings.cache_mesh_format = "BOBJECT"
    settings.domain_type = "LIQUID"
    for poly in mesh.polygons:
        poly.use_smooth = True
    bpy.context.view_layer.update()
    return {"domain": domain.name, "flows": [o.name for o in objects],
            "domain_size": [round(v, 3) for v in domain.dimensions]}


def _domain_settings(obj):
    for modifier in obj.modifiers:
        if modifier.type == "FLUID" and modifier.fluid_type == "DOMAIN":
            return modifier.domain_settings
    raise BridgeCommandError("invalid_param", f"{obj.name} is not a fluid domain", param="domain")


def bake_fluid(p):
    """Domain > Cache: type All, the frame range, Bake All -- the simulation runs once and is replayed from the
    cache (written in the allowed save folder). Resolution: the domain's voxel divisions (keep it low while trying)."""
    _leave_edit_mode()
    domain = _obj(p["domain"])
    settings = _domain_settings(domain)
    if getattr(bpy.app, "module", False):
        # The pip bpy module's Mantaflow can't run its solver scripts and aborts the process: refuse instead.
        raise BridgeCommandError("unsupported", "the fluid solver doesn't run inside the bpy Python module; bake in "
                                 "a full Blender (the setup steps work here)")
    if p["resolution"] is not None:
        if not 16 <= p["resolution"] <= 256:
            raise BridgeCommandError("invalid_param", "resolution 16..256", param="resolution")
        settings.resolution_max = p["resolution"]
    if p["frame_end"] <= p["frame_start"]:
        raise BridgeCommandError("invalid_param", "frame_end after frame_start", param="frame_end")
    allowed = _allowed_dirs("LUCIUS_ALLOWED_SAVE_DIRS")
    if not allowed:
        raise BridgeCommandError("path_not_allowed", "no folder is allowed for the simulation cache")
    cache = os.path.join(allowed[0], "caches", f"fluid_{uuid.uuid4().hex[:12]}")
    os.makedirs(cache, exist_ok=True)
    settings.cache_directory = cache
    settings.cache_type = "ALL"
    settings.cache_frame_start = p["frame_start"]
    settings.cache_frame_end = p["frame_end"]
    scene = bpy.context.scene
    scene.frame_set(p["frame_start"])
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain
    started = time.time()
    with bpy.context.temp_override(**_context_override(domain)):
        _op_result(bpy.ops.fluid.bake_all(), "bake_all")
    files = sum(len(names) for _root, _dirs, names in os.walk(cache))
    if not files:
        raise BridgeCommandError("operator_failed", "the bake wrote nothing")
    return {"domain": domain.name, "frames": [p["frame_start"], p["frame_end"]], "resolution":
            settings.resolution_max, "cache_files": files, "seconds": round(time.time() - started, 1)}


def _point_caches():
    for obj in bpy.context.scene.objects:
        for modifier in obj.modifiers:
            if modifier.type in ("CLOTH", "SOFT_BODY"):
                yield obj, modifier, modifier.point_cache
        for system in getattr(obj, "particle_systems", []):
            yield obj, system, system.point_cache


def bake_physics(p):
    """Timeline > Bake (Physics properties > Cache): the cloth, soft body and particle simulations run once over
    the frame range and are kept (in the .blend), so every frame -- the draped cloth at the end -- plays back
    without re-simulating. ``free`` throws the bake away (after changing a setting)."""
    _leave_edit_mode()
    scene = bpy.context.scene
    caches = list(_point_caches())
    if not caches:
        raise BridgeCommandError("invalid_param", "nothing to bake: add a CLOTH / SOFT_BODY modifier or particles")
    if p["frame_end"] <= p["frame_start"] or p["frame_end"] - p["frame_start"] > 2000:
        raise BridgeCommandError("invalid_param", "frame_end after frame_start (2000 frames at most)",
                                 param="frame_end")
    started = time.time()
    with bpy.context.temp_override(scene=scene):
        _op_result(bpy.ops.ptcache.free_bake_all(), "free_bake_all")
    if p["free"]:
        return {"freed": [obj.name for obj, _owner, _cache in caches]}
    for _obj_, _owner, cache in caches:
        cache.frame_start, cache.frame_end = p["frame_start"], p["frame_end"]
    scene.frame_start, scene.frame_end = min(scene.frame_start, p["frame_start"]), max(scene.frame_end,
                                                                                         p["frame_end"])
    with bpy.context.temp_override(scene=scene):
        _op_result(bpy.ops.ptcache.bake_all(bake=True), "bake_all")
    scene.frame_set(p["show_frame"] if p["show_frame"] is not None else p["frame_end"])
    return {"baked": sorted({obj.name for obj, _owner, cache in caches if cache.is_baked}),
            "frames": [p["frame_start"], p["frame_end"]], "frame": scene.frame_current,
            "seconds": round(time.time() - started, 1)}


ACTIONS = {
    "quick_liquid": (quick_liquid, {"objects": ("names", REQUIRED), "domain": ("name", None)}),
    "bake_fluid": (bake_fluid, {"domain": ("name", REQUIRED), "resolution": ("int", None),
                                "frame_start": ("int", 1), "frame_end": ("int", 50)}),
    "bake_physics": (bake_physics, {"frame_start": ("int", 1), "frame_end": ("int", 60), "show_frame": ("int", None),
                                    "free": ("bool", False)}),
}
