"""Materials, lights, camera, world and rendering (allowlisted, parameter-validated like every action).

Tutorials finish a model by shading, lighting and rendering it, and a rendered picture is how Lucius
(and the person watching it) judges what it built. ``render_image`` can frame the scene with a
temporary camera and studio lights when the scene has none, so work in progress can be rendered
without changing it; they are removed after the render.
"""

import math
import os

import bpy
from mathutils import Vector

from .actions import OBJ, REQUIRED, _check_path, _leave_edit_mode, _obj
from .protocol import BridgeCommandError

COLOR = ("vec3", None)
RENDER_EXTENSIONS = (".png", ".jpg", ".jpeg")
VIEW_DIRECTIONS = {  # camera direction from the scene centre (x, y, z) for automatic framing
    "three_quarter": (1.0, -1.25, 0.8), "front": (0.0, -1.0, 0.12), "side": (1.0, 0.0, 0.12),
    "top": (0.0, -0.001, 1.0), "low": (1.0, -1.3, 0.3),
}
PRINCIPLED_INPUTS = {  # parameter -> Principled BSDF input
    "base_color": "Base Color", "roughness": "Roughness", "metallic": "Metallic", "alpha": "Alpha",
    "emission_color": "Emission Color", "emission_strength": "Emission Strength",
    "transmission": "Transmission Weight", "subsurface": "Subsurface Weight", "coat": "Coat Weight",
    "ior": "IOR",
}


def _unit_color(value, name):
    if value is None:
        return None
    if any(c < 0.0 or c > 1.0 for c in value):
        raise BridgeCommandError("invalid_param", f"{name} components must be within 0..1", param=name)
    return (*value, 1.0)


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    if direction.length < 1e-9:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


# -- materials -----------------------------------------------------------------------------------

def set_material(p):
    """Create (or update) a Principled BSDF material and give it to the object.

    ``assign='replace'`` makes it the object's only material, ``'append'`` adds a slot, and
    ``'selected_faces'`` adds a slot used by the faces selected in edit mode (a two-colour object).
    """
    obj = _obj(p["object"])
    if obj.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT"):
        raise BridgeCommandError("not_a_mesh", f"{obj.name} cannot hold a material")
    name = p["name"] or f"{obj.name}_material"
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        output = next((n for n in material.node_tree.nodes if n.type == "OUTPUT_MATERIAL"), None) \
            or material.node_tree.nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(bsdf.outputs[0], output.inputs[0])
    applied = {}
    for key, socket in PRINCIPLED_INPUTS.items():
        value = p.get(key)
        if value is None or socket not in bsdf.inputs:
            continue
        if key in ("base_color", "emission_color"):
            value = _unit_color(value, key)
        bsdf.inputs[socket].default_value = value
        applied[key] = list(value) if isinstance(value, tuple) else value
    if p.get("base_color") is not None:
        material.diffuse_color = _unit_color(p["base_color"], "base_color")  # the solid viewport colour
    if p["assign"] == "replace":
        obj.data.materials.clear()
        obj.data.materials.append(material)
    elif p["assign"] == "append" or material.name not in obj.data.materials:
        obj.data.materials.append(material)
    if p["assign"] == "selected_faces":
        if obj.type != "MESH":
            raise BridgeCommandError("not_a_mesh", "face assignment needs a mesh")
        import bmesh

        index = list(obj.data.materials).index(material)
        was_editing = obj.mode == "EDIT"
        if not was_editing:
            raise BridgeCommandError("empty_selection", "select faces in edit mode first")
        bm = bmesh.from_edit_mesh(obj.data)
        count = 0
        for face in bm.faces:
            if face.select:
                face.material_index = index
                count += 1
        bmesh.update_edit_mesh(obj.data)
        applied["faces"] = count
    return {"object": obj.name, "material": material.name, "set": applied}


# -- lights, camera, world -----------------------------------------------------------------------

def _existing(name, kind):
    obj = bpy.data.objects.get(name) if name else None
    if obj is not None and obj.type != kind:
        raise BridgeCommandError("invalid_param", f"{name!r} exists and is not a {kind.lower()}", param="name")
    return obj


def add_light(p):
    """Add a light, or change the light of that name (the default scene's "Light")."""
    _leave_edit_mode()
    obj = _existing(p["name"], "LIGHT")
    if obj is None:
        data = bpy.data.lights.new(p["name"] or p["type"].title(), type=p["type"])
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(obj)
    else:
        obj.data.type = p["type"]
        data = obj.data
    data.energy = p["power"] if p["power"] is not None else {"SUN": 3.0, "POINT": 800.0, "SPOT": 800.0,
                                                              "AREA": 400.0}[p["type"]]
    if p["color"] is not None:
        data.color = _unit_color(p["color"], "color")[:3]
    if p["size"] is not None:
        if p["type"] == "AREA":
            data.size = p["size"]
        elif p["type"] == "SUN":
            data.angle = math.radians(min(180.0, p["size"]))
        else:
            data.shadow_soft_size = p["size"]
    obj.location = p["location"]
    if p["look_at"] is not None:
        _look_at(obj, p["look_at"])
    elif p["rotation"] is not None:
        obj.rotation_euler = p["rotation"]
    return {"object": obj.name, "type": p["type"], "power": data.energy}


def add_camera(p):
    """Add a camera, or move and set up the camera of that name (the default scene's "Camera")."""
    _leave_edit_mode()
    obj = _existing(p["name"], "CAMERA")
    if obj is None:
        data = bpy.data.cameras.new(p["name"] or "Camera")
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(obj)
    data = obj.data
    data.lens = p["lens"]
    obj.location = p["location"]
    if p["look_at"] is not None:
        _look_at(obj, p["look_at"])
    elif p["rotation"] is not None:
        obj.rotation_euler = p["rotation"]
    if p["dof_distance"] is not None:
        data.dof.use_dof = True
        data.dof.focus_distance = p["dof_distance"]
        data.dof.aperture_fstop = p["fstop"]
    if p["active"]:
        bpy.context.scene.camera = obj
    return {"object": obj.name, "lens": data.lens, "rotation": list(obj.rotation_euler)}


def set_world(p):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if background is None:
        raise BridgeCommandError("operator_failed", "the world has no background node")
    if p["color"] is not None:
        background.inputs["Color"].default_value = _unit_color(p["color"], "color")
    background.inputs["Strength"].default_value = p["strength"]
    return {"strength": p["strength"]}


def set_render(p):
    scene = bpy.context.scene
    scene.render.engine = p["engine"]
    scene.render.resolution_x, scene.render.resolution_y = p["width"], p["height"]
    scene.render.resolution_percentage = 100
    if p["engine"] == "CYCLES":
        scene.cycles.samples = p["samples"]
        scene.cycles.device = "CPU" if bpy.app.background else scene.cycles.device
        scene.cycles.use_denoising = p["denoise"]
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = p["samples"]
    return {"engine": scene.render.engine, "samples": p["samples"], "resolution": [p["width"], p["height"]]}


# -- rendering -----------------------------------------------------------------------------------

def _visible_bounds(names=None):
    points = []
    for obj in bpy.context.scene.objects:
        if names and obj.name not in names:
            continue
        if obj.type not in ("MESH", "CURVE", "FONT", "SURFACE", "META") or obj.hide_render or obj.hide_get():
            continue
        points += [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    if not points:
        return Vector((0, 0, 0)), 1.0
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return (lo + hi) / 2, max((hi - lo).length / 2, 0.05)


def _studio(view, names=None):
    """A temporary camera framing the visible objects (or ``names``) and, if the scene has no light, three lights."""
    scene = bpy.context.scene
    made = []
    centre, radius = _visible_bounds(names)
    cam_data = bpy.data.cameras.new("LuciusPreviewCamera")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new(cam_data.name, cam_data)
    scene.collection.objects.link(cam)
    made.append(cam)
    half_fov = math.atan(cam_data.sensor_width / 2 / cam_data.lens)
    aspect = scene.render.resolution_x / max(1, scene.render.resolution_y)
    distance = radius / math.sin(half_fov * min(1.0, aspect)) * 1.08
    direction = Vector(VIEW_DIRECTIONS[view]).normalized()
    cam.location = centre + direction * distance
    cam_data.clip_end = max(100.0, distance * 4)
    _look_at(cam, centre)
    if not any(o.type == "LIGHT" and not o.hide_render for o in scene.objects):
        for name, kind, energy, offset in (("LuciusKey", "AREA", 700.0, (1.2, -1.0, 1.6)),
                                           ("LuciusFill", "AREA", 250.0, (-1.4, -0.6, 0.8)),
                                           ("LuciusRim", "AREA", 400.0, (-0.3, 1.5, 1.4))):
            data = bpy.data.lights.new(name, type=kind)
            data.energy = energy * max(1.0, radius) ** 2
            data.size = max(1.0, radius * 1.5)
            light = bpy.data.objects.new(name, data)
            scene.collection.objects.link(light)
            light.location = centre + Vector(offset) * max(radius * 2.5, 2.0)
            _look_at(light, centre)
            made.append(light)
    return cam, made


def render_image(p):
    """Render the scene to a PNG/JPEG inside the allowed directories.

    ``camera='scene'`` uses the scene's camera (a finished tutorial shot); ``'auto'`` frames the visible
    objects from ``view`` with a temporary camera (and temporary lights if the scene has none).
    """
    path = _check_path(p["path"], "LUCIUS_ALLOWED_SAVE_DIRS", RENDER_EXTENSIONS)
    if not (16 <= p["width"] <= 4096 and 16 <= p["height"] <= 4096 and 1 <= p["samples"] <= 4096):
        raise BridgeCommandError("invalid_param", "render size 16..4096 px and 1..4096 samples")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _leave_edit_mode()
    scene = bpy.context.scene
    render = scene.render
    saved = {"engine": render.engine, "x": render.resolution_x, "y": render.resolution_y,
             "pct": render.resolution_percentage, "path": render.filepath, "camera": scene.camera,
             "format": render.image_settings.file_format}
    render.engine = p["engine"]
    render.resolution_x, render.resolution_y, render.resolution_percentage = p["width"], p["height"], 100
    if p["engine"] == "CYCLES":
        saved["samples"] = scene.cycles.samples
        scene.cycles.samples = p["samples"]
        if bpy.app.background:
            scene.cycles.device = "CPU"
    render.image_settings.file_format = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
    render.filepath = path
    temporary = []
    world_created = False
    try:
        if p["camera"] == "auto" or scene.camera is None:
            camera, temporary = _studio(p["view"], set(p["frame"] or []))
            scene.camera = camera
        if scene.world is None:
            scene.world = bpy.data.worlds.new("LuciusPreviewWorld")
            world_created = True
        bpy.ops.render.render(write_still=True)
    finally:
        for obj in temporary:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                (bpy.data.cameras if isinstance(data, bpy.types.Camera) else bpy.data.lights).remove(data)
        if world_created:
            bpy.data.worlds.remove(scene.world)
        scene.camera = saved["camera"]
        render.engine, render.resolution_x, render.resolution_y = saved["engine"], saved["x"], saved["y"]
        render.resolution_percentage, render.filepath = saved["pct"], saved["path"]
        render.image_settings.file_format = saved["format"]
        if "samples" in saved:
            scene.cycles.samples = saved["samples"]
    if not os.path.exists(path):
        raise BridgeCommandError("operator_failed", "the render wrote no image")
    return {"path": path, "camera": "auto" if temporary else "scene", "size": [p["width"], p["height"]]}


# -- scattering (particles) ----------------------------------------------------------------------

def _scatter_collection(instance):
    """Move the instanced object into a collection outside the scene: its copies render, the original does not."""
    name = f"LuciusScatter_{instance.name}"[:63]
    collection = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if instance.name not in collection.objects:
        collection.objects.link(instance)
    for other in list(instance.users_collection):
        if other != collection:
            other.objects.unlink(instance)
    instance.location = (0.0, 0.0, 0.0)
    return collection


def add_scatter(p):
    """A hair particle system that places copies of an object over a surface (sprinkles on icing)."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    instance = _obj(p["instance"])
    if instance == obj:
        raise BridgeCommandError("invalid_param", "an object cannot be scattered over itself", param="instance")
    modifier = obj.modifiers.new(name=p["name"] or "Scatter", type="PARTICLE_SYSTEM")
    settings = modifier.particle_system.settings
    settings.type = "HAIR"
    settings.use_advanced_hair = True
    settings.count = p["count"]
    settings.hair_length = 1.0  # hair instances are scaled by the hair length times the particle size
    settings.particle_size = p["scale"]
    settings.size_random = p["scale_random"]
    settings.use_rotations = True
    settings.rotation_mode = "NOR"
    settings.phase_factor_random = p["rotation_random"]
    settings.rotation_factor_random = p["rotation_random"] / 2.0
    modifier.particle_system.seed = p["seed"]
    if p["hide_instance"]:
        settings.render_type = "COLLECTION"
        settings.instance_collection = _scatter_collection(instance)
    else:
        settings.render_type = "OBJECT"
        settings.instance_object = instance
    obj.show_instancer_for_render = True
    return {"object": obj.name, "instance": instance.name, "count": settings.count}


ACTIONS = {
    "set_material": (set_material, {
        "object": OBJ, "name": ("name", None), "base_color": COLOR, "roughness": ("float", None),
        "metallic": ("float", None), "alpha": ("float", None), "emission_color": COLOR,
        "emission_strength": ("float", None), "transmission": ("float", None), "subsurface": ("float", None),
        "coat": ("float", None), "ior": ("float", None),
        "assign": (("replace", "append", "selected_faces"), "replace")}),
    "add_light": (add_light, {
        "type": (("POINT", "SUN", "SPOT", "AREA"), "AREA"), "name": ("name", None), "location": ("vec3", REQUIRED),
        "rotation": ("vec3", None), "look_at": ("vec3", None), "power": ("float", None), "color": COLOR,
        "size": ("float", None)}),
    "add_camera": (add_camera, {
        "name": ("name", None), "location": ("vec3", REQUIRED), "rotation": ("vec3", None), "look_at": ("vec3", None),
        "lens": ("float", 50.0), "active": ("bool", True), "dof_distance": ("float", None), "fstop": ("float", 2.8)}),
    "set_world": (set_world, {"color": COLOR, "strength": ("float", 1.0)}),
    "set_render": (set_render, {
        "engine": (("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"), "CYCLES"),
        "samples": ("int", 64), "width": ("int", 1280), "height": ("int", 720), "denoise": ("bool", True)}),
    "render_image": (render_image, {
        "path": ("path", REQUIRED), "camera": (("scene", "auto"), "auto"), "view": (tuple(VIEW_DIRECTIONS), "three_quarter"),
        "engine": (("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"), "CYCLES"),
        "samples": ("int", 24), "width": ("int", 800), "height": ("int", 600), "frame": ("names", None)}),
    "add_scatter": (add_scatter, {
        "object": OBJ, "instance": ("name", REQUIRED), "name": ("name", None), "count": ("int", 300),
        "scale": ("float", 1.0), "scale_random": ("float", 0.3),
        "rotation_random": ("float", 1.0), "seed": ("int", 0), "hide_instance": ("bool", True)}),
}

