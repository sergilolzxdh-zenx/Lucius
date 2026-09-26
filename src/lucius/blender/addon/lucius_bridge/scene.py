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
    if p["pattern"] is not None:
        applied["pattern"] = _pattern(material, bsdf, p)
    if p["bump"] is not None:
        applied["bump"] = _bump(material, bsdf, p)
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


def _node(tree, kind, name):
    node = tree.nodes.get(name)
    if node is None or node.bl_idname != kind:
        if node is not None:
            tree.nodes.remove(node)
        node = tree.nodes.new(kind)
        node.name = name
    return node


def _pattern(material, bsdf, p):
    """A procedural colour pattern on Base Color: brick (a check tablecloth: square bricks, coloured mortar
    lines), checker or noise, over the object's own coordinates."""
    tree = material.node_tree
    kind = {"brick": "ShaderNodeTexBrick", "checker": "ShaderNodeTexChecker", "noise": "ShaderNodeTexNoise"}[p["pattern"]]
    coords = _node(tree, "ShaderNodeTexCoord", "LuciusPatternCoords")
    texture = _node(tree, kind, "LuciusPattern")
    tree.links.new(coords.outputs["Object"], texture.inputs["Vector"])
    base = _unit_color(p["base_color"], "base_color") or tuple(bsdf.inputs["Base Color"].default_value)
    second = _unit_color(p["pattern_color"], "pattern_color") or base
    if p["pattern_scale"] is not None:
        texture.inputs["Scale"].default_value = p["pattern_scale"]
    if p["pattern"] == "brick":
        texture.inputs["Color1"].default_value = base
        texture.inputs["Color2"].default_value = second
        texture.inputs["Mortar"].default_value = _unit_color(p["line_color"], "line_color") or (0.1, 0.2, 0.6, 1.0)
        texture.inputs["Mortar Size"].default_value = p["mortar_size"]
        texture.inputs["Brick Width"].default_value = p["brick_width"]
        texture.inputs["Row Height"].default_value = p["row_height"]
        texture.offset_frequency = 1
        texture.offset = 0.0
        texture.squash_frequency = 1
        output = texture.outputs["Color"]
    elif p["pattern"] == "checker":
        texture.inputs["Color1"].default_value = base
        texture.inputs["Color2"].default_value = second
        output = texture.outputs["Color"]
    else:
        ramp = _node(tree, "ShaderNodeMix", "LuciusPatternMix")
        ramp.data_type = "RGBA"
        tree.links.new(texture.outputs["Fac"], ramp.inputs["Factor"])
        ramp.inputs["A"].default_value = base
        ramp.inputs["B"].default_value = second
        output = ramp.outputs["Result"]
    tree.links.new(output, bsdf.inputs["Base Color"])
    return p["pattern"]


def _bump(material, bsdf, p):
    """Surface relief from a procedural texture (a fabric's weave: magic texture, large scale, distorted)."""
    tree = material.node_tree
    kind = {"magic": "ShaderNodeTexMagic", "noise": "ShaderNodeTexNoise", "voronoi": "ShaderNodeTexVoronoi"}[p["bump"]]
    coords = _node(tree, "ShaderNodeTexCoord", "LuciusPatternCoords")
    texture = _node(tree, kind, "LuciusBumpTexture")
    tree.links.new(coords.outputs["Object"], texture.inputs["Vector"])
    texture.inputs["Scale"].default_value = p["bump_scale"]
    if p["bump"] == "magic":
        texture.inputs["Distortion"].default_value = p["bump_distortion"]
    elif "Distortion" in texture.inputs:
        texture.inputs["Distortion"].default_value = p["bump_distortion"]
    bump = _node(tree, "ShaderNodeBump", "LuciusBump")
    bump.inputs["Strength"].default_value = p["bump_strength"]
    tree.links.new(texture.outputs["Fac"] if "Fac" in texture.outputs else texture.outputs[0], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return p["bump"]


# -- lights, camera, world -----------------------------------------------------------------------

def _existing(name, kind):
    obj = bpy.data.objects.get(name) if name else None
    if obj is not None and obj.type != kind:
        raise BridgeCommandError("invalid_param", f"{name!r} exists and is not a {kind.lower()}", param="name")
    return obj


def add_light(p):
    """Add a light, or change the light of that name (the default scene's "Light"): only what is given changes."""
    _leave_edit_mode()
    obj = _existing(p["name"], "LIGHT")
    if obj is None:
        if p["location"] is None:
            raise BridgeCommandError("invalid_param", "a new light needs a location", param="location")
        light_type = p["type"] or "AREA"
        data = bpy.data.lights.new(p["name"] or light_type.title(), type=light_type)
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(obj)
        data.energy = {"SUN": 3.0, "POINT": 800.0, "SPOT": 800.0, "AREA": 400.0}[light_type]
    else:
        if p["type"] is not None and obj.data.type != p["type"]:
            obj.data.type = p["type"]   # the datablock becomes another light class: fetch it again below
        data = obj.data
    if p["power"] is not None:
        data.energy = p["power"]
    if p["color"] is not None:
        data.color = _unit_color(p["color"], "color")[:3]
    if p["temperature"] is not None:
        if not 800 <= p["temperature"] <= 20000:
            raise BridgeCommandError("invalid_param", "temperature is in kelvin, 800..20000", param="temperature")
        if hasattr(data, "use_temperature"):
            data.use_temperature = True
            data.temperature = p["temperature"]
        else:
            data.color = _kelvin(p["temperature"])
    if data.type == "SPOT":
        if p["spot_size"] is not None:
            data.spot_size = math.radians(max(1.0, min(180.0, p["spot_size"])))
        if p["spot_blend"] is not None:
            data.spot_blend = max(0.0, min(1.0, p["spot_blend"]))
    if p["size"] is not None:
        if data.type == "AREA":
            data.size = p["size"]
        elif data.type == "SUN":
            data.angle = math.radians(min(180.0, p["size"]))
        else:
            data.shadow_soft_size = p["size"]
    if p["location"] is not None:
        obj.location = p["location"]
    if p["look_at"] is not None:
        _look_at(obj, p["look_at"])
    elif p["rotation"] is not None:
        obj.rotation_euler = p["rotation"]
    return {"object": obj.name, "type": data.type, "power": data.energy, "location": list(obj.location)}


def _kelvin(kelvin):
    """Approximate RGB of a black body (for Blender versions without light temperature)."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return tuple(max(0.0, min(255.0, c)) / 255.0 for c in (r, g, b))


def add_camera(p):
    """Add a camera, or move and set up the camera of that name (the default scene's "Camera"): only what is
    given changes, so a later step can just set the focus or the lens."""
    _leave_edit_mode()
    obj = _existing(p["name"], "CAMERA")
    if obj is None:
        if p["location"] is None:
            raise BridgeCommandError("invalid_param", "a new camera needs a location", param="location")
        data = bpy.data.cameras.new(p["name"] or "Camera")
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(obj)
        data.lens = 50.0
    data = obj.data
    if p["lens"] is not None:
        data.lens = p["lens"]
    if p["location"] is not None:
        obj.location = p["location"]
    if p["look_at"] is not None:
        _look_at(obj, p["look_at"])
    elif p["rotation"] is not None:
        obj.rotation_euler = p["rotation"]
    if p["dof_distance"] is not None or p["focus_object"] is not None:
        # Depth of field: sharp at the focus object (or distance), blurred elsewhere; a lower f-stop blurs more.
        data.dof.use_dof = True
        if p["focus_object"] is not None:
            data.dof.focus_object = _obj(p["focus_object"])
        else:
            data.dof.focus_object = None
            data.dof.focus_distance = p["dof_distance"]
        data.dof.aperture_fstop = p["fstop"] if p["fstop"] is not None else 2.8
    elif p["fstop"] is not None:
        data.dof.aperture_fstop = p["fstop"]
    if p["dof"] is not None:
        data.dof.use_dof = p["dof"]
    if p["active"]:
        bpy.context.scene.camera = obj
    return {"object": obj.name, "lens": data.lens, "location": list(obj.location),
            "rotation": list(obj.rotation_euler), "dof": data.dof.use_dof}


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
    if p["view_transform"] is not None:
        scene.view_settings.view_transform = p["view_transform"]
    return {"engine": scene.render.engine, "samples": p["samples"], "resolution": [p["width"], p["height"]],
            "view_transform": scene.view_settings.view_transform}


def scale_scene(p):
    """A then S: scale every object of the scene about a point (to bring a model to real-world size).
    Lights keep their power -- at a new scale they usually need less (the tutor's lesson on working to scale)."""
    _leave_edit_mode()
    factor = p["factor"]
    if not 1e-4 <= factor <= 1e4:
        raise BridgeCommandError("invalid_param", "factor must be between 0.0001 and 10000", param="factor")
    pivot = Vector(p["pivot"])
    scaled = []
    for obj in bpy.context.scene.objects:
        if obj.parent is not None:
            continue   # children follow their parent
        obj.location = pivot + (obj.location - pivot) * factor
        obj.scale = obj.scale * factor
        scaled.append(obj.name)
    bpy.context.view_layer.update()
    return {"scaled": scaled, "factor": factor}


# -- rendering -----------------------------------------------------------------------------------

def _visible_bounds(names=None):
    bpy.context.view_layer.update()   # world matrices of objects just added or appended
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


def _studio(view, names=None, lights=True):
    """A temporary camera framing the visible objects (or ``names``) and (unless ``lights`` is false) three
    studio lights."""
    scene = bpy.context.scene
    made = []
    centre, radius = _visible_bounds(names)
    cam_data = bpy.data.cameras.new("LuciusPreviewCamera")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new(cam_data.name, cam_data)
    scene.collection.objects.link(cam)
    made.append(cam)
    # The sensor spans the image's longer side; the shorter side sees less, and limits the framing.
    half_long = math.atan(cam_data.sensor_width / 2 / cam_data.lens)
    aspect = scene.render.resolution_x / max(1, scene.render.resolution_y)
    half_short = math.atan(math.tan(half_long) / max(aspect, 1.0 / aspect))
    distance = radius / math.sin(half_short) * 1.05
    direction = Vector(VIEW_DIRECTIONS[view]).normalized()
    cam.location = centre + direction * distance
    cam_data.clip_end = max(100.0, distance * 4)
    _look_at(cam, centre)
    # Previews are for seeing the shape: studio lights (the scene's own lights are for its own camera).
    for name, kind, energy, offset in () if not lights else (("LuciusKey", "AREA", 450.0, (1.2, -1.0, 1.2)),
                                       ("LuciusFill", "AREA", 120.0, (-1.4, -0.6, 0.8)),
                                       ("LuciusRim", "AREA", 250.0, (-0.3, 1.5, 1.4))):
        data = bpy.data.lights.new(name, type=kind)
        data.energy = energy * max(1.0, radius) ** 2
        data.size = max(1.0, radius * 1.5)
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = centre + Vector(offset) * max(radius * 2.5, 2.0)
        _look_at(light, centre)
        made.append(light)
    return cam, made


def _background(world):
    if world is None:
        return None
    world.use_nodes = True
    return next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)


def render_image(p):
    """Render the scene to a PNG/JPEG inside the allowed directories.

    ``camera='scene'`` uses the scene's camera and lights (a finished tutorial shot); ``'auto'`` frames the
    visible objects (or ``frame``) from ``view`` with a temporary camera and temporary studio lights.
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
    if not p["scene_settings"]:
        # (with scene_settings: the engine, size and samples the scene was set up with -- a finished shot)
        render.engine = p["engine"]
        render.resolution_x, render.resolution_y, render.resolution_percentage = p["width"], p["height"], 100
        if p["engine"] == "CYCLES":
            saved["samples"] = scene.cycles.samples
            scene.cycles.samples = p["samples"]
    if render.engine == "CYCLES" and bpy.app.background:
        scene.cycles.device = "CPU"
    elif render.engine != "CYCLES" and bpy.app.background:
        render.engine = "CYCLES"   # Eevee needs a GPU context headless Blender does not have
    render.image_settings.file_format = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
    render.filepath = path
    temporary = []
    hidden = []
    world_created = False
    try:
        studio = p["lights"] == "studio"
        if p["camera"] == "auto" or scene.camera is None:
            camera, temporary = _studio(p["view"], set(p["frame"] or []), lights=studio)
        if p["camera"] == "auto" and studio:
            for obj in scene.objects:   # the scene's own lights stay out of a studio preview
                if obj.type == "LIGHT" and obj not in temporary and not obj.hide_render:
                    obj.hide_render = True
                    hidden.append(obj)
        if p["camera"] == "auto" or scene.camera is None:
            scene.camera = camera
        if scene.world is None:
            scene.world = bpy.data.worlds.new("LuciusPreviewWorld")
            world_created = True
        if p["camera"] == "auto" and studio:
            # A shape preview, like the viewport's solid mode: grey clay when nothing has a material yet, and soft
            # ambient light so hollows (a dish, the inside of a mug) read.
            meshes = [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
            if meshes and not any(slot.material for o in meshes for slot in o.material_slots):
                clay = bpy.data.materials.new("LuciusPreviewClay")
                clay.diffuse_color = (0.42, 0.42, 0.44, 1.0)
                clay.use_nodes = True
                bsdf = next(n for n in clay.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
                bsdf.inputs["Base Color"].default_value = (0.42, 0.42, 0.44, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.55
                saved["override"] = bpy.context.view_layer.material_override
                bpy.context.view_layer.material_override = clay
                saved["clay"] = clay
            background = _background(scene.world)
            if background is not None:
                saved["background"] = (tuple(background.inputs["Color"].default_value),
                                       background.inputs["Strength"].default_value)
                background.inputs["Color"].default_value = (0.25, 0.25, 0.27, 1.0)
                background.inputs["Strength"].default_value = 1.0
        bpy.ops.render.render(write_still=True)
    finally:
        if "clay" in saved:
            bpy.context.view_layer.material_override = saved["override"]
            bpy.data.materials.remove(saved["clay"])
        if "background" in saved and _background(scene.world) is not None:
            background = _background(scene.world)
            background.inputs["Color"].default_value, background.inputs["Strength"].default_value = saved["background"]
        for obj in hidden:
            obj.hide_render = False
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
    name = p["name"] or "Scatter"
    existing = obj.modifiers.get(name)
    if existing is not None and existing.type == "PARTICLE_SYSTEM":
        modifier = existing   # the same system again: its settings change (more sugar, bigger crystals)
    else:
        modifier = obj.modifiers.new(name=name, type="PARTICLE_SYSTEM")
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
        "assign": (("replace", "append", "selected_faces"), "replace"),
        "pattern": (("brick", "checker", "noise"), None), "pattern_color": COLOR, "line_color": COLOR,
        "pattern_scale": ("float", None), "mortar_size": ("float", 0.02), "brick_width": ("float", 0.5),
        "row_height": ("float", 0.5), "bump": (("magic", "noise", "voronoi"), None), "bump_scale": ("float", 200.0),
        "bump_distortion": ("float", 15.0), "bump_strength": ("float", 0.3)}),
    "add_light": (add_light, {
        "type": (("POINT", "SUN", "SPOT", "AREA"), None), "name": ("name", None), "location": ("vec3", None),
        "rotation": ("vec3", None), "look_at": ("vec3", None), "power": ("float", None), "color": COLOR,
        "size": ("float", None), "temperature": ("float", None), "spot_size": ("float", None),
        "spot_blend": ("float", None)}),
    "add_camera": (add_camera, {
        "name": ("name", None), "location": ("vec3", None), "rotation": ("vec3", None), "look_at": ("vec3", None),
        "lens": ("float", None), "active": ("bool", True), "dof_distance": ("float", None), "fstop": ("float", None),
        "focus_object": ("name", None), "dof": ("bool", None)}),
    "set_world": (set_world, {"color": COLOR, "strength": ("float", 1.0)}),
    "set_render": (set_render, {
        "engine": (("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"), "CYCLES"),
        "samples": ("int", 64), "width": ("int", 1280), "height": ("int", 720), "denoise": ("bool", True),
        "view_transform": (("Standard", "AgX", "Filmic", "Khronos PBR Neutral"), None)}),
    "scale_scene": (scale_scene, {"factor": ("float", REQUIRED), "pivot": ("vec3", [0.0, 0.0, 0.0])}),
    "render_image": (render_image, {
        "path": ("path", REQUIRED), "camera": (("scene", "auto"), "auto"), "view": (tuple(VIEW_DIRECTIONS), "three_quarter"),
        "engine": (("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"), "CYCLES"),
        "samples": ("int", 24), "width": ("int", 800), "height": ("int", 600), "frame": ("names", None),
        "scene_settings": ("bool", False), "lights": (("studio", "scene"), "studio")}),
    "add_scatter": (add_scatter, {
        "object": OBJ, "instance": ("name", REQUIRED), "name": ("name", None), "count": ("int", 300),
        "scale": ("float", 1.0), "scale_random": ("float", 0.3),
        "rotation_random": ("float", 1.0), "seed": ("int", 0), "hide_instance": ("bool", True)}),
}

