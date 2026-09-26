"""Node editors and properties: the shader editor (materials, the world, node groups), geometry nodes, and any
value of the scene set or keyframed by its property path; shape keys and curves.

A node step works like the editor: add nodes (Shift+A), set their values and settings, connect sockets. Only
node types of the shader, geometry and compositor editors can be made (no script nodes, no file import or
output nodes), settings are checked against Blender's own property definitions (numbers, switches and menu
choices; never text paths or pointers), and images load only from the allowed folders.
"""

import ctypes.util
import os
import re
import sys

import bpy

from .actions import OBJ, REQUIRED, _check_path, _leave_edit_mode, _obj
from .protocol import BridgeCommandError
from .rig import DATA_PATH

TREES = ("material", "world", "group", "geometry")
PREFIXES = ("ShaderNode", "GeometryNode", "FunctionNode", "CompositorNode")
EXTRA_NODES = {"NodeReroute", "NodeFrame", "NodeGroupInput", "NodeGroupOutput"}
BLOCKED = ("Script", "Import", "OutputFile", "Python", "Export")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr", ".hdr")
MAX_NODES = 300
SOCKET_TYPES = ("NodeSocketFloat", "NodeSocketInt", "NodeSocketBool", "NodeSocketVector", "NodeSocketColor",
                "NodeSocketShader", "NodeSocketGeometry", "NodeSocketObject", "NodeSocketCollection",
                "NodeSocketMaterial", "NodeSocketString", "NodeSocketRotation", "NodeSocketImage")
RAMP_INTERPOLATIONS = ("LINEAR", "CONSTANT", "EASE", "B_SPLINE", "CARDINAL")
INTERPOLATIONS = ("CONSTANT", "LINEAR", "BEZIER")
PROPERTY_TARGETS = ("object", "data", "material", "nodes", "world", "world_nodes", "shape_keys", "texture",
                    "scene", "particles", "group")
LAST = re.compile(r"^(?:(?P<owner>.+)\.)?(?P<attr>[A-Za-z_][A-Za-z0-9_]*)(?:\[(?P<index>\d{1,3})\])?$")


# ------------------------------------------------------------------ values
def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _color(value, param, linear=True):
    """A colour: [r, g, b] or [r, g, b, a] (linear, 0..1 -- brighter than 1 allowed for emission), or "#rrggbb"."""
    if isinstance(value, str):
        text = value.lstrip("#")
        if not re.match(r"^[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$", text):
            raise BridgeCommandError("invalid_param", f"{param}: colour as #rrggbb or [r, g, b]", param=param)
        rgb = [int(text[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        alpha = int(text[6:8], 16) / 255 if len(text) == 8 else 1.0
        return [*(_srgb_to_linear(c) if linear else c for c in rgb), alpha]
    if isinstance(value, (list, tuple)) and len(value) in (3, 4) and all(
            isinstance(c, (int, float)) and not isinstance(c, bool) for c in value):
        if any(c < 0 or c > 1000 for c in value):
            raise BridgeCommandError("invalid_param", f"{param}: colour components 0..1 (up to 1000 for glow)",
                                     param=param)
        return [float(c) for c in value] + ([1.0] if len(value) == 3 else [])
    raise BridgeCommandError("invalid_param", f"{param}: colour as #rrggbb or [r, g, b]", param=param)


def _number(value, param, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeCommandError("invalid_param", f"{param} must be a number", param=param)
    if abs(value) > 1e7:
        raise BridgeCommandError("invalid_param", f"{param} is out of range", param=param)
    return int(value) if integer else float(value)


TEXT_PROPS = ("attribute_name", "layer_name", "uv_map")
STUDIO_LIGHTS = ("city", "courtyard", "forest", "interior", "night", "studio", "sunrise", "sunset")


def _studio_light(name, param):
    """The HDRIs Blender ships for the viewport's look-dev shading (``studio:interior`` ...), usable in a world."""
    if name not in STUDIO_LIGHTS:
        raise BridgeCommandError("invalid_param", f"{param}: studio:<{'|'.join(STUDIO_LIGHTS)}>", param="nodes")
    folder = bpy.utils.system_resource("DATAFILES", path="studiolights/world")
    path = os.path.join(folder or "", f"{name}.exr")
    if not folder or not os.path.exists(path):
        raise BridgeCommandError("invalid_param", f"{param}: this Blender has no studio light {name}", param="nodes")
    return path


ASSET_KINDS = {"MATERIAL": "materials", "OBJECT": "objects", "NODE_GROUP": "node_groups", "WORLD": "worlds",
               "COLLECTION": "collections"}


def mark_asset(p):
    """Outliner > right click > Mark as Asset: the material (or object, node group, world, collection) joins the
    asset library -- saved with the file, it shows in the Asset Browser to drag onto other objects and scenes.
    ``clear`` takes the mark off again."""
    blocks = getattr(bpy.data, ASSET_KINDS[p["kind"]])
    block = blocks.get(p["name"])
    if block is None:
        raise BridgeCommandError("invalid_param", f"no {p['kind'].lower().replace('_', ' ')} {p['name']!r}",
                                 param="name")
    if p["clear"]:
        block.asset_clear()
        return {"name": block.name, "asset": False}
    block.asset_mark()
    data = block.asset_data
    if p["description"]:
        data.description = str(p["description"])[:300]
    for tag in (p["tags"] or [])[:16]:
        if isinstance(tag, str) and tag and tag not in data.tags:
            data.tags.new(tag[:63])
    try:
        block.asset_generate_preview()
    except Exception:  # needs a GPU context; the Asset Browser makes one later
        pass
    assets = sorted(b.name for kind in ASSET_KINDS.values() for b in getattr(bpy.data, kind) if b.asset_data)
    return {"name": block.name, "asset": True, "tags": [t.name for t in data.tags], "assets": assets[:100]}


def _rna_value(owner, attr, value, param, index=None, degrees=False):
    """``value`` checked against the property's definition: a number, a switch, a menu choice or an array of
    numbers (colours may be #rrggbb). Text and pointer properties are refused."""
    prop = owner.bl_rna.properties.get(attr)
    if prop is None or attr in ("rna_type", "name"):
        raise BridgeCommandError("invalid_param", f"{param}: no setting {attr!r} on {owner.bl_rna.identifier}",
                                 param=param)
    if prop.is_readonly:
        raise BridgeCommandError("invalid_param", f"{param}: {attr} is read-only", param=param)
    kind = prop.type
    if kind == "ENUM":
        items = {i.identifier for i in prop.enum_items}
        if prop.is_enum_flag:
            values = set(value if isinstance(value, (list, tuple)) else [value])
            if not values <= items:
                raise BridgeCommandError("invalid_param", f"{param}: choose from {sorted(items)}", param=param)
            return values
        if value not in items:
            raise BridgeCommandError("invalid_param", f"{param}: {value!r} is not one of {sorted(items)}",
                                     param=param)
        return value
    if kind not in ("BOOLEAN", "INT", "FLOAT"):
        raise BridgeCommandError("invalid_param", f"{param}: {attr} ({kind.lower()}) can't be set by a step",
                                 param=param)
    length = getattr(prop, "array_length", 0)
    scale = 3.141592653589793 / 180 if degrees else 1.0

    def one(v):
        if kind == "BOOLEAN":
            if not isinstance(v, bool):
                raise BridgeCommandError("invalid_param", f"{param} must be true or false", param=param)
            return v
        n = _number(v, param, integer=kind == "INT")
        if kind == "FLOAT":
            n *= scale
            lo, hi = prop.hard_min, prop.hard_max
            if n < lo or n > hi:
                raise BridgeCommandError("invalid_param", f"{param} must be within {lo:g}..{hi:g}", param=param)
        return n

    if length and index is None:
        if prop.subtype in ("COLOR", "COLOR_GAMMA") and (isinstance(value, str) or len(value or []) in (3, 4)):
            color = _color(value, param, linear=prop.subtype == "COLOR")
            return color[:length]
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise BridgeCommandError("invalid_param", f"{param} needs {length} values", param=param)
        return [one(v) for v in value]
    if index is not None and not length:
        raise BridgeCommandError("invalid_param", f"{param}: {attr} is not an array", param=param)
    if index is not None and index >= length:
        raise BridgeCommandError("invalid_param", f"{param}: index {index} past the {length} values", param=param)
    return one(value)


# ------------------------------------------------------------------ node trees
# Names tutorials (and older Blenders) use for nodes that were merged or renamed.
NODE_ALIASES = {"ShaderNodeBsdfGlossy": "ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfVelvet": "ShaderNodeBsdfSheen",
                "ShaderNodeMixRGB": "ShaderNodeMix", "ShaderNodeSeparateRGB": "ShaderNodeSeparateColor",
                "ShaderNodeCombineRGB": "ShaderNodeCombineColor"}


def _node_type(name):
    if isinstance(name, str) and getattr(bpy.types, name, None) is None and name in NODE_ALIASES:
        name = NODE_ALIASES[name]
    if name == "ShaderNodeTexMusgrave":
        raise BridgeCommandError("invalid_param", "the Musgrave texture was folded into ShaderNodeTexNoise "
                                 "(props normalize false, the noise_type choice)", param="nodes")
    if not isinstance(name, str) or not (name.startswith(PREFIXES) or name in EXTRA_NODES) \
            or any(b in name for b in BLOCKED):
        raise BridgeCommandError("invalid_param", f"node type {name!r} is not allowed (ShaderNode..., "
                                 "GeometryNode..., FunctionNode..., NodeReroute)", param="nodes")
    cls = getattr(bpy.types, name, None)
    if cls is None or not issubclass(cls, bpy.types.Node):
        raise BridgeCommandError("invalid_param", f"unknown node type {name!r}", param="nodes")
    return name


def _group_io(group):
    if not any(n.bl_idname == "NodeGroupInput" for n in group.nodes):
        node = group.nodes.new("NodeGroupInput")
        node.location = (-400, 0)
    if not any(n.bl_idname == "NodeGroupOutput" for n in group.nodes):
        node = group.nodes.new("NodeGroupOutput")
        node.location = (400, 0)


def _interface(group, specs):
    for spec in specs or []:
        if not isinstance(spec, dict) or not spec.get("name"):
            raise BridgeCommandError("invalid_param", "interface items are {name, in_out, type}", param="interface")
        in_out = spec.get("in_out", "INPUT")
        socket_type = spec.get("type", "NodeSocketFloat")
        if in_out not in ("INPUT", "OUTPUT") or socket_type not in SOCKET_TYPES:
            raise BridgeCommandError("invalid_param", f"in_out INPUT|OUTPUT, type one of {SOCKET_TYPES}",
                                     param="interface")
        name = str(spec["name"])[:63]
        item = next((i for i in group.interface.items_tree if getattr(i, "item_type", "") == "SOCKET"
                     and i.name == name and i.in_out == in_out), None)
        if item is None:
            item = group.interface.new_socket(name, in_out=in_out, socket_type=socket_type)
        for key in ("default", "min", "max"):
            if key in spec and spec[key] is not None:
                attr = {"default": "default_value", "min": "min_value", "max": "max_value"}[key]
                if not hasattr(item, attr):
                    continue
                value = spec[key]
                if socket_type == "NodeSocketColor" and key == "default":
                    value = _color(value, "interface")
                setattr(item, attr, value)


def _assign(obj, material, mode):
    if obj.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT"):
        raise BridgeCommandError("invalid_param", f"{obj.name} can't have a material", param="object")
    if mode == "replace":
        if not obj.material_slots:
            obj.data.materials.append(material)
        for slot in obj.material_slots:
            slot.material = material
    elif mode == "append" and material.name not in [s.material.name for s in obj.material_slots if s.material]:
        obj.data.materials.append(material)


def _tree(p):
    kind = p["tree"]
    if kind == "material":
        obj = _obj(p["object"]) if p["object"] else None
        name = p["material"] or (obj.active_material.name if obj is not None and obj.active_material else None)
        if not name:
            raise BridgeCommandError("invalid_param", "give the material's name", param="material")
        material = bpy.data.materials.get(name)
        if material is None:
            if p["copy_from"]:
                source = bpy.data.materials.get(p["copy_from"])
                if source is None:
                    raise BridgeCommandError("invalid_param", f"material {p['copy_from']!r} not found",
                                             param="copy_from")
                material = source.copy()   # the "2" next to the material: a single-user copy
                material.name = name
            else:
                material = bpy.data.materials.new(name)
        material.use_nodes = True
        if obj is not None and p["assign"] != "none":
            _assign(obj, material, p["assign"])
        return material.node_tree, material.name
    if kind == "world":
        scene = bpy.context.scene
        if scene.world is None:
            scene.world = bpy.data.worlds.new("World")
        scene.world.use_nodes = True
        return scene.world.node_tree, scene.world.name
    if kind == "group":
        name = p["group"]
        if not name:
            raise BridgeCommandError("invalid_param", "give the node group's name", param="group")
        group = bpy.data.node_groups.get(name)
        if group is None:
            group = bpy.data.node_groups.new(name, p["group_type"])
            _group_io(group)
        _interface(group, p["interface"])
        return group, group.name
    # geometry nodes: the object's Geometry Nodes modifier and its node group
    obj = _obj(p["object"])
    name = p["modifier"] or "GeometryNodes"
    modifier = obj.modifiers.get(name)
    if modifier is None:
        modifier = obj.modifiers.new(name, "NODES")
    elif modifier.type != "NODES":
        raise BridgeCommandError("invalid_param", f"modifier {name!r} is not Geometry Nodes", param="modifier")
    if modifier.node_group is None:
        group = bpy.data.node_groups.get(p["group"]) if p["group"] else None
        if group is None:
            group = bpy.data.node_groups.new(p["group"] or f"{obj.name} Nodes", "GeometryNodeTree")
            group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            _group_io(group)
            gin = next(n for n in group.nodes if n.bl_idname == "NodeGroupInput")
            gout = next(n for n in group.nodes if n.bl_idname == "NodeGroupOutput")
            group.links.new(gin.outputs[0], gout.inputs[0])
        modifier.node_group = group
    _interface(modifier.node_group, p["interface"])
    return modifier.node_group, modifier.node_group.name


def _socket(sockets, key, node, param):
    if isinstance(key, int) and not isinstance(key, bool):
        if not 0 <= key < len(sockets):
            raise BridgeCommandError("invalid_param", f"{node.name} has no socket {key}", param=param)
        return sockets[key]
    if not isinstance(key, str):
        raise BridgeCommandError("invalid_param", f"{param}: a socket name or number", param=param)
    shown = [s for s in sockets if s.enabled and not getattr(s, "hide", False)]
    for group in (shown, list(sockets)):
        for s in group:
            if s.name == key or s.identifier == key:
                return s
    for s in shown:
        if s.name.lower() == key.lower():
            return s
    names = sorted({s.name for s in shown})
    raise BridgeCommandError("invalid_param", f"{node.name} has no socket {key!r} (has: {', '.join(names)})",
                             param=param)


def _datablock(kind, name, param):
    collection = {"OBJECT": bpy.data.objects, "COLLECTION": bpy.data.collections, "MATERIAL": bpy.data.materials,
                  "IMAGE": bpy.data.images}[kind]
    block = collection.get(name) if isinstance(name, str) else None
    if block is None:
        raise BridgeCommandError("invalid_param", f"{param}: {kind.lower()} {name!r} not found", param=param)
    return block


def _set_socket(socket, value, param, degrees=False):
    kind = socket.type
    if kind in ("SHADER", "GEOMETRY", "CUSTOM") or not hasattr(socket, "default_value"):
        raise BridgeCommandError("invalid_param", f"{param}: a {kind.lower()} socket takes a link, not a value",
                                 param=param)
    scale = 3.141592653589793 / 180 if degrees else 1.0
    if kind == "VALUE":
        socket.default_value = _number(value, param) * scale
    elif kind == "INT":
        socket.default_value = _number(value, param, integer=True)
    elif kind == "BOOLEAN":
        if not isinstance(value, bool):
            raise BridgeCommandError("invalid_param", f"{param} must be true or false", param=param)
        socket.default_value = value
    elif kind in ("VECTOR", "ROTATION"):
        size = len(socket.default_value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = [value] * size
        if not isinstance(value, (list, tuple)) or len(value) != size:
            raise BridgeCommandError("invalid_param", f"{param} needs {size} numbers", param=param)
        socket.default_value = [_number(v, param) * scale for v in value]
    elif kind == "RGBA":
        socket.default_value = _color(value, param)
    elif kind in ("STRING", "MENU"):
        if not isinstance(value, str) or len(value) > 1024:
            raise BridgeCommandError("invalid_param", f"{param} must be text (up to 1024 characters)", param=param)
        socket.default_value = value
    elif kind in ("OBJECT", "COLLECTION", "MATERIAL", "IMAGE"):
        socket.default_value = _datablock(kind, value, param)
    else:
        raise BridgeCommandError("invalid_param", f"{param}: {kind.lower()} sockets can't be set by a step",
                                 param=param)


def _ramp(node, stops, interpolation, param):
    ramp = getattr(node, "color_ramp", None)
    if ramp is None:
        raise BridgeCommandError("invalid_param", f"{node.name} has no colour ramp", param=param)
    if stops is not None:
        if not isinstance(stops, list) or not 1 <= len(stops) <= 32:
            raise BridgeCommandError("invalid_param", f"{param}: 1..32 stops [position, colour]", param=param)
        clean = []
        for stop in stops:
            if not isinstance(stop, (list, tuple)) or len(stop) != 2:
                raise BridgeCommandError("invalid_param", f"{param}: each stop is [position, colour]", param=param)
            position = _number(stop[0], param)
            if not 0 <= position <= 1:
                raise BridgeCommandError("invalid_param", f"{param}: positions 0..1", param=param)
            clean.append((position, _color(stop[1], param)))
        elements = ramp.elements
        while len(elements) > len(clean):
            elements.remove(elements[-1])
        while len(elements) < len(clean):
            elements.new(0.5)
        # positions first (Blender keeps the stops sorted), then the colours in that order
        for element, (position, _) in zip(list(elements), sorted(clean)):
            element.position = position
        for element, (_, color) in zip(sorted(elements, key=lambda e: e.position), sorted(clean)):
            element.color = color
    if interpolation is not None:
        if interpolation not in RAMP_INTERPOLATIONS:
            raise BridgeCommandError("invalid_param", f"ramp interpolation one of {RAMP_INTERPOLATIONS}",
                                     param=param)
        ramp.interpolation = interpolation


def _layout(tree, placed):
    """New nodes without a location go in columns by how far they are from the output (like Node Arrange)."""
    new = [n for n in tree.nodes if n.name not in placed]
    if not new:
        return
    downstream = {n.name: set() for n in tree.nodes}
    for link in tree.links:
        downstream[link.from_node.name].add(link.to_node.name)
    depth = {}

    def walk(name, seen=()):
        if name in depth:
            return depth[name]
        if name in seen:
            return 0
        after = [walk(d, (*seen, name)) + 1 for d in downstream[name]]
        depth[name] = max(after, default=0)
        return depth[name]

    sinks_x = [n.location.x for n in tree.nodes if not downstream[n.name] and n.name in placed]
    right = max(sinks_x, default=300.0)
    rows = {}
    for node in new:
        column = walk(node.name)
        row = rows.setdefault(column, 0)
        node.location = (right - 280 * column, 250 - 220 * row)
        rows[column] = row + 1


def edit_nodes(p):
    """The node editor: add or change nodes (type, values on their inputs, settings such as a Math node's
    operation, a colour ramp's stops, an image, a texture coordinate's object), connect outputs to inputs, remove
    nodes or links. Trees: a material (made if new, assigned to ``object``), the world, a node group (with its
    interface), or an object's Geometry Nodes modifier."""
    _leave_edit_mode()
    tree, owner = _tree(p)
    if p["clear"]:
        for node in list(tree.nodes):
            if node.bl_idname not in ("NodeGroupInput", "NodeGroupOutput"):
                tree.nodes.remove(node)
    for name in p["remove"] or []:
        node = tree.nodes.get(name)
        if node is None:
            raise BridgeCommandError("invalid_param", f"no node {name!r} to remove", param="remove")
        tree.nodes.remove(node)
    specs = p["nodes"] or []
    if not isinstance(specs, list) or len(tree.nodes) + len(specs) > MAX_NODES:
        raise BridgeCommandError("invalid_param", f"nodes: a list (a tree holds up to {MAX_NODES})", param="nodes")
    placed = {n.name for n in tree.nodes}
    made = []
    for i, spec in enumerate(specs):
        param = f"nodes[{i}]"
        if not isinstance(spec, dict):
            raise BridgeCommandError("invalid_param", f"{param} must be an object", param="nodes")
        name = spec.get("name")
        node = tree.nodes.get(name) if name else None
        if node is None:
            if not spec.get("type"):
                raise BridgeCommandError("invalid_param", f"{param}: a new node needs a type", param="nodes")
            try:
                node = tree.nodes.new(_node_type(spec["type"]))
            except RuntimeError as exc:
                raise BridgeCommandError("invalid_param", f"{param}: {spec['type']} can't go in this tree "
                                         f"({exc})", param="nodes") from exc
            if name:
                node.name = str(name)[:63]
            made.append(node.name)
        elif spec.get("type") and node.bl_idname != _node_type(spec["type"]):
            raise BridgeCommandError("invalid_param", f"{param}: {name!r} is a {node.bl_idname}", param="nodes")
        if spec.get("location") is not None:
            loc = spec["location"]
            if not isinstance(loc, (list, tuple)) or len(loc) != 2:
                raise BridgeCommandError("invalid_param", f"{param}: location is [x, y]", param="nodes")
            node.location = (_number(loc[0], param), _number(loc[1], param))
            placed.add(node.name)
        if spec.get("label") is not None:
            node.label = str(spec["label"])[:63]
        if spec.get("mute") is not None:
            node.mute = bool(spec["mute"])
        for key, value in (spec.get("props") or {}).items():
            degrees = isinstance(key, str) and key.endswith("_deg")   # sun_elevation_deg: 30
            attr = key[:-4] if degrees else key
            setattr(node, attr, _rna_value(node, attr, value, f"{param}.props.{key}", degrees=degrees))
        if "ramp" in spec or "ramp_interpolation" in spec:
            _ramp(node, spec.get("ramp"), spec.get("ramp_interpolation"), f"{param}.ramp")
        if spec.get("image") is not None:
            if not hasattr(node, "image"):
                raise BridgeCommandError("invalid_param", f"{param}: {node.bl_idname} takes no image", param="nodes")
            if isinstance(spec["image"], str) and spec["image"].startswith("studio:"):
                path = _studio_light(spec["image"][7:], param)   # one of the HDRIs that ship with Blender
            elif isinstance(spec["image"], str) and spec["image"].startswith("textures/"):
                from .actions import _texture_path

                path = _texture_path(spec["image"], write=False)   # a texture a bake_texture step made
            else:
                path = _check_path(spec["image"], "LUCIUS_ALLOWED_READ_DIRS", IMAGE_EXTENSIONS)
            node.image = bpy.data.images.load(path, check_existing=True)
            if spec.get("non_color"):
                node.image.colorspace_settings.name = "Non-Color"
            if spec.get("projection"):
                if spec["projection"] not in ("FLAT", "BOX", "SPHERE", "TUBE"):
                    raise BridgeCommandError("invalid_param", f"{param}: projection FLAT|BOX|SPHERE|TUBE",
                                             param="nodes")
                node.projection = spec["projection"]
        if spec.get("generated_image") is not None:
            # Image > New: a picture Blender makes itself (a UV grid / colour grid to check unwraps, or a colour)
            gen = spec["generated_image"]
            if not hasattr(node, "image") or not isinstance(gen, dict):
                raise BridgeCommandError("invalid_param", f"{param}: generated_image {{name, type, width, height}} "
                                         "on an image node", param="nodes")
            kind = gen.get("type", "UV_GRID")
            if kind not in ("UV_GRID", "COLOR_GRID", "BLANK"):
                raise BridgeCommandError("invalid_param", f"{param}: type UV_GRID|COLOR_GRID|BLANK", param="nodes")
            width, height = int(gen.get("width", 1024)), int(gen.get("height", 1024))
            if not (16 <= width <= 4096 and 16 <= height <= 4096):
                raise BridgeCommandError("invalid_param", f"{param}: 16..4096 pixels", param="nodes")
            name = str(gen.get("name") or "Generated")[:63]
            image = bpy.data.images.get(name) or bpy.data.images.new(name, width, height)
            image.generated_type = kind
            if gen.get("color") is not None:
                image.generated_color = _color(gen["color"], param)
            node.image = image
        for key, value in (spec.get("text") or {}).items():
            # the name fields of a node: an Attribute node's attribute, a Color Attribute / UV Map node's layer
            if key not in TEXT_PROPS or not hasattr(node, key):
                raise BridgeCommandError("invalid_param", f"{param}: text {key!r} is not one of {TEXT_PROPS} on "
                                         f"{node.bl_idname}", param="nodes")
            if not isinstance(value, str) or len(value) > 63:
                raise BridgeCommandError("invalid_param", f"{param}: {key} is a name (up to 63 characters)",
                                         param="nodes")
            setattr(node, key, value)
        if spec.get("object") is not None:
            if not hasattr(node, "object"):
                raise BridgeCommandError("invalid_param", f"{param}: {node.bl_idname} takes no object", param="nodes")
            node.object = _obj(spec["object"])
        if spec.get("group") is not None:
            group = bpy.data.node_groups.get(spec["group"])
            if not hasattr(node, "node_tree") or group is None or group.bl_idname != tree.bl_idname:
                raise BridgeCommandError("invalid_param", f"{param}: node group {spec['group']!r} not found for "
                                         "this editor", param="nodes")
            node.node_tree = group
        for field, sockets in (("inputs", node.inputs), ("outputs", node.outputs)):
            for key, value in (spec.get(field) or {}).items():
                degrees = isinstance(key, str) and key.endswith("_deg")
                socket_key = key[:-4] if degrees else key
                if isinstance(socket_key, str) and socket_key.isdigit():
                    socket_key = int(socket_key)
                socket = _socket(sockets, socket_key, node, f"{param}.{field}.{key}")
                _set_socket(socket, value, f"{param}.{field}.{key}", degrees=degrees)
    for i, link in enumerate(p["links"] or []):
        param = f"links[{i}]"
        if not isinstance(link, dict) or "from" not in link or "to" not in link:
            raise BridgeCommandError("invalid_param", f"{param}: {{from, output, to, input}}", param="links")
        a, b = tree.nodes.get(link["from"]), tree.nodes.get(link["to"])
        if a is None or b is None:
            missing = link["from"] if a is None else link["to"]
            raise BridgeCommandError("invalid_param", f"{param}: no node {missing!r}", param="links")
        out = _socket(a.outputs, link.get("output", 0), a, param)
        inp = _socket(b.inputs, link.get("input", 0), b, param)
        tree.links.new(out, inp)
    for i, cut in enumerate(p["unlink"] or []):
        param = f"unlink[{i}]"
        node = tree.nodes.get(cut.get("to")) if isinstance(cut, dict) else None
        if node is None:
            raise BridgeCommandError("invalid_param", f"{param}: {{to, input}} of an existing node", param="unlink")
        socket = _socket(node.inputs, cut.get("input", 0), node, param)
        for link in list(socket.links):
            tree.links.remove(link)
    _layout(tree, placed)
    bpy.context.view_layer.update()
    dangling = []
    for node in tree.nodes:
        if node.bl_idname in ("ShaderNodeOutputMaterial", "ShaderNodeOutputWorld", "NodeGroupOutput") and \
                node.inputs and not any(s.is_linked for s in node.inputs):
            dangling.append(node.name)
    return {"tree": tree.name, "owner": owner, "made": made, "nodes": len(tree.nodes), "links": len(tree.links),
            "unconnected_outputs": dangling}


# ------------------------------------------------------------------ any property
def _property_id(p):
    target, name = p["target"], p["name"]
    scene = bpy.context.scene
    if target == "scene":
        return scene
    if target in ("world", "world_nodes"):
        if scene.world is None:
            raise BridgeCommandError("invalid_param", "the scene has no world", param="target")
        if target == "world_nodes" and scene.world.node_tree is None:
            raise BridgeCommandError("invalid_param", "the world has no nodes", param="target")
        return scene.world if target == "world" else scene.world.node_tree
    if not name:
        raise BridgeCommandError("invalid_param", f"give the {target}'s name", param="name")
    if target in ("material", "nodes"):
        material = bpy.data.materials.get(name)
        if material is None:
            raise BridgeCommandError("invalid_param", f"material {name!r} not found", param="name")
        if target == "nodes" and material.node_tree is None:
            raise BridgeCommandError("invalid_param", f"{name} has no nodes", param="name")
        return material if target == "material" else material.node_tree
    if target == "texture":
        texture = bpy.data.textures.get(name)
        if texture is None:
            raise BridgeCommandError("invalid_param", f"texture {name!r} not found", param="name")
        return texture
    if target == "group":
        group = bpy.data.node_groups.get(name)
        if group is None:
            raise BridgeCommandError("invalid_param", f"node group {name!r} not found", param="name")
        return group
    obj = _obj(name)
    if target == "object":
        return obj
    if target == "data":
        if obj.data is None:
            raise BridgeCommandError("invalid_param", f"{obj.name} has no data", param="name")
        return obj.data
    if target == "shape_keys":
        keys = getattr(obj.data, "shape_keys", None)
        if keys is None:
            raise BridgeCommandError("invalid_param", f"{obj.name} has no shape keys", param="name")
        return keys
    systems = getattr(obj, "particle_systems", None)
    if not systems:
        raise BridgeCommandError("invalid_param", f"{obj.name} has no particle system", param="name")
    return systems.active.settings


def set_property(p):
    """Any setting of the scene by its property path (the tooltip's Python path), and with ``frame`` keyed there
    (hover it, I): a modifier's strength, a curve's bevel end, a material's blend mode, a colour ramp handle's
    position, a shape key's value, a texture's size, a render setting. Numbers, switches and menu choices only."""
    _leave_edit_mode()
    block = _property_id(p)
    path = p["path"]
    match = LAST.match(path) if DATA_PATH.match(path) and "__" not in path else None
    if match is None:
        raise BridgeCommandError("invalid_param", "path must be a property path like modifiers[\"Displace\"]."
                                 "strength or location[2]", param="path")
    owner_path, attr = match.group("owner"), match.group("attr")
    index = int(match.group("index")) if match.group("index") is not None else None
    try:
        owner = block.path_resolve(owner_path) if owner_path else block
    except ValueError as exc:
        raise BridgeCommandError("invalid_param", f"{owner_path} not found on {block.name}", param="path") from exc
    if not hasattr(owner, "bl_rna"):
        raise BridgeCommandError("invalid_param", f"{owner_path} is not a settings block", param="path")
    value = _rna_value(owner, attr, p["value"], "value", index=index, degrees=p["degrees"])
    if p["frame"] is not None:
        bpy.context.scene.frame_set(p["frame"])   # go to the frame, then change the value and key it
    if index is None:
        setattr(owner, attr, value)
    else:
        getattr(owner, attr)[index] = value
    keyed = False
    if p["frame"] is not None:
        full = f"{owner_path}.{attr}" if owner_path else attr
        try:
            block.keyframe_insert(data_path=full, index=-1 if index is None else index, frame=p["frame"])
        except (TypeError, RuntimeError) as exc:
            raise BridgeCommandError("invalid_param", f"{path} can't be keyframed here ({exc})", param="path") from exc
        from .anim import _fcurves

        for curve in _fcurves(block):
            if curve.data_path == full and (index is None or curve.array_index == index):
                for point in curve.keyframe_points:
                    if abs(point.co.x - p["frame"]) < 1e-4:
                        point.interpolation = p["interpolation"]
                curve.update()
        keyed = True
    bpy.context.view_layer.update()
    current = getattr(owner, attr)
    if index is not None:
        current = current[index]
    elif hasattr(current, "__len__") and not isinstance(current, str):
        current = [round(v, 5) if isinstance(v, float) else v for v in current]
    return {"target": p["target"], "owner": block.name, "path": path,
            "value": round(current, 5) if isinstance(current, float) else
            (sorted(current) if isinstance(current, set) else current), "keyed": keyed}


# ------------------------------------------------------------------ shape keys and curves
def shape_key(p):
    """Shape keys (Object Data > Shape Keys): the object's Basis is made first; a new key becomes the active one,
    so the edit-mode steps that follow shape it; its value (0 = Basis, 1 = the key), range and a key on a frame."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    if obj.type not in ("MESH", "CURVE", "SURFACE", "LATTICE"):
        raise BridgeCommandError("invalid_param", f"{obj.name} can't have shape keys", param="object")
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    blocks = obj.data.shape_keys.key_blocks
    key = blocks.get(p["name"])
    made = key is None
    if made:
        key = obj.shape_key_add(name=p["name"], from_mix=False)
    if made or p["active"]:
        obj.active_shape_key_index = list(blocks).index(key)
    if p["frame"] is not None:
        bpy.context.scene.frame_set(p["frame"])
    for field in ("slider_min", "slider_max"):
        if p[field] is not None:
            if not -10 <= p[field] <= 10:
                raise BridgeCommandError("invalid_param", f"{field} within -10..10", param=field)
            setattr(key, field, p[field])
    if p["value"] is not None:
        if not key.slider_min <= p["value"] <= key.slider_max:
            raise BridgeCommandError("invalid_param", f"value within {key.slider_min:g}..{key.slider_max:g}",
                                     param="value")
        key.value = p["value"]
    if p["frame"] is not None:
        key.keyframe_insert("value", frame=p["frame"])
        from .anim import _fcurves

        for curve in _fcurves(obj.data.shape_keys):
            if curve.data_path == f'key_blocks["{key.name}"].value':
                for point in curve.keyframe_points:
                    if abs(point.co.x - p["frame"]) < 1e-4:
                        point.interpolation = p["interpolation"]
                curve.update()
    bpy.context.view_layer.update()
    return {"object": obj.name, "key": key.name, "made": made, "value": round(key.value, 4),
            "active": obj.active_shape_key.name if obj.active_shape_key else None,
            "keys": [b.name for b in blocks]}


HANDLE_TYPES = ("AUTO", "VECTOR", "ALIGNED", "FREE")


def add_curve(p):
    """Shift+A > Curve, drawn point by point: one curve object holding one or more splines (Bezier with auto or
    vector handles, or poly), with a thickness (bevel depth), per-point radius (taper), and fill."""
    _leave_edit_mode()
    splines = p["splines"]
    if not isinstance(splines, list) or not 1 <= len(splines) <= 100:
        raise BridgeCommandError("invalid_param", "splines: 1..100 of {points, handle, radius, cyclic}",
                                 param="splines")
    name = p["name"] or "Curve"
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = p["resolution"]
    data.bevel_depth = p["bevel_depth"]
    data.bevel_resolution = p["bevel_resolution"]
    data.extrude = p["extrude"]
    data.use_fill_caps = p["fill_caps"]
    for i, spec in enumerate(splines):
        param = f"splines[{i}]"
        points = spec.get("points") if isinstance(spec, dict) else None
        if not isinstance(points, list) or not 2 <= len(points) <= 500:
            raise BridgeCommandError("invalid_param", f"{param}: 2..500 points [x, y, z]", param="splines")
        coords = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 3:
                raise BridgeCommandError("invalid_param", f"{param}: points are [x, y, z]", param="splines")
            coords.append([_number(v, param) for v in pt])
        radius = spec.get("radius")
        if radius is not None and (not isinstance(radius, list) or len(radius) != len(coords)):
            raise BridgeCommandError("invalid_param", f"{param}: one radius per point", param="splines")
        kind = spec.get("type", "BEZIER")
        handle = spec.get("handle", "AUTO")
        if kind not in ("BEZIER", "POLY") or handle not in HANDLE_TYPES:
            raise BridgeCommandError("invalid_param", f"{param}: type BEZIER|POLY, handle {HANDLE_TYPES}",
                                     param="splines")
        spline = data.splines.new(kind)
        if kind == "BEZIER":
            spline.bezier_points.add(len(coords) - 1)
            for j, (bp, co) in enumerate(zip(spline.bezier_points, coords)):
                bp.co = co
                bp.handle_left_type = bp.handle_right_type = handle
                bp.radius = _number(radius[j], param) if radius else 1.0
        else:
            spline.points.add(len(coords) - 1)
            for j, (pt, co) in enumerate(zip(spline.points, coords)):
                pt.co = (*co, 1.0)
                pt.radius = _number(radius[j], param) if radius else 1.0
        spline.use_cyclic_u = bool(spec.get("cyclic", False))
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = p["location"]
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return {"object": obj.name, "splines": len(data.splines),
            "points": sum(len(s.bezier_points) or len(s.points) for s in data.splines)}


# ------------------------------------------------------------------ render engine
def eevee_available():
    """Eevee needs an OpenGL/EGL context; headless Linux has one only with EGL installed (e.g. Mesa)."""
    if not bpy.app.background or not sys.platform.startswith("linux"):
        return True
    return ctypes.util.find_library("EGL") is not None


def chosen_engine(scene):
    """The engine a render uses: Eevee when a step chose it (set_render) and it can run here, else Cycles."""
    wanted = scene.get("lucius_engine")
    if wanted in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT") and eevee_available():
        return "BLENDER_EEVEE"
    if wanted == "BLENDER_WORKBENCH":
        return "BLENDER_WORKBENCH"
    return "CYCLES"


TEXT_ALIGN_X = ("LEFT", "CENTER", "RIGHT", "JUSTIFY", "FLUSH")
TEXT_ALIGN_Y = ("TOP_BASELINE", "TOP", "CENTER", "BOTTOM", "BOTTOM_BASELINE")


def add_text(p):
    """Shift+A > Text, typed in edit mode (a new line with Enter: ``\n``), and its Object Data settings: geometry
    extrude and bevel (depth, resolution) for 3D letters, size and shear (slant), paragraph alignment and
    character / word / line spacing, preview resolution, fill mode, and Text on Curve. An existing text object of
    that name changes only the values given."""
    _leave_edit_mode()
    name = p["name"] or "Text"
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type != "FONT":
        raise BridgeCommandError("invalid_param", f"{name} exists and is not a text object", param="name")
    if obj is None:
        data = bpy.data.curves.new(name, "FONT")
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        data.body = "Text"
    data = obj.data
    if p["body"] is not None:
        if not isinstance(p["body"], str) or not 0 < len(p["body"]) <= 500:
            raise BridgeCommandError("invalid_param", "body: 1..500 characters", param="body")
        data.body = p["body"]
    if p["location"] is not None:
        obj.location = p["location"]
    if p["rotation"] is not None:
        obj.rotation_euler = p["rotation"]
    settings = {"extrude": "extrude", "bevel_depth": "bevel_depth", "bevel_resolution": "bevel_resolution",
                "size": "size", "shear": "shear", "space_character": "space_character",
                "space_word": "space_word", "space_line": "space_line", "resolution": "resolution_u",
                "offset_x": "offset_x", "offset_y": "offset_y"}
    for key, attr in settings.items():
        if p[key] is not None:
            setattr(data, attr, _rna_value(data, attr, p[key], key))
    if p["align_x"] is not None:
        data.align_x = p["align_x"]
    if p["align_y"] is not None:
        data.align_y = p["align_y"]
    if p["fill_mode"] is not None:
        data.fill_mode = p["fill_mode"]
    if p["follow_curve"] is not None:
        curve = _obj(p["follow_curve"]) if p["follow_curve"] else None
        if curve is not None and curve.type != "CURVE":
            raise BridgeCommandError("invalid_param", f"{curve.name} is not a curve", param="follow_curve")
        data.follow_curve = curve
    bpy.context.view_layer.update()
    return {"object": obj.name, "body": data.body, "lines": data.body.count("\n") + 1,
            "dimensions": [round(v, 3) for v in obj.dimensions]}


def convert_to_mesh(p):
    """Object > Convert > Mesh: a text or curve (or an object with modifiers, applied) becomes plain editable mesh
    -- vertices, edges and faces to select and give materials; its text settings and their keyframes go."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    if obj.type not in ("FONT", "CURVE", "SURFACE", "META", "MESH"):
        raise BridgeCommandError("invalid_param", f"{obj.name} ({obj.type}) can't become a mesh", param="object")
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    from .actions import _context_override, _op_result

    override = dict(_context_override(obj), selected_objects=[obj], selected_editable_objects=[obj])
    with bpy.context.temp_override(**override):
        _op_result(bpy.ops.object.convert(target="MESH"), "convert")
    obj = bpy.context.view_layer.objects.active or obj
    return {"object": obj.name, "type": obj.type, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)}


ACTIONS = {
    "edit_nodes": (edit_nodes, {
        "tree": (TREES, "material"), "material": ("name", None), "object": ("name", None), "group": ("name", None),
        "group_type": (("ShaderNodeTree", "GeometryNodeTree"), "ShaderNodeTree"), "modifier": ("name", None),
        "copy_from": ("name", None), "assign": (("replace", "append", "none"), "replace"),
        "interface": ("list", None), "clear": ("bool", False), "remove": ("names", None), "nodes": ("list", None),
        "links": ("list", None), "unlink": ("list", None)}),
    "mark_asset": (mark_asset, {"kind": (tuple(ASSET_KINDS), "MATERIAL"), "name": ("name", REQUIRED),
                                "description": ("path_expr", None), "tags": ("list", None), "clear": ("bool", False)}),
    "set_property": (set_property, {
        "target": (PROPERTY_TARGETS, "object"), "name": ("name", None), "path": ("path_expr", REQUIRED),
        "value": ("any", REQUIRED), "degrees": ("bool", False), "frame": ("int", None),
        "interpolation": (INTERPOLATIONS, "BEZIER")}),
    "shape_key": (shape_key, {
        "object": OBJ, "name": ("name", REQUIRED), "value": ("float", None), "slider_min": ("float", None),
        "slider_max": ("float", None), "active": ("bool", False), "frame": ("int", None),
        "interpolation": (INTERPOLATIONS, "BEZIER")}),
    "add_text": (add_text, {
        "name": ("name", None), "body": ("path_expr", None), "location": ("vec3", None), "rotation": ("vec3", None),
        "extrude": ("float", None), "bevel_depth": ("float", None), "bevel_resolution": ("int", None),
        "size": ("float", None), "shear": ("float", None), "space_character": ("float", None),
        "space_word": ("float", None), "space_line": ("float", None), "resolution": ("int", None),
        "offset_x": ("float", None), "offset_y": ("float", None), "align_x": (TEXT_ALIGN_X, None), "align_y": (TEXT_ALIGN_Y, None),
        "fill_mode": (("NONE", "BACK", "FRONT", "BOTH"), None), "follow_curve": ("any", None)}),
    "convert_to_mesh": (convert_to_mesh, {"object": OBJ}),
    "add_curve": (add_curve, {
        "name": ("name", None), "splines": ("list", REQUIRED), "location": ("vec3", [0.0, 0.0, 0.0]),
        "bevel_depth": ("float", 0.0), "bevel_resolution": ("int", 4), "extrude": ("float", 0.0),
        "resolution": ("int", 12), "fill_caps": ("bool", True)}),
}
