"""Rigging: armatures, binding meshes to bones, weights, poses, constraints and drivers.

The operations follow a rigging tutorial: build the skeleton in edit mode (bones with a head, a tail, a
parent, connected or not, deforming or not), bind the model to it (each rigid part to one bone, or a
deforming mesh with automatic or hand-assigned weights), then add controls in pose mode: constraints
such as inverse kinematics, and drivers that let one value move another.
"""

import math
import re

import bpy
from mathutils import Matrix, Vector

from .actions import OBJ, REQUIRED, _context_override, _ensure_mode, _leave_edit_mode, _obj, _op_result
from .protocol import BridgeCommandError

MAX_BONES = 500
BONE_NAME = re.compile(r"^[^\x00-\x1f\"\\]{1,63}$")
CONSTRAINTS = ("IK", "CHILD_OF", "COPY_ROTATION", "COPY_LOCATION", "COPY_SCALE", "COPY_TRANSFORMS", "DAMPED_TRACK",
               "TRACK_TO", "STRETCH_TO", "LIMIT_ROTATION", "LIMIT_LOCATION", "LIMIT_SCALE", "MAINTAIN_VOLUME",
               "FLOOR")
CONSTRAINT_PROPS = {  # extra settings a constraint step may give
    "use_tail": bool, "use_stretch": bool, "iterations": int, "head_tail": float, "use_x": bool, "use_y": bool,
    "use_z": bool, "invert_x": bool, "invert_y": bool, "invert_z": bool, "use_offset": bool, "mix_mode": str,
    "use_limit_x": bool, "use_limit_y": bool, "use_limit_z": bool, "min_x": float, "max_x": float, "min_y": float,
    "max_y": float, "min_z": float, "max_z": float, "owner_space": str, "target_space": str, "track_axis": str,
    "up_axis": str, "free_axis": str, "volume": float, "keep_axis": str, "rest_length": float, "bulge": float,
    "use_min_x": bool, "use_max_x": bool, "use_min_y": bool, "use_max_y": bool, "use_min_z": bool,
    "use_max_z": bool, "use_transform_limit": bool,
}
TRANSFORMS = ("LOC_X", "LOC_Y", "LOC_Z", "ROT_X", "ROT_Y", "ROT_Z", "ROT_W", "SCALE_X", "SCALE_Y", "SCALE_Z",
              "SCALE_AVG")
SPACES = ("WORLD_SPACE", "TRANSFORM_SPACE", "LOCAL_SPACE")
# Driver expressions are arithmetic on the variables: no calls except these, no attributes, no strings.
EXPRESSION = re.compile(r"^[A-Za-z0-9_\s\.\+\-\*/%\(\),<>=!]*$")
EXPRESSION_FUNCS = {"sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sqrt", "abs", "min", "max", "pi",
                    "radians", "degrees", "floor", "ceil", "round", "frame", "pow", "exp", "log", "and", "or", "not",
                    "if", "else", "True", "False", "clamp", "smoothstep", "lerp", "e"}
DATA_PATH = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\[("[^"\\]{1,63}"|\d{1,3})\]|\.[A-Za-z_][A-Za-z0-9_]*)*$')


def _armature(name):
    obj = _obj(name)
    if obj.type != "ARMATURE":
        raise BridgeCommandError("invalid_param", f"{obj.name} is not an armature", param="armature")
    return obj


def _bone_name(value, param):
    if not isinstance(value, str) or not BONE_NAME.match(value):
        raise BridgeCommandError("invalid_param", f"{param} must be a bone name", param=param)
    return value


def _vec(value, param):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 1e6 for v in value):
        raise BridgeCommandError("invalid_param", f"{param} must be [x, y, z]", param=param)
    return Vector(value)


def _bones_spec(bones):
    if not isinstance(bones, list) or not 1 <= len(bones) <= MAX_BONES:
        raise BridgeCommandError("invalid_param", f"bones must be a list of 1..{MAX_BONES} bones", param="bones")
    out = []
    for i, item in enumerate(bones):
        if not isinstance(item, dict):
            raise BridgeCommandError("invalid_param", f"bone {i} must be an object", param="bones")
        unknown = set(item) - {"name", "head", "tail", "parent", "connect", "deform", "roll_deg"}
        if unknown:
            raise BridgeCommandError("invalid_param", f"bone {i}: unknown keys {sorted(unknown)}", param="bones")
        out.append({
            "name": _bone_name(item.get("name"), "bones.name"),
            "head": _vec(item["head"], "bones.head") if item.get("head") is not None else None,
            "tail": _vec(item["tail"], "bones.tail") if item.get("tail") is not None else None,
            "parent": _bone_name(item["parent"], "bones.parent") if item.get("parent") else None,
            "connect": bool(item.get("connect", False)),
            "deform": item.get("deform"),
            "roll": math.radians(float(item.get("roll_deg") or 0.0)),
        })
    return out


def _edit(obj):
    _ensure_mode(obj, "EDIT")
    return obj.data.edit_bones


def add_armature(p):
    """Shift+A > Armature, then build the bones in edit mode: each bone has a head (its pivot), a tail, a
    parent (connected: its head on the parent's tail, like an extruded chain) and whether it deforms the mesh.
    The same armature name again adds or changes bones. Coordinates are the armature's own (local)."""
    _leave_edit_mode()
    bones = _bones_spec(p["bones"])
    obj = bpy.data.objects.get(p["name"]) if p["name"] else None
    if obj is None:
        data = bpy.data.armatures.new(p["name"] or "Armature")
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = p["location"]
    elif obj.type != "ARMATURE":
        raise BridgeCommandError("invalid_param", f"{obj.name} exists and is not an armature", param="name")
    if p["display"] is not None:
        obj.data.display_type = p["display"]
    obj.show_in_front = p["in_front"]
    edit_bones = _edit(obj)
    for spec in bones:
        bone = edit_bones.get(spec["name"]) or edit_bones.new(spec["name"])
        parent = edit_bones.get(spec["parent"]) if spec["parent"] else None
        if spec["parent"] and parent is None:
            raise BridgeCommandError("invalid_param", f"parent bone {spec['parent']!r} of {spec['name']!r} not found "
                                     "(list parents first)", param="bones")
        head = spec["head"] if spec["head"] is not None else (parent.tail.copy() if parent else None)
        if head is None or (spec["tail"] is None and bone.length == 0):
            raise BridgeCommandError("invalid_param", f"bone {spec['name']!r} needs a head and a tail", param="bones")
        bone.head = head
        if spec["tail"] is not None:
            bone.tail = spec["tail"]
        if (bone.tail - bone.head).length < 1e-5:
            raise BridgeCommandError("invalid_param", f"bone {spec['name']!r} has no length", param="bones")
        bone.roll = spec["roll"]
        bone.parent = parent
        bone.use_connect = bool(parent) and spec["connect"]
        if spec["deform"] is not None:
            bone.use_deform = bool(spec["deform"])
    names = [b.name for b in edit_bones]
    _ensure_mode(obj, "OBJECT")
    return {"object": obj.name, "bones": names}


def _mirror_name(name):
    for left, right in ((".L", ".R"), ("_L", "_R"), (".l", ".r"), ("_l", "_r"), (".Left", ".Right")):
        if name.endswith(left):
            return name[: -len(left)] + right
    return None


def symmetrize_bones(p):
    """Right click > Symmetrize: every left bone (``.L``) gets a mirrored right twin (``.R``) across X."""
    obj = _armature(p["armature"])
    edit_bones = _edit(obj)
    made = []
    for bone in [b for b in edit_bones if _mirror_name(b.name)]:
        twin_name = _mirror_name(bone.name)
        twin = edit_bones.get(twin_name) or edit_bones.new(twin_name)
        twin.head = Vector((-bone.head.x, bone.head.y, bone.head.z))
        twin.tail = Vector((-bone.tail.x, bone.tail.y, bone.tail.z))
        twin.roll = -bone.roll
        twin.use_deform = bone.use_deform
        made.append((bone, twin))
    for bone, twin in made:   # parents after every twin exists
        parent = bone.parent
        if parent is not None:
            twin.parent = edit_bones.get(_mirror_name(parent.name) or parent.name) or parent
        twin.use_connect = bone.use_connect
    names = [t.name for _b, t in made]
    _ensure_mode(obj, "OBJECT")
    return {"armature": obj.name, "mirrored": names}


def set_bone(p):
    """Bone properties: move its head or tail, change its parent (empty string clears it), connect it, whether
    it deforms, hide it, its colour, or show it as a custom shape (a control)."""
    obj = _armature(p["armature"])
    name = _bone_name(p["bone"], "bone")
    edit_changes = any(p[k] is not None for k in ("head", "tail", "parent", "connect", "deform"))
    if edit_changes:
        edit_bones = _edit(obj)
        bone = edit_bones.get(name)
        if bone is None:
            raise BridgeCommandError("invalid_param", f"bone {name!r} not found", param="bone")
        if p["head"] is not None:
            bone.head = p["head"]
        if p["tail"] is not None:
            bone.tail = p["tail"]
        if p["parent"] is not None:
            if p["parent"] == "":
                bone.parent = None
            else:
                parent = edit_bones.get(p["parent"])
                if parent is None or parent == bone:
                    raise BridgeCommandError("invalid_param", f"parent bone {p['parent']!r} not found", param="parent")
                bone.parent = parent
        if p["connect"] is not None:
            bone.use_connect = p["connect"] and bone.parent is not None
        if p["deform"] is not None:
            bone.use_deform = p["deform"]
        _ensure_mode(obj, "OBJECT")
    bone = obj.data.bones.get(name)
    if bone is None:
        raise BridgeCommandError("invalid_param", f"bone {name!r} not found", param="bone")
    if p["new_name"]:
        # F2: vertex groups, constraints and bone-parented objects follow the new name.
        bone.name = _bone_name(p["new_name"], "new_name")
        name = bone.name
    if p["hide"] is not None:
        bone.hide = p["hide"]
    if p["color"] is not None:
        bone.color.palette = p["color"]
    if p["shape"] is not None:
        obj.pose.bones[name].custom_shape = _obj(p["shape"]) if p["shape"] else None
    return {"armature": obj.name, "bone": name, "parent": bone.parent.name if bone.parent else None,
            "deform": bone.use_deform, "hidden": bone.hide}


def _bone_matrix(arm, name):
    bpy.context.view_layer.update()
    pose_bone = arm.pose.bones.get(name)
    if pose_bone is None:
        raise BridgeCommandError("invalid_param", f"bone {name!r} not found", param="bone")
    # A child of a bone hangs from its tail.
    return arm.matrix_world @ pose_bone.matrix @ Matrix.Translation((0.0, pose_bone.bone.length, 0.0))


def bind_to_armature(p):
    """Ctrl+P: parent objects to the rig. ``BONE``: each (rigid) object follows one bone. ``AUTOMATIC``: the
    mesh deforms with automatic weights. ``EMPTY``: empty vertex groups named after the bones, to fill with
    weights. ``ENVELOPE``: weights from the bones' envelopes. The objects stay where they are."""
    _leave_edit_mode()
    arm = _armature(p["armature"])
    objects = [_obj(n) for n in p["objects"]]
    if not objects:
        raise BridgeCommandError("invalid_param", "give the objects to bind", param="objects")
    if p["mode"] == "BONE":
        if not p["bone"]:
            raise BridgeCommandError("invalid_param", "BONE mode needs the bone", param="bone")
        parent_matrix = _bone_matrix(arm, p["bone"])
        for obj in objects:
            world = obj.matrix_world.copy()
            obj.parent = arm
            obj.parent_type = "BONE"
            obj.parent_bone = p["bone"]
            obj.matrix_parent_inverse = parent_matrix.inverted()
            obj.matrix_world = world
        bpy.context.view_layer.update()
        return {"armature": arm.name, "bone": p["bone"], "objects": [o.name for o in objects]}
    for obj in objects:
        if obj.type != "MESH":
            raise BridgeCommandError("not_a_mesh", f"{obj.name} is not a mesh (convert curves first)")
    kind = {"AUTOMATIC": "ARMATURE_AUTO", "EMPTY": "ARMATURE_NAME", "ENVELOPE": "ARMATURE_ENVELOPE"}[p["mode"]]
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in (*objects, arm):
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    with bpy.context.temp_override(**_context_override(arm), selected_objects=[*objects, arm],
                                   selected_editable_objects=[*objects, arm]):
        _op_result(bpy.ops.object.parent_set(type=kind), "parent_set")
    groups = {o.name: len(o.vertex_groups) for o in objects}
    return {"armature": arm.name, "mode": p["mode"], "vertex_groups": groups}


def assign_weights(p):
    """Vertex group > Assign: the vertices selected in edit mode get this weight for this bone's group (1 =
    they follow it fully). ``exclusive`` takes them out of every other group (rigid parts)."""
    obj = _obj(p["object"])
    if obj.type != "MESH":
        raise BridgeCommandError("not_a_mesh", f"{obj.name} is not a mesh")
    if obj.mode == "EDIT":
        import bmesh

        bm = bmesh.from_edit_mesh(obj.data)
        indices = [v.index for v in bm.verts if v.select]
        _leave_edit_mode()
    else:
        indices = [v.index for v in obj.data.vertices if v.select]
    if not indices:
        raise BridgeCommandError("empty_selection", "select the vertices first (edit mode)")
    group = obj.vertex_groups.get(p["group"]) or obj.vertex_groups.new(name=p["group"])
    if p["exclusive"]:
        for other in obj.vertex_groups:
            if other != group:
                other.remove(indices)
    group.add(indices, p["weight"], p["mode"])
    return {"object": obj.name, "group": group.name, "vertices": len(indices)}


def _set_key_types(arm, bone, frame, interpolation, handle):
    """The interpolation and handle type of a bone's keys on one frame (T and V on those keys)."""
    from .anim import _fcurves

    prefix = f'pose.bones["{bone}"]'
    for curve in _fcurves(arm):
        if not curve.data_path.startswith(prefix):
            continue
        for point in curve.keyframe_points:
            if abs(point.co.x - frame) < 1e-4:
                if interpolation is not None:
                    point.interpolation = interpolation
                if handle is not None:
                    point.handle_left_type = point.handle_right_type = handle
        curve.update()


def pose_bone(p):
    """Pose mode: move, turn or scale a bone (rotation in degrees, XYZ); with ``frame`` it is keyed there (I),
    with an interpolation and handle type for those keys. ``visual`` (Pose > Apply > Visual Transform) gives the
    bone the pose its constraints give it right now, e.g. to match FK bones to the IK pose before switching."""
    _leave_edit_mode()
    arm = _armature(p["armature"])
    name = _bone_name(p["bone"], "bone")
    pb = arm.pose.bones.get(name)
    if pb is None:
        raise BridgeCommandError("invalid_param", f"bone {name!r} not found", param="bone")
    if p["frame"] is not None:
        bpy.context.scene.frame_set(p["frame"])
    if p["reset"]:
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)
    keyed = []
    if p["visual"]:
        if any(p[k] is not None for k in ("location", "rotation", "scale")):
            raise BridgeCommandError("invalid_param", "visual takes the constraint pose; give no values with it",
                                     param="visual")
        bpy.context.view_layer.update()
        pb.rotation_mode = "XYZ"
        pb.matrix_basis = arm.convert_space(pose_bone=pb, matrix=pb.matrix, from_space="POSE", to_space="LOCAL")
        keyed += ["location", "rotation_euler", "scale"]
    if p["location"] is not None:
        pb.location = p["location"]
        keyed.append("location")
    if p["rotation"] is not None:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = p["rotation"]
        keyed.append("rotation_euler")
    if p["scale"] is not None:
        pb.scale = p["scale"]
        keyed.append("scale")
    if p["frame"] is not None:
        for path in keyed or ["location", "rotation_euler" if pb.rotation_mode == "XYZ" else "rotation_quaternion",
                              "scale"]:
            pb.keyframe_insert(path, frame=p["frame"])
        if p["interpolation"] is not None or p["handle"] is not None:
            _set_key_types(arm, name, p["frame"], p["interpolation"], p["handle"])
    elif p["interpolation"] is not None or p["handle"] is not None:
        raise BridgeCommandError("invalid_param", "interpolation and handle apply to keys: give a frame",
                                 param="frame")
    bpy.context.view_layer.update()
    return {"armature": arm.name, "bone": name, "head": [round(v, 4) for v in arm.matrix_world @ pb.head],
            "tail": [round(v, 4) for v in arm.matrix_world @ pb.tail], "keyed": keyed if p["frame"] else []}


def add_constraint(p):
    """Bone (or object) constraints: rules such as inverse kinematics (IK: the chain reaches for a target bone,
    bending towards a pole), Child Of, copy or limit a transform, track or stretch to a target."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    if p["bone"]:
        if obj.type != "ARMATURE" or obj.pose.bones.get(p["bone"]) is None:
            raise BridgeCommandError("invalid_param", f"bone {p['bone']!r} not found on {obj.name}", param="bone")
        owner = obj.pose.bones[p["bone"]]
    else:
        owner = obj
    # Unnamed: the owner's constraint of that type is changed, or a new one keeps Blender's name ("IK", "Child Of").
    if p["name"]:
        con = owner.constraints.get(p["name"])
    else:
        con = next((c for c in owner.constraints if c.type == p["type"]), None)
    if con is None or con.type != p["type"]:
        con = owner.constraints.new(p["type"])
        if p["name"]:
            con.name = p["name"]
    target = _obj(p["target"]) if p["target"] else None
    if target is not None:
        con.target = target
        if p["subtarget"]:
            if target.type != "ARMATURE" or target.pose.bones.get(p["subtarget"]) is None:
                raise BridgeCommandError("invalid_param", f"bone {p['subtarget']!r} not found", param="subtarget")
            con.subtarget = p["subtarget"]
    if p["type"] == "IK":
        if p["pole_target"]:
            con.pole_target = _obj(p["pole_target"])
            if p["pole_subtarget"]:
                con.pole_subtarget = p["pole_subtarget"]
            con.pole_angle = math.radians(p["pole_angle_deg"])
        con.chain_count = p["chain_count"]
    for key, value in (p["props"] or {}).items():
        kind = CONSTRAINT_PROPS.get(key)
        if kind is None or not hasattr(con, key):
            raise BridgeCommandError("invalid_param", f"setting {key!r} not allowed for {p['type']}", param="props")
        if kind is float:
            value = math.radians(value) if key.startswith(("min_", "max_")) and "ROTATION" in p["type"] else value
        setattr(con, key, kind(value) if kind is not str else str(value))
    con.influence = p["influence"]
    if p["type"] == "CHILD_OF" and target is not None:
        # Set Inverse: the owner stays where it is and from now on follows the target.
        bpy.context.view_layer.update()
        if p["subtarget"]:
            target_matrix = target.matrix_world @ target.pose.bones[p["subtarget"]].matrix
        else:
            target_matrix = target.matrix_world
        con.inverse_matrix = target_matrix.inverted()
        if p["bone"]:
            con.set_inverse_pending = False
    bpy.context.view_layer.update()
    return {"object": obj.name, "bone": p["bone"], "constraint": con.name, "type": con.type}


def key_constraint(p):
    """A constraint's influence, set and keyed on a frame (hover the value, I): e.g. an IK/FK switch, the IK
    constraint on at 1 while a foot is planted and off at 0 while the legs swing freely. Keys are constant by
    default so the switch happens on that frame."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    if p["bone"]:
        if obj.type != "ARMATURE" or obj.pose.bones.get(p["bone"]) is None:
            raise BridgeCommandError("invalid_param", f"bone {p['bone']!r} not found on {obj.name}", param="bone")
        owner = obj.pose.bones[p["bone"]]
    else:
        owner = obj
    con = owner.constraints.get(p["constraint"])
    if con is None:
        have = ", ".join(c.name for c in owner.constraints) or "none"
        raise BridgeCommandError("invalid_param", f"no constraint {p['constraint']!r} (has: {have})",
                                 param="constraint")
    if not 0 <= p["influence"] <= 1:
        raise BridgeCommandError("invalid_param", "influence is 0..1", param="influence")
    bpy.context.scene.frame_set(p["frame"])
    con.influence = p["influence"]
    con.keyframe_insert("influence", frame=p["frame"])
    from .anim import _fcurves

    path = con.path_from_id("influence")
    for curve in _fcurves(obj):
        if curve.data_path == path:
            for point in curve.keyframe_points:
                if abs(point.co.x - p["frame"]) < 1e-4:
                    point.interpolation = p["interpolation"]
            curve.update()
    bpy.context.view_layer.update()
    return {"object": obj.name, "bone": p["bone"], "constraint": con.name, "frame": p["frame"],
            "influence": con.influence}


def _check_expression(expression, variables):
    if not EXPRESSION.match(expression) or len(expression) > 300 or "__" in expression:
        raise BridgeCommandError("invalid_param", "the expression may only use numbers, the variables, arithmetic "
                                 "and a few math functions", param="expression")
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression):
        if word not in variables and word not in EXPRESSION_FUNCS:
            raise BridgeCommandError("invalid_param", f"unknown name {word!r} in the expression", param="expression")


def add_driver(p):
    """Right click > Add Driver: a value computed from other values (a scripted expression over variables that
    read another object's or bone's transform, or any property by its data path)."""
    _leave_edit_mode()
    if p["material"]:
        material = bpy.data.materials.get(p["material"])
        if material is None or material.node_tree is None:
            raise BridgeCommandError("invalid_param", f"material {p['material']!r} not found", param="material")
        obj = material.node_tree   # a node value, e.g. nodes["LuciusMapping"].inputs["Location"].default_value
    else:
        obj = _obj(p["object"])
    path = p["path"]
    if not DATA_PATH.match(path):
        raise BridgeCommandError("invalid_param", "path must be a property path like location or "
                                 "constraints[\"IK\"].influence", param="path")
    if p["bone"]:
        if p["material"] or obj.type != "ARMATURE" or obj.pose.bones.get(p["bone"]) is None:
            raise BridgeCommandError("invalid_param", f"bone {p['bone']!r} not found", param="bone")
        path = f'pose.bones["{p["bone"]}"].{path}'
    variables = p["variables"] or []
    if not isinstance(variables, list) or not 1 <= len(variables) <= 8:
        raise BridgeCommandError("invalid_param", "give 1..8 variables", param="variables")
    names = []
    specs = []
    for i, var in enumerate(variables):
        if not isinstance(var, dict):
            raise BridgeCommandError("invalid_param", f"variable {i} must be an object", param="variables")
        vname = var.get("name") or "var"
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,30}$", vname) or vname in EXPRESSION_FUNCS:
            raise BridgeCommandError("invalid_param", f"bad variable name {vname!r}", param="variables")
        kind = var.get("type", "TRANSFORMS")
        if kind not in ("TRANSFORMS", "SINGLE_PROP"):
            raise BridgeCommandError("invalid_param", "variable type TRANSFORMS or SINGLE_PROP", param="variables")
        source = _obj(var.get("object"))
        if kind == "TRANSFORMS":
            transform = var.get("transform", "LOC_X")
            space = var.get("space", "WORLD_SPACE")
            if transform not in TRANSFORMS or space not in SPACES:
                raise BridgeCommandError("invalid_param", f"transform one of {TRANSFORMS}, space one of {SPACES}",
                                         param="variables")
            bone = var.get("bone")
            if bone and (source.type != "ARMATURE" or source.pose.bones.get(bone) is None):
                raise BridgeCommandError("invalid_param", f"bone {bone!r} not found", param="variables")
            specs.append((vname, kind, source, bone, transform, space, None))
        else:
            var_path = var.get("path", "")
            if not DATA_PATH.match(var_path):
                raise BridgeCommandError("invalid_param", "variable path must be a property path", param="variables")
            specs.append((vname, kind, source, None, None, None, var_path))
        names.append(vname)
    expression = p["expression"] or names[0]
    _check_expression(expression, set(names))
    try:
        fcurve = obj.driver_add(path, p["index"]) if p["index"] is not None else obj.driver_add(path)
    except TypeError as exc:
        raise BridgeCommandError("invalid_param", f"{path} cannot take a driver: {exc}", param="path")
    if isinstance(fcurve, list):
        raise BridgeCommandError("invalid_param", f"{path} is a vector: give its index", param="index")
    driver = fcurve.driver
    driver.type = "SCRIPTED"
    for old in list(driver.variables):
        driver.variables.remove(old)
    for vname, kind, source, bone, transform, space, var_path in specs:
        var = driver.variables.new()
        var.name = vname
        var.type = kind
        target = var.targets[0]
        target.id = source
        if kind == "TRANSFORMS":
            if bone:
                target.bone_target = bone
            target.transform_type = transform
            target.transform_space = space
        else:
            target.data_path = var_path
    driver.expression = expression
    for modifier in list(fcurve.modifiers):   # the default generator would offset the value
        fcurve.modifiers.remove(modifier)
    bpy.context.view_layer.update()
    return {"object": p["material"] or obj.name, "path": path, "index": p["index"], "expression": expression,
            "valid": bool(driver.is_valid)}


ACTIONS = {
    "add_armature": (add_armature, {
        "name": ("name", None), "location": ("vec3", [0.0, 0.0, 0.0]), "bones": ("list", REQUIRED),
        "display": (("OCTAHEDRAL", "STICK", "BBONE", "ENVELOPE", "WIRE"), None), "in_front": ("bool", True)}),
    "symmetrize_bones": (symmetrize_bones, {"armature": ("name", REQUIRED)}),
    "set_bone": (set_bone, {
        "armature": ("name", REQUIRED), "bone": ("name", REQUIRED), "head": ("vec3", None), "tail": ("vec3", None),
        "parent": ("bone_or_empty", None), "connect": ("bool", None), "deform": ("bool", None), "hide": ("bool", None),
        "color": (tuple(f"THEME{i:02d}" for i in range(1, 21)) + ("DEFAULT",), None), "shape": ("bone_or_empty", None),
        "new_name": ("name", None)}),
    "bind_to_armature": (bind_to_armature, {
        "objects": ("names", REQUIRED), "armature": ("name", REQUIRED),
        "mode": (("BONE", "AUTOMATIC", "EMPTY", "ENVELOPE"), "AUTOMATIC"), "bone": ("name", None)}),
    "assign_weights": (assign_weights, {
        "object": OBJ, "group": ("name", REQUIRED), "weight": ("float", 1.0),
        "mode": (("REPLACE", "ADD", "SUBTRACT"), "REPLACE"), "exclusive": ("bool", False)}),
    "pose_bone": (pose_bone, {
        "armature": ("name", REQUIRED), "bone": ("name", REQUIRED), "location": ("vec3", None),
        "rotation": ("vec3", None), "scale": ("vec3", None), "frame": ("int", None), "reset": ("bool", False),
        "visual": ("bool", False), "interpolation": (("BEZIER", "LINEAR", "CONSTANT"), None),
        "handle": (("AUTO_CLAMPED", "AUTO", "VECTOR", "ALIGNED", "FREE"), None)}),
    "key_constraint": (key_constraint, {
        "object": OBJ, "bone": ("name", None), "constraint": ("name", REQUIRED), "influence": ("float", REQUIRED),
        "frame": ("int", REQUIRED), "interpolation": (("CONSTANT", "LINEAR", "BEZIER"), "CONSTANT")}),
    "add_constraint": (add_constraint, {
        "object": OBJ, "bone": ("name", None), "type": (CONSTRAINTS, REQUIRED), "name": ("name", None),
        "target": ("name", None), "subtarget": ("name", None), "pole_target": ("name", None),
        "pole_subtarget": ("name", None), "pole_angle_deg": ("float", 0.0), "chain_count": ("int", 2),
        "influence": ("float", 1.0), "props": ("dict", None)}),
    "add_driver": (add_driver, {
        "object": OBJ, "material": ("name", None), "bone": ("name", None), "path": ("path_expr", REQUIRED),
        "index": ("int", None),
        "expression": ("expression", None), "variables": ("list", REQUIRED)}),
}
