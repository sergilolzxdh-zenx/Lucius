"""Allowlisted Blender API actions.

The bridge never evaluates code. Each action is a named, parameter-validated operation.
Geometry is addressed by resolution-independent predicates (``normalized`` bounding-box
coordinates, face normals) instead of screen coordinates or vertex indices, which is what lets
a learned skill apply to a new object instance.
"""

import json
import os
import re
import tempfile

import bmesh
import bpy
from mathutils import Matrix, Vector

from .protocol import BridgeCommandError
from .state import find_view3d

PRIMITIVES = {
    "cube": lambda p: bpy.ops.mesh.primitive_cube_add(size=p["size"], location=p["location"], rotation=p["rotation"]),
    "plane": lambda p: bpy.ops.mesh.primitive_plane_add(size=p["size"], location=p["location"], rotation=p["rotation"]),
    "cylinder": lambda p: bpy.ops.mesh.primitive_cylinder_add(
        vertices=p["vertices"], radius=p["size"] / 2, depth=p["size"], location=p["location"], rotation=p["rotation"]),
    "cone": lambda p: bpy.ops.mesh.primitive_cone_add(
        vertices=p["vertices"], radius1=p["size"] / 2, depth=p["size"], location=p["location"], rotation=p["rotation"]),
    "uv_sphere": lambda p: bpy.ops.mesh.primitive_uv_sphere_add(
        segments=max(3, p["vertices"]), radius=p["size"] / 2, location=p["location"], rotation=p["rotation"]),
    "ico_sphere": lambda p: bpy.ops.mesh.primitive_ico_sphere_add(radius=p["size"] / 2, location=p["location"]),
    "torus": lambda p: bpy.ops.mesh.primitive_torus_add(location=p["location"], rotation=p["rotation"]),
    "monkey": lambda p: bpy.ops.mesh.primitive_monkey_add(
        size=p["size"], location=p["location"], rotation=p["rotation"]),
    # A ring of vertices without a face (tutorials often start pipes and bottles from one).
    "circle": lambda p: bpy.ops.mesh.primitive_circle_add(
        vertices=p["vertices"], radius=p["size"] / 2, location=p["location"], rotation=p["rotation"]),
}

MODIFIER_PROPS = {
    "MIRROR": {"use_axis": "bool3", "use_clip": "bool", "use_mirror_merge": "bool"},
    "BEVEL": {"width": "float", "segments": "int", "limit_method": ("NONE", "ANGLE", "WEIGHT", "VGROUP")},
    "SUBSURF": {"levels": "int", "render_levels": "int"},
    "SOLIDIFY": {"thickness": "float"},
    "ARRAY": {"count": "int", "relative_offset_displace": "vec3"},
    "SIMPLE_DEFORM": {"deform_method": ("TWIST", "BEND", "TAPER", "STRETCH"), "factor": "float",
                      "deform_axis": ("X", "Y", "Z")},
    "WEIGHTED_NORMAL": {},
    "TRIANGULATE": {},
    "DECIMATE": {"ratio": "float"},
}

AXES = {"x": 0, "y": 1, "z": 2}
SNAPSHOT_TAG = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
OBJECT_NAME = re.compile(r"^[^\x00-\x1f]{1,63}$")


# -- parameter validation --------------------------------------------------------------------

REQUIRED = object()


def _check(value, kind, name):
    if isinstance(kind, tuple):
        if value not in kind:
            raise BridgeCommandError("invalid_param", f"{name} must be one of {list(kind)}", param=name)
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BridgeCommandError("invalid_param", f"{name} must be a number", param=name)
        if value != value or abs(value) > 1e6:
            raise BridgeCommandError("invalid_param", f"{name} out of range", param=name)
        return float(value)
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int) or abs(value) > 100000:
            raise BridgeCommandError("invalid_param", f"{name} must be an integer", param=name)
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            raise BridgeCommandError("invalid_param", f"{name} must be a boolean", param=name)
        return value
    if kind in ("vec3", "bool3"):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise BridgeCommandError("invalid_param", f"{name} must be a 3-vector", param=name)
        inner = "bool" if kind == "bool3" else "float"
        return [_check(v, inner, name) for v in value]
    if kind == "name":
        if not isinstance(value, str) or not OBJECT_NAME.match(value):
            raise BridgeCommandError("invalid_param", f"{name} must be a valid object name", param=name)
        return value
    if kind == "names":
        if not isinstance(value, (list, tuple)) or len(value) > 500:
            raise BridgeCommandError("invalid_param", f"{name} must be a list of names", param=name)
        return [_check(v, "name", name) for v in value]
    if kind == "positions":
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 64:
            raise BridgeCommandError("invalid_param", f"{name} must be a list of 1..64 numbers", param=name)
        return [_check(v, "float", name) for v in value]
    if kind == "dict":
        if not isinstance(value, dict):
            raise BridgeCommandError("invalid_param", f"{name} must be an object", param=name)
        return value
    if kind == "path":
        if not isinstance(value, str) or not 0 < len(value) < 1024 or "\x00" in value:
            raise BridgeCommandError("invalid_param", f"{name} must be a path string", param=name)
        return value
    raise BridgeCommandError("invalid_spec", f"unknown parameter kind {kind}")


def validate(spec, args):
    unknown = set(args) - set(spec)
    if unknown:
        raise BridgeCommandError("invalid_param", f"unknown parameters: {sorted(unknown)}")
    clean = {}
    for name, (kind, default) in spec.items():
        if name in args and args[name] is not None:
            clean[name] = _check(args[name], kind, name)
        elif default is REQUIRED:
            raise BridgeCommandError("missing_param", f"{name} is required", param=name)
        else:
            clean[name] = default
    return clean



# -- helpers -------------------------------------------------------------------------------------

def _obj(name):
    if name is None:
        obj = bpy.context.view_layer.objects.active
        if obj is None:
            raise BridgeCommandError("no_active_object", "no active object")
        return obj
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise BridgeCommandError("object_not_found", f"object {name!r} not found", object=name)
    return obj


def _active():
    return bpy.context.view_layer.objects.active


def _context_override(obj=None):
    """Explicit context for operators.

    Bridge commands run from a timer or socket thread where ``bpy.context`` lacks the
    window/area/object members operators expect, so every operator call gets them explicitly.
    """
    window, area = find_view3d()
    override = {}
    if window is not None:
        override["window"] = window
        override["screen"] = window.screen
    if area is not None:
        override["area"] = area
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is not None:
            override["region"] = region
    obj = obj if obj is not None else _active()
    if obj is not None:
        override["active_object"] = obj
        override["object"] = obj
        if obj.mode == "EDIT":
            override["edit_object"] = obj
    return override


def _leave_edit_mode():
    obj = _active()
    if obj is not None and obj.mode != "OBJECT":
        with bpy.context.temp_override(**_context_override(obj)):
            bpy.ops.object.mode_set(mode="OBJECT")


def _ensure_mode(obj, mode):
    if _active() != obj:
        _leave_edit_mode()
        bpy.context.view_layer.objects.active = obj
    if obj.mode != mode:
        with bpy.context.temp_override(**_context_override(obj)):
            _op_result(bpy.ops.object.mode_set(mode=mode), "mode_set")


def _edit_bmesh(obj):
    if obj.type != "MESH":
        raise BridgeCommandError("not_a_mesh", f"{obj.name} is not a mesh", object=obj.name)
    _ensure_mode(obj, "EDIT")
    return bmesh.from_edit_mesh(obj.data)


def _selected_verts(bm):
    verts = [v for v in bm.verts if v.select]
    if not verts:
        raise BridgeCommandError("empty_selection", "no vertices selected")
    return verts


def _local_bbox(bm):
    xs = [v.co for v in bm.verts]
    if not xs:
        raise BridgeCommandError("empty_mesh", "mesh has no vertices")
    lo = Vector([min(c[i] for c in xs) for i in range(3)])
    hi = Vector([max(c[i] for c in xs) for i in range(3)])
    return lo, hi


def _op_result(result, name):
    if "FINISHED" not in result:
        raise BridgeCommandError("operator_failed", f"{name} returned {sorted(result)}", operator=name)


# -- actions -------------------------------------------------------------------------------------

def add_primitive(p):
    before = set(bpy.data.objects.keys())
    _leave_edit_mode()
    with bpy.context.temp_override(**_context_override()):
        _op_result(PRIMITIVES[p["kind"]](p), f"add_{p['kind']}")
    obj = _active()
    if p["name"]:
        obj.name = p["name"]
        obj.data.name = p["name"]
    created = sorted(set(bpy.data.objects.keys()) - before)
    return {"object": obj.name, "created": created}


def select_objects(p):
    _leave_edit_mode()
    if p["deselect_others"]:
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
    objs = [_obj(n) for n in p["names"]]
    for o in objs:
        o.select_set(True)
    if p["active"]:
        bpy.context.view_layer.objects.active = _obj(p["active"])
    elif objs:
        bpy.context.view_layer.objects.active = objs[0]
    return {"selected": [o.name for o in bpy.context.view_layer.objects if o.select_get()]}


def delete_objects(p):
    _leave_edit_mode()
    removed = []
    for name in p["names"]:
        obj = _obj(name)
        bpy.data.objects.remove(obj, do_unlink=True)
        removed.append(name)
    return {"removed": removed}


def set_mode(p):
    obj = _obj(p["object"])
    _ensure_mode(obj, p["mode"])
    return {"object": obj.name, "mode": obj.mode, "context_mode": bpy.context.mode}


def transform_object(p):
    obj = _obj(p["object"])
    if p["location"] is not None:
        obj.location = Vector(p["location"]) + (obj.location if p["relative"] else Vector())
    if p["rotation"] is not None:
        rot = Vector(p["rotation"])
        obj.rotation_euler = (Vector(obj.rotation_euler) + rot) if p["relative"] else rot
    if p["scale"] is not None:
        s = Vector(p["scale"])
        obj.scale = Vector(a * b for a, b in zip(obj.scale, s)) if p["relative"] else s
    bpy.context.view_layer.update()
    return {"object": obj.name, "location": list(obj.location), "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale), "dimensions": list(obj.dimensions)}


def set_dimensions(p):
    obj = _obj(p["object"])
    obj.dimensions = p["dimensions"]
    bpy.context.view_layer.update()
    return {"object": obj.name, "dimensions": list(obj.dimensions)}


def apply_transform(p):
    obj = _obj(p["object"])
    _ensure_mode(obj, "OBJECT")
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    with bpy.context.temp_override(**_context_override(obj), selected_editable_objects=[obj]):
        _op_result(bpy.ops.object.transform_apply(location=p["location"], rotation=p["rotation"],
                                                  scale=p["scale"]), "transform_apply")
    return {"object": obj.name, "dimensions": list(obj.dimensions)}


def rename_object(p):
    obj = _obj(p["object"])
    obj.name = p["new_name"]
    return {"object": obj.name}


def select_elements(p):
    """Select vertices (or faces by centre) matching an axis predicate.

    ``space='normalized'`` maps the mesh bounding box to [0, 1] per axis so "the top 10% of
    the blade" means the same thing on a long sword and a short dagger.
    """
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    lo, hi = _local_bbox(bm)
    axis = AXES[p["axis"]]
    span = (hi[axis] - lo[axis]) or 1.0

    def coord(co):
        if p["space"] == "normalized":
            return (co[axis] - lo[axis]) / span
        if p["space"] == "world":
            return (obj.matrix_world @ co)[axis]
        return co[axis]

    low, high = p["min"], p["max"]

    def match(co):
        c = coord(co)
        return (low is None or c >= low - 1e-6) and (high is None or c <= high + 1e-6)

    if not p["extend"]:
        for elem in (*bm.verts, *bm.edges, *bm.faces):
            elem.select = False
    if p["element"] == "FACE":
        bm.select_mode = {"FACE"}
        for face in bm.faces:
            if match(face.calc_center_median()):
                face.select = True
    else:
        bm.select_mode = {"VERT"}
        for vert in bm.verts:
            if match(vert.co):
                vert.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "verts": sum(v.select for v in bm.verts),
            "faces": sum(f.select for f in bm.faces)}


def select_faces_by_normal(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    direction = Vector(p["direction"]).normalized()
    if not p["extend"]:
        for elem in (*bm.verts, *bm.edges, *bm.faces):
            elem.select = False
    bm.select_mode = {"FACE"}
    count = 0
    for face in bm.faces:
        if face.normal.dot(direction) >= p["min_dot"]:
            face.select = True
            count += 1
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "faces": count}


def select_all(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    for elem in (*bm.verts, *bm.edges, *bm.faces):
        elem.select = p["action"] == "SELECT"
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "verts": sum(v.select for v in bm.verts)}


def _select_only(bm, geom):
    for elem in (*bm.verts, *bm.edges, *bm.faces):
        elem.select = False
    for elem in geom:
        elem.select = True
    bm.select_flush(True)


def extrude(p):
    """Extrude the selected region (faces, else edges, else vertices) and move it by ``offset``.

    Implemented with bmesh so it behaves identically in the GUI and in headless Blender: the
    original region is replaced by the moved cap connected through side walls, and the cap
    stays selected, like Blender's own extrude.
    """
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    faces = [f for f in bm.faces if f.select]
    edges = [e for e in bm.edges if e.select]
    verts = _selected_verts(bm)
    before = len(bm.verts)
    if faces:
        ret = bmesh.ops.extrude_face_region(bm, geom=faces)
        bmesh.ops.delete(bm, geom=faces, context="FACES_ONLY")
    elif edges:
        ret = bmesh.ops.extrude_edge_only(bm, edges=edges)
    else:
        ret = bmesh.ops.extrude_vert_indiv(bm, verts=verts)
        ret = {"geom": ret["verts_out"] + ret["edges_out"]}
    new_geom = ret["geom"]
    new_verts = [g for g in new_geom if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=Vector(p["offset"]), verts=new_verts)
    _select_only(bm, new_geom)
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "new_verts": len(bm.verts) - before, "mode": "faces" if faces else
            "edges" if edges else "verts"}


def translate_selection(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = _selected_verts(bm)
    bmesh.ops.translate(bm, vec=Vector(p["offset"]), verts=verts)
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "moved": len(verts)}


def scale_selection(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = _selected_verts(bm)
    if p["pivot"] == "median":
        pivot = sum((v.co for v in verts), Vector()) / len(verts)
    else:
        lo, hi = _local_bbox(bm)
        pivot = (lo + hi) / 2
    bmesh.ops.scale(bm, vec=Vector(p["factor"]), space=Matrix.Translation(-pivot), verts=verts)
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "scaled": len(verts)}


def taper_selection(p):
    """Scale cross-section progressively along an axis (1 at ``start`` .. 1-amount at the end)."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = _selected_verts(bm)
    along = AXES[p["along"]]
    targets = [AXES[a] for a in p["affect"]]
    lo = min(v.co[along] for v in verts)
    hi = max(v.co[along] for v in verts)
    span = (hi - lo) or 1.0
    centre = sum((v.co for v in verts), Vector()) / len(verts)
    for v in verts:
        t = (v.co[along] - lo) / span
        if p["reverse"]:
            t = 1.0 - t
        factor = 1.0 - p["amount"] * max(0.0, (t - p["start"]) / (1.0 - p["start"] or 1.0))
        for axis in targets:
            v.co[axis] = centre[axis] + (v.co[axis] - centre[axis]) * factor
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "tapered": len(verts)}


def inset(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    faces = [f for f in bm.faces if f.select]
    if not faces:
        raise BridgeCommandError("empty_selection", "no faces selected")
    result = bmesh.ops.inset_region(bm, faces=faces, thickness=p["thickness"], depth=p["depth"])
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "new_faces": len(result.get("faces", []))}


def bevel(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    _selected_verts(bm)
    before = len(bm.faces)
    if p["affect"] == "EDGES":
        geom = [e for e in bm.edges if e.select]
        if not geom:
            raise BridgeCommandError("empty_selection", "no edges selected")
    else:
        geom = [v for v in bm.verts if v.select]
    bmesh.ops.bevel(bm, geom=geom, offset=p["offset"], offset_type="OFFSET", segments=p["segments"],
                    profile=0.5, affect=p["affect"], clamp_overlap=True)
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "new_faces": len(bm.faces) - before}


def loop_cut_axis(p):
    """Insert edge loops perpendicular to an axis at normalised positions (a loop cut that
    does not depend on hovering an edge in a viewport)."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    lo, hi = _local_bbox(bm)
    axis = AXES[p["axis"]]
    normal = Vector((0, 0, 0))
    normal[axis] = 1.0
    cuts = 0
    for pos in p["positions"]:
        if not 0.0 < pos < 1.0:
            raise BridgeCommandError("invalid_param", "positions must be inside (0, 1)", param="positions")
        co = (lo + hi) / 2
        co[axis] = lo[axis] + (hi[axis] - lo[axis]) * pos
        geom = [*bm.verts, *bm.edges, *bm.faces]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=normal)
        cuts += 1
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "cuts": cuts, "verts": len(bm.verts)}


def merge_by_distance(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=p["distance"])
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "removed": before - len(bm.verts)}


def add_modifier(p):
    obj = _obj(p["object"])
    allowed = MODIFIER_PROPS[p["type"]]
    props = p["props"] or {}
    if not isinstance(props, dict):
        raise BridgeCommandError("invalid_param", "props must be an object", param="props")
    clean = {}
    for key, value in props.items():
        if key not in allowed:
            raise BridgeCommandError("invalid_param", f"property {key!r} not allowed for {p['type']}", param=key)
        clean[key] = _check(value, allowed[key], key)
    modifier = obj.modifiers.new(name=p["name"] or p["type"].title(), type=p["type"])
    for key, value in clean.items():
        setattr(modifier, key, value)
    return {"object": obj.name, "modifier": modifier.name, "type": modifier.type}


def remove_modifier(p):
    obj = _obj(p["object"])
    modifier = obj.modifiers.get(p["modifier"])
    if modifier is None:
        raise BridgeCommandError("modifier_not_found", f"modifier {p['modifier']!r} not found")
    obj.modifiers.remove(modifier)
    return {"object": obj.name, "removed": p["modifier"]}


def apply_modifier(p):
    obj = _obj(p["object"])
    _ensure_mode(obj, "OBJECT")
    if obj.modifiers.get(p["modifier"]) is None:
        raise BridgeCommandError("modifier_not_found", f"modifier {p['modifier']!r} not found")
    with bpy.context.temp_override(**_context_override(obj)):
        _op_result(bpy.ops.object.modifier_apply(modifier=p["modifier"]), "modifier_apply")
    return {"object": obj.name, "applied": p["modifier"]}


def set_symmetry(p):
    obj = _obj(p["object"])
    if obj.type != "MESH":
        raise BridgeCommandError("not_a_mesh", f"{obj.name} is not a mesh")
    obj.data.use_mirror_x, obj.data.use_mirror_y, obj.data.use_mirror_z = p["axes"]
    return {"object": obj.name, "axes": p["axes"]}


def shade(p):
    obj = _obj(p["object"])
    if obj.type != "MESH":
        raise BridgeCommandError("not_a_mesh", f"{obj.name} is not a mesh")
    for poly in obj.data.polygons:
        poly.use_smooth = p["smooth"]
    return {"object": obj.name, "smooth": p["smooth"]}


def reset_scene(p):
    _leave_edit_mode()
    keep = {"CAMERA", "LIGHT"} if p["keep_camera_light"] else set()
    removed = [o.name for o in list(bpy.data.objects) if o.type not in keep]
    for name in removed:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    return {"removed": removed}


# -- GUI-only view control ---------------------------------------------------------------------

def _require_gui():
    override = _context_override()
    if bpy.app.background or "area" not in override:
        raise BridgeCommandError("unsupported_in_background", "requires an interactive 3D viewport")
    return override


def set_view(p):
    override = _require_gui()
    with bpy.context.temp_override(**override):
        _op_result(bpy.ops.view3d.view_axis(type=p["view"]), "view_axis")
        r3d = override["area"].spaces.active.region_3d
        r3d.view_perspective = "ORTHO" if p["ortho"] else "PERSP"
    return {"view": p["view"], "ortho": p["ortho"]}


def orbit_view(p):
    override = _require_gui()
    with bpy.context.temp_override(**override):
        _op_result(bpy.ops.view3d.view_orbit(angle=p["angle"], type=p["direction"]), "view_orbit")
    return {"direction": p["direction"], "angle": p["angle"]}


def frame_selected(p):
    override = _require_gui()
    with bpy.context.temp_override(**override):
        _op_result(bpy.ops.view3d.view_selected(), "view_selected")
    return {}


def undo(p):
    override = _require_gui()
    with bpy.context.temp_override(**override):
        for _ in range(p["steps"]):
            _op_result(bpy.ops.ed.undo(), "undo")
    return {"steps": p["steps"]}


def redo(p):
    override = _require_gui()
    with bpy.context.temp_override(**override):
        for _ in range(p["steps"]):
            _op_result(bpy.ops.ed.redo(), "redo")
    return {"steps": p["steps"]}


# -- files & snapshots -----------------------------------------------------------------------

def _allowed_dirs(env_name):
    raw = os.environ.get(env_name, "")
    return [os.path.realpath(p) for p in raw.split(os.pathsep) if p]


def _check_path(path, env_name, extensions):
    real = os.path.realpath(path)
    allowed = _allowed_dirs(env_name)
    if not any(real == d or real.startswith(d + os.sep) for d in allowed):
        raise BridgeCommandError("path_not_allowed", f"{path} is outside the directories allowed by {env_name}")
    if not real.lower().endswith(extensions):
        raise BridgeCommandError("path_not_allowed", f"{path} must end with one of {extensions}")
    return real


def save_file(p):
    path = _check_path(p["path"], "LUCIUS_ALLOWED_SAVE_DIRS", (".blend",))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _leave_edit_mode()
    with bpy.context.temp_override(**_context_override()):
        _op_result(bpy.ops.wm.save_as_mainfile(filepath=path, copy=p["copy"]), "save_as_mainfile")
    return {"path": path}


def load_reference_image(p):
    path = _check_path(p["path"], "LUCIUS_ALLOWED_READ_DIRS", (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))
    image = bpy.data.images.load(path, check_existing=True)
    empty = bpy.data.objects.new(p["name"] or "Reference", None)
    empty.empty_display_type = "IMAGE"
    empty.data = image
    empty.empty_display_size = p["size"]
    rotations = {"FRONT": (1.5708, 0, 0), "SIDE": (1.5708, 0, 1.5708), "TOP": (0, 0, 0)}
    empty.rotation_euler = rotations[p["view"]]
    bpy.context.scene.collection.objects.link(empty)
    return {"object": empty.name, "image": image.name}


def import_blend(p):
    """Replace the scene's objects with the objects of a .blend file for inspection.

    Uses library appending (data only): scripts embedded in the file are never executed.
    """
    path = _check_path(p["path"], "LUCIUS_ALLOWED_READ_DIRS", (".blend",))
    _leave_edit_mode()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    _purge_orphans()
    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    names = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.scene.collection.objects.link(obj)
            names.append(obj.name)
    return {"objects": sorted(names)}


def _snapshot_dir():
    path = os.path.join(tempfile.gettempdir(), f"lucius_snapshots_{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    return path


def _purge_orphans():
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights, bpy.data.curves,
                       bpy.data.images):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def snapshot(p):
    """Save the scene to a private snapshot file (headless recovery; the GUI also has undo)."""
    if not SNAPSHOT_TAG.match(p["tag"]):
        raise BridgeCommandError("invalid_param", "invalid snapshot tag", param="tag")
    was_editing = bpy.context.mode == "EDIT_MESH"
    _leave_edit_mode()  # flushes edit-mode changes into the mesh before saving
    path = os.path.join(_snapshot_dir(), p["tag"] + ".blend")
    with bpy.context.temp_override(**_context_override()):
        _op_result(bpy.ops.wm.save_as_mainfile(filepath=path, copy=True), "snapshot")
        if was_editing and _active() is not None:
            # A snapshot must not change what the user (or a keyboard sequence) sees: back to edit mode.
            bpy.ops.object.mode_set(mode="EDIT")
    active = _active()
    meta = {"active": active.name if active else None,
            "selected": [o.name for o in bpy.context.view_layer.objects if o.select_get()]}
    with open(path + ".json", "w") as handle:
        json.dump(meta, handle)
    return {"tag": p["tag"], "objects": len(bpy.data.objects)}


def restore(p):
    """Replace the scene's objects with those of a snapshot.

    Objects are appended from the snapshot file rather than re-opening it: re-opening the main
    file from a bridge thread is unsafe, and appending keeps the current UI untouched.
    """
    if not SNAPSHOT_TAG.match(p["tag"]):
        raise BridgeCommandError("invalid_param", "invalid snapshot tag", param="tag")
    path = os.path.join(_snapshot_dir(), p["tag"] + ".blend")
    if not os.path.exists(path):
        raise BridgeCommandError("snapshot_not_found", f"no snapshot {p['tag']!r}")
    _leave_edit_mode()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    _purge_orphans()
    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    scene_collection = bpy.context.scene.collection
    restored = []
    for obj in data_to.objects:
        if obj is not None:
            scene_collection.objects.link(obj)
            restored.append(obj.name)
    meta_path = path + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            meta = json.load(handle)
        for obj in bpy.context.view_layer.objects:
            obj.select_set(obj.name in meta.get("selected", []))
        active = meta.get("active")
        # Blender 5 raises TypeError for `None in bpy.data.objects` (4.x returned False).
        if isinstance(active, str) and active in bpy.data.objects:
            bpy.context.view_layer.objects.active = bpy.data.objects[active]
    return {"tag": p["tag"], "restored": sorted(restored)}


V3 = ("vec3", [0.0, 0.0, 0.0])
OBJ = ("name", None)

ACTIONS = {
    "add_primitive": (add_primitive, {
        "kind": (tuple(PRIMITIVES), REQUIRED), "size": ("float", 2.0), "location": V3, "rotation": V3,
        "vertices": ("int", 32), "name": ("name", None)}),
    "select_objects": (select_objects, {"names": ("names", REQUIRED), "active": OBJ, "deselect_others": ("bool", True)}),
    "delete_objects": (delete_objects, {"names": ("names", REQUIRED)}),
    "set_mode": (set_mode, {"object": OBJ, "mode": (("OBJECT", "EDIT", "SCULPT"), REQUIRED)}),
    "transform_object": (transform_object, {
        "object": OBJ, "location": ("vec3", None), "rotation": ("vec3", None), "scale": ("vec3", None),
        "relative": ("bool", False)}),
    "set_dimensions": (set_dimensions, {"object": OBJ, "dimensions": ("vec3", REQUIRED)}),
    "apply_transform": (apply_transform, {
        "object": OBJ, "location": ("bool", False), "rotation": ("bool", True), "scale": ("bool", True)}),
    "rename_object": (rename_object, {"object": OBJ, "new_name": ("name", REQUIRED)}),
    "select_elements": (select_elements, {
        "object": OBJ, "axis": (tuple(AXES), REQUIRED), "min": ("float", None), "max": ("float", None),
        "space": (("normalized", "local", "world"), "normalized"), "element": (("VERT", "FACE"), "VERT"),
        "extend": ("bool", False)}),
    "select_faces_by_normal": (select_faces_by_normal, {
        "object": OBJ, "direction": ("vec3", REQUIRED), "min_dot": ("float", 0.9), "extend": ("bool", False)}),
    "select_all": (select_all, {"object": OBJ, "action": (("SELECT", "DESELECT"), "SELECT")}),
    "extrude": (extrude, {"object": OBJ, "offset": ("vec3", REQUIRED)}),
    "translate_selection": (translate_selection, {"object": OBJ, "offset": ("vec3", REQUIRED)}),
    "scale_selection": (scale_selection, {
        "object": OBJ, "factor": ("vec3", REQUIRED), "pivot": (("median", "bbox_center"), "median")}),
    "taper_selection": (taper_selection, {
        "object": OBJ, "along": (tuple(AXES), REQUIRED), "affect": (("x", "y", "z", "xy", "xz", "yz"), "x"),
        "amount": ("float", REQUIRED), "start": ("float", 0.0), "reverse": ("bool", False)}),
    "inset": (inset, {"object": OBJ, "thickness": ("float", REQUIRED), "depth": ("float", 0.0)}),
    "bevel": (bevel, {"object": OBJ, "offset": ("float", REQUIRED), "segments": ("int", 1),
                      "affect": (("EDGES", "VERTICES"), "EDGES")}),
    "loop_cut_axis": (loop_cut_axis, {"object": OBJ, "axis": (tuple(AXES), REQUIRED), "positions": ("positions", REQUIRED)}),
    "merge_by_distance": (merge_by_distance, {"object": OBJ, "distance": ("float", 0.0001)}),
    "add_modifier": (add_modifier, {
        "object": OBJ, "type": (tuple(MODIFIER_PROPS), REQUIRED), "name": ("name", None), "props": ("dict", None)}),
    "remove_modifier": (remove_modifier, {"object": OBJ, "modifier": ("name", REQUIRED)}),
    "apply_modifier": (apply_modifier, {"object": OBJ, "modifier": ("name", REQUIRED)}),
    "set_symmetry": (set_symmetry, {"object": OBJ, "axes": ("bool3", REQUIRED)}),
    "shade": (shade, {"object": OBJ, "smooth": ("bool", True)}),
    "reset_scene": (reset_scene, {"keep_camera_light": ("bool", True)}),
    "set_view": (set_view, {"view": (("FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM"), REQUIRED),
                            "ortho": ("bool", True)}),
    "orbit_view": (orbit_view, {"direction": (("ORBITLEFT", "ORBITRIGHT", "ORBITUP", "ORBITDOWN"), REQUIRED),
                                "angle": ("float", 0.2618)}),
    "frame_selected": (frame_selected, {}),
    "undo": (undo, {"steps": ("int", 1)}),
    "redo": (redo, {"steps": ("int", 1)}),
    "save_file": (save_file, {"path": ("path", REQUIRED), "copy": ("bool", True)}),
    "load_reference_image": (load_reference_image, {
        "path": ("path", REQUIRED), "view": (("FRONT", "SIDE", "TOP"), "FRONT"), "size": ("float", 5.0),
        "name": ("name", None)}),
    "import_blend": (import_blend, {"path": ("path", REQUIRED)}),
    "snapshot": (snapshot, {"tag": ("path", REQUIRED)}),
    "restore": (restore, {"tag": ("path", REQUIRED)}),
}

GUI_ONLY = {"set_view", "orbit_view", "frame_selected", "undo", "redo"}


def execute_action(name, args):
    if name not in ACTIONS:
        raise BridgeCommandError("unknown_action", f"action {name!r} is not allowlisted", action=name)
    handler, spec = ACTIONS[name]
    if not isinstance(args, dict):
        raise BridgeCommandError("invalid_param", "args must be an object")
    params = validate(spec, args)
    result = handler(params)
    return {"action": name, "result": result}


def describe_actions():
    return {
        name: {param: (list(kind) if isinstance(kind, tuple) else kind) for param, (kind, _d) in spec.items()}
        for name, (_h, spec) in ACTIONS.items()
    }
