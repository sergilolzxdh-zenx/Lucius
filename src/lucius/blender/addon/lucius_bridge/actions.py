"""Allowlisted Blender API actions.

The bridge never evaluates code. Each action is a named, parameter-validated operation.
Geometry is addressed by resolution-independent predicates (``normalized`` bounding-box
coordinates, face normals) instead of screen coordinates or vertex indices, which is what lets
a learned skill apply to a new object instance.
"""

import json
import math
import os
import re
import tempfile

import bmesh
import bpy
from mathutils import Matrix, Vector

from .protocol import BridgeCommandError
from .state import find_view3d

def _radius(p):
    return p["radius"] if p.get("radius") is not None else p["size"] / 2


def _depth(p):
    return p["depth"] if p.get("depth") is not None else p["size"]


PRIMITIVES = {
    "cube": lambda p: bpy.ops.mesh.primitive_cube_add(size=p["size"], location=p["location"], rotation=p["rotation"]),
    "plane": lambda p: bpy.ops.mesh.primitive_plane_add(size=p["size"], location=p["location"], rotation=p["rotation"]),
    "cylinder": lambda p: bpy.ops.mesh.primitive_cylinder_add(
        vertices=p["vertices"], radius=_radius(p), depth=_depth(p), location=p["location"], rotation=p["rotation"]),
    "cone": lambda p: bpy.ops.mesh.primitive_cone_add(
        vertices=p["vertices"], radius1=_radius(p), radius2=p.get("radius2") or 0.0, depth=_depth(p),
        location=p["location"], rotation=p["rotation"]),
    "uv_sphere": lambda p: bpy.ops.mesh.primitive_uv_sphere_add(
        segments=max(3, p["vertices"]), ring_count=max(3, p.get("rings") or p["vertices"] // 2), radius=_radius(p),
        location=p["location"], rotation=p["rotation"]),
    "ico_sphere": lambda p: bpy.ops.mesh.primitive_ico_sphere_add(radius=_radius(p), location=p["location"]),
    "torus": lambda p: bpy.ops.mesh.primitive_torus_add(
        major_radius=p.get("major_radius") or 1.0, minor_radius=p.get("minor_radius") or 0.25,
        major_segments=p.get("major_segments") or 48, minor_segments=p.get("minor_segments") or 12,
        location=p["location"], rotation=p["rotation"]),
    "monkey": lambda p: bpy.ops.mesh.primitive_monkey_add(
        size=p["size"], location=p["location"], rotation=p["rotation"]),
    # A lattice: a cage of points that deforms whatever uses it in a Lattice modifier.
    "lattice": lambda p: bpy.ops.object.add(type="LATTICE", radius=p["size"] / 2, location=p["location"],
                                            rotation=p["rotation"]),
    # A metaball: a blob that melts into other metaballs near it (smoke puffs, liquids).
    "metaball": lambda p: bpy.ops.object.metaball_add(type="BALL", radius=_radius(p), location=p["location"],
                                                      rotation=p["rotation"]),
    # An empty: an object with no geometry, a handle to move, scale or parent things with.
    "empty": lambda p: bpy.ops.object.empty_add(type="PLAIN_AXES", radius=p["size"] / 2, location=p["location"],
                                                rotation=p["rotation"]),
    # A ring of vertices (tutorials often start pipes, plates and croissants from one); ``fill`` adds the n-gon.
    "circle": lambda p: bpy.ops.mesh.primitive_circle_add(
        vertices=p["vertices"], radius=_radius(p), fill_type="NGON" if p.get("fill") else "NOTHING",
        location=p["location"], rotation=p["rotation"]),
}

MODIFIER_PROPS = {
    "MIRROR": {"use_axis": "bool3", "use_bisect_axis": "bool3", "use_clip": "bool", "use_mirror_merge": "bool"},
    "BEVEL": {"width": "float", "segments": "int", "limit_method": ("NONE", "ANGLE", "WEIGHT", "VGROUP")},
    "SUBSURF": {"levels": "int", "render_levels": "int"},
    "SOLIDIFY": {"thickness": "float", "offset": "float", "use_even_offset": "bool"},
    "ARRAY": {"count": "int", "relative_offset_displace": "vec3"},
    "BOOLEAN": {"object": "object", "operation": ("DIFFERENCE", "UNION", "INTERSECT")},
    "DISPLACE": {"strength": "float", "mid_level": "float",
                 "texture": ("CLOUDS", "VORONOI", "MUSGRAVE", "NOISE", "MARBLE", "WOOD", "STUCCI", "DISTORTED_NOISE",
                             "MAGIC", "BLEND"),
                 "texture_scale": "float", "texture_coords": ("LOCAL", "GLOBAL", "OBJECT", "UV"),
                 "texture_coords_object": "object", "direction": ("X", "Y", "Z", "NORMAL", "RGB_TO_XYZ")},
    "SKIN": {"use_smooth_shade": "bool", "branch_smoothing": "float"},
    "COLLISION": {},
    "WAVE": {"height": "float", "width": "float", "speed": "float", "narrowness": "float"},
    "SMOOTH": {"factor": "float", "iterations": "int"},
    "CAST": {"factor": "float", "cast_type": ("SPHERE", "CYLINDER", "CUBOID")},
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
    if kind == "vec2":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise BridgeCommandError("invalid_param", f"{name} must be 2 numbers", param=name)
        return [_check(v, "float", name) for v in value]
    if kind in ("vec3", "bool3"):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise BridgeCommandError("invalid_param", f"{name} must be a 3-vector", param=name)
        inner = "bool" if kind == "bool3" else "float"
        return [_check(v, inner, name) for v in value]
    if kind == "bounds3":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise BridgeCommandError("invalid_param", f"{name} must be 3 numbers or nulls", param=name)
        return [None if v is None else _check(v, "float", name) for v in value]
    if kind == "object":
        return _obj(_check(value, "name", name))
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
    if kind == "list":
        if not isinstance(value, list) or len(value) > 500:
            raise BridgeCommandError("invalid_param", f"{name} must be a list (up to 500 items)", param=name)
        return value
    if kind == "bone_or_empty":
        if value == "":
            return value
        return _check(value, "name", name)
    if kind == "any":
        # a plain JSON value for a setting: a number, a switch, a short text, or a short list of those
        def plain(v, depth=0):
            if isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 200):
                return True
            return isinstance(v, list) and depth < 2 and len(v) <= 64 and all(plain(x, depth + 1) for x in v)

        if not plain(value):
            raise BridgeCommandError("invalid_param", f"{name} must be a number, true/false, a short text or a list",
                                     param=name)
        return value
    if kind in ("path_expr", "expression"):
        if not isinstance(value, str) or not 0 < len(value) <= 300:
            raise BridgeCommandError("invalid_param", f"{name} must be a short text", param=name)
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
    if p["into"]:
        # Shift+A in edit mode: the new shape becomes part of an existing mesh, selected on its own.
        target = _obj(p["into"])
        if target.type != "MESH" or target == obj:
            raise BridgeCommandError("invalid_param", "into must be another mesh object", param="into")
        obj.data.transform(target.matrix_world.inverted() @ obj.matrix_world)
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bm = _edit_bmesh(target)
        old = set(bm.verts)
        bm.from_mesh(mesh)
        bpy.data.meshes.remove(mesh)
        bm.verts.ensure_lookup_table()
        new_verts = [v for v in bm.verts if v not in old]
        new_set = set(new_verts)
        _select_only(bm, [*new_verts, *(e for e in bm.edges if all(v in new_set for v in e.verts)),
                          *(f for f in bm.faces if all(v in new_set for v in f.verts))])
        bm.normal_update()
        bmesh.update_edit_mesh(target.data)
        return {"object": target.name, "added_verts": len(new_verts)}
    if p["name"]:
        obj.name = p["name"]
        if obj.data is not None:
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


def hide_objects(p):
    """H (hide in the viewport) and the camera icon (hide from renders): control shapes, helpers, references."""
    _leave_edit_mode()
    done = []
    for name in p["names"]:
        obj = _obj(name)
        obj.hide_set(p["hide"])
        if p["render"]:
            obj.hide_render = p["hide"]
        done.append(obj.name)
    return {"objects": done, "hidden": p["hide"]}


def parent_object(p):
    """Ctrl+P > Object: the child follows the parent (moves, turns and scales with it) and keeps where it is now;
    ``parent`` null clears the parent (Alt+P, keeping the transform)."""
    _leave_edit_mode()
    child = _obj(p["object"])
    world = child.matrix_world.copy()
    if p["parent"] is None:
        child.parent = None
        child.matrix_world = world
        return {"object": child.name, "parent": None}
    parent = _obj(p["parent"])
    ancestor = parent
    while ancestor is not None:
        if ancestor == child:
            raise BridgeCommandError("invalid_param", "an object cannot be parented to its own child", param="parent")
        ancestor = ancestor.parent
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    child.matrix_world = world
    bpy.context.view_layer.update()
    return {"object": child.name, "parent": parent.name}


def join_objects(p):
    """Ctrl+J: merge objects into one (the ``into`` object keeps its name, origin and modifiers; the others'
    modifiers are lost, as in Blender -- apply them first)."""
    _leave_edit_mode()
    target = _obj(p["into"])
    others = [_obj(n) for n in p["names"] if n != target.name]
    if not others:
        raise BridgeCommandError("invalid_param", "nothing to join", param="names")
    for o in (target, *others):
        if o.type != "MESH":
            raise BridgeCommandError("not_a_mesh", f"{o.name} is not a mesh")
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in (target, *others):
        o.select_set(True)
    bpy.context.view_layer.objects.active = target
    joined = [o.name for o in others]
    with bpy.context.temp_override(**_context_override(target), selected_objects=[target, *others],
                                   selected_editable_objects=[target, *others]):
        _op_result(bpy.ops.object.join(), "join")
    return {"object": target.name, "joined": joined, "verts": len(target.data.vertices)}


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


def _to_local(obj, co, space):
    return obj.matrix_world.inverted() @ Vector(co) if space == "world" else Vector(co)


def knife_cut(p):
    """K (knife) for a straight cut: from ``start`` to ``end`` as seen along ``view`` (e.g. [0,-1,0] = from the
    front); only the selected faces are cut unless ``through`` (C: cut through, every face behind as well). The
    new edges are selected."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    a, b = _to_local(obj, p["start"], p["space"]), _to_local(obj, p["end"], p["space"])
    view = obj.matrix_world.inverted().to_3x3() @ Vector(p["view"]) if p["space"] == "world" else Vector(p["view"])
    normal = (b - a).cross(view)
    if normal.length < 1e-9:
        raise BridgeCommandError("invalid_param", "the cut can't run along the view direction", param="view")
    faces = list(bm.faces) if p["through"] else [f for f in bm.faces if f.select]
    if not faces:
        raise BridgeCommandError("empty_selection", "select the faces to cut (or cut through)", param="through")
    lo, hi = min(a, b, key=lambda v: v.dot(b - a)), max(a, b, key=lambda v: v.dot(b - a))
    edges = list({e for f in faces for e in f.edges})
    geom = list({v for e in edges for v in e.verts}) + edges + faces
    result = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=a, plane_no=normal.normalized())
    cut = [g for g in result["geom_cut"] if isinstance(g, bmesh.types.BMEdge)]
    # a knife stroke ends where it ends: keep only the new edges between the two clicks
    along = (b - a).normalized()
    inside = [e for e in cut if all(lo.dot(along) - 1e-5 <= v.co.dot(along) <= hi.dot(along) + 1e-5
                                    for v in e.verts)]
    _select_only(bm, inside)
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "new_edges": len(inside)}


def bisect(p):
    """Bisect: cut the whole mesh (or the selection) with a plane through ``point`` facing ``normal``; clear the
    inner side (behind the normal) or the outer side, and fill the cut with a face."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    selected = [f for f in bm.faces if f.select]
    faces = selected or list(bm.faces)
    edges = list({e for f in faces for e in f.edges})
    geom = list({v for e in edges for v in e.verts}) + edges + faces
    co = _to_local(obj, p["point"], p["space"])
    no = (obj.matrix_world.inverted().to_3x3() @ Vector(p["normal"])) if p["space"] == "world" else Vector(p["normal"])
    result = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=no.normalized(),
                                    clear_inner=p["clear_inner"], clear_outer=p["clear_outer"])
    cut = [g for g in result["geom_cut"] if isinstance(g, bmesh.types.BMEdge) and g.is_valid]
    filled = 0
    if p["fill"] and cut:
        made = bmesh.ops.contextual_create(bm, geom=cut)
        filled = len(made.get("faces", []))
    _select_only(bm, cut)
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "cut_edges": len(cut), "filled_faces": filled}


def spin(p):
    """The spin tool: the selected edges or faces swept round an axis through ``center`` by ``angle`` (radians;
    recipes give angle_deg) in ``steps`` segments (a pipe bend, a vase from a profile)."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = [v for v in bm.verts if v.select]
    if not verts:
        raise BridgeCommandError("empty_selection", "select the profile to spin")
    geom = verts + [e for e in bm.edges if e.select] + [f for f in bm.faces if f.select]
    axis = Vector({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[p["axis"]])
    center = _to_local(obj, p["center"], p["space"])
    result = bmesh.ops.spin(bm, geom=geom, cent=center, axis=axis, angle=p["angle"],
                            steps=max(1, p["steps"]), use_duplicate=False)
    if any(isinstance(g, bmesh.types.BMFace) for g in geom):
        bmesh.ops.delete(bm, geom=[f for f in geom if isinstance(f, bmesh.types.BMFace) and f.is_valid],
                         context="FACES_ONLY")
    _select_only(bm, result["geom_last"])
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "steps": p["steps"], "verts": len(bm.verts)}


def slide_selection(p):
    """G G: slide the selected vertices (an edge, a loop) along the edges leading away from them, towards
    ``toward`` (a direction), by ``factor`` of those edges' length -- the shape keeps its surface."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = [v for v in bm.verts if v.select]
    if not verts:
        raise BridgeCommandError("empty_selection", "select the edge or vertices to slide")
    if not 0 <= p["factor"] <= 1:
        raise BridgeCommandError("invalid_param", "factor 0..1", param="factor")
    toward = obj.matrix_world.inverted().to_3x3() @ Vector(p["toward"])
    selected = set(verts)
    moves = []
    for v in verts:
        options = [e.other_vert(v) for e in v.link_edges if e.other_vert(v) not in selected]
        if not options:
            continue
        target = max(options, key=lambda o: (o.co - v.co).normalized().dot(toward))
        if (target.co - v.co).dot(toward) <= 0:
            continue
        moves.append((v, v.co.lerp(target.co, p["factor"])))
    for v, co in moves:
        v.co = co
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "slid": len(moves)}


def shrink_fatten(p):
    """Alt+S (shrink/fatten): the selected vertices pushed out along their normals (positive) or in."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = [v for v in bm.verts if v.select]
    if not verts:
        raise BridgeCommandError("empty_selection", "select faces to shrink or fatten")
    bm.normal_update()
    moves = [(v, v.co + v.normal * p["distance"]) for v in verts]
    for v, co in moves:
        v.co = co
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "moved": len(moves)}


def move_lattice_points(p):
    """Edit a lattice: the points inside a box (the lattice's own coordinates run -0.5..0.5) moved by
    ``offset`` and/or scaled about their centre by ``factor`` -- whatever uses the lattice deforms with it."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    if obj.type != "LATTICE":
        raise BridgeCommandError("invalid_param", f"{obj.name} is not a lattice", param="object")
    low, high = p["min"], p["max"]

    def inside(co):
        return all((low[i] is None or co[i] >= low[i] - 1e-6) and (high[i] is None or co[i] <= high[i] + 1e-6)
                   for i in range(3))

    points = [pt for pt in obj.data.points if inside(pt.co)]
    if not points:
        raise BridgeCommandError("empty_selection", "no lattice points inside the box (they run -0.5..0.5)")
    centre = sum((Vector(pt.co_deform) for pt in points), Vector()) / len(points)
    factor = Vector(p["factor"])
    for pt in points:
        co = Vector(pt.co_deform) - centre
        pt.co_deform = centre + Vector([co[i] * factor[i] for i in range(3)]) + Vector(p["offset"])
    obj.data.update_tag()
    bpy.context.view_layer.update()
    return {"object": obj.name, "points": len(points)}


def add_hook(p):
    """Ctrl+H > Hook to New Object: the vertices selected in edit mode follow an empty (made at their centre,
    or an existing object); move the empty and they move, with the rest of the mesh unaffected."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    selected = [v for v in bm.verts if v.select]
    if not selected:
        raise BridgeCommandError("empty_selection", "select the vertices to hook in edit mode")
    indices = [v.index for v in selected]
    centre = sum((v.co for v in selected), Vector()) / len(selected)
    _leave_edit_mode()
    name = p["hook"] or f"Hook-{obj.name}"
    empty = bpy.data.objects.get(name)
    if empty is None:
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = p["size"]
        bpy.context.collection.objects.link(empty)
        empty.location = obj.matrix_world @ centre
        bpy.context.view_layer.update()
    modifier = obj.modifiers.new(name=f"Hook-{empty.name}", type="HOOK")
    modifier.object = empty
    modifier.vertex_indices_set(indices)
    modifier.center = centre
    # like Hook Reset: nothing jumps now, only later moves of the empty count
    modifier.matrix_inverse = (obj.matrix_world.inverted() @ empty.matrix_world).inverted()
    bpy.context.view_layer.update()
    return {"object": obj.name, "hook": empty.name, "vertices": len(indices), "modifier": modifier.name}


BIND_OPERATORS = {"MESH_DEFORM": "meshdeform_bind", "SURFACE_DEFORM": "surfacedeform_bind",
                  "LAPLACIANDEFORM": "laplaciandeform_bind", "CORRECTIVE_SMOOTH": "correctivesmooth_bind"}


def bind_modifier(p):
    """The Bind button of Mesh Deform, Surface Deform, Laplacian Deform and Corrective Smooth: the object
    remembers its shape relative to the cage / target / anchors, so editing those deforms it."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    modifier = obj.modifiers.get(p["modifier"])
    if modifier is None or modifier.type not in BIND_OPERATORS:
        raise BridgeCommandError("invalid_param", f"no bindable modifier {p['modifier']!r} on {obj.name}",
                                 param="modifier")
    operator = getattr(bpy.ops.object, BIND_OPERATORS[modifier.type])
    bpy.context.view_layer.objects.active = obj
    with bpy.context.temp_override(**_context_override(obj)):
        _op_result(operator(modifier=modifier.name), BIND_OPERATORS[modifier.type])
    bpy.context.view_layer.update()   # a mesh deform binds on the next evaluation
    bound = getattr(modifier, "is_bound", True)
    return {"object": obj.name, "modifier": modifier.name, "bound": bool(bound)}


UNWRAP = ("UNWRAP", "SMART_PROJECT", "CUBE_PROJECT", "CYLINDER_PROJECT", "SPHERE_PROJECT", "RESET")


def uv_unwrap(p):
    """U in edit mode: lay the selected faces (all, if none) out flat in the UV map -- Unwrap (by seams), Smart UV
    Project (by angle), Cube / Cylinder / Sphere Projection, Reset (every face fills the square)."""
    import math

    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    if not any(f.select for f in bm.faces):
        for f in bm.faces:
            f.select = True
        bmesh.update_edit_mesh(obj.data)
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    method = p["method"]
    with bpy.context.temp_override(**_context_override(obj)):
        if method == "UNWRAP":
            result = bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=p["margin"])
        elif method == "SMART_PROJECT":
            result = bpy.ops.uv.smart_project(angle_limit=math.radians(p["angle_limit_deg"]), island_margin=p["margin"])
        elif method == "CUBE_PROJECT":
            result = bpy.ops.uv.cube_project(cube_size=p["size"])
        elif method == "CYLINDER_PROJECT":
            result = bpy.ops.uv.cylinder_project(direction="ALIGN_TO_OBJECT", scale_to_bounds=True)
        elif method == "SPHERE_PROJECT":
            result = bpy.ops.uv.sphere_project(direction="ALIGN_TO_OBJECT", scale_to_bounds=True)
        else:
            result = bpy.ops.uv.reset()
    _op_result(result, method.lower())
    bm = _edit_bmesh(obj)
    return {"object": obj.name, "method": method, "faces": sum(1 for f in bm.faces if f.select)}


def uv_transform(p):
    """The UV editor: the selected faces' UVs (all, if none) rotated, scaled and moved about their centre --
    the texture on them turns, grows or slides the opposite way."""
    import math

    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    layer = bm.loops.layers.uv.active
    if layer is None:
        raise BridgeCommandError("invalid_param", f"{obj.name} has no UV map (uv_unwrap first)", param="object")
    faces = [f for f in bm.faces if f.select] or list(bm.faces)
    loops = [loop for f in faces for loop in f.loops]
    centre = sum((loop[layer].uv.copy() for loop in loops), Vector((0.0, 0.0))) / len(loops)
    angle = math.radians(p["rotate_deg"])
    c, s_ = math.cos(angle), math.sin(angle)
    su, sv = p["scale"]
    du, dv = p["offset"]
    for loop in loops:
        u, v = loop[layer].uv - centre
        u, v = u * su, v * sv
        loop[layer].uv = (centre.x + u * c - v * s_ + du, centre.y + u * s_ + v * c + dv)
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "faces": len(faces)}


def skin_radius(p):
    """Ctrl+A in edit mode with a Skin modifier: the thickness the skin gives the selected vertices."""
    obj = _obj(p["object"])
    if not any(m.type == "SKIN" for m in obj.modifiers):
        raise BridgeCommandError("invalid_param", f"{obj.name} has no Skin modifier", param="object")
    if not 0 < p["radius"] <= 100:
        raise BridgeCommandError("invalid_param", "radius must be positive", param="radius")
    bm = _edit_bmesh(obj)
    layer = bm.verts.layers.skin.verify()
    verts = [v for v in bm.verts if v.select]
    for v in verts:
        v[layer].radius = (p["radius"], p["radius"])
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "vertices": len(verts), "radius": p["radius"]}


def select_nth(p):
    """Select > Checker Deselect: of the selected elements, keep every nth (``skip`` deselected, ``nth`` kept,
    starting at ``offset``), walking along the mesh from the active element -- e.g. every other vertex of a circle."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    selected = [v for v in bm.verts if v.select]
    if not selected:
        raise BridgeCommandError("invalid_param", "select something first", param="object")
    if not bm.select_history:
        bm.select_history.add(selected[0])   # the walk starts at the active element
    bmesh.update_edit_mesh(obj.data)
    with bpy.context.temp_override(**_context_override(obj)):
        _op_result(bpy.ops.mesh.select_nth(skip=p["skip"], nth=p["nth"], offset=p["offset"]), "select_nth")
    bm = _edit_bmesh(obj)
    return {"object": obj.name, "selected_verts": sum(1 for v in bm.verts if v.select)}


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
    region is moved and connected through side walls, the original faces are removed when they
    would end up inside the mesh (and kept for a lone region, as Blender does), and the moved
    region stays selected, like Blender's own extrude.
    """
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    faces = [f for f in bm.faces if f.select]
    edges = [e for e in bm.edges if e.select]
    verts = _selected_verts(bm)
    if p["offset"] is not None:
        offset = Vector(p["offset"])
    elif p["distance"] is not None:
        # Like pressing E on faces: along their average normal.
        normal = sum((f.normal for f in faces), Vector()) if faces else Vector((0, 0, 1))
        offset = (normal.normalized() if normal.length > 1e-9 else Vector((0, 0, 1))) * p["distance"]
    else:
        raise BridgeCommandError("missing_param", "extrude needs offset or distance", param="offset")
    before = len(bm.verts)
    if faces and p["individual"]:
        # Extrude Individual Faces (Alt+E): each face pushed out along its own normal by ``distance``.
        if p["distance"] is None:
            raise BridgeCommandError("missing_param", "individual extrusion needs a distance", param="distance")
        ret = bmesh.ops.extrude_discrete_faces(bm, faces=faces)
        for face in ret["faces"]:
            bmesh.ops.translate(bm, vec=face.normal * p["distance"], verts=list(face.verts))
        _select_only(bm, ret["faces"])   # (the originals are replaced by the extruded ones)
        bm.normal_update()
        bmesh.update_edit_mesh(obj.data)
        return {"object": obj.name, "new_verts": len(bm.verts) - before, "mode": "individual faces"}
    if faces:
        region = set(faces)
        # Blender's rule: the original faces go only when the region is attached to other faces (a cylinder's
        # cap); a lone region (a filled circle, a plane) keeps them, so extruding a disc makes a closed cup.
        attached = any(other not in region for face in faces for edge in face.edges for other in edge.link_faces)
        ret = bmesh.ops.extrude_face_region(bm, geom=faces)
        if attached:
            bmesh.ops.delete(bm, geom=faces, context="FACES")
    elif edges:
        ret = bmesh.ops.extrude_edge_only(bm, edges=edges)
    else:
        ret = bmesh.ops.extrude_vert_indiv(bm, verts=verts)
        # (Blender 5 names the outputs verts / edges; older versions verts_out / edges_out)
        ret = {"geom": list(ret.get("verts_out", ret.get("verts", []))) + list(ret.get("edges_out", ret.get("edges", [])))}
    new_geom = ret["geom"]
    new_verts = [g for g in new_geom if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=offset, verts=new_verts)
    # Like E: what stays selected is the moved region -- for edges, the new edge loop, not the new walls.
    _select_only(bm, new_geom if faces else [g for g in new_geom if not isinstance(g, bmesh.types.BMFace)])
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "new_verts": len(bm.verts) - before, "mode": "faces" if faces else
            "edges" if edges else "verts"}


FALLOFFS = {  # proportional editing falloffs, as Blender's (t = 1 at the selection .. 0 at the radius)
    "SMOOTH": lambda t: 3 * t * t - 2 * t * t * t, "SPHERE": lambda t: math.sqrt(max(0.0, 2 * t - t * t)),
    "ROOT": lambda t: math.sqrt(t), "SHARP": lambda t: t * t, "LINEAR": lambda t: t, "CONSTANT": lambda t: 1.0,
}


def _transform_selection(bm, verts, transform, p):
    """Apply ``transform`` to the selected vertices; with ``proportional`` (O), vertices within that radius of
    the selection follow it partly, fading with the distance (Blender's proportional editing)."""
    followed = 0
    if p.get("proportional"):
        from mathutils.kdtree import KDTree

        radius = p["proportional"]
        falloff = FALLOFFS[p.get("falloff") or "SMOOTH"]
        tree = KDTree(len(verts))
        for i, v in enumerate(verts):
            tree.insert(v.co, i)
        tree.balance()
        selected = set(verts)
        moves = []
        for v in bm.verts:
            if v in selected:
                continue
            _co, _index, distance = tree.find(v.co)
            if distance < radius:
                weight = falloff(1.0 - distance / radius)
                moves.append((v, v.co + (transform(v.co.copy()) - v.co) * weight))
        for v, co in moves:
            v.co = co
        followed = len(moves)
    for v in verts:
        v.co = transform(v.co.copy())
    bm.normal_update()
    return followed


def translate_selection(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = _selected_verts(bm)
    offset = Vector(p["offset"])
    followed = _transform_selection(bm, verts, lambda co: co + offset, p)
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "moved": len(verts), "followed": followed}


def scale_selection(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = _selected_verts(bm)
    if p["pivot"] == "median":
        pivot = sum((v.co for v in verts), Vector()) / len(verts)
    elif p["pivot"] == "origin":
        pivot = Vector()
    else:
        lo, hi = _local_bbox(bm)
        pivot = (lo + hi) / 2
    factor = p["factor"]
    followed = _transform_selection(
        bm, verts, lambda co: Vector([pivot[i] + (co[i] - pivot[i]) * factor[i] for i in range(3)]), p)
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "scaled": len(verts), "followed": followed}


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
    # Blender's I key insets open borders too and keeps the rim even; bmesh's own defaults do neither
    # (a lone face -- a filled circle, a plane -- would not inset at all).
    if p["individual"]:
        # I pressed twice: every face inset on its own (tiles, stickers, panels), each with its own rim.
        result = bmesh.ops.inset_individual(bm, faces=faces, thickness=p["thickness"], depth=p["depth"],
                                            use_even_offset=True)
    else:
        result = bmesh.ops.inset_region(bm, faces=faces, thickness=p["thickness"], depth=p["depth"],
                                        use_boundary=True, use_even_offset=True)
    _select_only(bm, [f for f in faces if f.is_valid])   # like I: the inner faces stay selected
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


def _box_coord(obj, bm, space):
    if space == "normalized":
        lo, hi = _local_bbox(bm)
        span = [(hi[i] - lo[i]) or 1.0 for i in range(3)]
        return lambda co: [(co[i] - lo[i]) / span[i] for i in range(3)]
    if space == "world":
        return lambda co: list(obj.matrix_world @ co)
    return lambda co: list(co)


def select_box(p):
    """Select vertices, edges or faces inside a box (any bound may be null = unbounded).

    Faces are tested by their centre, edges by both vertices. ``facing`` keeps only faces whose normal
    points that way (``min_dot``); ``sharp_deg`` keeps only edges whose faces meet at least at that
    angle, and boundary edges (a rim, the lip of a mug) -- what Alt+click on an edge loop usually picks;
    ``boundary`` keeps only edges of holes and open rims (what bridge_edge_loops joins).
    """
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    coord = _box_coord(obj, bm, p["space"])
    low, high = p["min"], p["max"]

    def inside(co):
        c = coord(co)
        return all((low[i] is None or c[i] >= low[i] - 1e-6) and (high[i] is None or c[i] <= high[i] + 1e-6)
                   for i in range(3))

    if not p["extend"]:
        for elem in (*bm.verts, *bm.edges, *bm.faces):
            elem.select = False
    facing = Vector(p["facing"]).normalized() if p["facing"] is not None else None
    count = 0
    fallback = None
    if p["element"] == "FACE":
        bm.select_mode = {"FACE"}
        for face in bm.faces:
            if inside(face.calc_center_median()) and (facing is None or face.normal.dot(facing) >= p["min_dot"]):
                face.select = True
                count += 1
    elif p["element"] == "EDGE":
        import math
        bm.select_mode = {"EDGE"}
        limit = math.radians(p["sharp_deg"]) if p["sharp_deg"] is not None else None
        boxed = [e for e in bm.edges if all(inside(v.co) for v in e.verts)]
        chosen = [e for e in boxed if len(e.link_faces) == 1] if p["boundary"] else boxed
        if p["boundary"] and not chosen:
            # A closed mesh has no open rim: what a loop selection there means is its sharp edges (a mug's lip).
            limit = limit if limit is not None else math.radians(30.0)
            chosen = boxed
            fallback = "sharp edges (the box holds no open rim)"
        for edge in chosen:
            if limit is not None and len(edge.link_faces) == 2 and edge.calc_face_angle(0.0) < limit:
                continue
            edge.select = True
            count += 1
    else:
        bm.select_mode = {"VERT"}
        for vert in bm.verts:
            if inside(vert.co):
                vert.select = True
                count += 1
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    if count == 0:
        lo, hi = _local_bbox(bm)
        raise BridgeCommandError("empty_selection", "nothing inside the box (the mesh spans x {:.3g}..{:.3g}, "
                                 "y {:.3g}..{:.3g}, z {:.3g}..{:.3g} in local coordinates)".format(
                                     lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]), object=obj.name)
    out = {"object": obj.name, "selected": count, "element": p["element"],
           "verts": sum(v.select for v in bm.verts), "faces": sum(f.select for f in bm.faces)}
    if fallback:
        out["fallback"] = fallback
    return out


def rotate_selection(p):
    """Rotate the selected elements about an axis through the selection (R, axis, angle)."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    verts = _selected_verts(bm)
    if p["pivot"] == "median":
        pivot = sum((v.co for v in verts), Vector()) / len(verts)
    elif p["pivot"] == "bbox_center":
        lo, hi = _local_bbox(bm)
        pivot = (lo + hi) / 2
    else:
        pivot = Vector()
    rotation = Matrix.Rotation(p["angle"], 3, p["axis"].upper())
    followed = _transform_selection(bm, verts, lambda co: pivot + rotation @ (co - pivot), p)
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "rotated": len(verts), "followed": followed}


def duplicate_selection(p):
    """Shift+D in edit mode: copy the selected part of the mesh, moved by ``offset``; the copy stays selected."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    _selected_verts(bm)
    geom = [*(v for v in bm.verts if v.select), *(e for e in bm.edges if e.select), *(f for f in bm.faces if f.select)]
    copy = bmesh.ops.duplicate(bm, geom=geom)["geom"]
    verts = [g for g in copy if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=Vector(p["offset"]), verts=verts)
    _select_only(bm, copy)
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "copied_verts": len(verts)}


def select_linked(p):
    """L / Ctrl+L: grow the selection to everything connected to it (a whole part of the mesh)."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    seen = set(_selected_verts(bm))
    todo = list(seen)
    while todo:
        v = todo.pop()
        for e in v.link_edges:
            other = e.other_vert(v)
            if other not in seen:
                seen.add(other)
                todo.append(other)
    _select_only(bm, [*seen, *(e for e in bm.edges if e.verts[0] in seen), *(f for f in bm.faces if f.verts[0] in seen)])
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "selected_verts": len(seen)}


def delete_elements(p):
    """X in edit mode: delete the selected vertices, edges, faces, or only the faces (keeping the rim)."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    context = {"VERTS": "VERTS", "EDGES": "EDGES", "FACES": "FACES", "ONLY_FACES": "FACES_ONLY"}[p["what"]]
    if p["what"] == "VERTS":
        geom = [v for v in bm.verts if v.select]
    elif p["what"] == "EDGES":
        geom = [e for e in bm.edges if e.select]
    else:
        geom = [f for f in bm.faces if f.select]
    if not geom:
        raise BridgeCommandError("empty_selection", f"no {p['what'].lower()} selected")
    bmesh.ops.delete(bm, geom=geom, context=context)
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "deleted": len(geom), "verts": len(bm.verts), "faces": len(bm.faces)}


def bridge_edge_loops(p):
    """Join two selected edge loops (or two holes) with a band of faces."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    faces = [f for f in bm.faces if f.select]
    before = len(bm.faces)
    if faces:
        # Two faces selected (a slat's front and back): Blender removes them and bridges their rims -- a clean
        # hole through the mesh, walls included.
        edges = list({e for f in faces for e in f.edges if sum(1 for g in e.link_faces if g.select) == 1})
        bmesh.ops.delete(bm, geom=faces, context="FACES_ONLY")
    else:
        edges = [e for e in bm.edges if e.select]
    if len(edges) < 2:
        raise BridgeCommandError("empty_selection", "select two edge loops (or two faces) to bridge")
    result = bmesh.ops.bridge_loops(bm, edges=edges)
    if p["cuts"]:
        inner = [e for e in result.get("edges", []) if e not in edges]
        if inner:
            bmesh.ops.subdivide_edges(bm, edges=inner, cuts=p["cuts"], use_grid_fill=True)
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "new_faces": len(bm.faces) - before}


def fill(p):
    """F: make a face (or faces) from the selected boundary."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    edges = [e for e in bm.edges if e.select]
    verts = [v for v in bm.verts if v.select]
    if not verts:
        raise BridgeCommandError("empty_selection", "nothing selected to fill")
    before = len(bm.faces)
    if p["grid"] and edges:
        bmesh.ops.grid_fill(bm, edges=edges)
    else:
        bmesh.ops.contextual_create(bm, geom=[*verts, *edges])
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "new_faces": len(bm.faces) - before}


def subdivide(p):
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    edges = [e for e in bm.edges if e.select]
    if not edges:
        raise BridgeCommandError("empty_selection", "no edges selected")
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=p["cuts"], use_grid_fill=True, smooth=p["smoothness"])
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "verts": len(bm.verts)}


def separate_selection(p):
    """Shift+D, P: copy (or move) the selected faces into a new object -- icing on a donut."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    if not any(f.select for f in bm.faces):
        raise BridgeCommandError("empty_selection", "no faces selected")
    before = set(bpy.data.objects.keys())
    with bpy.context.temp_override(**_context_override(obj)):
        if p["duplicate"]:
            _op_result(bpy.ops.mesh.duplicate(), "duplicate")
        _op_result(bpy.ops.mesh.separate(type="SELECTED"), "separate")
        bpy.ops.object.mode_set(mode="OBJECT")
    created = sorted(set(bpy.data.objects.keys()) - before)
    if not created:
        raise BridgeCommandError("operator_failed", "separate made no object")
    new = bpy.data.objects[created[0]]
    if p["new_name"]:
        new.name = p["new_name"]
        new.data.name = p["new_name"]
    for o in bpy.context.view_layer.objects:
        o.select_set(o == new)
    bpy.context.view_layer.objects.active = new
    return {"object": new.name, "from": obj.name, "faces": len(new.data.polygons)}


def recalc_normals(p):
    """Shift+N: make every face point outwards (or inwards) consistently."""
    obj = _obj(p["object"])
    bm = _edit_bmesh(obj)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    if p["inside"]:
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    bmesh.update_edit_mesh(obj.data)
    return {"object": obj.name, "faces": len(bm.faces)}


def duplicate_object(p):
    """Shift+D on an object: a copy (with its modifiers and materials), optionally moved."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    new = obj.copy()
    if obj.data is not None and not p["linked"]:
        new.data = obj.data.copy()
    for collection in obj.users_collection:
        collection.objects.link(new)
    if p["new_name"]:
        new.name = p["new_name"]
    new.location = obj.location + Vector(p["offset"])
    if p["rotation"] is not None:
        new.rotation_euler = Vector(obj.rotation_euler) + Vector(p["rotation"])
    for o in bpy.context.view_layer.objects:
        o.select_set(o == new)
    bpy.context.view_layer.objects.active = new
    return {"object": new.name, "location": list(new.location)}


# Every mesh modifier a step may add (not the ones that read files or have their own actions: Geometry Nodes ->
# edit_nodes, particle systems -> add_particles, fluid -> quick_liquid).
MODIFIER_TYPES = (
    "ARRAY", "BEVEL", "BOOLEAN", "BUILD", "DECIMATE", "EDGE_SPLIT", "MASK", "MIRROR", "MULTIRES", "REMESH", "SCREW",
    "SKIN", "SOLIDIFY", "SUBSURF", "TRIANGULATE", "WELD", "WIREFRAME", "CAST", "CURVE", "DISPLACE", "HOOK",
    "LAPLACIANDEFORM", "LATTICE", "MESH_DEFORM", "SHRINKWRAP", "SIMPLE_DEFORM", "SMOOTH", "CORRECTIVE_SMOOTH",
    "LAPLACIANSMOOTH", "SURFACE_DEFORM", "WARP", "WAVE", "WEIGHTED_NORMAL", "NORMAL_EDIT", "UV_PROJECT", "UV_WARP",
    "VERTEX_WEIGHT_EDIT", "VERTEX_WEIGHT_MIX", "VERTEX_WEIGHT_PROXIMITY", "DATA_TRANSFER", "CLOTH", "SOFT_BODY",
    "COLLISION", "EXPLODE", "OCEAN", "PARTICLE_INSTANCE", "ARMATURE")
NAME_SETTINGS = ("vertex_group", "vertex_group_a", "vertex_group_b", "subtarget", "uv_layer", "bone_from",
                 "bone_to", "mask_vertex_group")


def _modifier_setting(modifier, key, value):
    """A modifier setting by its Python name, checked against Blender's definition: numbers, switches, menu
    choices, objects / collections / textures by name, vertex groups by name; ``<name>_deg`` takes degrees."""
    from .nodes import _rna_value

    degrees = key.endswith("_deg")
    attr = key[:-4] if degrees else key
    prop = modifier.bl_rna.properties.get(attr)
    if prop is None:
        raise BridgeCommandError("invalid_param", f"{modifier.type} has no setting {attr!r}", param=key)
    if prop.type == "POINTER":
        kind = prop.fixed_type.identifier
        if kind == "Object":
            setattr(modifier, attr, _obj(value) if value else None)
        elif kind == "Collection":
            collection = bpy.data.collections.get(value) if isinstance(value, str) else None
            if collection is None:
                raise BridgeCommandError("invalid_param", f"collection {value!r} not found", param=key)
            setattr(modifier, attr, collection)
        elif kind == "Texture":
            texture = bpy.data.textures.get(value) if isinstance(value, str) else None
            if texture is None:
                raise BridgeCommandError("invalid_param", f"texture {value!r} not found", param=key)
            setattr(modifier, attr, texture)
        else:
            raise BridgeCommandError("invalid_param", f"{attr} ({kind}) can't be set by a step", param=key)
        return
    if prop.type == "STRING":
        if attr not in NAME_SETTINGS and not attr.endswith("vertex_group"):
            raise BridgeCommandError("invalid_param", f"{attr} is text and can't be set by a step", param=key)
        if not isinstance(value, str) or len(value) > 63:
            raise BridgeCommandError("invalid_param", f"{attr} is a name (up to 63 characters)", param=key)
        setattr(modifier, attr, value)
        return
    setattr(modifier, attr, _rna_value(modifier, attr, value, key, degrees=degrees))


def add_modifier(p):
    obj = _obj(p["object"])
    known = MODIFIER_PROPS.get(p["type"], {})
    props = p["props"] or {}
    if not isinstance(props, dict):
        raise BridgeCommandError("invalid_param", "props must be an object", param="props")
    clean = {key: _check(value, known[key], key) for key, value in props.items() if key in known}
    name = p["name"] or p["type"].title()
    existing = obj.modifiers.get(name)
    if existing is not None and existing.type == p["type"]:
        modifier = existing   # the same modifier again: its values change, as in the properties panel
    else:
        modifier = obj.modifiers.new(name=name, type=p["type"])
        if modifier is None:
            raise BridgeCommandError("invalid_param", f"{obj.name} ({obj.type}) can't take a {p['type']} modifier",
                                     param="type")
    texture_kind = clean.pop("texture", None)
    texture_scale = clean.pop("texture_scale", None)
    if texture_kind is not None:
        # A procedural texture makes the displacement irregular (a lumpy donut, a rough stone).
        texture = bpy.data.textures.new(f"{obj.name}_{texture_kind.lower()}", type=texture_kind)
        if texture_scale is not None and hasattr(texture, "noise_scale"):
            texture.noise_scale = texture_scale
        modifier.texture = texture
    for key, value in clean.items():
        setattr(modifier, key, value)
    for key, value in props.items():
        if key not in known:
            _modifier_setting(modifier, key, value)
    bpy.context.view_layer.update()
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
    editing = obj.mode == "EDIT"
    # In edit mode the edit mesh is written back on Tab: set it there, or the change is lost.
    bm = bmesh.from_edit_mesh(obj.data) if editing else bmesh.new()
    if not editing:
        bm.from_mesh(obj.data)
    for face in bm.faces:
        face.smooth = p["smooth"]
    sharp = 0
    if p["auto_smooth_deg"] is not None:
        # Auto smooth / smooth by angle: edges whose faces meet at more than the angle stay sharp.
        limit = math.radians(p["auto_smooth_deg"])
        for edge in bm.edges:
            hard = len(edge.link_faces) == 2 and edge.calc_face_angle(0.0) > limit
            edge.smooth = not hard
            sharp += hard
    if editing:
        bmesh.update_edit_mesh(obj.data)
    else:
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
    return {"object": obj.name, "smooth": p["smooth"], "sharp_edges": sharp}


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


TEXTURE_NAME = re.compile(r"^textures/[A-Za-z0-9_\-]{1,64}\.(png|jpg)$")


def _texture_path(path, write):
    """A texture file for a recipe: ``textures/<name>.png`` lives in the folder Lucius may both write and read
    (so a course replays anywhere); an absolute path must be inside the allowed folders."""
    if os.path.isabs(path):
        return _check_path(path, "LUCIUS_ALLOWED_SAVE_DIRS" if write else "LUCIUS_ALLOWED_READ_DIRS",
                           (".png", ".jpg", ".jpeg"))
    if not TEXTURE_NAME.match(path):
        raise BridgeCommandError("invalid_param", "a texture is textures/<name>.png (letters, digits, _ and -)",
                                 param="path")
    read, save = _allowed_dirs("LUCIUS_ALLOWED_READ_DIRS"), _allowed_dirs("LUCIUS_ALLOWED_SAVE_DIRS")
    if write:
        base = next((d for d in save if d in read), save[0] if save else None)
        if base is None:
            raise BridgeCommandError("path_not_allowed", "no folder is allowed for textures")
        return os.path.join(base, path)
    for base in read:
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    raise BridgeCommandError("invalid_param", f"{path} has not been made yet (bake_texture writes it)", param="path")


BAKE_TYPES = ("DIFFUSE", "ROUGHNESS", "NORMAL", "AO", "EMIT", "COMBINED", "GLOSSY")


def bake_texture(p):
    """Render > Bake (Cycles) into a new image and save it: the colour (Diffuse, colour only), roughness, a
    tangent normal map, ambient occlusion, or anything routed into an emission (e.g. a height map) of the
    object's active material, on its UV map. The image lands at ``path`` (textures/<name>.png)."""
    _leave_edit_mode()
    obj = _obj(p["object"])
    if obj.type != "MESH" or not obj.data.uv_layers:
        raise BridgeCommandError("invalid_param", f"{obj.name} needs a UV map (uv_unwrap) to bake", param="object")
    material = obj.active_material
    if material is None or not material.use_nodes:
        raise BridgeCommandError("invalid_param", f"{obj.name} has no node material to bake", param="object")
    if not (16 <= p["width"] <= 4096 and 16 <= p["height"] <= 4096):
        raise BridgeCommandError("invalid_param", "16..4096 pixels", param="width")
    path = _texture_path(p["path"], write=True)
    image = bpy.data.images.new(os.path.splitext(os.path.basename(path))[0], p["width"], p["height"], alpha=False)
    if p["type"] in ("ROUGHNESS", "NORMAL", "AO", "EMIT") and p["non_color"]:
        image.colorspace_settings.name = "Non-Color"
    tree = material.node_tree
    target = tree.nodes.new("ShaderNodeTexImage")
    target.image = image
    for node in tree.nodes:
        node.select = False
    target.select = True
    tree.nodes.active = target
    scene = bpy.context.scene
    saved = (scene.render.engine, scene.cycles.samples, scene.cycles.device)
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = p["samples"]
        kwargs = {"type": p["type"], "margin": p["margin"], "use_clear": True}
        if p["type"] in ("DIFFUSE", "GLOSSY"):
            kwargs["pass_filter"] = {"COLOR"}
        if p["type"] == "NORMAL":
            kwargs["normal_space"] = "TANGENT"
        override = dict(_context_override(obj), selected_objects=[obj], selected_editable_objects=[obj])
        with bpy.context.temp_override(**override):
            _op_result(bpy.ops.object.bake(**kwargs), "bake")
    finally:
        tree.nodes.remove(target)
        scene.render.engine, scene.cycles.samples, scene.cycles.device = saved
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    return {"object": obj.name, "type": p["type"], "path": p["path"], "size": [p["width"], p["height"]]}


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
    """Replace the scene with the scene of a .blend file (to continue it or inspect it): its objects, its
    collections (including ones used only by particle systems), camera, world and render settings.

    Uses library appending (data only): scripts embedded in the file are never executed.
    """
    path = _check_path(p["path"], "LUCIUS_ALLOWED_READ_DIRS", (".blend",))
    _leave_edit_mode()
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for child in list(scene.collection.children):
        scene.collection.children.unlink(child)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    _purge_orphans()
    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        data_to.scenes = list(data_from.scenes[:1])
    saved = data_to.scenes[0] if data_to.scenes else None
    if saved is None:
        raise BridgeCommandError("invalid_param", "the file has no scene", param="path")
    for child in list(saved.collection.children):
        scene.collection.children.link(child)
    for obj in list(saved.collection.objects):
        scene.collection.objects.link(obj)
    # A saved scene continues where it stopped: camera, world, render and frame settings come back too.
    scene.world = saved.world
    render, old = scene.render, saved.render
    render.engine = old.engine
    render.resolution_x, render.resolution_y = old.resolution_x, old.resolution_y
    render.resolution_percentage = old.resolution_percentage
    render.fps = old.fps
    scene.frame_start, scene.frame_end, scene.frame_current = saved.frame_start, saved.frame_end, saved.frame_current
    scene.view_settings.view_transform = saved.view_settings.view_transform
    if hasattr(scene, "cycles") and hasattr(saved, "cycles"):
        scene.cycles.samples = saved.cycles.samples
        scene.cycles.use_denoising = saved.cycles.use_denoising
    if hasattr(scene, "eevee") and hasattr(saved, "eevee"):
        scene.eevee.taa_render_samples = saved.eevee.taa_render_samples
    render.use_motion_blur = old.use_motion_blur
    render.film_transparent = old.film_transparent
    for key in list(saved.keys()):
        if key.startswith("lucius_") and isinstance(saved[key], (str, int, float)):
            scene[key] = saved[key]   # choices a step made for the scene, e.g. the render engine
    camera = saved.camera
    bpy.data.scenes.remove(saved)
    bpy.context.view_layer.update()   # appended objects get their world matrices only on an update
    names = sorted(o.name for o in scene.objects)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if meshes and bpy.context.view_layer.objects.active is None:
        bpy.context.view_layer.objects.active = meshes[-1]   # as after working on it: something is active
    cameras = sorted((o for o in scene.objects if o.type == "CAMERA"), key=lambda o: o.name)
    if camera is not None and camera.name in scene.objects:
        scene.camera = camera
    elif cameras and (scene.camera is None or scene.camera.name not in scene.objects):
        scene.camera = cameras[0]
    return {"objects": names, "camera": scene.camera.name if scene.camera else None,
            "collections": [c.name for c in scene.collection.children],
            "engine": scene.get("lucius_engine") or scene.render.engine}


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
    bpy.context.view_layer.update()
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
        "vertices": ("int", 32), "name": ("name", None), "radius": ("float", None), "radius2": ("float", None),
        "depth": ("float", None), "major_radius": ("float", None), "minor_radius": ("float", None),
        "major_segments": ("int", None), "minor_segments": ("int", None), "fill": ("bool", False),
        "into": ("name", None), "rings": ("int", None)}),
    "duplicate_selection": (duplicate_selection, {"object": OBJ, "offset": V3}),
    "select_linked": (select_linked, {"object": OBJ}),
    "select_objects": (select_objects, {"names": ("names", REQUIRED), "active": OBJ, "deselect_others": ("bool", True)}),
    "delete_objects": (delete_objects, {"names": ("names", REQUIRED)}),
    "join_objects": (join_objects, {"names": ("names", REQUIRED), "into": ("name", REQUIRED)}),
    "parent_object": (parent_object, {"object": OBJ, "parent": ("name", None)}),
    "hide_objects": (hide_objects, {"names": ("names", REQUIRED), "hide": ("bool", True), "render": ("bool", True)}),
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
    "skin_radius": (skin_radius, {"object": OBJ, "radius": ("float", REQUIRED)}),
    "bake_texture": (bake_texture, {"object": OBJ, "type": (BAKE_TYPES, "DIFFUSE"), "path": ("path", REQUIRED),
                                    "width": ("int", 1024), "height": ("int", 1024), "samples": ("int", 16),
                                    "margin": ("int", 8), "non_color": ("bool", True)}),
    "uv_unwrap": (uv_unwrap, {"object": OBJ, "method": (UNWRAP, "SMART_PROJECT"), "margin": ("float", 0.02),
                              "angle_limit_deg": ("float", 66.0), "size": ("float", 1.0)}),
    "uv_transform": (uv_transform, {"object": OBJ, "rotate_deg": ("float", 0.0), "scale": ("vec2", [1.0, 1.0]),
                                    "offset": ("vec2", [0.0, 0.0])}),
    "move_lattice_points": (move_lattice_points, {
        "object": OBJ, "min": ("bounds3", [None, None, None]), "max": ("bounds3", [None, None, None]),
        "offset": ("vec3", [0.0, 0.0, 0.0]), "factor": ("vec3", [1.0, 1.0, 1.0])}),
    "add_hook": (add_hook, {"object": OBJ, "hook": ("name", None), "size": ("float", 0.3)}),
    "bind_modifier": (bind_modifier, {"object": OBJ, "modifier": ("name", REQUIRED)}),
    "knife_cut": (knife_cut, {"object": OBJ, "start": ("vec3", REQUIRED), "end": ("vec3", REQUIRED),
                              "view": ("vec3", [0.0, -1.0, 0.0]), "through": ("bool", False),
                              "space": (("local", "world"), "local")}),
    "bisect": (bisect, {"object": OBJ, "point": ("vec3", [0.0, 0.0, 0.0]), "normal": ("vec3", [0.0, 0.0, 1.0]),
                        "clear_inner": ("bool", False), "clear_outer": ("bool", False), "fill": ("bool", False),
                        "space": (("local", "world"), "local")}),
    "spin": (spin, {"object": OBJ, "axis": (("x", "y", "z"), "z"), "center": ("vec3", [0.0, 0.0, 0.0]),
                    "angle": ("float", 1.5707963), "steps": ("int", 8), "space": (("local", "world"), "local")}),
    "slide_selection": (slide_selection, {"object": OBJ, "toward": ("vec3", REQUIRED), "factor": ("float", 0.5)}),
    "shrink_fatten": (shrink_fatten, {"object": OBJ, "distance": ("float", REQUIRED)}),
    "select_nth": (select_nth, {"object": OBJ, "skip": ("int", 1), "nth": ("int", 1), "offset": ("int", 0)}),
    "select_box": (select_box, {
        "object": OBJ, "min": ("bounds3", [None, None, None]), "max": ("bounds3", [None, None, None]),
        "element": (("VERT", "EDGE", "FACE"), "FACE"), "space": (("normalized", "local", "world"), "normalized"),
        "facing": ("vec3", None), "min_dot": ("float", 0.7), "sharp_deg": ("float", None), "boundary": ("bool", False),
        "extend": ("bool", False)}),
    "extrude": (extrude, {"object": OBJ, "offset": ("vec3", None), "distance": ("float", None),
                          "individual": ("bool", False)}),
    "rotate_selection": (rotate_selection, {
        "object": OBJ, "axis": (tuple(AXES), REQUIRED), "angle": ("float", REQUIRED),
        "pivot": (("median", "bbox_center", "origin"), "median"), "proportional": ("float", None), "falloff": (tuple(FALLOFFS), "SMOOTH")}),
    "delete_elements": (delete_elements, {"object": OBJ, "what": (("VERTS", "EDGES", "FACES", "ONLY_FACES"), "FACES")}),
    "bridge_edge_loops": (bridge_edge_loops, {"object": OBJ, "cuts": ("int", 0)}),
    "fill": (fill, {"object": OBJ, "grid": ("bool", False)}),
    "subdivide": (subdivide, {"object": OBJ, "cuts": ("int", 1), "smoothness": ("float", 0.0)}),
    "recalc_normals": (recalc_normals, {"object": OBJ, "inside": ("bool", False)}),
    "separate_selection": (separate_selection, {"object": OBJ, "new_name": ("name", None), "duplicate": ("bool", True)}),
    "duplicate_object": (duplicate_object, {"object": OBJ, "new_name": ("name", None), "offset": V3,
                                            "rotation": ("vec3", None), "linked": ("bool", False)}),
    "translate_selection": (translate_selection, {"object": OBJ, "offset": ("vec3", REQUIRED), "proportional": ("float", None), "falloff": (tuple(FALLOFFS), "SMOOTH")}),
    "scale_selection": (scale_selection, {
        "object": OBJ, "factor": ("vec3", REQUIRED), "pivot": (("median", "bbox_center", "origin"), "median"), "proportional": ("float", None), "falloff": (tuple(FALLOFFS), "SMOOTH")}),
    "taper_selection": (taper_selection, {
        "object": OBJ, "along": (tuple(AXES), REQUIRED), "affect": (("x", "y", "z", "xy", "xz", "yz"), "x"),
        "amount": ("float", REQUIRED), "start": ("float", 0.0), "reverse": ("bool", False)}),
    "inset": (inset, {"object": OBJ, "thickness": ("float", REQUIRED), "depth": ("float", 0.0),
                      "individual": ("bool", False)}),
    "bevel": (bevel, {"object": OBJ, "offset": ("float", REQUIRED), "segments": ("int", 1),
                      "affect": (("EDGES", "VERTICES"), "EDGES")}),
    "loop_cut_axis": (loop_cut_axis, {"object": OBJ, "axis": (tuple(AXES), REQUIRED), "positions": ("positions", REQUIRED)}),
    "merge_by_distance": (merge_by_distance, {"object": OBJ, "distance": ("float", 0.0001)}),
    "add_modifier": (add_modifier, {
        "object": OBJ, "type": (MODIFIER_TYPES, REQUIRED), "name": ("name", None), "props": ("dict", None)}),
    "remove_modifier": (remove_modifier, {"object": OBJ, "modifier": ("name", REQUIRED)}),
    "apply_modifier": (apply_modifier, {"object": OBJ, "modifier": ("name", REQUIRED)}),
    "set_symmetry": (set_symmetry, {"object": OBJ, "axes": ("bool3", REQUIRED)}),
    "shade": (shade, {"object": OBJ, "smooth": ("bool", True), "auto_smooth_deg": ("float", None)}),
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



def registry():
    """Every allowlisted action: these plus materials, lights, camera and rendering (their own module)."""
    from . import anim, nodes, paint, physics, rig, scene

    return {**ACTIONS, **scene.ACTIONS, **anim.ACTIONS, **rig.ACTIONS, **nodes.ACTIONS, **physics.ACTIONS,
            **paint.ACTIONS}


def execute_action(name, args):
    actions = registry()
    if name not in actions:
        raise BridgeCommandError("unknown_action", f"action {name!r} is not allowlisted", action=name)
    handler, spec = actions[name]
    if not isinstance(args, dict):
        raise BridgeCommandError("invalid_param", "args must be an object")
    params = validate(spec, args)
    result = handler(params)
    return {"action": name, "result": result}


def describe_actions():
    return {
        name: {param: (list(kind) if isinstance(kind, tuple) else kind) for param, (kind, _d) in spec.items()}
        for name, (_h, spec) in registry().items()
    }
