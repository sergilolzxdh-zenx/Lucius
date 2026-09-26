"""Vertex paint: colour attributes (Object Data > Color Attributes), brush strokes painted into them, and
Paint > Dirty Vertex Colors. A material shows them through an Attribute node named after the attribute."""

import math

import bpy

from .actions import REQUIRED, _context_override, _leave_edit_mode, _obj, _op_result
from .nodes import _color
from .protocol import BridgeCommandError

MAX_STROKE_POINTS = 256


def _mesh_object(name):
    obj = _obj(name)
    if obj.type != "MESH":
        raise BridgeCommandError("invalid_param", f"{obj.name} is not a mesh", param="object")
    return obj


def _attribute(mesh, name, param="attribute"):
    if name is None:
        attr = mesh.color_attributes.active_color
        if attr is None:
            raise BridgeCommandError("invalid_param", f"{mesh.name} has no colour attribute yet (color_attribute)",
                                     param=param)
        return attr
    attr = mesh.color_attributes.get(name)
    if attr is None:
        names = [a.name for a in mesh.color_attributes]
        raise BridgeCommandError("invalid_param", f"no colour attribute {name!r} (has: {', '.join(names) or 'none'})",
                                 param=param)
    return attr


def _make_active(mesh, attr):
    mesh.color_attributes.active_color = attr
    if hasattr(mesh.color_attributes, "render_color_index"):
        for i, a in enumerate(mesh.color_attributes):
            if a.name == attr.name:
                mesh.color_attributes.render_color_index = i


def color_attribute(p):
    """Object Data > Color Attributes: + adds one (a name, the colour it starts filled with, per face corner or
    per vertex, byte or float), - removes one; the new one becomes the active one vertex paint writes into."""
    _leave_edit_mode()
    obj = _mesh_object(p["object"])
    mesh = obj.data
    name = p["name"]
    if not isinstance(name, str) or not 0 < len(name) <= 63:
        raise BridgeCommandError("invalid_param", "name: 1..63 characters", param="name")
    if p["remove"]:
        attr = _attribute(mesh, name, "name")
        mesh.color_attributes.remove(attr)
        return {"object": obj.name, "removed": name, "attributes": [a.name for a in mesh.color_attributes]}
    attr = mesh.color_attributes.get(name)
    if attr is None:
        if len(mesh.color_attributes) >= 16:
            raise BridgeCommandError("invalid_param", "16 colour attributes at most", param="name")
        attr = mesh.color_attributes.new(name, p["data_type"], p["domain"])
        rgba = _color(p["color"], "color")
        values = rgba * len(attr.data)
        attr.data.foreach_set("color", values)
    if p["active"]:
        _make_active(mesh, attr)
    mesh.update()
    return {"object": obj.name, "attribute": attr.name, "domain": attr.domain, "data_type": attr.data_type,
            "attributes": [a.name for a in mesh.color_attributes]}


def _stroke_points(stroke, param, to_local):
    from mathutils import Vector

    points = stroke.get("points")
    if not isinstance(points, list) or not 1 <= len(points) <= MAX_STROKE_POINTS:
        raise BridgeCommandError("invalid_param", f"{param}: points is a list of 1..{MAX_STROKE_POINTS} [x, y, z]",
                                 param="strokes")
    out = []
    for pt in points:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 3 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in pt)):
            raise BridgeCommandError("invalid_param", f"{param}: each point is [x, y, z]", param="strokes")
        out.append(to_local(Vector(pt)))
    return out


def _segment_distance(co, a, b):
    ab = b - a
    length = ab.length_squared
    if length == 0.0:
        return (co - a).length
    t = max(0.0, min(1.0, (co - a).dot(ab) / length))
    return (co - (a + ab * t)).length


FALLOFFS = {
    "SMOOTH": lambda x: 3 * (1 - x) ** 2 - 2 * (1 - x) ** 3,
    "LINEAR": lambda x: 1 - x,
    "CONSTANT": lambda x: 1.0,
    "SPHERE": lambda x: math.sqrt(max(0.0, 1 - x * x)),
}


def _blend(mode, base, color, weight):
    if mode == "MIX":
        target = color
    elif mode == "MULTIPLY":
        target = [b * c for b, c in zip(base, color)]
    elif mode == "ADD":
        target = [b + c for b, c in zip(base, color)]
    elif mode == "SUBTRACT":
        target = [max(0.0, b - c) for b, c in zip(base, color)]
    elif mode == "DARKEN":
        target = [min(b, c) for b, c in zip(base, color)]
    else:   # LIGHTEN
        target = [max(b, c) for b, c in zip(base, color)]
    return [b + (t - b) * weight for b, t in zip(base, target)]


def vertex_paint(p):
    """Vertex paint mode: brush strokes of a colour into a colour attribute. Each stroke is a line through
    points (the brush dragged over the surface) with a radius; the paint fades out towards the edge of the brush
    (falloff) and ``strength`` is the brush strength. ``selected`` paints the selected faces instead (the face
    selection mask), ``fill`` the whole mesh."""
    from mathutils import Vector

    obj = _mesh_object(p["object"])
    was_edit = obj.mode == "EDIT"
    if was_edit:
        obj.update_from_editmode()
    _leave_edit_mode()
    mesh = obj.data
    attr = _attribute(mesh, p["attribute"])
    color = _color(p["color"], "color")
    strength = p["strength"]
    if not 0.0 < strength <= 1.0:
        raise BridgeCommandError("invalid_param", "strength 0..1", param="strength")
    inverse = obj.matrix_world.inverted()
    to_local = (lambda v: inverse @ v) if p["space"] == "world" else (lambda v: v)
    weights = [0.0] * len(mesh.vertices)
    if p["fill"]:
        weights = [strength] * len(mesh.vertices)
    elif p["selected"]:
        chosen = {v for poly in mesh.polygons if poly.select for v in poly.vertices}
        if not chosen:
            chosen = {v.index for v in mesh.vertices if v.select}
        if not chosen:
            raise BridgeCommandError("invalid_param", "nothing is selected to paint", param="selected")
        for i in chosen:
            weights[i] = strength
    else:
        strokes = p["strokes"]
        if not isinstance(strokes, list) or not 1 <= len(strokes) <= 64:
            raise BridgeCommandError("invalid_param", "strokes: 1..64 of {points, radius}", param="strokes")
        falloff = FALLOFFS[p["falloff"]]
        for s, stroke in enumerate(strokes):
            param = f"strokes[{s}]"
            if not isinstance(stroke, dict):
                raise BridgeCommandError("invalid_param", f"{param}: {{points, radius}}", param="strokes")
            radius = stroke.get("radius", p["radius"])
            if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not 0 < radius <= 1000:
                raise BridgeCommandError("invalid_param", f"{param}: radius > 0", param="strokes")
            # the radius is in the space of the points; in object space it scales with the object
            local_radius = radius / (max(obj.matrix_world.to_scale()) or 1.0) if p["space"] == "world" else radius
            pts = _stroke_points(stroke, param, to_local)
            segments = list(zip(pts, pts[1:])) or [(pts[0], pts[0])]
            lo = Vector([min(v[i] for v in pts) - local_radius for i in range(3)])
            hi = Vector([max(v[i] for v in pts) + local_radius for i in range(3)])
            for v in mesh.vertices:
                co = v.co
                if any(co[i] < lo[i] or co[i] > hi[i] for i in range(3)):
                    continue
                d = min(_segment_distance(co, a, b) for a, b in segments)
                if d < local_radius:
                    w = strength * falloff(d / local_radius)
                    weights[v.index] = max(weights[v.index], w)
    painted = sum(1 for w in weights if w > 0.0)
    if not painted:
        raise BridgeCommandError("invalid_param", "the strokes don't reach the surface (check the points and the "
                                 "radius against the object's size)", param="strokes")
    data = [0.0] * (len(attr.data) * 4)
    attr.data.foreach_get("color", data)
    if attr.domain == "POINT":
        owners = range(len(mesh.vertices))
    else:
        owners = [loop.vertex_index for loop in mesh.loops]
    for i, vert in enumerate(owners):
        w = weights[vert]
        if w > 0.0:
            base = data[i * 4:i * 4 + 4]
            data[i * 4:i * 4 + 3] = _blend(p["blend"], base[:3], color[:3], w)
            data[i * 4 + 3] = base[3]
    attr.data.foreach_set("color", data)
    mesh.update()
    if was_edit:
        with bpy.context.temp_override(**_context_override(obj)):
            bpy.ops.object.mode_set(mode="EDIT")
    return {"object": obj.name, "attribute": attr.name, "painted_verts": painted,
            "total_verts": len(mesh.vertices)}


def dirty_vertex_colors(p):
    """Vertex paint > Paint > Dirty Vertex Colors: darkens the active colour attribute in the crevices (concave
    places) -- quick dirt/grime. Normalize off spreads the darkness over the whole object; Dirt Only leaves the
    rest as it was."""
    _leave_edit_mode()
    obj = _mesh_object(p["object"])
    mesh = obj.data
    if p["attribute"] is not None:
        _make_active(mesh, _attribute(mesh, p["attribute"]))
    elif mesh.color_attributes.active_color is None:
        attr = mesh.color_attributes.new("Attribute", "BYTE_COLOR", "CORNER")
        attr.data.foreach_set("color", [1.0, 1.0, 1.0, 1.0] * len(attr.data))
        _make_active(mesh, attr)
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if not 0 <= p["blur_iterations"] <= 40:
        raise BridgeCommandError("invalid_param", "blur_iterations 0..40", param="blur_iterations")
    with bpy.context.temp_override(**_context_override(obj)):
        _op_result(bpy.ops.paint.vertex_color_dirt(
            blur_strength=max(0.01, min(1.0, p["blur_strength"])), blur_iterations=p["blur_iterations"],
            clean_angle=math.radians(max(0.0, min(180.0, p["clean_angle_deg"]))),
            dirt_angle=math.radians(max(0.0, min(180.0, p["dirt_angle_deg"]))),
            dirt_only=p["dirt_only"], normalize=p["normalize"]), "vertex_color_dirt")
    mesh.update()
    return {"object": obj.name, "attribute": mesh.color_attributes.active_color.name}


ACTIONS = {
    "color_attribute": (color_attribute, {"object": ("name", REQUIRED), "name": ("name", REQUIRED),
                                          "color": ("any", "#FFFFFF"), "domain": (("CORNER", "POINT"), "CORNER"),
                                          "data_type": (("BYTE_COLOR", "FLOAT_COLOR"), "BYTE_COLOR"),
                                          "active": ("bool", True), "remove": ("bool", False)}),
    "vertex_paint": (vertex_paint, {"object": ("name", REQUIRED), "attribute": ("name", None),
                                    "color": ("any", REQUIRED), "strokes": ("list", None), "radius": ("float", 0.1),
                                    "strength": ("float", 1.0), "falloff": (tuple(FALLOFFS), "SMOOTH"),
                                    "blend": (("MIX", "MULTIPLY", "ADD", "SUBTRACT", "DARKEN", "LIGHTEN"), "MIX"),
                                    "space": (("local", "world"), "world"), "selected": ("bool", False),
                                    "fill": ("bool", False)}),
    "dirty_vertex_colors": (dirty_vertex_colors, {"object": ("name", REQUIRED), "attribute": ("name", None),
                                                  "blur_strength": ("float", 1.0), "blur_iterations": ("int", 1),
                                                  "clean_angle_deg": ("float", 180.0),
                                                  "dirt_angle_deg": ("float", 0.0), "dirt_only": ("bool", False),
                                                  "normalize": ("bool", True)}),
}
