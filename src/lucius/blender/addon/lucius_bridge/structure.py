"""Structural inspection used by the evaluator (level-2 checks and measured silhouettes).

Returns measured facts only: counts, bounding boxes, modifiers, mirror-symmetry error and
orthographic silhouette triangles. Interpretation (does it *pass*?) happens in Lucius.
"""

import bpy
from mathutils import kdtree

from .state import object_summary

# Orthographic projections: view name -> (screen-x world axis, screen-y world axis).
VIEW_AXES = {"front": (0, 2), "side": (1, 2), "top": (0, 1)}


def _round(values, digits=5):
    return [round(float(v), digits) for v in values]


def _symmetry_errors(points, bbox_min, bbox_max):
    """Mean distance from each mirrored vertex to its nearest vertex, per axis.

    Normalised by the bounding-box diagonal so the number is scale independent; 0 means a
    perfect mirror across the bounding-box centre plane.
    """
    n = len(points)
    if n == 0:
        return None
    tree = kdtree.KDTree(n)
    for i, p in enumerate(points):
        tree.insert(p, i)
    tree.balance()
    diag = sum((b - a) ** 2 for a, b in zip(bbox_min, bbox_max)) ** 0.5 or 1.0
    errors = {}
    for axis, label in enumerate("xyz"):
        centre = (bbox_min[axis] + bbox_max[axis]) / 2.0
        total = 0.0
        for p in points:
            mirrored = list(p)
            mirrored[axis] = 2 * centre - p[axis]
            _co, _index, dist = tree.find(mirrored)
            total += dist
        errors[label] = round(total / n / diag, 6)
    return errors


def inspect_structure(names=None, views=("front", "side", "top"), max_triangles=20000):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    wanted = set(names) if names else None
    objects = [o for o in scene.objects if wanted is None or o.name in wanted]
    out = {
        "mode": bpy.context.mode,
        "object_count": len(scene.objects),
        "mesh_object_count": sum(1 for o in scene.objects if o.type == "MESH"),
        "missing": sorted(wanted - {o.name for o in objects}) if wanted else [],
        "objects": [],
    }
    for obj in objects:
        info = object_summary(obj)
        if obj.type == "MESH":
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            try:
                matrix = obj.matrix_world
                points = [tuple(matrix @ v.co) for v in mesh.vertices]
                if points:
                    bbox_min = [min(p[i] for p in points) for i in range(3)]
                    bbox_max = [max(p[i] for p in points) for i in range(3)]
                else:
                    bbox_min = bbox_max = [0.0, 0.0, 0.0]
                mesh.calc_loop_triangles()
                tris = mesh.loop_triangles
                step = max(1, len(tris) // max_triangles) if max_triangles else 1
                silhouettes = {}
                for view in views:
                    ax, ay = VIEW_AXES[view]
                    flat = []
                    for ti in range(0, len(tris), step):  # bpy collections do not support stepped slices
                        for vi in tris[ti].vertices:
                            p = points[vi]
                            flat.extend((round(p[ax], 5), round(p[ay], 5)))
                    silhouettes[view] = flat
                info["evaluated"] = {
                    "verts": len(mesh.vertices), "edges": len(mesh.edges), "faces": len(mesh.polygons),
                    "triangles": len(tris), "triangles_sampled_every": step,
                    "bbox_min": _round(bbox_min), "bbox_max": _round(bbox_max),
                    "size": _round([b - a for a, b in zip(bbox_min, bbox_max)]),
                    "symmetry_error": _symmetry_errors(points, bbox_min, bbox_max),
                    "non_manifold_edges": _non_manifold_edges(mesh),
                }
                info["silhouettes"] = silhouettes
            finally:
                evaluated.to_mesh_clear()
            info["modifiers"] = [_modifier_info(m) for m in obj.modifiers]
        out["objects"].append(info)
    return out


def _non_manifold_edges(mesh):
    counts = {}
    for poly in mesh.polygons:
        for key in poly.edge_keys:
            counts[key] = counts.get(key, 0) + 1
    return sum(1 for c in counts.values() if c != 2)


_MODIFIER_FIELDS = {
    "MIRROR": ("use_axis", "use_clip", "use_mirror_merge"),
    "BEVEL": ("width", "segments", "limit_method"),
    "SUBSURF": ("levels", "render_levels"),
    "SOLIDIFY": ("thickness",),
    "ARRAY": ("count", "relative_offset_displace"),
    "SIMPLE_DEFORM": ("deform_method", "factor", "deform_axis"),
}


def _modifier_info(modifier):
    info = {"name": modifier.name, "type": modifier.type, "show_viewport": bool(modifier.show_viewport)}
    for field in _MODIFIER_FIELDS.get(modifier.type, ()):
        value = getattr(modifier, field, None)
        if value is None:
            continue
        if isinstance(value, (int, float, str, bool)):
            info[field] = value
        else:
            try:
                info[field] = [v if isinstance(v, (bool, str)) else float(v) for v in value]
            except TypeError:
                info[field] = str(value)
    return info
