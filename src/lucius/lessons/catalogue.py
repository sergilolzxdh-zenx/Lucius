"""The Blender actions a recipe may use, described for a model, and argument normalisation.

A recipe is written by a model (from a tutorial it watched, or for a new task) in terms of the
bridge's allowlisted actions. The descriptions below are the whole contract: what each action does
in Blender terms (the hotkey a tutorial would use), its arguments and their units. Angles are
written in degrees and colours as hex, the way people read them off Blender's UI; they are
converted here before the bridge validates them.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

# action -> (Blender equivalent, arguments). Units: metres, degrees, colours "#RRGGBB".
PROPORTIONAL = (", proportional (O: radius in metres; nearby vertices follow, fading out), falloff "
                "SMOOTH|SPHERE|ROOT|SHARP|LINEAR|CONSTANT")
ACTION_DOCS: dict[str, tuple[str, str]] = {
    "add_primitive": ("Shift+A > Mesh", "kind: cube|plane|cylinder|cone|uv_sphere|ico_sphere|torus|circle|monkey|empty, "
                      "name, location [x,y,z], rotation_deg [x,y,z], size (cube/plane edge, default 2), radius, "
                      "depth (cylinder/cone height), radius2 (cone top), vertices (cylinder/cone/circle/sphere "
                      "segments, default 32), major_radius, minor_radius, major_segments, minor_segments (torus), "
                      "fill (circle: true adds the face, like F), into (an existing mesh: Shift+A in edit mode, the shape joins that object, selected)"),
    "delete_objects": ("X in object mode", "names [..]"),
    "hide_objects": ("H / Alt+H, and the camera icon", "names [..], hide (false shows them again), render (also "
                     "hidden from renders; default true)"),
    "join_objects": ("Ctrl+J", "names [..] (the objects to merge), into (the one they merge into; keeps its name, "
                     "origin and modifiers -- apply the others' modifiers first)"),
    "duplicate_object": ("Shift+D", "object, new_name, offset [x,y,z] from the original, rotation_deg [x,y,z] added"),
    "transform_object": ("G / R / S in object mode", "object, location [x,y,z], rotation_deg [x,y,z], scale [x,y,z], "
                         "relative (true: add location/rotation and multiply scale)"),
    "set_dimensions": ("N panel > Dimensions", "object, dimensions [x,y,z] in metres (scales the object)"),
    "apply_transform": ("Ctrl+A", "object, location, rotation, scale (booleans; applies them to the mesh)"),
    "rename_object": ("F2", "object, new_name"),
    "set_mode": ("Tab", "object, mode: OBJECT|EDIT"),
    "select_all": ("A / Alt+A in edit mode", "object, action: SELECT|DESELECT"),
    "select_box": ("clicking / Alt+click loops in edit mode",
                   "object, element: VERT|EDGE|FACE, min [x,y,z], max [x,y,z] (null = unbounded), "
                   "space: local (object coordinates in metres, recommended) | normalized (0..1 of the bounding box) "
                   "| world, facing [x,y,z] (faces whose normal points that way; min_dot 0.7), sharp_deg (edges "
                   "where faces meet at >= this angle, plus open rims), boundary (edges of holes and open rims; on a "
                   "closed mesh it falls back to the sharp edges in the box, like Alt+click on a rim loop), "
                   "extend (add to the selection). Fails if nothing is inside the box."),
    "extrude": ("E", "object, offset [x,y,z] (move the new region by this vector) OR distance (along the faces' "
                "normal). Extrudes the selected faces (else edges, else vertices); the new cap stays selected."),
    "inset": ("I", "object, thickness (metres, must stay below half the face width), depth"),
    "bevel": ("Ctrl+B", "object, offset (metres), segments (1 = chamfer, 3+ = rounded), affect: EDGES|VERTICES "
              "(bevels the selected edges)"),
    "loop_cut_axis": ("Ctrl+R", "object, axis: x|y|z (cuts perpendicular to it), positions [0..1 of the object's "
                      "extent along the axis], e.g. [0.25, 0.5, 0.75] = three cuts"),
    "translate_selection": ("G in edit mode", "object, offset [x,y,z]" + PROPORTIONAL),
    "scale_selection": ("S in edit mode", "object, factor [x,y,z], pivot: median|bbox_center|origin" + PROPORTIONAL),
    "rotate_selection": ("R in edit mode", "object, axis: x|y|z, angle_deg, pivot: median|bbox_center|origin"
                         + PROPORTIONAL),
    "taper_selection": ("proportional scaling along an axis", "object, along: x|y|z, affect: x|y|z|xy|xz|yz, "
                        "amount (0..1, how much the far end shrinks), start (0..1), reverse"),
    "duplicate_selection": ("Shift+D in edit mode", "object, offset [x,y,z]: copies the selected part of the mesh "
                            "(an eye, a button) and moves the copy, which stays selected"),
    "select_linked": ("L / Ctrl+L", "object: grows the selection to everything connected to it (a whole part)"),
    "delete_elements": ("X in edit mode", "object, what: VERTS|EDGES|FACES|ONLY_FACES (ONLY_FACES keeps the rim "
                        "edges, which is what bridging needs)"),
    "bridge_edge_loops": ("Edge > Bridge Edge Loops", "object, cuts (extra loops along the bridge). Joins two "
                          "selected edge loops / holes (select them with select_box boundary=true)."),
    "fill": ("F", "object, grid (grid fill instead of one face). Makes faces from the selected boundary."),
    "subdivide": ("Edge > Subdivide", "object, cuts, smoothness"),
    "merge_by_distance": ("M > By Distance", "object, distance"),
    "recalc_normals": ("Shift+N", "object, inside (false: faces point outwards)"),
    "separate_selection": ("Shift+D then P > Selection", "object, new_name, duplicate (true: copy the selected "
                           "faces into a new object, e.g. donut icing; false: move them)"),
    "add_modifier": ("Properties > Modifiers (Ctrl+1..3 adds Subdivision)",
                     "object, type, name, props {...}. Types and props: SUBSURF {levels, render_levels}, "
                     "MIRROR {use_axis [bool,bool,bool], use_bisect_axis, use_clip, use_mirror_merge}, "
                     "SOLIDIFY {thickness, offset (-1..1), use_even_offset}, BEVEL {width, segments, limit_method: "
                     "NONE|ANGLE}, ARRAY {count, relative_offset_displace [x,y,z]}, SIMPLE_DEFORM {deform_method: "
                     "TWIST|BEND|TAPER|STRETCH, factor, deform_axis}, BOOLEAN {object, operation: "
                     "DIFFERENCE|UNION|INTERSECT}, DISPLACE {strength, texture: CLOUDS|VORONOI|NOISE, texture_scale} (with a "
                     "texture: an irregular, hand-deformed surface), SMOOTH {factor, iterations}, "
                     "CAST {factor, cast_type}, DECIMATE {ratio}, WEIGHTED_NORMAL {}, TRIANGULATE {}"),
    "apply_modifier": ("Ctrl+A over the modifier", "object, modifier (its name)"),
    "remove_modifier": ("X on the modifier", "object, modifier (its name)"),
    "shade": ("right click > Shade Smooth / Flat (Auto Smooth)", "object, smooth (true|false), auto_smooth_deg "
              "(smooth by angle: edges sharper than this stay crisp, e.g. 30)"),
    "set_material": ("Material properties > Principled BSDF",
                     "object, name, base_color \"#RRGGBB\", roughness 0..1, metallic 0..1, alpha, emission_color, "
                     "emission_strength, transmission 0..1 (glass), subsurface 0..1 (soft food, skin), coat 0..1, "
                     "ior, assign: replace (only material) | append | selected_faces (faces selected in edit mode get "
                     "it). Patterns on the colour: pattern brick|checker|noise|dots|wave, pattern_color (second colour), "
                     "line_color (brick mortar), pattern_scale, mortar_size (0..0.125), brick_width, row_height "
                     "(0.5 and 0.5: square tiles, a check tablecloth), dot_size (dots: 0..0.5 of a cell; round "
                     "spots of pattern_color, e.g. a mushroom cap). Relief: bump magic|noise|voronoi, "
                     "bump_scale, bump_distortion, bump_strength (a fabric: magic, 200, 15). The same name on "
                     "another object shares the material (Ctrl+L)."),
    "add_light": ("Shift+A > Light (an existing light's name changes only the values given)",
                  "type: POINT|SUN|SPOT|AREA, name, location [x,y,z], look_at [x,y,z] or "
                  "rotation_deg, power (watts; sun: strength ~1-10), color \"#RRGGBB\" or temperature (kelvin: "
                  "3200 warm, 6500 neutral), size (area size, point radius -- softer shadows, sun angle in "
                  "degrees), spot_size (cone, degrees), spot_blend (0..1 soft edge)"),
    "add_camera": ("Shift+A > Camera, Ctrl+Numpad0 (an existing camera's name, e.g. the default \"Camera\", "
                   "changes only the values given)", "name, location [x,y,z], look_at [x,y,z] or rotation_deg, "
                   "lens (mm, default 50), active (true), depth of field: focus_object (sharp there) or dof_distance "
                   "(metres), fstop (lower = more blur, default 2.8), dof (false turns the blur off)"),
    "set_render": ("Render / Output / Color Management properties", "engine CYCLES|BLENDER_EEVEE, samples, motion_blur, width, "
                   "height (pixels), denoise, view_transform Standard|AgX|Filmic"),
    "scale_scene": ("A, then S", "factor, pivot [x,y,z]: every object scaled about the pivot (bring a scene to "
                    "real-world size; lights then need less power)"),
    "drop_object": ("G Z by eye until it rests on the surface", "object, onto [names] (default: everything), "
                    "floor (true: the ground at z=0 counts), gap (metres): the object falls straight down onto the "
                    "top surfaces below it, or comes up out of one it sank into"),
    "move_to_collection": ("M", "names [..], collection (created if new): group objects in a collection"),
    "set_frames": ("Timeline / Output properties", "start, end (frames), fps, current (go to that frame)"),
    "insert_keyframe": ("go to the frame, place it, I", "object, frame, and what to key there: location [x,y,z], "
                        "rotation_deg or look_at [x,y,z], scale; for a camera also lens, focus_distance (metres, "
                        "animated focus), fstop; axes x|z|xz|.. (key only those components, e.g. only X), "
                        "interpolation BEZIER|LINEAR|CONSTANT, handle AUTO_CLAMPED|VECTOR|ALIGNED|FREE (V in the graph "
                        "editor: VECTOR = sharp, e.g. a bounce). Blender fills the frames between"),
    "parent_object": ("Ctrl+P > Object (Alt+P clears)", "object (the child), parent (it follows the parent's moves, "
                      "turns and scale; null clears)"),
    "clear_animation": ("Alt+I / Clear Keyframes", "object: removes all its keyframes (it stays where it is now)"),
    "add_shake": ("camera shake add-on (Camera Shakify)", "object, strength (metres), rotation_strength (radians), "
                  "scale (frames per wobble, higher = slower), influence 0..1: handheld noise on top of the animation"),
    "add_armature": ("Shift+A > Armature, then edit mode (E to extrude bones)", "name, location, bones [{name, "
                     "head [x,y,z], tail [x,y,z], parent, connect (head on the parent's tail), deform (false for "
                     "control bones), roll_deg}] in the armature's coordinates (a head may be left out when "
                     "connected: it starts at the parent's tail), display OCTAHEDRAL|STICK|BBONE, in_front. The "
                     "same name again adds or changes bones"),
    "symmetrize_bones": ("right click > Symmetrize", "armature: every .L bone gets a mirrored .R twin across X"),
    "set_bone": ("Bone properties (F2 renames)", "armature, bone, new_name, head, tail, parent (\"\" clears), connect, "
                 "deform, hide, color THEME01..THEME20, shape (an object shown as the bone's control)"),
    "bind_to_armature": ("Ctrl+P > Bone / With Automatic Weights / With Empty Groups", "objects [..], armature, "
                         "mode BONE (rigid: each object follows one bone; give bone) | AUTOMATIC (the mesh deforms, "
                         "weights computed) | EMPTY (vertex groups named after the bones, fill with assign_weights) "
                         "| ENVELOPE"),
    "assign_weights": ("Vertex group > Assign (or weight painting)", "object, group (the bone's name), weight 0..1, "
                       "mode REPLACE|ADD|SUBTRACT, exclusive (take the vertices out of other groups): for the "
                       "vertices selected in edit mode"),
    "pose_bone": ("Pose mode: G, R, S on a bone (I to key)", "armature, bone, location, rotation_deg [x,y,z], "
                  "scale, frame (key it there), reset (back to the rest pose first)"),
    "add_constraint": ("Bone (or object) constraints", "object (the armature, or an object), bone, type IK|CHILD_OF|"
                       "COPY_ROTATION|COPY_LOCATION|DAMPED_TRACK|TRACK_TO|STRETCH_TO|LIMIT_ROTATION|MAINTAIN_VOLUME|"
                       "..., target (object), subtarget (bone), IK: pole_target, pole_subtarget, pole_angle_deg, "
                       "chain_count; influence 0..1; props {use_tail, use_stretch, owner_space, min_x.. (degrees "
                       "for rotation limits), free_axis, ...}"),
    "add_driver": ("right click > Add Driver", "object, or material (a node value such as "
                   "nodes[\"LuciusMapping\"].inputs[\"Location\"].default_value: moves a pattern), bone (optional), "
                   "path (location, rotation_euler, "
                   "constraints[\"IK\"].influence, ...), index (0/1/2 for a vector), expression (arithmetic on the "
                   "variables, e.g. var/2 or one - two), variables [{name, type TRANSFORMS|SINGLE_PROP, object, bone, "
                   "transform LOC_X..ROT_Z..SCALE_Z, space WORLD_SPACE|TRANSFORM_SPACE|LOCAL_SPACE, path}]"),
    "set_world": ("World properties", "color \"#RRGGBB\", strength; or sky true (a physical sky with a sun, "
                  "like an outdoor HDRI: it lights everything), sun_elevation_deg, sun_rotation_deg, strength ~0.3"),
    "add_scatter": ("Particle system (hair, render as object or collection)", "object (surface), instance (object "
                    "copied over it, hidden itself) or collection (its objects copied, a random one each: grass kinds), "
                    "name (the same name again changes that system), count, children (interpolated copies between "
                    "particles, e.g. 20), scale (size of the copies), scale_random 0..1, rotation_random 0..1, "
                    "rotation_axis NOR|OB_X|OB_Y|OB_Z (Advanced > Rotation: which way copies stand), seed"),
}

# Always usable: making, placing and selecting objects (what anyone knows after opening Blender once).
BASIC_ACTIONS = ("add_primitive", "delete_objects", "transform_object", "set_dimensions", "apply_transform",
                 "rename_object", "set_mode", "select_all", "select_box")
RECIPE_ACTIONS = tuple(ACTION_DOCS)

# Bridge parameters given in radians, which recipes write in degrees.
DEGREE_ARGS = {"rotation_deg": "rotation", "angle_deg": "angle"}
COLOR_ARGS = ("base_color", "emission_color", "color", "pattern_color", "line_color")
HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def catalogue(actions: tuple[str, ...] | list[str] | set[str] | None = None) -> str:
    """The action reference given to a model, limited to ``actions``."""
    names = [a for a in RECIPE_ACTIONS if actions is None or a in actions]
    return "\n".join(f"- {name} ({ACTION_DOCS[name][0]}): {ACTION_DOCS[name][1]}" for name in names)


def srgb_to_linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _color(value: Any) -> Any:
    """'#RRGGBB' (as shown in Blender's colour picker) -> linear RGB floats; 0-255 triples likewise."""
    if isinstance(value, str) and HEX.match(value.strip()):
        digits = HEX.match(value.strip()).group(1)  # type: ignore[union-attr]
        return [round(srgb_to_linear(int(digits[i:i + 2], 16) / 255.0), 5) for i in (0, 2, 4)]
    if isinstance(value, (list, tuple)) and len(value) in (3, 4) and all(isinstance(v, (int, float)) for v in value):
        rgb = [float(v) for v in value[:3]]
        if max(rgb) > 1.0:
            return [round(srgb_to_linear(min(255.0, v) / 255.0), 5) for v in rgb]
        return rgb
    return value


AXIS_FLAGS = ("use_axis", "use_bisect_axis")


def _prop(key: str, value: Any) -> Any:
    """Modifier properties as people write them: "use_axis": "X" or true (the X axis) -> [true, false, false]."""
    if key.endswith("color"):
        return _color(value)
    if key in AXIS_FLAGS:
        if isinstance(value, bool):
            return [value, False, False]
        if isinstance(value, str) and set(value.upper()) <= set("XYZ") and value:
            return ["X" in value.upper(), "Y" in value.upper(), "Z" in value.upper()]
    return value


def normalize_args(action: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Degrees -> radians, hex colours -> linear floats. Returns the bridge arguments and notes."""
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in args.items():
        if key in DEGREE_ARGS:
            target = DEGREE_ARGS[key]
            if isinstance(value, (list, tuple)):
                out[target] = [math.radians(float(v)) if isinstance(v, (int, float)) else v for v in value]
            elif isinstance(value, (int, float)):
                out[target] = math.radians(float(value))
            else:
                out[target] = value
        elif key in ("rotation", "angle") and _looks_like_degrees(value):
            # A model wrote degrees under the radian name ("angle": 45): nobody means 45 radians.
            out[key] = [math.radians(float(v)) for v in value] if isinstance(value, (list, tuple)) \
                else math.radians(float(value))
            notes.append(f"{key} read as degrees")
        elif key in COLOR_ARGS:
            out[key] = _color(value)
        elif key == "props" and isinstance(value, (dict, str)):
            if isinstance(value, str):   # a JSON object written as a string inside the arguments
                try:
                    value = json.loads(value) if value.strip() else {}
                except json.JSONDecodeError:
                    out[key] = value
                    continue
            out[key] = {k: _prop(k, v) for k, v in value.items()} if isinstance(value, dict) else value
        else:
            out[key] = value
    if action == "add_primitive" and isinstance(out.get("kind"), str):
        out["kind"] = out["kind"].lower().replace(" ", "_")
    return out, notes


def fix_keys(action: str, args: dict[str, Any], known: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Match misspelled argument names to the action's real ones (".max" -> "max", "offest" -> "offset").

    Models writing recipes make such slips; the bridge would reject the step, and a model asked to correct it
    often repeats the slip. Only close, unambiguous matches are renamed; anything else is left for the bridge
    to reject."""
    import difflib

    if not known:
        return args, []
    out: dict[str, Any] = {}
    notes: list[str] = []
    accepted = known | set(DEGREE_ARGS)
    for key, value in args.items():
        if key in accepted:
            out[key] = value
            continue
        cleaned = re.sub(r"[^a-z0-9_]", "", key.strip().lower().replace(" ", "_").replace("-", "_"))
        match = cleaned if cleaned in accepted else next(iter(difflib.get_close_matches(cleaned, sorted(accepted),
                                                                                     n=1, cutoff=0.8)), None)
        if match is not None and match not in args and match not in out:
            out[match] = value
            notes.append(f"argument {key!r} read as {match!r}")
        else:
            out[key] = value
    return out, notes


def _looks_like_degrees(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return abs(value) > 2 * math.pi + 1e-6
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (int, float)) for v in value):
        return any(abs(v) > 2 * math.pi + 1e-6 for v in value)
    return False
