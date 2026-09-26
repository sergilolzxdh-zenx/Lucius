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
    "add_primitive": ("Shift+A > Mesh", "kind: cube|plane|cylinder|cone|uv_sphere|ico_sphere|torus|circle|monkey|empty|metaball|lattice, "
                      "name, location [x,y,z], rotation_deg [x,y,z], size (cube/plane edge, default 2), radius, "
                      "depth (cylinder/cone height), radius2 (cone top), vertices (cylinder/cone/circle/sphere "
                      "segments, default 32), rings (UV sphere rings, default half the segments), major_radius, minor_radius, major_segments, minor_segments (torus), "
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
    "extrude": ("E (Alt+E > Individual Faces)", "object, offset [x,y,z] (move the new region by this vector) OR "
                "distance (along the faces' normal); individual (each selected face out along its own normal, "
                "with distance). Extrudes the selected faces (else edges, else vertices); the new cap stays "
                "selected."),
    "inset": ("I (I twice: individual faces)", "object, thickness (metres, must stay below half the face width), "
              "depth (raises or sinks the inner faces), individual (each selected face gets its own rim: tiles, "
              "stickers, panels)"),
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
                          "selected edge loops / holes (select them with select_box boundary=true); with two "
                          "facing faces selected (a front and a back) it removes them and tunnels through: a clean "
                          "hole, e.g. the gaps between a chair back's slats."),
    "fill": ("F", "object, grid (grid fill instead of one face). Makes faces from the selected boundary."),
    "subdivide": ("Edge > Subdivide", "object, cuts, smoothness"),
    "merge_by_distance": ("M > By Distance", "object, distance"),
    "recalc_normals": ("Shift+N", "object, inside (false: faces point outwards)"),
    "separate_selection": ("Shift+D then P > Selection", "object, new_name, duplicate (true: copy the selected "
                           "faces into a new object, e.g. donut icing; false: move them)"),
    "add_modifier": ("Properties > Modifiers (Ctrl+1..3 adds Subdivision)",
                     "object, type, name, props {...}: any mesh modifier -- generate ARRAY BEVEL BOOLEAN BUILD "
                     "DECIMATE EDGE_SPLIT MASK MIRROR MULTIRES REMESH SCREW SKIN SOLIDIFY SUBSURF TRIANGULATE WELD "
                     "WIREFRAME, deform CAST CURVE DISPLACE HOOK LAPLACIANDEFORM LATTICE MESH_DEFORM SHRINKWRAP "
                     "SIMPLE_DEFORM SMOOTH CORRECTIVE_SMOOTH LAPLACIANSMOOTH SURFACE_DEFORM WARP WAVE, and "
                     "WEIGHTED_NORMAL, UV_PROJECT, CLOTH, SOFT_BODY, COLLISION, EXPLODE, OCEAN, ... props use the "
                     "modifier's Python setting names (the tooltip): numbers, switches, menu choices, objects and "
                     "collections by name (object, offset_object, target, origin, mirror_object, collection), vertex "
                     "group names; a name ending in _deg takes degrees (angle_deg, screw angle_deg). Common: SUBSURF "
                     "{levels, render_levels}, MIRROR {use_axis [x,y,z], use_bisect_axis, use_clip}, SOLIDIFY "
                     "{thickness, offset}, BEVEL {width, segments, affect VERTICES|EDGES, limit_method}, ARRAY "
                     "{count, relative_offset_displace [x,y,z], use_relative_offset, use_object_offset, "
                     "offset_object}, BOOLEAN {object, operation DIFFERENCE|UNION|INTERSECT}, SIMPLE_DEFORM "
                     "{deform_method TWIST|BEND|TAPER|STRETCH, angle_deg, factor, deform_axis}, DISPLACE {strength, "
                     "texture CLOUDS|VORONOI|MUSGRAVE|..., texture_scale, texture_coords, texture_coords_object} "
                     "(the texture is named <object>_<kind>), SCREW {angle_deg, screw_offset, steps, axis}, REMESH "
                     "{mode BLOCKS|SMOOTH|SHARP|VOXEL, voxel_size, octree_depth}, WIREFRAME {thickness}, DECIMATE "
                     "{ratio}, BUILD {frame_start, frame_duration}, CAST {factor, cast_type}, WAVE {height, width, "
                     "speed}, SHRINKWRAP {target, wrap_method}, LATTICE / CURVE {object}; physics settings via "
                     "set_property (modifiers[\"Cloth\"].settings.quality)"),
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
    "set_render": ("Render / Output / Color Management properties", "engine CYCLES|BLENDER_EEVEE (renders then use "
                   "it; Eevee-only nodes such as Shader to RGB need BLENDER_EEVEE), samples, motion_blur, width, "
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
    "pose_bone": ("Pose mode: G, R, S on a bone (I to key)", "armature, bone, location, rotation_deg [x,y,z] "
                  "(XYZ Euler, so a full 360 flip works; quaternions take the short way round), scale, frame (key it "
                  "there), interpolation CONSTANT|BEZIER|LINEAR and handle for those keys, reset (back to the rest "
                  "pose first), visual (Pose > Apply > Visual Transform: take the pose the constraints give it now, "
                  "e.g. FK bones matched to the IK pose before an IK/FK switch)"),
    "key_constraint": ("hover the constraint's influence (an IK/FK switch), I", "object, bone, constraint (its name), "
                       "influence 0..1, frame, interpolation (CONSTANT by default: the switch happens on that frame)"),
    "set_interpolation": ("graph editor: select keys (A), T / V, or an ease add-on", "object, bones [..] and channels "
                          "[location|rotation|scale|influence|shape|other] and axes (the object's keys, its data's, shape keys' and materials') (optional: only those curves), start/end "
                          "frames (optional), interpolation CONSTANT (blocking: poses pop) | BEZIER | LINEAR, handle "
                          "AUTO_CLAMPED | VECTOR (no slowing down: take-off, landing, free fall), ease_in / ease_out "
                          "(0..100 % of the gap to the neighbouring key, flat handles: a gentle arrival / departure)"),
    "add_particles": ("Particle properties > + (Emitter)", "object (the emitter mesh), name, count, frame_start, "
                      "frame_end, lifetime, lifetime_random 0..1, emit_from FACE|VERT|VOLUME, normal_velocity, "
                      "velocity [x,y,z] (object-aligned, e.g. [0,0,2] rises), random_velocity, gravity 0..1 (Field "
                      "weights), size, size_random, instance (render each particle as that object, e.g. a metaball), "
                      "show_emitter; other settings via set_property target particles"),
    "quick_liquid": ("F3 > Quick Liquid", "objects [meshes that emit liquid], domain (name of the new domain box). "
                     "Then set_property target object on the flows (path modifiers[\"Fluid\"].flow_settings."
                     "use_initial_velocity true, ...velocity_coord [x,y,z]) and the domain (modifiers[\"Fluid\"]."
                     "domain_settings.use_mesh true, .particle_radius, .use_flip_particles false); scale the domain "
                     "so the liquid fits (it can't leave it)"),
    "bake_fluid": ("Domain > Cache: type All, Bake All", "domain, resolution (voxel divisions: 64 to try, 100-150 "
                   "final), frame_start, frame_end: the simulation is computed once and replayed"),
    "knife_cut": ("K (knife), C to cut through", "object, start [x,y,z], end [x,y,z] (the two clicks), view (the "
                  "direction you look along, e.g. [0,-1,0] from the front), through (cut every face behind too; "
                  "otherwise only the selected faces), space local|world: the new edges are selected"),
    "bisect": ("the Bisect tool", "object, point, normal (the cutting plane), clear_inner / clear_outer (delete the "
               "side behind / in front of the normal), fill (a face over the cut), space"),
    "spin": ("the Spin tool", "object, axis x|y|z, center (the pivot, where the 3D cursor would be), angle_deg, "
             "steps: the selection swept round (a pipe bend, a lathe profile)"),
    "slide_selection": ("G G (edge / vertex slide)", "object, toward [x,y,z] (which way), factor 0..1 of the "
                        "neighbouring edges: moves the selection along the surface instead of off it"),
    "shrink_fatten": ("Alt+S (shrink / fatten)", "object, distance: the selection pushed along its normals (negative "
                      "shrinks)"),
    "move_lattice_points": ("lattice edit mode: select points, G / S", "object (a lattice: add_primitive kind "
                            "lattice, resolution via set_property target data points_u/points_v/points_w), min/max "
                            "(a box in the lattice's -0.5..0.5 coordinates), offset [x,y,z], factor [x,y,z]"),
    "add_hook": ("Ctrl+H > Hook to New Object", "object (vertices selected in edit mode), hook (the empty's name; "
                 "made at their centre if new), size: then moving the empty drags those vertices"),
    "bind_modifier": ("the modifier's Bind button", "object, modifier (a MESH_DEFORM / SURFACE_DEFORM / "
                      "LAPLACIANDEFORM / CORRECTIVE_SMOOTH): bind before editing the cage / target / anchors"),
    "uv_unwrap": ("U in edit mode", "object, method UNWRAP (by seams) | SMART_PROJECT | CUBE_PROJECT | "
                  "CYLINDER_PROJECT | SPHERE_PROJECT | RESET, margin, angle_limit_deg, size (cube projection): lays "
                  "the selected faces (all if none) flat in the UV map; texture coordinate UV then follows it"),
    "uv_transform": ("UV editor: A, then R / S / G", "object, rotate_deg, scale [u, v], offset [u, v]: the selected "
                     "faces' UVs (all if none) about their centre -- the texture turns / shrinks / slides"),
    "color_attribute": ("Object Data > Color Attributes (+ / -)", "object, name, color (what it starts filled "
                        "with, default white), domain CORNER|POINT, data_type BYTE_COLOR|FLOAT_COLOR, active, remove: "
                        "a colour layer vertex paint writes into; a material reads it with an Attribute node "
                        "(edit_nodes text {attribute_name: <name>}, case sensitive)"),
    "vertex_paint": ("Vertex Paint mode: the brush", "object, attribute (default the active one), color, strokes "
                     "[{points [[x,y,z], ...] (where the brush is dragged over the surface), radius}], radius "
                     "(default for strokes), strength 0..1, falloff SMOOTH|LINEAR|CONSTANT|SPHERE, blend MIX|MULTIPLY|"
                     "ADD|SUBTRACT|DARKEN|LIGHTEN, space world|local; or selected true (paint the selected faces: the "
                     "face selection mask) or fill true (the whole mesh)"),
    "dirty_vertex_colors": ("Vertex Paint > Paint > Dirty Vertex Colors", "object, attribute, blur_strength, "
                            "blur_iterations, clean_angle_deg, dirt_angle_deg, dirt_only, normalize (off spreads the "
                            "darkness over the whole object): darkens the crevices of the colour attribute"),
    "bake_physics": ("Physics properties > Cache > Bake All Dynamics", "frame_start, frame_end, show_frame (the "
                     "frame to stop on, default the end), free: runs the cloth / soft body / particle simulations "
                     "(add_modifier CLOTH on the cloth, COLLISION on what it lands on; settings through set_property "
                     "modifiers[\"Cloth\"].settings.quality or .collision_settings.use_self_collision) and keeps "
                     "the result"),
    "light_link": ("Object properties > Shading > Light Linking", "light, receivers [objects it lights], "
                   "blockers [objects that cast its shadows], clear: a light (or emissive object) that lights only "
                   "some objects -- separate lighting set-ups in one scene"),
    "mark_asset": ("Outliner > right click > Mark as Asset", "kind MATERIAL|OBJECT|NODE_GROUP|WORLD|COLLECTION, "
                   "name, description, tags, clear: puts it in the asset library (Asset Browser) so other files "
                   "can drag it in"),
    "bake_texture": ("Render properties > Bake (Cycles)", "object (with a UV map and a node material), type "
                     "DIFFUSE (colour only) | ROUGHNESS | NORMAL (tangent) | AO | EMIT (whatever is routed into an "
                     "emission, e.g. a height map) | COMBINED | GLOSSY, path textures/<name>.png, width, height, "
                     "samples, non_color: bakes the material into an image file an Image Texture node can load"),
    "skin_radius": ("Ctrl+A in edit mode (Skin modifier)", "object, radius: the skin's thickness at the selected "
                    "vertices"),
    "select_nth": ("Select > Checker Deselect", "object, skip, nth, offset: of the selection keep every nth element "
                   "walking along the mesh (e.g. every other vertex of a circle, then scale them out: a spiky star)"),
    "edit_nodes": ("the shader / geometry node editor: Shift+A, drag links", "tree material|world|group|geometry, "
                   "material (made if new; copy_from another = the single-user copy), object (gets the material; "
                   "geometry: its Geometry Nodes modifier), group + group_type + interface [{name, in_out, type "
                   "NodeSocketFloat|Color|Vector|..., default, min, max}] for a node group (Ctrl+G), clear, remove "
                   "[names], nodes [{name, type (ShaderNodeTexNoise, ShaderNodeValToRGB, ShaderNodeMath, "
                   "ShaderNodeMix, ShaderNodeMapping, ShaderNodeTexCoord, ShaderNodeEmission, ShaderNodeMixShader, "
                   "ShaderNodeBsdfTransparent, ShaderNodeShaderToRGB, GeometryNode..., ...), inputs {socket name: "
                   "value; Rotation_deg for degrees}, props {operation, blend_type, data_type, ...; a name ending _deg takes degrees, e.g. sun_elevation_deg}, ramp [[pos, "
                   "colour], ...], ramp_interpolation LINEAR|CONSTANT|B_SPLINE|EASE, image (a file, textures/<name>.png made by bake_texture, or studio:interior|studio|city|"
                   "courtyard|forest|night|sunrise|sunset -- the HDRIs Blender ships, for an Environment Texture), "
                   "non_color (data maps), projection FLAT|BOX, text {attribute_name | layer_name | uv_map: "
                   "a name}, generated_image {name, type UV_GRID|COLOR_GRID|BLANK, width, height, color} "
                   "(Image > New), "
                   "object (texture coordinate object), group (a node group), location [x, y]}], links [{from, output, to, input}] "
                   "(socket names or numbers), unlink [{to, input}]. Keep the material output's Surface linked"),
    "set_property": ("hover any field (its Python tooltip path); I to key it", "target object|data|material|nodes|"
                     "world|world_nodes|shape_keys|texture|scene|particles|group, name (the object / material / "
                     "texture ...), path (e.g. modifiers[\"Displace\"].strength, bevel_factor_end, "
                     "surface_render_method, nodes[\"Ramp\"].color_ramp.elements[0].position, "
                     "key_blocks[\"Grow\"].value, distance_metric, location[2]), value (number, true/false, menu "
                     "choice, list), degrees (value in degrees), frame (key it there), interpolation"),
    "shape_key": ("Object Data > Shape Keys (+)", "object, name: made with a Basis first and made active, so the "
                  "edit-mode steps after it shape the key (set active to go back to one); value 0..1, slider_min "
                  "(negative values allowed), slider_max, frame (key the value), interpolation"),
    "add_curve": ("Shift+A > Curve, then edit mode", "name, splines [{points [[x,y,z], ...], type BEZIER|POLY, handle "
                  "AUTO|VECTOR|ALIGNED|FREE, radius [per point: taper], cyclic}] (several splines = one object, like "
                  "Shift+D in edit mode), location, bevel_depth (thickness), bevel_resolution, extrude, resolution, "
                  "fill_caps. Animate bevel_factor_start / bevel_factor_end with set_property (target data) to draw "
                  "it on"),
    "retime_keys": ("dope sheet / graph editor: select keys, S X (2D cursor pivot) or G X", "object, bones, channels, "
                    "start/end frames of the keys to move, scale (below 1 = faster) around pivot (a frame, default "
                    "start), offset (frames); keys snap to whole frames. Overlapping action: offset a follower's keys "
                    "(the forearm after the upper arm) by a few frames"),
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
                   "transform LOC_X..ROT_Z..SCALE_Z, space WORLD_SPACE|TRANSFORM_SPACE|LOCAL_SPACE, path}]; no variables "
                   "for a value that follows the timeline: expression frame / 10 (typed #frame/10 in a field)"),
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
