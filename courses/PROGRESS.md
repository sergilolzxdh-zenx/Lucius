# Overnight progress (2026-09-26)

What was asked: remove the half-learned (Gemini) skills, have Claude Code teach Lucius instead, redo the
first video from scratch until every chapter is 9/10 or better, make sure Lucius can do more than the
examples, then go through the other tutorials, saving everything.

How to get it on your PC (PowerShell, in the Lucius folder):

```powershell
git pull
.venv\Scripts\Activate.ps1
pip install -e ".[all,dev]" bpy
lucius skills remove --learned                 # forget the half-learned Gemini skills
lucius teach course courses\lrlpwIumFnE        # Lucius learns the Spanish course (about 10 minutes, no API)
lucius teach course courses\objects            # and the 10 objects below
lucius teach course courses\mHnoznNqlis        # Mastercourse Part 1: 37 chapters (the longest pack)
lucius make "hazme una espada"                 # rebuilds the sword, no API needed
lucius serve                                   # http://127.0.0.1:8765/ -> Projects: every chapter's pictures
```

## 1. Clean slate

- `lucius skills remove --learned` removed all 210 learned skills (the Gemini-extracted tutorial
  patterns, the Gemini lesson chapters and the three Gemini swords) here; the command is in the repo for
  your PC too.

## 2. First video, redone from scratch: all chapters 9/10

[`courses/lrlpwIumFnE`](lrlpwIumFnE) -- see `overview.jpg` (tutor's frame next to Lucius' build for
every chapter) and `final.jpg` (the final Cycles render).

| Chapter | Before | Now |
|---|---|---|
| 4 Interface | 10 (Gemini, random transforms) | 9 (every step the tutor shows, incl. proportional editing) |
| 5 Mug and plate | 7 | 9 (the tutor's curled-extrusion handle, deeper plate) |
| 6 Donut and croissant | 8 | 9 (fat rolled croissant with drooping horns, donut leaning and sagging) |
| 7 Particles and composition | 8 | 9 |
| 8 Materials | 8 | 9 |
| 9 Lighting | 8 | 9 |
| 10 Camera | 8 | 9 |
| 11 Render | 8 | 9 |

Bugs found and fixed on the way: Shade Smooth given in edit mode was silently lost (every "smooth"
object was flat-shaded), deleting faces left loose edges inside bridged handles. New abilities:
proportional editing, dropping an object onto what is below it, changing a modifier by name.

## 3. Beyond the examples: 10 new objects, all 9/10

[`objects/gallery.jpg`](objects/gallery.jpg): sword, wooden chair, cafe table, desk lamp, wine bottle,
low-poly tree, house, candle, toy rocket, and a Mario mushroom built **from a reference image** (cropped
from a tutorial thumbnail, kept in `objects/references/`). Each uses only techniques from the course and
is a validated skill. `lucius make "<thing>"` rebuilds one without any API (Spanish or English names:
espada, silla, mesa, lampara, botella, arbol, casa, vela, cohete, seta); with Gemini configured, the
planner starts from the closest taught object instead of from nothing.

New abilities added for this: a polka-dot material pattern (the mushroom's spots), teacher-made objects
as skills, offline recall, objects packs (`lucius teach objects --out ...`, `lucius teach course ...`).

How to make one from YOUR reference image: send it to me (Claude Code) in the chat, or in the control
center (Projects -> Choose Files -> Make it; that path needs Gemini).

## 4. Second Spanish course (Blender 4.0 guide): all chapters 9/10

[`O-tV7uBf5LI/overview.jpg`](O-tV7uBf5LI/overview.jpg) and the rendered animation
[`O-tV7uBf5LI/final.mp4`](O-tV7uBf5LI/final.mp4): the Mario pipe, mushroom and Bob-omb on a lawn of
particle grass, textured, sky-lit, with a keyframed camera move (handheld shake, focus pulling from the
mushroom to the pipe). `lucius teach course courses\O-tV7uBf5LI` learns it on your PC (about 30 minutes,
most of it rendering the two videos).

Stand-ins where headless Blender can't do what the tutor did: a physical sky instead of the downloaded
HDRI, noise on the camera instead of the Camera Shakify add-on, Cycles instead of Eevee, booleans for the
wind-up key's holes instead of the knife tool.

## 5. Animation tutorial (How to Animate ANYTHING): all 5 build chapters 9/10

[`JQT9sT1YuAI`](JQT9sT1YuAI): keyframes, the graph editor, the bouncing ball, and (after the rigging course)
the backflip: [`final.mp4`](JQT9sT1YuAI/final.mp4) and [`backflip_frames.jpg`](JQT9sT1YuAI/backflip_frames.jpg).
The tutor's rigged Mixamo character is replaced by a pink mannequin with the rig he describes (IK legs with
poles, an IK/FK switch that also moves the feet between the IK handle and the shin). Chapter 9 blocks it
pose to pose at 60 fps with constant keys: stand, wind-up, take-off on the toes, FK in the air (the FK legs
first keyed where the IK had them), the tuck upside down, legs reaching, IK back on the landing frame,
crouch, stand. Chapter 11 polishes it in the graph editor the way he does: Bezier, linear travel in the
air, vector handles at take-off and landing, the wind-up eased 36/36 (his Graph Pilot values), the feet
pointing, the elbows a few frames behind the arms, and his retime (the slow beginning scaled in, the rest
moved in). New tools for this: `key_constraint` (the IK/FK switch), `pose_bone visual` (Apply Visual
Transform), `set_interpolation` (T / V / ease on any selection of keys), `retime_keys` (S X / G X on keys).

## 6. Rigging tutorial (How to Rig ANYTHING): all 15 build chapters 9/10

[`Y2SWwZmwrwM/overview.jpg`](Y2SWwZmwrwM/overview.jpg) and the animation
[`Y2SWwZmwrwM/final.mp4`](Y2SWwZmwrwM/final.mp4). Lucius learns: armatures (extruded chains, branches,
Symmetrize), a rigid robot (parts parented to bones), a mannequin skinned with automatic weights and its
weights fixed, IK legs with pole targets (pole angles measured, not guessed), bone colours and hidden
bones, drivers (a valve turning a screw, a cube following two spheres), then the tutor's full project: a
mechanical bacteriophage modelled, its skeleton, parts bound, cables weighted, IK controls with poles,
root/body split, custom control shapes, a material driven by a leg, and a short animation made only by
moving the controls. `lucius teach course courses\Y2SWwZmwrwM` learns it on your PC (about 20 minutes).

Talk-only chapters (intro, skinning theory, extra tips, outro) have nothing to build and are skipped.
Stand-ins: bones are drawn as thin shapes in the pictures (armatures don't render); the valve's driver
reads the bone's local X rotation like the tutor's.

## 7. Stylized 2D VFX (SharpWind): all 8 effects 9/10, rendered in Eevee

[`ooF_vBB41xw/overview.jpg`](ooF_vBB41xw/overview.jpg), every effect's video in
[`ooF_vBB41xw/videos`](ooF_vBB41xw/videos) and all of them in [`showreel.mp4`](ooF_vBB41xw/showreel.mp4):
the flame (displacement and voronoi masks driven by `#frame`), the comic-book halftone shader (a reusable node
group with the tutor's exposed controls, on Suzanne and a torus, dots following a circling sun), the anime
waterfall, the toon explosion with its ring, energy lines and smoke ring, the blood squirt, metaball smoke,
the spark impact and the energy-line ribbons.

New abilities (usable by any course): the node editor (`edit_nodes`: shader, world, node-group and geometry-node
graphs), keying any setting by its path (`set_property`), shape keys, curves, emitter particles, metaballs,
Checker Deselect, the Skin modifier, frame drivers. Eevee: renders use it when a recipe picks it; on a Linux
server without a GPU that needs Mesa's EGL (`apt install libegl1 libegl-mesa0`), otherwise Cycles is used.

One substitution: the blood squirt is a fluid simulation in the video; Blender's fluid solver (Mantaflow) can't
run inside the `bpy` Python module used here, so the same set-up is made with particles rendered as metaballs.
`quick_liquid` and `bake_fluid` exist for a full Blender.

## 8. Complete Blender Mastercourse part 1 (12 hours): all 37 build chapters 9/10

Pack: `courses/mHnoznNqlis` (a recipe per chapter, `frames/`, `videos/` for the animated chapters,
`overview.jpg`, `grid.jpg`). Interface/installation chapters (1, 2, 4, 5) have nothing to build. Each chapter builds
on the previous one's scene, as the course does.

| Ch. | Topic | What Lucius built (and beyond the tutorial) |
|---|---|---|
| 3 | interface | Rubik's cube: subdivided, bevelled, stickers per face |
| 6 | objects | table scene: joined table, Suzanne, two point lights |
| 7 | editing objects | computer on the table, screen material, hidden Suzanne |
| 8 | editing tools | extrude / individual / inset / bevel / loop cuts, one object each |
| 9 | basic modelling | slatted chair, treasure chest with lock, grenade |
| 10 | advanced tools | knife, bisect, spin (pipe bend), edge slide, shrink/fatten |
| 11-13 | modifiers | array; every Generate modifier; every Deform modifier (lattice, hooks, bound deformers, animated cast) |
| 14-16 | node components | three-point lighting, texture coordinates, maps, bump, brick wall |
| 17-18 | textures | brick, checker, gradient, magic, noise, Musgrave, voronoi, wave |
| 19 | image textures | PBR chain; Lucius bakes its own colour/normal/roughness/height/AO set (new `bake_texture`) |
| 20 | vertex paint | dirty vertex colours, painted eyes/head; + a vertex-painted toadstool (new paint actions) |
| 21 | Object Info | random / index colours; + colour by location |
| 22 | basic shaders 1 | studio backdrop, Blender's HDRIs, chrome, brushed aluminium, gold... as assets (`mark_asset`) |
| 23 | basic shaders 2 | clear coat, glass, a modelled candle with flame and wax, a baked cloth simulation (`bake_physics`) |
| 24 | procedural shaders 1 | every shader node on its own Suzanne, car paint; + a second colourway |
| 25 | procedural shaders 2 | sand shader, sky texture; + sand dunes |
| 26 | lights and cameras | the four light types side by side; + light linking (`light_link`) |
| 27 | animation basics | keyframes, material keys, the hopping-balls loop (portrait video) |
| 28 | graph editor | six cubes racing with Bezier / linear / sine / hand handles / noise / bounce, all cyclic |
| 29 | NLA | named actions, push down, mute/solo, Combine; + the same actions reused on a second Suzanne |
| 30 | 3D text | extruded, bevelled two-colour title converted to mesh; + an arched text on a curve |
| 31 | geometry nodes 1 | a snowman from spheres in nodes; + carrot nose, coal eyes/buttons on a Mesh Line, twig arms, hat |
| 32 | geometry nodes 2 | a city: cubes on a grid's points, random heights, voronoi windows; + set-back rooftops |
| 33 | geometry nodes 3 | a fence instanced along curves with random gaps; + a closed square pen |
| 34 | geometry nodes 4 | rolling hills (noise in Set Position), scattered trees; + the fence dropped onto the hills (Raycast) |
| 35 | geometry nodes 5 | a procedural grass blade, patches, density/height inputs; + a mix of kinds, random greens |
| 36 | geometry nodes 6 | procedural rocks, duplicated with random rotation/scale; + rocks raycast onto the hills: the finished landscape |
| 37 | geometry nodes 7 | a house generator (walls by normal, windows instanced per wall face, roof); + a long house, a two-storey one, roof tiles, stone plinth |
| 38 | geometry nodes 8 | donuts from drawn curves: frosting on the top faces, sprinkles on the top only; + a thinner third donut |
| 39 | geometry nodes 9 | futuristic towers on a street grid blown apart by a meteor (faces pushed out along normals); + the meteor animated falling in at dusk |
| 40 | geometry nodes 10 | an L-shaped sci-fi corridor: hexagon floor (Dual Mesh), inset wall panels, strip lights; + a keyframed fly-through round the corner |
| 41 | geometry nodes 11 | a neon billboard: text in nodes, words on lines, the title typed in and out, a counter 0 -> 41; + the counter counts Lucius's chapters |

New bridge abilities written for these chapters (all tested): colour attributes and vertex painting, texture
baking, studio HDRIs, asset marking, cloth/soft-body baking, light linking, legacy node names, easing curves and
hand-placed handles, F-curve modifiers, actions and the NLA editor, text objects and convert to mesh.
For the geometry nodes chapters: group input defaults and `modifier_inputs` reach the modifier panel, dynamic
node items (Capture Attribute values, repeat/simulation zone state), and a String node's text.

Lessons from the geometry nodes chapters, now in the recipes: give every new object a fresh name (a reused name
silently becomes "Name.001"); hiding a collection's source objects hides Collection Info instances (Object Info
with hide-in-render is fine); don't name an attribute "color" (Cycles has a built-in of that name); a loop cut
exactly on an existing loop makes degenerate faces; clear the last chapter's camera keys before placing the camera;
a multi-input socket (Join Strings, Join Geometry) stacks the newest link first.

Part 1 is finished. Part 2 has not been started (waiting for the go-ahead).
