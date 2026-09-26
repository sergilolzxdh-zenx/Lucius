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
