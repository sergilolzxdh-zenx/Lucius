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

## 5. Animation tutorial (How to Animate ANYTHING): keyframes, graph editor, bouncing ball, 9/10 each

[`JQT9sT1YuAI`](JQT9sT1YuAI). The backflip chapters need a rigged character: they come after the rigging
tutorial.
