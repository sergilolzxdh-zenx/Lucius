---
name: lucius-make
description: Make something in Blender with Lucius from a description or reference photos (a sword, a chair, a house, a creature...), judged from its renders until it is really 9/10, then kept so Lucius rebuilds it later without any model. Use whenever the user asks Lucius (or you) to make, build or model a thing, passes reference images of something to build, or asks to teach Lucius an object or a whole course pack.
---

# Make things with Lucius (you are the teacher)

Lucius builds Blender scenes from **recipes**: JSON lists of bridge actions with exact values. You write the
recipe, Lucius builds and renders it in a background Blender, you look at the renders and fix the recipe until
it is right, then you keep it. A kept object becomes a skill: `lucius make "<its name>"` rebuilds it with no
model, and model-driven `lucius make` runs start from it for similar requests. Every object you keep is one more
thing Lucius knows.

Run `lucius` from the repo's virtual environment: `.venv\Scripts\lucius` on Windows, `.venv/bin/lucius`
elsewhere (or activate the venv first). Every command prints JSON.

## 0. Is Lucius set up?

`lucius status`, then `lucius skills list`. If no course lessons are listed, teach the packs first (no model
needed; the Mastercourse pack is long, run it in the background if you can):

```
lucius teach course courses/lrlpwIumFnE
lucius teach course courses/O-tV7uBf5LI
lucius teach course courses/Y2SWwZmwrwM
lucius teach course courses/JQT9sT1YuAI
lucius teach course courses/ooF_vBB41xw
lucius teach course courses/mHnoznNqlis
lucius teach course courses/objects
```

A chapter reported as `failed` is a real problem: read its error and say what failed. Don't hide it.

If the request names something already taught (`courses/objects/*.json`: sword/espada, chair/silla, café
table/mesa, desk lamp/lampara, wine bottle/botella, tree/arbol, house/casa, candle/vela, rocket/cohete,
mushroom/seta, and whatever was added since), `lucius make "<name>"` rebuilds it. Only write a new recipe when
the user wants something different.

## 1. Understand what to build

- Reference photos: open every one with Read and study it. Write down the parts, each part's size in metres
  (estimate real-world scale from the photo: a sword is about 1 m, a door 2 m), the silhouette, and the colours
  as hex values.
- No photo: decide what a person would expect "a <thing>" to look like and name its main parts.
- If it needs what the bridge cannot do well, tell the user before building: realistic people and faces,
  sculpted organic detail, cloth folds, hair. Don't build a blob and call it done.

## 2. Find how to build it

- Every action and its arguments: `src/lucius/lessons/catalogue.py` (`ACTION_DOCS`). Use only those names.
- Worked examples, kept at 9/10: `courses/objects/*.json` for objects, `courses/<video>/*.json` for every
  tutorial chapter. Find techniques by action name, e.g. `grep -l '"spin"' courses/*/*.json`. Reuse their
  approach and values: blades are cubes scaled to a point at the tip, handles are curled extrusions, organic
  shapes are a subdivided cage, hard edges get a small bevel.

## 3. Write the recipe

Save it as `recipes/<slug>.json`:

```json
{"title": "Greatsword", "summary": "what it is and how it is built",
 "expected_result": "what the render should show",
 "aliases": ["espadon", "great sword", "mandoble"],
 "steps": [{"action": "add_primitive", "args": {"kind": "cube", "name": "Blade", "size": 1,
            "location": [0, 0, 0.9]}, "note": "the blade starts as a cube"}]}
```

- Metres, real-world scale, resting on the ground (z = 0); angles as `rotation_deg` / `angle_deg`; colours as
  hex strings.
- `aliases`: other names people will use, Spanish and English. `lucius make` recalls the object by them.
- Name every part uniquely. A name that already exists becomes "Name.001" silently, and later steps then edit
  the wrong object.
- `apply_transform` scale before edit-mode work, so edit-mode sizes are in metres. `select_box` with
  `"space": "local"` uses the object's own coordinates.
- Give it materials, a light and a camera (`set_material`, `add_light`, `add_camera`), like the examples. Grey
  geometry is never finished.
- Pitfalls already paid for: a loop cut exactly on an existing edge loop makes degenerate faces; don't name an
  attribute "color" (Cycles has a built-in of that name); AgX view transform washes out saturated colours; a
  camera with keys from an earlier step ignores a new location (clear its animation first); multi-input
  sockets (Join Geometry, Join Strings) stack the newest link first.

## 4. Build and look

```
lucius teach task recipes/<slug>.json --task "<the user's request>" --reference <photo> [--reference <photo2>]
```

- `"ok": false`: `error` names the step that failed and `steps_before_failure` shows what came before. Fix
  that step and run again.
- `"ok": true`: open `sheet` (the reference photos next to Lucius's renders) with Read, and the single renders
  listed in `renders` when you need detail.

## 5. Judge honestly, like a 3D artist reviewing a portfolio

- 9-10: a finished model a person would accept as the thing: right silhouette, every main part, believable
  proportions, proper materials; with a photo, it reads as the same object side by side.
- 7: clearly the thing, but missing detail or materials. 5: recognisable but crude. 3: only hints at it (a
  ball on a stick for a mushroom). Grey untextured geometry is 5 at most.
- When unsure between two scores, give the lower one. Write down the three biggest differences from what was
  asked or from the photo, fix them in the recipe, and build again.
- Keep going until it is 9. If something keeps it below 9 that the bridge cannot do, stop and tell the user
  exactly what and why.

## 6. Keep it

```
lucius teach task recipes/<slug>.json --task "<the request>" --reference <photo> --score 9 --note "what makes it right"
lucius teach objects --out courses/objects
```

The first command keeps it as a skill. The second adds it to the objects pack in the repo, so it survives a
fresh install (`lucius teach course courses/objects`). Show the user the final sheet, and tell them that
`lucius make "<alias>"` now rebuilds it with no model.

## Whole scenes and courses

The same loop works for scenes (several objects, lights, camera, animation; see `courses/*/` for keyframes,
rigs, geometry nodes and simulations). To teach a new tutorial chapter, see how the existing course packs
pair each recipe with the tutorial's frame (`lucius teach chapter --video <id> --chapter <n>`), and judge the
renders against that frame instead of a photo.
