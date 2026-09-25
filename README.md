# Lucius

Lucius is a continual-learning agent for Blender. You show it how you work (**WATCH ME**) or give it
media (videos, screenshots, before/after pairs, reference images, `.blend` files, written steps).
It turns these into structured, versioned **skills** with parameters, checkpoints, known failures
and recovery actions. It then reuses them on new tasks, verifies its own results, and learns from
your corrections when it gets stuck.

Everything it learns is inspectable data with provenance. You can edit it, roll it back, disable it
or delete it. Nothing enters a training dataset without your consent, and no model weights are
trained behind the scenes.

→ Design details: [docs/architecture.md](docs/architecture.md)

## Quick start

Requirements: Python 3.11, Blender 4.2+ for live use. Headless practice, validation and benchmarks
use the `bpy` module (`pip install bpy`, which needs Python 3.11) or a `blender` binary on `PATH`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[api,media,capture]"      # add ",gemini" or ",anthropic" for model-assisted features
pip install bpy                             # optional: headless Blender for practice/validation/benchmarks

lucius addon --output lucius_bridge.zip     # Blender: Edit > Preferences > Add-ons > Install…, enable "Lucius Bridge"
lucius serve                                # open http://127.0.0.1:8765/
```

In Blender, the add-on starts a loopback-only bridge (N-panel **Lucius**; it autostarts by default)
and writes a discovery file to `~/.config/lucius/bridge.json`. Lucius connects to it automatically.

Then:

1. **Watch Me.** Type what you are about to demonstrate, press *Start WATCH ME*, work in Blender as
   usual, press *Stop & save*. Processing runs in the background.
2. **Skills.** Inspect what was learned: parameters, phases, checkpoints, evidence and confidence
   breakdown, versions. Confirm, edit, roll back, merge or split.
3. **Agent.** Ask for a task ("make another simple sword blockout with a longer blade"). The run is
   verified. If a check fails and no learned recovery works, Lucius asks you to take over. Your
   correction becomes a recovery action on a new skill version.
4. **Practice / Benchmarks.** Let it practise sampled variations in headless Blender and measure
   whether memory actually helps (A/B arms with confidence intervals).

Everything is also scriptable: `lucius --help` (record, run, practice, import, dataset, benchmark,
export/delete sessions…). Data lives in `.lucius/` (override with `--data-dir` or `LUCIUS_DATA_DIR`).

### Model providers (optional)

Without any model, Lucius records, segments, extracts skills, retrieves, plans, executes and measures
results. Checks that need judgement are sent to you. Model-assisted segment labelling, media
analysis and a visual judge can use **Gemini** or **Claude**:

- **Gemini:** install the `gemini` extra and set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the
  environment. Run `lucius models --provider gemini --check` to see which models your key can
  actually call. The API also lists retired models, models with no quota on your tier, and
  audio/image models; `--check` sends each one small JSON+image request and labels it. Then set
  the model (the model is never guessed):

  ```bash
  lucius config --set providers.gemini_model=gemini-3.5-flash-lite \
                --set providers.llm=gemini --set providers.vlm=gemini --set providers.evaluation=gemini
  lucius config --set providers.requests_per_minute=10   # optional: stay under a free-tier per-minute quota
  ```

  The same settings are editable in the control center's **Settings**.
- **Claude:** install the `anthropic` extra, set `ANTHROPIC_API_KEY`, and set the three roles to
  `anthropic`.

Lucius never stores credentials.
Model outputs are schema-constrained JSON with reason codes (no hidden reasoning is requested or
stored), are labelled `model_inferred`, are confidence-discounted, and never count as objective
verification.

## Status

The table separates what is built and tested from what is built but unverified here, what is an
interface only, and what is future work.

### Implemented and tested

The full learning loop runs in an automated test against real headless Blender 4.5:
demonstrate → trajectory → segments → intents → skill → retrieve → execute → verify → fail →
human takeover → recovery rule → the next attempt recovers without a human.

| Area | Notes |
|---|---|
| Storage, sessions, event bus, crash recovery | SQLite/WAL, content-addressed frames, journal replay after a crash |
| WATCH ME recorder | Real X11 capture under Xvfb: Blender-only scoping, dropped input in other windows, numpad/hotkeys, compressed mouse moves, frames |
| Blender bridge add-on | Typed action vocabulary, structure inspection, silhouettes, snapshots/restore, rejection of unknown actions, bad arguments and disallowed paths |
| Trajectory, segmentation, intents | Deterministic stage; model relabelling (with a scripted test provider); human edits preserved across re-processing |
| Skills | Extraction, generalisation across demonstrations, result-parameterised actions, versioning/rollback/merge/split, evidence-based promotion and demotion |
| Memory | Episodic, semantic (repetition-gated), failure rules with promotion, workflow preferences, learning graph |
| Retrieval, planning, execution | Hybrid retrieval with reason codes, guards from failure rules, recovery, explicit state machine, human takeover and learning from corrections (DAgger) |
| Evaluation | Structural and measured-visual checks; `needs_human`/`subjective_pass`/`executed_unverified` never reported as success; human review with deferred skill credit |
| External media | Video change analysis (OpenCV), before/after pairs, reference measurement, `.blend` inspection in headless Blender, written instructions, human confirmation of inferred actions, validation by reproduction (only the skill under test is planned and credited) |
| Practice | Curricula with sampled tasks and mastery gates; honest `requires_gui` / `needs_demonstration` |
| Datasets | Consent-based eligibility, quality validation, provenance per sample, session/skill bundles, training-file formatters |
| Benchmarks | Baseline / retrieval-only / memory-enhanced / raw-demonstration arms, bootstrap CIs |
| API, UI, CLI | Loopback server with Host check and token; every UI view checked in Chromium; CLI |
| Tutorials (live) | `lucius tutorial URL`: chapters become demonstrations; captions (json3/VTT/SRT, rolling duplicates removed) give spoken actions in English and Spanish and a narration-to-action lag estimate; when the video cannot be downloaded, Gemini watches the URL chunk by chunk. Run on four real tutorials (see below) |
| Keyboard and mouse control (live, X11) | `lucius run --backend gui`: actions typed with Blender's default shortcuts and verified against the operator log; mouse click-select and middle-drag orbit; tested against a windowed Blender 5.0.1 under Xvfb, including a full learned-skill run |
| Gemini provider (live) | All four output schemas (segment labelling, video transitions, reference analysis, visual judge) accepted by the live API with `gemini-3.5-flash-lite`; quota errors classified (zero quota / daily / per-minute with the server's retry delay); model probing |

### Implemented, not verified in this environment

- **The recorder against a live Blender window.** The add-on's GUI-thread dispatcher and viewport
  actions now run in a windowed Blender (keyboard/mouse tests), but WATCH ME has not recorded one.
- **Windows and macOS.** Window providers and keyboard/mouse input (pynput: SendInput, Quartz) are
  implemented; only X11 was exercised. Wayland-native windows cannot receive synthesised input.
- **Anthropic provider.** Request construction (JSON-schema output, images), refusal/truncation
  handling, error mapping and retries are tested with fake SDK clients only; it has not been called
  against the live API.
- **VLM media analysis** (`ingestion/vision.py`) and the **sentence-transformers** embedding
  option: the code paths exist but have not been run.

### Partial

- **Keyboard and mouse coverage.** Mode switches, select all/none, extrude, move/scale/rotate along one
  axis or uniformly, inset, bevel, views, orbit, frame selected, undo/redo, delete and click-selecting an
  object are typed. Region selections, loop cuts, modifiers and adding primitives have no exact keyboard
  form and go through the add-on, which also observes and verifies everything (no add-on, no control).
- **Skills learned from video** know *which* operations were done but rarely *how much* (distances
  are seldom visible or said). Such parameters stay unresolved, so these skills cannot be validated
  by reproduction until a demonstration or a person supplies the values.
- **Sculpting and organic work.** Those practice stages report `requires_gui`/`needs_demonstration`.
  The bridge has no brush-stroke action vocabulary yet.
- **Default embeddings** are a lexical hashing embedding (documented as such). Use
  sentence-transformers for semantic similarity.

### Interface only

- **Training.** `TrainerBackend`, `TrainerRegistry` and `TrainingJobSpec`, plus a training advisor.
  Formatters write behaviour-cloning and SFT files from consented datasets, but no trainer ships
  and nothing trains automatically. The `trained_policy` benchmark arm reports itself unavailable.

### Future work

Sculpt stroke capture and replay, keyboard forms for loop cuts and menus, multi-user and remote
deployment, and a trained policy as an experiment arm.

## Learning from tutorial videos

```bash
pip install -e ".[all]"            # includes the youtube extra (yt-dlp and a JavaScript runtime for it)
lucius tutorial "https://www.youtube.com/watch?v=..." --start 9:24 --end 19:24   # a test slice first
lucius tutorial "https://www.youtube.com/watch?v=..."                            # every chapter
```

Each chapter becomes one demonstration whose task is the chapter title (translated to English
when the video is in another language). Introductions, installation and promotion chapters are
skipped (`--no-skip` keeps them). Chapters already processed are skipped when the command is run
again, so a run stopped by a model quota continues where it stopped.

* **Video.** Downloaded at <=720p without audio and analysed by frame changes. If the platform
  blocks downloads from your network, `--remote` (or the automatic fallback) lets a video-capable
  model (Gemini) watch the URL in chunks of up to 5 minutes instead. `--cookies cookies.txt` from
  your own signed-in browser is the other remedy.
* **Captions.** The original-language track (manual if published, else speech recognition) with
  per-word times. Spoken actions ("press E to extrude", "le damos a la G") support the visual
  evidence around them, the narration goes to the model with each chunk, and the lag between
  words and actions is estimated and applied only when it beats chance.
* **Quota.** `lucius config --set providers.requests_per_minute=8` keeps a free-tier key under its
  per-minute limit; `processing.max_vision_pairs` caps frame pairs sent per chapter.

External videos are learning material only: they are never training-eligible, and the licence
and URL travel with every sample.

**First runs (2026-09-24/25, free-tier `gemini-3.5-flash-lite`, videos watched by URL).** Thirteen
tutorials: two Spanish beginner courses, the English *Complete Blender Mastercourse* (Parts 1 and 2,
22.6 h), a topology course (4.1 h), a sculpting course (4.4 h), three body-modelling videos, and
shorter ones on rigging, animation, VFX and texturing. 199 chapters, 8,037 steps of which 7,536 were
identified, 199 skills (15 validated or high-confidence), 919 model requests over two days (the free
tier allows about 500 a day).

What that knowledge is worth, honestly:

* Most skills lack values: a video shows which operations were done but rarely how much.
* "Validated" means little for video-derived skills. A video does not show Blender's state, so their
  only checks are "the object exists", and validation proves that the steps replay, not that they
  build the right thing.
* Skills are merged across demonstrations by object role and phase. Tutorial objects mostly have
  generic names, so unrelated chapters (texturing and an animated backflip) end up in one large
  "object blockout" skill, and such merged skills are the ones that validate most easily. Scoping
  video-derived skills by chapter is the next step before relying on them.

They are useful now as workflow knowledge for retrieval and planning, and become reliable after a
WATCH ME demonstration of the same workflow or a person's review in **Skills**. The runs found and
fixed several evidence bugs: unknown object kinds were guessed to be cubes, validation credited
skills whose actions could not run, validation credited every skill retrieved into the plan rather
than the one under test, and the agent's own validation runs were sent for model relabelling
(about 40% of a day's requests).

## Lessons: learn a tutorial by rebuilding it

`lucius tutorial` (above) extracts operations; `lucius learn` goes further and checks that Lucius can
actually reproduce what the tutor built:

```bash
lucius learn "https://www.youtube.com/watch?v=..."          # all chapters, in order
lucius learn "https://www.youtube.com/watch?v=..." --chapters 5-6 --practice 2
lucius projects                                              # what it built, with scores
lucius projects good <project-id> --note "nice mug"          # or: bad <project-id> --note "no handle"
```

For each chapter (a course builds one project across chapters, so each chapter continues from the
previous chapter's scene):

1. **Watch.** A video model (Gemini) watches the chapter by URL, in pieces of up to 5 minutes, at
   high resolution with the narration, and writes lesson notes: each operation, what was selected,
   the values read off the screen or heard (or estimated, and marked so), and the objects at the end.
2. **Recipe.** The notes become a recipe of Blender actions with concrete values.
3. **Rebuild.** The recipe runs in headless Blender. A failing step goes back to the model with the
   error and the scene, and is corrected.
4. **Compare.** The result is rendered (Cycles) and the model compares the renders with the end of
   the chapter in the video: a score from 0 to 10 and a list of differences.
5. **Practise.** The recipe is revised from the differences and rebuilt (`--practice` rounds, or
   until `--target` is reached). The best attempt is kept.
6. **Keep.** The best recipe becomes a skill: *validated* if it rebuilt without errors and was judged
   at least 6/10. Everything is saved in `.lucius/projects/<id>/`: every attempt's renders and
   recipe, the tutorial's own frame, `sheet.png` (tutorial next to Lucius' attempts) and
   `scene.blend` (open it in Blender).

The score is a model's judgement, not a measurement. The project folder and `sheet.png` exist so
you can check it, and your rating (`lucius projects good|bad`, or the buttons in the control
center) confirms or rejects the skill. Chapters already learned are skipped when you run the
command again, and the notes are cached, so a run stopped by the model's daily quota resumes where
it stopped.

The actions a recipe can use: mesh primitives with sizes, box selection of faces, edges or vertices
(Alt+click loops), extrude, inset, bevel, loop cuts, rotate/scale/move a selection, delete faces,
bridge edge loops, fill, subdivide, separate, duplicate, modifiers (subdivision, mirror, solidify,
bevel, array, boolean, displace…), shade smooth, Principled BSDF materials, lights, camera, world
colour and particle scattering (sprinkles). There is no sculpting, curves, texture images or node
editing yet: recipes approximate them and say so.

**First lesson (2026-09-25, the Spanish beginner course, free-tier Gemini).** Every chapter's
notes came from the models still answering that day (mostly `gemini-3.5-flash-lite`: 3.8-flash's free
daily quota is small and the other flash models were overloaded), and the free daily quotas of all
of them ran out after three chapters:

| Chapter | Result | Best score |
|---|---|---|
| Interface (ends with the mug's base) | learned | 10/10 |
| Mug and plate | learned: hollow mug with rim and handle, dished plate | 7/10 |
| Donut and croissant | partial: lumpy donut and a segmented croissant, badly placed | 4/10 |
| Particles, materials, lights, camera, render | not yet (quota) | – |

What it took to get there, and what it says about the approach:

* Recipes written by a small model fail on details (a typo in an argument name, a selection box
  that misses the geometry). Corrections that see the error, the object's local bounds and what the
  previous steps selected fix most of them; a step that still fails is skipped and recorded.
* Practice helps when it edits a recipe (change, insert or delete steps by index) and hurts when the
  model rewrites it: rewrites of the 71-step mug recipe came back with 20 steps and scored lower.
  Relearning continues from the best recipe so far, so learning accumulates across runs.
* The runs found real bugs: extrude removed a lone face's original (a filled circle became a tube,
  not a cup), previews were framed before loaded objects had their positions, and failed practice
  runs counted against the skill they were practising.
* Judging is noisy: the same mug recipe was scored 3, 4 and 7 by different models. The best score is
  kept, which favours lucky judgements; your rating in **Projects** is the correction.

## Make something new, or from a reference image

```bash
lucius make "a sword with a long thin blade and a round guard"
lucius make "a mug like this" --reference photo_of_my_mug.jpg
```

Or in the control center (`lucius serve`, then http://127.0.0.1:8765/ → **Projects**): type what
to make, optionally drop reference images (a photo, a drawing, a screenshot), press **Make it**, and
the result appears with its pictures. Open it and press **Good** or **Bad**.

The maker plans a recipe **only with the techniques Lucius has learned** (actions used by lesson
recipes it passed) plus basic object handling, and uses the learned recipes as worked examples. A
technique it has not learned is left out and named under "not learned yet" (`--allow-unlearned`
lifts the restriction; the project records it). It builds, renders, has the model judge the result
against your words and reference image, revises (three tries by default) and keeps the best try in
a project folder like a lesson's. What it made becomes a skill that stays a candidate until you
rate it good.

### Sending a reference image

Any of these works:

* **Control center:** `lucius serve`, open http://127.0.0.1:8765/, go to **Projects**, type what to
  make, press *Choose Files* and pick the image (photo, drawing or screenshot), then **Make it**.
* **Command line:** `lucius make "a sword like this" --reference path/to/image.jpg` (repeat
  `--reference` for several views).
* **Through Claude Code:** attach the image in the chat and ask for it to be made; it is saved next to
  the project and passed with `--reference`.

The image goes to the model that plans and judges the build, and is kept in the project folder next
to the renders so you can compare them.

### Windows (PowerShell)

```powershell
cd C:\path\to\Lucius
.venv\Scripts\Activate.ps1
$env:GEMINI_API_KEY = "your key"
lucius learn "https://www.youtube.com/watch?v=lrlpwIumFnE"
lucius make "a sword" --reference "C:\Users\you\Pictures\sword.jpg"
lucius projects
lucius serve          # then open http://127.0.0.1:8765/ and go to Projects
```

Pictures of every project are in `.lucius\projects\<project-id>\` (open `sheet.png`).

## Keyboard and mouse control

`lucius run "<task>" --backend gui` performs the plan in your running Blender by keyboard and
mouse, like a person would. The Lucius Bridge add-on must be enabled: it tells Lucius where the 3D
viewport is and confirms every keystroke sequence ran the right operator with the typed values.

* Blender must be the focused window and have no menu or popup open (the splash screen takes key
  presses). Lucius presses Esc once before it starts.
* Linux: an X11 session, or Blender under XWayland (start it with `WAYLAND_DISPLAY=` unset). macOS:
  allow your terminal in *System Settings → Privacy & Security → Accessibility*. Windows: run
  Lucius at the same privilege level as Blender.
* Default keymap and median-point pivot are assumed; anything else is detected and done through the
  add-on instead.
* Move the mouse to take over: the agent presses Esc, stops and hands the run to you. Actions it
  types are recorded as the agent's, not yours.

## Privacy and safety

- Capture happens only while recording and is clearly indicated. Only Blender windows are captured.
  Keystrokes typed in other applications and windows with sensitive-looking titles are dropped.
  Recordings can be deleted (frames included).
- Live demonstrations are excluded from training datasets unless you opt in. External media are
  never training-eligible by default, and licence and consent travel with every sample. Private
  memory is kept separate from datasets.
- Model-proposed and agent actions are untrusted. They are limited to a fixed, type-checked bridge
  vocabulary with path allowlists, with no arbitrary Python or OS commands and no uploads.
- The API binds to loopback, rejects non-loopback `Host` headers and requires a token for every
  change.

## Development

```bash
pip install -e ".[all,dev]" bpy
pytest                              # full suite; Blender-, Xvfb-, OpenCV- and browser-dependent tests skip when unavailable
pytest -m "not bpy"                 # without Blender
ruff check src tests
```

Test markers: `bpy` (headless Blender), `xvfb` (real X11 capture), `media` (OpenCV), `ui`
(Playwright + Chromium; `pip install -e ".[ui-test]"`).
