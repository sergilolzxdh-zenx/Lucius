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
