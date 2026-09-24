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
| External media | Video change analysis (OpenCV), before/after pairs, reference measurement, `.blend` inspection in headless Blender, written instructions, human confirmation of inferred actions, validation by reproduction |
| Practice | Curricula with sampled tasks and mastery gates; honest `requires_gui` / `needs_demonstration` |
| Datasets | Consent-based eligibility, quality validation, provenance per sample, session/skill bundles, training-file formatters |
| Benchmarks | Baseline / retrieval-only / memory-enhanced / raw-demonstration arms, bootstrap CIs |
| API, UI, CLI | Loopback server with Host check and token; every UI view checked in Chromium; CLI |
| Gemini provider (live) | All four output schemas (segment labelling, video transitions, reference analysis, visual judge) accepted by the live API with `gemini-3.5-flash-lite`; quota errors classified (zero quota / daily / per-minute with the server's retry delay); model probing |

### Implemented, not verified in this environment

- **Live interactive Blender.** The add-on's GUI-thread dispatcher, operator watching during real
  interactive use, and viewport navigation actions. All tests used headless Blender. The recorder
  has not been run against a live Blender window.
- **Windows and macOS window providers.** Only X11 was exercised.
- **Anthropic provider.** Request construction (JSON-schema output, images), refusal/truncation
  handling, error mapping and retries are tested with fake SDK clients only; it has not been called
  against the live API.
- **VLM media analysis** (`ingestion/vision.py`) and the **sentence-transformers** embedding
  option: the code paths exist but have not been run.

### Partial

- **Execution layers.** Actions run through the Blender add-on. Pure mouse/keyboard GUI actuation
  (driving Blender without the add-on) is not implemented; such actions are rejected with
  `unsupported_layer`. Hotkey hints are recorded on skills for a future actuator.
- **Sculpting and organic work.** Those practice stages report `requires_gui`/`needs_demonstration`.
  The bridge has no brush-stroke action vocabulary yet.
- **Default embeddings** are a lexical hashing embedding (documented as such). Use
  sentence-transformers for semantic similarity.

### Interface only

- **Training.** `TrainerBackend`, `TrainerRegistry` and `TrainingJobSpec`, plus a training advisor.
  Formatters write behaviour-cloning and SFT files from consented datasets, but no trainer ships
  and nothing trains automatically. The `trained_policy` benchmark arm reports itself unavailable.

### Future work

A GUI actuator (synthesised input attributed through the agent ledger), sculpt stroke capture and
replay, multi-user and remote deployment, and a trained policy as an experiment arm.

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
