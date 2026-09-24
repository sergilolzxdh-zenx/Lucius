# Lucius architecture

Lucius learns Blender workflows from demonstrations and reuses them. Everything it learns is
structured, versioned data with provenance, never hidden weights. A person can inspect, edit,
roll back or delete any of it. This document describes the system as built. The
[README](../README.md) gives the capability status (implemented, partial, interface only, future work).

```
             ┌────────────── WATCH ME (live) ──────────────┐   ┌──── Import (external media) ────┐
             │ screen frames · input · window · add-on ops │   │ video · images · pairs · refs · │
             │ undo/redo · Blender state · annotations     │   │ .blend · written instructions   │
             └───────────────────────┬─────────────────────┘   └───────────────┬─────────────────┘
                                     ▼                                         ▼
                           session + events + frames  ◄──── same representation (provenance-tagged)
                                     ▼
   trajectory ─► segmentation ─► (model refinement) ─► intents ─► skills ─► episode ─► semantic ─► preferences ─► index
                                     ▼
            skill library (versioned) · failure memory · episodic/semantic memory · learning graph
                                     ▼
   task ─► hybrid retrieval ─► planner ─► state machine (OBSERVE→PLAN→EXECUTE→VERIFY→RECOVER→HUMAN_TAKEOVER→…)
                                     ▼                                       ▲
                     Blender add-on bridge (validated actions)               │ corrections (DAgger)
                                     ▼                                       │
                5-level evaluation (execution · structural · measured visual · human/model · task)
```

## 1. Packages

| Package | Responsibility |
|---|---|
| `lucius.app` | Composition root: wires every service onto one data directory. |
| `lucius.config` | Validated config (`<data_dir>/config.json` + `LUCIUS_*` env). Secrets are never stored. |
| `lucius.storage` | SQLite (WAL, foreign keys, per-thread connections, migrations), content-addressed frame store, fsync'd JSONL journals. |
| `lucius.events` | Event bus. Decisions are persisted to `event_log`; capture noise (frames, mouse moves) is streamed only. |
| `lucius.sessions` | Sessions, events, frames, lifecycle, deletion (unreferenced frame files are removed too). |
| `lucius.recorder` | WATCH ME: screen grabber (mss), input (pynput + raw X keycodes), window context (X11/Win32/Quartz), privacy filter, mouse-move compression, agent/human attribution ledger, crash-recovery journal. |
| `lucius.blender` | Client for the **Lucius Bridge** add-on (loopback JSON-lines TCP, token auth); `HeadlessBlender` spawns a private Blender for practice, validation, benchmarks and `.blend` inspection. |
| `lucius.blender.addon.lucius_bridge` | The add-on (stdlib + `bpy` only): action handlers, state/structure inspection, operator/undo/depsgraph watching, snapshot/restore. |
| `lucius.trajectory` | Events → typed `TrajectoryStep`s (action type, parameters, source, evidence kind, confidence, candidates, undo links), compression. |
| `lucius.segmentation` | Deterministic boundary signals and labels → optional model relabelling → human edits (lock, split, merge, relabel). |
| `lucius.intent` | Structured intent hypotheses per segment (category/target/scope/confidence/evidence), human override. |
| `lucius.skills` | Skill schema, extraction/generalisation, library (versions, lineage, promotion, rollback, merge, split), recovery learning from takeovers, seeded capabilities. |
| `lucius.memory` | Episodic, semantic (repetition-gated), failure memory (rules with promotion), workflow preferences. |
| `lucius.retrieval` | Vector index + hybrid retriever (vector, lexical, metadata, quality, graph expansion, failures, episodes); every retrieval is recorded. |
| `lucius.planner` | Skill selection, parameter resolution, guards from failure rules, recovery plans, compilation to bridge actions. |
| `lucius.executor` | Explicit run state machine, execution engine, safety policy, human-takeover channel. |
| `lucius.evaluation` | Checkpoint checks (structural, silhouette IoU/taper from rasterised geometry), verdict rules, human evaluations. |
| `lucius.ingestion` | External media: validation, content-addressed media store, video change analysis, before/after transitions, reference measurement, optional VLM analysis, `.blend` inspection, text instructions. |
| `lucius.practice` | Curricula, sampled practice tasks, mastery gates, synthetic references. |
| `lucius.dataset` / `lucius.training` | Dataset assembly with eligibility and quality validation, bundles, training-file formatters, trainer interfaces and advisor. |
| `lucius.benchmarks` | Benchmarks, A/B experiment arms, bootstrap confidence intervals. |
| `lucius.providers` | Provider-neutral LLM/VLM/embedding/evaluation interfaces; Anthropic implementation; hashing and sentence-transformers embeddings. |
| `lucius.api` | Local FastAPI server, job manager, control-center SPA. |
| `lucius.cli` | `lucius` command line. |

## 2. Capture (WATCH ME)

* **Scope.** Capture only happens while recording is active; the UI and the terminal show the
  state at all times. The privacy filter admits only Blender windows (process/title patterns) and
  blocks titles that look sensitive. Keystrokes typed while another window is focused are dropped
  entirely, not redacted afterwards. Frames of other windows are never stored.
* **Signals.** Frames at a configurable learning rate (default 6 FPS, identical frames skipped),
  mouse/keyboard (moves compressed by rate and direction change), window context, and from the
  add-on: operator reports (`window_manager.operators` diff), undo/redo handlers, depsgraph
  geometry changes and periodic Blender state (mode, active/selected objects, dimensions,
  modifiers).
* **Attribution.** Inputs the agent injects are registered in the `AgentActionLedger` and recorded
  with `actor=agent`. Human input during a takeover is recorded as the correction.
* **Durability.** Events go to an fsync'd journal before the database. After a crash,
  `recover_interrupted_sessions` replays the journal and marks the session `interrupted`, and
  nothing is lost.

## 3. From events to skills

1. **Trajectory.** Hotkeys and modal transforms are matched with operator reports. A step confirmed
   by an operator is `observed`/`direct_blender_event`; one known only from input is `inferred`.
   Cancelled modals, undo links and navigation spans are explicit.
2. **Segmentation.** Deterministic boundaries (pauses, mode changes, object changes, undo, inspection
   spans, visual change) and labels come first. If a provider is configured, a model may relabel
   unlocked segments. It sees compressed action spans and at most a few frames, its confidence is
   discounted, and the deterministic label is kept. Human edits lock a segment, and later
   processing never overwrites a locked segment.
3. **Intents.** Several ranked hypotheses per segment. No chain-of-thought is requested or stored:
   model outputs are schema-constrained JSON with reason codes.
4. **Skill extraction.** Segments are grouped into units per object role and phase. Concrete values
   become parameters when they vary across demonstrations or are result-parameterised (a scale
   becomes a `target_size`). Selections are stored as normalised regions, not indices. Checkpoints
   (dimensions, ratios, symmetry, taper, silhouettes), failure conditions (from undone work and
   notes) and recovery actions are attached. Re-processing is idempotent.
5. **Promotion.** `candidate_pattern` (one demonstration) → `candidate_skill` (≥2 demonstrations or a
   human confirmation) → `validated` (an objectively verified execution) → `high_confidence` (≥3
   objective successes on ≥2 distinct instances, success rate ≥0.8, no net human rejection). Skills
   are demoted when the success rate drops below 0.5 after 4 uses, or on repeated human rejection.
6. **Confidence.** `alpha/(alpha+beta)` with `alpha = 1 + 0.6·weighted demos + successes +
   1.5·validations + 1.5·confirmations + 0.5·(instances−1)` and `beta = 1 + failures +
   2·rejections + contradictions`. Demonstrations are weighted by source (user demo 1.0, `.blend`
   0.8, video 0.6, image 0.4…) and by evidence kind (Blender event 1.0 … VLM inference 0.45, weak
   visual guess 0.25). Every score stores its breakdown.

## 4. Memory

* **Episodic**: one structured episode per processed session (phases, objects, corrections, notes).
* **Semantic**: general statements mined from episodes and failures. They need repetition, can be
  contradicted, and can be accepted or rejected by the user.
* **Procedural**: the skill library.
* **Failure**: signature-deduplicated records with symptoms, likely cause, correction, a future rule,
  a guard checkpoint and the premature trigger action. A rule starts as a `candidate`, becomes
  `promoted` once corrections keep working, and can be `confirmed` or `rejected` by the user.
  Every rule that is not rejected and names a trigger action becomes a planner guard when the skill
  has the guard checkpoint (deferring a premature action is cheap). Promotion or confirmation raises
  the rule's retrieval priority; rejected rules are never applied.
* **Preferences**: workflow habits (e.g. symmetry early) learned only from the user's own
  demonstrations. They bias retrieval and never override a task.
* **Learning graph**: typed edges (`extracted_from`, `derived_from`, `validated_by`, `co_used`,
  `follows`, `causes_failure`, `recovers_with`, `modified_by`, `conditioned_by`…) with evidence counts.

## 5. Retrieval and planning

The hybrid retriever scores each skill as `0.55·vector + 0.35·lexical + metadata` (object class,
categories, phases), scaled by quality (status × confidence) and adjusted by success rate and
preferences. It expands the top results along `follows`/`requires`/`co_used` edges and also returns
relevant failures, semantic statements and similar episodes. Each result carries score components
and reason codes, and every retrieval is persisted (see *Memory → Retrieval debug* in the UI).
Numbers in the task are treated as parameters, not vocabulary.

The planner selects skills: learned skills first, and seeded capabilities only when the task
explicitly mentions them. It resolves parameters from the task, reference hints, task parameters,
generalised defaults or derivation. It adds guards for non-rejected failure rules whose trigger
action appears in a step, and recovery plans. It compiles templates to validated bridge actions. Missing
required parameters become `unresolved` entries with reason codes, not guesses.

## 6. Execution

Every run creates its own session (`agent_execution`, `practice`, `benchmark`, `validation`) and
follows an explicit state machine. Every transition is recorded with a reason code, evidence and
context:

```
IDLE → OBSERVE → PLAN → EXECUTE ⇄ VERIFY → SUCCESS
                          │  ▲      │
                          ▼  │      ▼
                        RECOVER   HUMAN_TAKEOVER → RESUME → OBSERVE/VERIFY/EXECUTE
                          └──────────┴──────────► FAILURE
```

* **Layers.** `blender_api` actions go through the add-on and are validated against a fixed
  vocabulary with typed arguments. Paths are restricted to allowed directories, and there is no
  arbitrary Python or OS command execution. `observation` actions inspect state. `internal` actions
  cover snapshots and bookkeeping. View navigation that needs a viewport runs only when a GUI is
  available. Pure mouse/keyboard GUI actuation (a `gui` layer) is **not** implemented; see the README.
* **Guards.** An action that a (non-rejected) failure rule marks as premature is deferred until the
  rule's guard checkpoint passes.
* **Recovery.** When a checkpoint fails, the engine restores the step's snapshot and re-runs the step
  with the learned recovery actions. Only if that fails does it ask for a human takeover.
* **Takeover (DAgger).** The run waits for the person, who corrects the scene in Blender. The
  correction's actions are recorded, and a reason is stored only if the person gives one.
  Processing the run session then turns the correction into a new skill version with a recovery
  action conditioned on the failed checkpoints. The next run recovers without asking.

## 7. Evaluation and false-success prevention

| Level | What | Examples |
|---|---|---|
| 1 | Execution | every action succeeded |
| 2 | Structural | object exists, dimensions/ratio within tolerance, symmetry, modifiers, vertex count, manifold |
| 3 | Measured visual | silhouette IoU against reference masks (rasterised from evaluated geometry, no renderer), taper profile |
| 4 | Human / model judgement | human review; optional vision-model judge (always marked subjective) |
| 5 | Task criteria | practice/benchmark success criteria |

Verdict rules: any failed required check or execution error → `failure`; any required check that
could not be evaluated → `needs_human`; otherwise `success` only if at least one objective (level
2/3) check passed; model-only support → `subjective_pass`; nothing verifiable →
`executed_unverified`. Skill credit for `needs_human` runs is deferred until a person reviews the
run. A reviewed run without objective checks becomes `subjective_pass`, never `success`.

## 8. External media (addendum 10A–10Z)

All media converge on the same session/trajectory representation, tagged with how each fact is
known: `observed`, `inferred`, `model_inferred` or `human_confirmed`.

* **Video.** Coarse sampling (2 FPS) with an adaptive change threshold on a per-cell grid plus a
  header band (mode and tool labels). Dense sampling (8 FPS) around changes finds the stable frames
  before and after, and representative frames come from static spans. Only these frames would ever
  reach a vision model.
* **Before/after transitions** are classified (camera, UI, geometry, selection,
  material/lighting) with ranked candidate operations and silhouette-shape evidence.
* **References** yield measured constraints (silhouette mask, aspect ratio, symmetry, width profile,
  taper, colours) that become planning hints and evaluation targets, never procedures.
* **`.blend` files** are opened in a private headless Blender and their real scene state is
  recorded as measured project constraints.
* **Written instructions** become low-weight `text_instruction` steps.
* **Validation by reproduction.** When headless Blender is available, each extracted skill is run in
  `validation` mode against the demonstration's references. The verified verdict is recorded like
  any other use (so it can promote or demote the skill). Without Blender, the stage reports
  `pending`.
* **Rights.** External material is never training-eligible by default. Licence, reference-only
  status and consent are stored with the media and every dataset sample.

## 9. Practice, datasets, training, benchmarks

* **Practice.** Curricula (hard surface, topology, reference matching, sculpting) of staged task
  templates with sampled parameters. Mastery gates cover attempts, completion, checkpoint pass rate,
  takeover rate, false-success rate, consistency and generalisation (distinct instances). Stages
  that need a GUI or an undemonstrated capability report `requires_gui` or `needs_demonstration`
  instead of practising. People can override a stage status, and the override is recorded.
* **Datasets.** Training datasets include only sessions whose policy allows training, with granted
  consent, that are not reference-only. Quality validation checks frames, timestamps, ordering,
  action vocabulary, outcomes, references, duplicates and Blender versions. Each sample carries its
  provenance. Session and skill bundles (ZIP/JSON) round-trip with remapped ids.
* **Training** is an interface: formatters write behaviour-cloning/SFT pairs (never from undone
  work) and a registry accepts trainer backends. No trainer ships and nothing trains automatically.
  The advisor reports whether training is even advisable given the data.
* **Benchmarks.** Arms are `baseline` (no memory), `retrieval_only`, `memory_enhanced` and
  `raw_demonstrations` (replay the most similar demonstration verbatim). `trained_policy` is listed
  as unavailable. Every arm runs on a reset scene, and results report bootstrap 95% CIs and deltas
  against the baseline.

## 10. API and UI security

* The server refuses to bind to non-loopback addresses and rejects requests whose `Host` is not
  loopback (DNS rebinding).
* Mutating requests need `X-Lucius-Token`. The token lives in `<data_dir>/api_token` (0600) and is
  injected into the control-center page, which cross-origin pages cannot read. The page sends
  `X-Frame-Options: DENY` and `Referrer-Policy: no-referrer`.
* All UI text is inserted as text nodes (no HTML injection from recorded or model content).
* The Blender bridge is loopback-only with its own token (discovery file written by the add-on).
  Its action vocabulary is fixed and every argument is type-checked.

## 11. Storage

One SQLite database per data directory, with tables for users, sessions, events, frames,
media_assets, demonstrations, trajectory_steps, segments, human_edits, intents, taxonomy_terms,
skills, skill_versions, skill_examples, graph_edges, memory_episodes, semantic_memories,
failure_records, runs, run_transitions, corrections, retrieval_records, evaluations,
reference_constraints, benchmarks, experiments, benchmark_results, practice_tasks,
mastery_records, datasets, dataset_samples, embeddings, preferences, processing_jobs, model_calls
and event_log. Frames and media are content-addressed files next to it. Processing is resumable per
stage (`processing_jobs`), so an interrupted pipeline continues where it stopped.

## 12. Extension points

* **Providers**: implement `LLMProvider.complete_json` (schema-constrained JSON),
  `EvaluationProvider.judge` or `EmbeddingProvider.embed`, then register them in
  `providers.build_providers`.
* **Trainer backends**: implement `TrainerBackend` and register it with `TrainerRegistry`. The
  benchmark arm `trained_policy` becomes available once a policy is registered.
* **Bridge actions**: add a handler plus an argument schema to `ACTIONS` in the add-on. The
  planner can only emit actions that exist there.
