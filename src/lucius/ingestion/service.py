"""TEACH FROM MEDIA: external demonstration ingestion (10A-10Z).

A demonstration bundles any of: videos, image sequences, before/after pairs, reference images,
Blender projects and text instructions. All of it converges on the same representation as a
live demonstration -- a session with a trajectory of steps -- and then goes through the same
segmentation, intent, skill-extraction and memory pipeline. What differs is provenance:
every step records whether it was observed, inferred or model-inferred, with its evidence,
media timestamp and frames, and every skill extracted from it starts as a candidate that needs
validation (reproduction in Blender, more demonstrations, or human confirmation).
"""

from __future__ import annotations

import re
import threading
import traceback
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image
from pydantic import BaseModel, Field

from lucius.errors import LuciusError, MediaError, NotFoundError
from lucius.events.bus import EventType
from lucius.ids import new_id
from lucius.ingestion.media import MediaAsset, MediaKind, MediaRole, MediaStore
from lucius.ingestion.reference import ReferenceStore
from lucius.ingestion.transitions import Transition, analyze_transition
from lucius.ingestion.video import VideoAnalyzer, VideoReader
from lucius.ingestion.vision import VisionAnalyzer
from lucius.logging_setup import get_logger
from lucius.provenance import EVIDENCE_WEIGHT, ActionSource, DataPolicy, EvidenceKind, SourceClass
from lucius.sessions.models import Actor, CapturedEvent, EventKind, Outcome, SessionKind, SessionStatus
from lucius.storage.db import dumps, loads
from lucius.storage.frames import thumbnail
from lucius.taxonomy import classify_task
from lucius.timeutil import now
from lucius.trajectory.builder import link_undo_relations
from lucius.trajectory.model import CandidateAction, TrajectoryStep

if TYPE_CHECKING:
    from lucius.app import Lucius

log = get_logger("ingestion")

MIN_ACTION_CONFIDENCE = 0.5
KEYWORD_ACTIONS = [
    (r"\badd (?:a |an )?(cube|cylinder|plane|sphere|cone|torus)\b", "add_primitive"), (r"\bextrud", "extrude"),
    (r"\bloop ?cut", "loop_cut"), (r"\bbevel", "bevel"), (r"\binset", "inset"), (r"\bscale|\bresize", "scale"),
    (r"\bmove|\bgrab|\btranslate", "translate"), (r"\brotate", "rotate"), (r"\bmirror", "add_modifier"),
    (r"\bsubdivi", "subdivide"), (r"\bedit mode|\btab into", "mode_change"), (r"\bfront view|\bside view", "view_preset"),
]


class DemoStatus(StrEnum):
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    EXTRACTING = "EXTRACTING"
    ANALYZING = "ANALYZING"
    SEGMENTING = "SEGMENTING"
    INFERRING_ACTIONS = "INFERRING_ACTIONS"
    EXTRACTING_SKILLS = "EXTRACTING_SKILLS"
    VALIDATING_SKILLS = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"


class MediaInput(BaseModel):
    path: str
    filename: str | None = None
    role: MediaRole = MediaRole.DEMONSTRATION
    view: str = "front"                   # for reference images
    kind: MediaKind | None = None


class Demonstration(BaseModel):
    id: str
    title: str
    source_types: list[str] = Field(default_factory=list)
    status: DemoStatus
    status_history: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    session_id: str | None = None
    instructions: list[str] = Field(default_factory=list)
    task_text: str | None = None
    source_class: str
    policy: DataPolicy
    created_at: float
    updated_at: float


def _decode(row: Any) -> Demonstration:
    return Demonstration(
        id=row["id"], title=row["title"], source_types=loads(row["source_types"], []), status=DemoStatus(row["status"]),
        status_history=loads(row["status_history"], []), error=loads(row["error"]), session_id=row["session_id"],
        instructions=loads(row["instructions"], []), task_text=row["task_text"], source_class=row["source_class"],
        policy=DataPolicy.model_validate(loads(row["policy"], {})), created_at=row["created_at"],
        updated_at=row["updated_at"])


class IngestionService:
    def __init__(self, app: Lucius) -> None:
        self.app = app
        self.media = MediaStore(app.db, app.config.media_dir)
        self.references = ReferenceStore(app.db, app.config.media_dir)
        self.vision = VisionAnalyzer(app.providers, app.config.providers.max_images_per_call)
        self.video = VideoAnalyzer()
        self._threads: dict[str, threading.Thread] = {}

    # -- lifecycle -------------------------------------------------------------------------------------
    def create(self, *, title: str, task_text: str | None, inputs: list[MediaInput],
               instructions: list[str] | None = None, policy: DataPolicy | None = None) -> Demonstration:
        kinds = []
        for item in inputs:
            ext = Path(item.filename or item.path).suffix.lower()
            kinds.append(item.kind.value if item.kind else ext.lstrip("."))
        if instructions:
            kinds.append("text")
        primary = self._primary_source(inputs, instructions)
        policy = policy or DataPolicy.for_external(primary)
        t = now()
        uploaded = {"status": "UPLOADED", "ts": t, "inputs": [i.model_dump(mode="json") for i in inputs]}
        demo = Demonstration(id=new_id("demonstration"), title=title, source_types=sorted(set(kinds)),
                             status=DemoStatus.UPLOADED, status_history=[uploaded],
                             instructions=instructions or [], task_text=task_text, source_class=primary.value,
                             policy=policy, created_at=t, updated_at=t)
        self.app.db.insert("demonstrations", {
            "id": demo.id, "title": title, "source_types": dumps(demo.source_types), "status": demo.status.value,
            "status_history": dumps(demo.status_history), "error": None, "session_id": None,
            "instructions": dumps(demo.instructions), "task_text": task_text, "source_class": primary.value,
            "policy": dumps(policy), "created_at": t, "updated_at": t})
        self.app.bus.publish(EventType.MEDIA_STATUS, demo.id, status="UPLOADED", title=title)
        return demo

    @staticmethod
    def _primary_source(inputs: list[MediaInput], instructions: list[str] | None) -> SourceClass:
        roles = [i for i in inputs if i.role in (MediaRole.DEMONSTRATION, MediaRole.BEFORE, MediaRole.AFTER,
                                                 MediaRole.INTERMEDIATE, MediaRole.PROJECT)]
        for item in roles or inputs:
            name = (item.filename or item.path).lower()
            if item.kind == MediaKind.VIDEO or name.endswith((".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")):
                return SourceClass.EXTERNAL_VIDEO
            if item.kind == MediaKind.BLENDER_PROJECT or name.endswith(".blend"):
                return SourceClass.EXTERNAL_PROJECT
        if inputs:
            return SourceClass.EXTERNAL_IMAGE
        return SourceClass.EXTERNAL_DOCUMENTATION if instructions else SourceClass.EXTERNAL_IMAGE

    def get(self, demo_id: str) -> Demonstration:
        row = self.app.db.query_one("SELECT * FROM demonstrations WHERE id = ?", (demo_id,))
        if row is None:
            raise NotFoundError(f"demonstration {demo_id} not found")
        return _decode(row)

    def list(self) -> list[Demonstration]:
        return [_decode(r) for r in self.app.db.query("SELECT * FROM demonstrations ORDER BY created_at DESC")]

    def _status(self, demo_id: str, status: DemoStatus, **info: Any) -> None:
        demo = self.get(demo_id)
        history = demo.status_history + [{"status": status.value, "ts": now(), **info}]
        values: dict[str, Any] = {"status": status.value, "status_history": dumps(history), "updated_at": now()}
        if "error" in info:  # partial errors recorded earlier are kept unless a new error replaces them
            values["error"] = dumps(info["error"])
        self.app.db.update("demonstrations", "id", demo_id, values)
        self.app.bus.publish(EventType.MEDIA_STATUS, demo_id, status=status.value, **info)

    def start(self, demo_id: str, *, background: bool = True, validate: bool = True) -> None:
        if background:
            thread = threading.Thread(target=self.process, args=(demo_id,), kwargs={"validate": validate},
                                      name=f"lucius-ingest-{demo_id}", daemon=True)
            self._threads[demo_id] = thread
            thread.start()
        else:
            self.process(demo_id, validate=validate)

    def wait(self, demo_id: str, timeout: float = 600.0) -> Demonstration:
        thread = self._threads.get(demo_id)
        if thread is not None:
            thread.join(timeout)
        return self.get(demo_id)

    # -- processing ------------------------------------------------------------------------------------
    def process(self, demo_id: str, *, validate: bool = True) -> Demonstration:
        demo = self.get(demo_id)
        # Inputs are persisted with the UPLOADED status, so processing survives a restart.
        uploaded = next((h for h in demo.status_history if h.get("status") == "UPLOADED"), {})
        inputs = [MediaInput.model_validate(i) for i in uploaded.get("inputs", [])]
        stage = DemoStatus.VALIDATING
        try:
            self._status(demo_id, stage)
            assets = self._ingest_files(demo, inputs)
            stage = DemoStatus.EXTRACTING
            self._status(demo_id, stage, media=[a.id for a in assets])
            session = self._create_session(demo, assets)
            extracted = self._extract(session.id, assets)
            stage = DemoStatus.ANALYZING
            self._status(demo_id, stage, frames=extracted["frames"], transitions=len(extracted["transitions"]))
            analysis = self._analyze(demo, session.id, assets, extracted)
            stage = DemoStatus.INFERRING_ACTIONS
            self._status(demo_id, stage)
            steps = self._infer_actions(demo, session.id, analysis)
            self.app.sessions.finalize(session.id, end_time=max([s.t_end for s in steps] + [session.start_time]),
                                       outcome=Outcome.UNKNOWN)
            stage = DemoStatus.SEGMENTING
            self._status(demo_id, stage, steps=len(steps))
            report = self.app.pipeline.process(session.id, stages=["trajectory", "segmentation", "refinement", "intents"])
            stage = DemoStatus.EXTRACTING_SKILLS
            self._status(demo_id, stage, segments=report["segmentation"].get("result", {}).get("segments"))
            report = self.app.pipeline.process(session.id)
            skills = report["skills"].get("result", {})
            stage = DemoStatus.VALIDATING_SKILLS
            self._status(demo_id, stage, skills=skills.get("skills", []))
            validation = self._validate(demo, session.id, skills.get("skills", [])) if validate else {"skipped": "disabled"}
            self._status(demo_id, DemoStatus.READY, validation=validation)
        except Exception as exc:  # structured failure; media and extracted evidence stay intact
            error = exc.to_dict() if isinstance(exc, LuciusError) else {
                "code": type(exc).__name__, "message": str(exc), "details": {"traceback": traceback.format_exc(limit=6)}}
            log.exception("ingestion of %s failed at %s", demo_id, stage)
            self._status(demo_id, DemoStatus.FAILED, error={**error, "stage": stage.value})
        return self.get(demo_id)

    def _ingest_files(self, demo: Demonstration, inputs: list[MediaInput]) -> list[MediaAsset]:
        assets = []
        errors = []
        for item in inputs:
            try:
                policy = demo.policy.model_copy(update={"reference_only": item.role == MediaRole.REFERENCE or
                                                        demo.policy.reference_only})
                asset = self.media.ingest(Path(item.path), filename=item.filename, role=item.role, kind=item.kind,
                                          demonstration_id=demo.id, policy=policy)
                self.media.update_analysis(asset.id, {"view": item.view})
                assets.append(self.media.get(asset.id))
            except MediaError as exc:
                errors.append({"file": item.filename or Path(item.path).name, **exc.to_dict()})
        if errors and not assets and not demo.instructions:
            raise MediaError("no usable media in the demonstration", files=errors)
        if errors:
            self.app.db.execute("UPDATE demonstrations SET error = ? WHERE id = ?",
                                (dumps({"partial": errors}), demo.id))
        return assets

    def _create_session(self, demo: Demonstration, assets: list[MediaAsset]) -> Any:
        task_class, _ = classify_task(demo.task_text or demo.title)
        references = [a.id for a in assets if a.role == MediaRole.REFERENCE]
        session = self.app.sessions.create(
            user_id=self.app.config.user_id, kind=SessionKind.EXTERNAL_MEDIA, policy=demo.policy,
            task_text=demo.task_text or demo.title, task_class=task_class, reference_ids=references,
            status=SessionStatus.RECORDING, start_time=demo.created_at,
            meta={"demonstration_id": demo.id, "media": [a.id for a in assets],
                  "capture_sources": {"media": True, "blender_bridge": False}})
        self.app.db.execute("UPDATE demonstrations SET session_id = ? WHERE id = ?", (session.id, demo.id))
        for asset in assets:
            self.app.graph.link(("media", asset.id), "demonstrated_in", ("session", session.id))
        return session

    # -- extraction: frames and transitions --------------------------------------------------------------
    def _extract(self, session_id: str, assets: list[MediaAsset]) -> dict[str, Any]:
        seq = 0
        frames = 0
        transitions: list[dict[str, Any]] = []
        session = self.app.sessions.get(session_id)
        offset = 0.0
        for asset in [a for a in assets if a.kind == MediaKind.VIDEO and a.role == MediaRole.DEMONSTRATION]:
            reader = VideoReader(self.media.path(asset))
            try:
                samples, threshold = self.video.coarse(reader)
                events = [self.video.refine(reader, e, threshold) for e in self.video.events(samples, threshold)]
                keyframes: dict[float, str] = {}
                previous_thumb = None

                def store(t: float, reason: str) -> str | None:
                    nonlocal seq, frames, previous_thumb
                    key = round(t, 3)
                    if key in keyframes:
                        return keyframes[key]
                    image = reader.image_at(t)
                    if image is None:
                        return None
                    record = self.app.sessions.add_frame(
                        session_id, image, seq=seq, ts=session.start_time + offset + t, source="video",
                        media_asset_id=asset.id, media_timestamp=key, previous_thumb=previous_thumb,
                        meta={"reason": reason})
                    previous_thumb = thumbnail(image)
                    seq += 1
                    frames += 1
                    keyframes[key] = record.id
                    return record.id

                for t in self.video.representative_times(samples, events, reader.duration):
                    store(t, "representative")
                for event in events:
                    before_id = store(event.t_before, "before_change")
                    after_id = store(event.t_after, "after_change")
                    before, after = reader.image_at(event.t_before), reader.image_at(event.t_after)
                    if before is None or after is None:
                        continue
                    analysis = analyze_transition(before, after, t_before=event.t_before, t_after=event.t_after)
                    transitions.append({"asset": asset.id, "offset": offset, "before_frame": before_id,
                                        "after_frame": after_id, "analysis": analysis, "images": (before, after)})
                self.media.update_analysis(asset.id, {"coarse_threshold": round(threshold, 5), "events": len(events),
                                                      "samples": len(samples)})
                offset += reader.duration + 1.0
            finally:
                reader.close()
        # Image sequences and before/after pairs are ordered stills: every consecutive pair is a transition.
        stills = [a for a in assets if a.kind in (MediaKind.IMAGE, MediaKind.SCREENSHOT)
                  and a.role in (MediaRole.BEFORE, MediaRole.INTERMEDIATE, MediaRole.AFTER, MediaRole.DEMONSTRATION)]
        order = {MediaRole.BEFORE: 0, MediaRole.DEMONSTRATION: 1, MediaRole.INTERMEDIATE: 1, MediaRole.AFTER: 2}
        stills.sort(key=lambda a: (order.get(a.role, 1), a.created_at))
        previous: tuple[MediaAsset, Image.Image, str] | None = None
        for index, asset in enumerate(stills):
            image = Image.open(self.media.path(asset)).convert("RGB")
            t = offset + index * 2.0
            record = self.app.sessions.add_frame(session_id, image, seq=seq, ts=session.start_time + t, source="image",
                                                 media_asset_id=asset.id, media_timestamp=None,
                                                 meta={"role": asset.role.value})
            seq += 1
            frames += 1
            if previous is not None:
                analysis = analyze_transition(previous[1], image, t_before=t - 2.0, t_after=t)
                transitions.append({"asset": asset.id, "offset": 0.0, "before_frame": previous[2],
                                    "after_frame": record.id, "analysis": analysis, "images": (previous[1], image),
                                    "pair": [previous[0].id, asset.id]})
            previous = (asset, image, record.id)
        return {"frames": frames, "transitions": transitions, "next_seq": seq, "duration": offset}

    # -- analysis: vision, references, projects, instructions -------------------------------------------
    def _analyze(self, demo: Demonstration, session_id: str, assets: list[MediaAsset],
                 extracted: dict[str, Any]) -> dict[str, Any]:
        transitions = extracted["transitions"]
        model = self.vision.describe_transitions([t["images"] for t in transitions],
                                                 context=demo.task_text or demo.title) if transitions else {}
        references = []
        for asset in [a for a in assets if a.role in (MediaRole.REFERENCE, MediaRole.TARGET)
                      and a.kind in (MediaKind.IMAGE, MediaKind.SCREENSHOT)]:
            image = Image.open(self.media.path(asset)).convert("RGB")
            view = asset.analysis.get("view", "front")
            constraints = self.references.measure(asset.id, image, view)
            description = self.vision.describe_reference(image)
            if description:
                self.references.add(asset.id, "object_class", "object", {"object_class": description["object_class"],
                                                                         "style": description["style"]},
                                    source="model_inferred", confidence=0.6)
                for part in description.get("parts", []):
                    self.references.add(asset.id, "part_proportion", part["name"],
                                        {"fraction_of_total_length": part["fraction_of_total_length"]},
                                        source="model_inferred", confidence=0.5)
            references.append({"asset": asset.id, "constraints": [c.id for c in constraints]})
            self.app.graph.link(("session", session_id), "conditioned_by", ("media", asset.id))
        projects = [self._inspect_project(session_id, a) for a in assets if a.kind == MediaKind.BLENDER_PROJECT]
        instructions = list(demo.instructions) + [self.media.path(a).read_text(encoding="utf-8")
                                                  for a in assets if a.kind == MediaKind.TEXT]
        return {"transitions": transitions, "model": model, "references": references,
                "projects": [p for p in projects if p], "instructions": instructions,
                "next_seq": extracted["next_seq"], "duration": extracted["duration"]}

    def _inspect_project(self, session_id: str, asset: MediaAsset) -> dict[str, Any] | None:
        """Load a .blend in a private headless Blender and record its actual scene state."""
        from lucius.blender.headless import HeadlessBlender, headless_available

        if headless_available() is None:
            self.media.update_analysis(asset.id, {"project_state": "unavailable: no headless Blender"})
            return None
        with HeadlessBlender(allowed_read_dirs=[str(self.media.root)]) as bridge:
            bridge.execute("import_blend", {"path": str(self.media.path(asset))})
            structure = bridge.inspect_structure(views=())
        objects = [{"name": o["name"], "type": o["type"], "dimensions": o.get("dimensions"),
                    "modifiers": [m.get("type") for m in o.get("modifiers", [])],
                    "mesh": o.get("evaluated", {}).get("verts")} for o in structure.get("objects", [])]
        for obj in objects:
            if obj["type"] == "MESH" and obj.get("dimensions"):
                self.references.add(asset.id, "project_dimensions", obj["name"].lower(),
                                    {"dimensions": dict(zip("xyz", obj["dimensions"])), "modifiers": obj["modifiers"]},
                                    source="measured", confidence=0.95)
        self.media.update_analysis(asset.id, {"project_state": {"objects": objects}})
        self.app.graph.link(("session", session_id), "conditioned_by", ("media", asset.id))
        return {"asset": asset.id, "objects": objects}

    # -- action inference -------------------------------------------------------------------------------
    def _infer_actions(self, demo: Demonstration, session_id: str, analysis: dict[str, Any]) -> list[TrajectoryStep]:
        session = self.app.sessions.get(session_id)
        seq = analysis["next_seq"]
        events: list[CapturedEvent] = []
        steps: list[TrajectoryStep] = []
        mode: str | None = None
        for i, item in enumerate(analysis["transitions"]):
            tr: Transition = item["analysis"]
            model = analysis["model"].get(i)
            candidates = {c.action_type: c for c in tr.candidates}
            source, evidence_kind = ActionSource.INFERRED, EvidenceKind.VISUAL_STATE_TRANSITION
            evidence = list(tr.evidence)
            state_after: dict[str, Any] = {"_inferred": True}
            if model:
                for c in model.get("candidate_operations", []):
                    op = c["operation"]
                    merged = 0.4 * (candidates[op].confidence if op in candidates else 0.0) + 0.6 * float(c["confidence"])
                    candidates[op] = CandidateAction(action_type=op, confidence=round(merged, 3),
                                                     evidence=[f"model: {c.get('evidence', '')}"])
                evidence.append(f"model ({model.get('model')}): {model.get('description', '')}")
                if model.get("visible_mode"):
                    visible = str(model["visible_mode"]).upper()
                    state_after["mode"] = "EDIT_MESH" if "EDIT" in visible else "OBJECT" if "OBJECT" in visible else visible
                if model.get("visible_keystrokes"):
                    evidence.append("visible keystrokes: " + ", ".join(model["visible_keystrokes"]))
            ranked = sorted(candidates.values(), key=lambda c: c.confidence, reverse=True)
            top = ranked[0] if ranked else None
            if model and top is not None and top.evidence and top.evidence[0].startswith("model:"):
                source = ActionSource.MODEL_INFERRED
                evidence_kind = EvidenceKind.VISIBLE_SHORTCUT if model.get("visible_keystrokes") else EvidenceKind.VLM_INFERENCE
            confident = top is not None and top.confidence >= MIN_ACTION_CONFIDENCE and top.action_type != "unknown_action"
            action_type = top.action_type if confident else "unknown_action"
            if tr.kind == "camera" and top is not None and top.action_type.startswith("viewport"):
                action_type = top.action_type  # navigation is certain enough to support inspection segmentation
            t0 = session.start_time + item["offset"] + tr.t_before
            t1 = session.start_time + item["offset"] + tr.t_after
            if state_after.get("mode"):
                mode = state_after["mode"]
            params = {"estimated_shape_change": tr.shape} if tr.shape else {}
            steps.append(TrajectoryStep(
                id=new_id("step"), session_id=session_id, idx=0, t_start=t0, t_end=t1,
                frame_before_id=item["before_frame"], frame_after_id=item["after_frame"], action_type=action_type,
                action_payload={"params": params, "change_kind": tr.kind, "bbox": tr.bbox},
                action_source=source, evidence_kind=evidence_kind,
                action_confidence=round(top.confidence if top else 0.0, 3), evidence=evidence,
                candidate_actions=ranked[:6], mode_label=mode, actor="human",
                media_timestamp=round(item["offset"] + tr.t_before, 3), state_after=state_after if len(state_after) > 1 else None,
                meta={"media_asset_id": item["asset"], "requires_validation": not confident or source != ActionSource.OBSERVED,
                      **({"pair": item["pair"]} if "pair" in item else {})},
            ))
            events.append(CapturedEvent(seq=seq, ts=t1, kind=EventKind.MEDIA_OBSERVATION, actor=Actor.SYSTEM,
                                        payload={"transition": tr.model_dump(), "model": model, "asset": item["asset"]}))
            seq += 1
        instruction_steps = self._instruction_steps(session, analysis, steps)
        steps = sorted(steps + instruction_steps, key=lambda s: s.t_start)
        for idx, step in enumerate(steps):
            step.idx = idx
        link_undo_relations(steps)
        for text in analysis["instructions"]:
            events.append(CapturedEvent(seq=seq, ts=session.start_time, kind=EventKind.ANNOTATION, actor=Actor.HUMAN,
                                        payload={"text": text[:2000], "label": "instruction"}))
            seq += 1
        self.app.sessions.append_events(session_id, events)
        self.app.trajectories.replace(session_id, steps)
        return steps

    def _instruction_steps(self, session: Any, analysis: dict[str, Any], media_steps: list[TrajectoryStep]) -> list[TrajectoryStep]:
        """Explicit text instructions: steps when there is no visual trajectory, evidence otherwise."""
        sentences = [s.strip() for text in analysis["instructions"] for s in re.split(r"[.\n;]+", text) if s.strip()]
        out = []
        for i, sentence in enumerate(sentences):
            lowered = sentence.lower()
            matches = [(action, m) for pattern, action in KEYWORD_ACTIONS if (m := re.search(pattern, lowered))]
            if not matches:
                continue
            action, match = matches[0]
            params: dict[str, Any] = {}
            if action == "add_primitive":
                params["kind"] = match.group(1).replace("sphere", "uv_sphere")
            numbers = re.findall(r"\d+(?:\.\d+)?", lowered)
            if numbers:
                params["value_mentioned"] = float(numbers[0])
            axis = re.search(r"\b(?:along|on|in)\s+(?:the\s+)?([xyz])\b", lowered)
            if axis:
                params["axis"] = axis.group(1)
            if media_steps:
                continue  # with a visual trajectory, instructions are evidence (annotations), not extra steps
            t = session.start_time + analysis["duration"] + i * 2.0
            out.append(TrajectoryStep(
                id=new_id("step"), session_id=session.id, idx=0, t_start=t, t_end=t + 1.0, action_type=action,
                action_payload={"params": params, "instruction": sentence}, action_source=ActionSource.INFERRED,
                evidence_kind=EvidenceKind.TEXT_INSTRUCTION,
                action_confidence=round(0.6 * EVIDENCE_WEIGHT[EvidenceKind.TEXT_INSTRUCTION] + 0.3, 3),
                evidence=[f"instruction: {sentence}"], actor="human", meta={"synthetic_time": True,
                                                                           "requires_validation": True}))
        return out

    # -- validation by reproduction (10U) ----------------------------------------------------------------
    def _validate(self, demo: Demonstration, session_id: str, skill_ids: list[str]) -> dict[str, Any]:
        from lucius.blender.headless import headless_available

        if not skill_ids:
            return {"status": "nothing_to_validate"}
        if headless_available() is None:
            return {"status": "pending", "reason": "no headless Blender available for reproduction"}
        session = self.app.sessions.get(session_id)
        silhouettes = self.references.silhouettes(session.reference_ids)
        results = {}
        backend = self.app.headless_backend()
        for skill_id in skill_ids:
            skill = self.app.library.get(skill_id)
            run = self.app.engine.run(skill.definition.name, backend, mode="validation", references=silhouettes,
                                      reference_ids=session.reference_ids, reset_scene=True)
            results[skill_id] = {"run_id": run.run_id, "verdict": run.verdict,
                                 "unresolved": len(run.plan.unresolved) if run.plan else None,
                                 "status_after": self.app.library.get(skill_id).status.value}
        return {"status": "done", "runs": results}

    # -- human corrections (10R) --------------------------------------------------------------------------
    def confirm_action(self, step_id: str, action_type: str, params: dict[str, Any] | None = None,
                       user_id: str = "local") -> None:
        step = self.app.trajectories.get(step_id)
        if step is None:
            raise NotFoundError(f"step {step_id} not found")
        self.app.trajectories.confirm_action(step_id, action_type, params)
        self.app.db.insert("human_edits", {"id": new_id("edit"), "subject_kind": "trajectory_step", "subject_id": step_id,
                                           "op": "confirm_action",
                                           "before": dumps({"action_type": step.action_type,
                                                            "source": step.action_source.value}),
                                           "after": dumps({"action_type": action_type, "params": params}),
                                           "user_id": user_id, "created_at": now()})

    def set_policy(self, demo_id: str, policy: DataPolicy) -> None:
        demo = self.get(demo_id)
        self.app.db.execute("UPDATE demonstrations SET policy = ? WHERE id = ?", (dumps(policy), demo_id))
        for asset in self.media.list(demonstration_id=demo_id):
            self.media.set_policy(asset.id, policy)
        if demo.session_id:
            self.app.sessions.update(demo.session_id, policy=policy)
