"""Dataset assembly, export and import (sections 44-45, 86).

Training eligibility is explicit and conservative: a session enters a *training* dataset only if
its policy allows training, consent is granted and it is not reference-only. Export datasets
require ``export_allowed``. External media never enter either silently -- their policy must be
changed by the user first. Every sample carries its provenance.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from lucius.dataset.validation import DatasetValidator, QualityReport
from lucius.errors import ConflictError, NotFoundError, PolicyViolation, ValidationError
from lucius.events.bus import EventType
from lucius.ids import new_id
from lucius.provenance import DataPolicy
from lucius.sessions.models import CapturedEvent, EnvironmentInfo, Outcome, Session, SessionKind, SessionStatus
from lucius.skills.schema import SkillDefinition, SkillStatus
from lucius.storage.db import dumps, loads
from lucius.storage.jsonl import write_jsonl
from lucius.timeutil import now
from lucius.trajectory.model import TrajectoryStep

if TYPE_CHECKING:
    from lucius.app import Lucius

Purpose = Literal["training", "export"]


class DatasetFilters(BaseModel):
    kinds: list[str] = Field(default_factory=lambda: [k.value for k in SessionKind])
    sources: list[str] | None = None
    task_class: str | None = None
    min_quality: float = 0.5
    session_ids: list[str] | None = None


class DatasetResult(BaseModel):
    id: str
    name: str
    purpose: str
    path: str
    samples: int
    excluded: dict[str, list[str]] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)


def eligible(policy: DataPolicy, purpose: Purpose) -> tuple[bool, str]:
    if purpose == "training":
        if not policy.training_allowed:
            return False, "training_not_allowed"
        if policy.consent_status.value != "granted":
            return False, "no_consent"
        if policy.reference_only:
            return False, "reference_only"
        return True, "ok"
    if not policy.export_allowed:
        return False, "export_not_allowed"
    return True, "ok"


class DatasetService:
    def __init__(self, app: Lucius) -> None:
        self.app = app
        self.validator = DatasetValidator(app.sessions, app.trajectories, app.ingestion.media)
        self.root = Path(app.config.exports_dir)

    # -- samples ------------------------------------------------------------------------------------
    def sample(self, session: Session, *, include_media: bool) -> dict[str, Any]:
        app = self.app
        steps = app.trajectories.for_session(session.id)
        segments = app.segments.for_session(session.id)
        events = app.sessions.events(session.id, kinds=["blender_state"])
        initial = events[0].payload if events else None
        final = events[-1].payload if events else None
        references = []
        for media_id in session.reference_ids:
            try:
                asset = app.ingestion.media.get(media_id)
            except NotFoundError:
                continue
            allowed, _ = eligible(asset.policy, "export")
            references.append({"media_id": media_id, "sha256": asset.sha256, "license": asset.policy.license,
                               "included": include_media and allowed, "role": asset.role.value})
        runs = app.db.query("SELECT id FROM runs WHERE session_id = ?", (session.id,))
        evaluations = [e for r in runs for e in app.evaluator.for_subject("run", r["id"])]
        skill_links = [{"skill_id": r["skill_id"], "role": r["role"], "segment_ids": loads(r["segment_ids"], [])}
                       for r in app.db.query("SELECT skill_id, role, segment_ids FROM skill_examples WHERE session_id = ?",
                                             (session.id,))]
        failures = [{"id": f.id, "phase": f.phase, "problem": f.observed_problem, "future_rule": f.future_rule,
                     "rule_status": f.rule_status} for f in app.failures.list(limit=5000)
                    if any(e.session_id == session.id for e in f.evidence)]
        corrections = [c.model_dump() for c in app.corrections.list(limit=5000) if c.session_id == session.id]
        return {
            "sample_id": new_id("sample"), "session_id": session.id, "task_text": session.task_text,
            "task_class": session.task_class, "kind": session.kind.value, "reference_images": references,
            "initial_blender_state": initial, "trajectory_steps": [s.model_dump(mode="json") for s in steps],
            "segments": [s.model_dump(mode="json") for s in segments],
            "intent_labels": {seg.id: app.intents.summary(app.intents.primary(seg.id)) for seg in segments},
            "skill_links": skill_links, "evaluation": evaluations, "failure_records": failures,
            "user_corrections": corrections, "final_artifacts": {"final_state": final},
            "outcome": session.outcome.value if session.outcome else None,
            "license": session.policy.license, "consent": session.policy.consent_status.value,
            "provenance": {"session_id": session.id, "source": session.source.value, "content_hash": session.content_hash,
                           "blender_version": session.environment.blender_version,
                           "recorded_at": session.start_time, "exported_at": now()},
        }

    def build(self, name: str, *, purpose: Purpose = "training", filters: DatasetFilters | None = None,
              include_media: bool = False) -> DatasetResult:
        filters = filters or DatasetFilters()
        dataset_id = new_id("dataset")
        folder = self.root / "datasets" / dataset_id
        candidates = [s for s in self.app.sessions.list(limit=100000)
                      if s.kind.value in filters.kinds and s.status in (SessionStatus.PROCESSED, SessionStatus.FINALIZED,
                                                                         SessionStatus.INTERRUPTED)
                      and (filters.sources is None or s.source.value in filters.sources)
                      and (filters.task_class is None or s.task_class == filters.task_class)
                      and (filters.session_ids is None or s.id in filters.session_ids)]
        excluded: dict[str, list[str]] = {}
        allowed = []
        for s in candidates:
            ok, reason = eligible(s.policy, purpose)
            if ok:
                allowed.append(s)
            else:
                excluded.setdefault(reason, []).append(s.id)
        report = self.validator.validate(allowed)
        quality = {q.session_id: q for q in report.sessions}
        kept = []
        for s in allowed:
            if quality[s.id].score < filters.min_quality:
                excluded.setdefault("low_quality", []).append(s.id)
            else:
                kept.append(s)
        samples = [self.sample(s, include_media=include_media) for s in kept]
        path = folder / "samples.jsonl"
        write_jsonl(path, samples)
        manifest = {"id": dataset_id, "name": name, "purpose": purpose, "created_at": now(),
                    "filters": filters.model_dump(), "samples": len(samples), "excluded": excluded,
                    "quality": DatasetValidator.summary(report), "schema_version": 1}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.app.db.insert("datasets", {"id": dataset_id, "name": name, "filters": dumps(filters.model_dump()),
                                        "path": str(folder), "stats": dumps({"samples": len(samples),
                                                                             "excluded": {k: len(v) for k, v in excluded.items()}}),
                                        "quality_report": dumps(manifest["quality"]), "created_at": now()})
        rows = [(new_id("sample"), dataset_id, sample["session_id"], i, quality[sample["session_id"]].score,
                 dumps([iss.model_dump() for iss in quality[sample["session_id"]].issues]),
                 dumps(sample["provenance"]), now()) for i, sample in enumerate(samples)]
        self.app.db.executemany("INSERT INTO dataset_samples (id, dataset_id, session_id, line_no, quality_score, issues,"
                                " provenance, created_at) VALUES (?,?,?,?,?,?,?,?)", rows)
        self.app.bus.publish(EventType.DATASET_BUILT, dataset_id, name=name, purpose=purpose, samples=len(samples),
                             excluded={k: len(v) for k, v in excluded.items()})
        return DatasetResult(id=dataset_id, name=name, purpose=purpose, path=str(folder), samples=len(samples),
                             excluded=excluded, quality=manifest["quality"])

    def list(self) -> list[dict[str, Any]]:
        return [{**dict(r), "filters": loads(r["filters"], {}), "stats": loads(r["stats"], {}),
                 "quality_report": loads(r["quality_report"], {})}
                for r in self.app.db.query("SELECT * FROM datasets ORDER BY created_at DESC")]

    def validate(self, session_ids: list[str] | None = None) -> QualityReport:
        sessions = [self.app.sessions.get(i) for i in session_ids] if session_ids else self.app.sessions.list(limit=100000)
        return self.validator.validate(sessions)

    # -- session bundles ----------------------------------------------------------------------------
    def export_session(self, session_id: str, *, include_frames: bool = True) -> Path:
        app = self.app
        session = app.sessions.get(session_id)
        ok, reason = eligible(session.policy, "export")
        if not ok:
            raise PolicyViolation(f"session cannot be exported: {reason}", session_id=session_id)
        path = self.root / "sessions" / f"{session_id}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = app.sessions.frames_for(session_id)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("manifest.json", json.dumps({"type": "lucius.session", "version": 1, "session_id": session_id,
                                                         "exported_at": now(), "frames_included": include_frames}))
            bundle.writestr("session.json", session.model_dump_json(indent=2))
            bundle.writestr("events.jsonl", "\n".join(e.model_dump_json() for e in app.sessions.events(session_id)))
            bundle.writestr("trajectory.jsonl", "\n".join(s.model_dump_json() for s in app.trajectories.for_session(session_id)))
            bundle.writestr("segments.json", json.dumps([s.model_dump(mode="json") for s in app.segments.for_session(session_id)]))
            bundle.writestr("frames.json", json.dumps([f.model_dump(mode="json") for f in frames]))
            runs = app.db.query("SELECT id FROM runs WHERE session_id = ?", (session_id,))
            bundle.writestr("evaluations.json", json.dumps([e for r in runs for e in app.evaluator.for_subject("run", r["id"])],
                                                           default=str))
            if include_frames:
                for f in frames:
                    bundle.write(app.sessions.frames.path(f.path), f"frames/{f.path}")
            for media_id in session.reference_ids:
                try:
                    asset = app.ingestion.media.get(media_id)
                except NotFoundError:
                    continue
                if eligible(asset.policy, "export")[0]:
                    bundle.write(app.ingestion.media.path(asset), f"references/{asset.path}")
        return path

    def import_session(self, path: Path) -> Session:
        """Import a bundle as a new session (all ids remapped; nothing is overwritten)."""
        app = self.app
        with zipfile.ZipFile(path) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
            if manifest.get("type") != "lucius.session":
                raise ValidationError("not a Lucius session bundle")
            original = Session.model_validate_json(bundle.read("session.json"))
            policy = original.policy.model_copy(update={"notes": f"imported from bundle {original.id}"})
            session = app.sessions.create(
                user_id=app.config.user_id, kind=original.kind, policy=policy, task_text=original.task_text,
                task_class=original.task_class, environment=EnvironmentInfo.model_validate(original.environment.model_dump()),
                recording_config=original.recording_config, status=SessionStatus.RECORDING,
                start_time=original.start_time, meta={**original.meta, "imported_from": original.id})
            events = [CapturedEvent.model_validate_json(line) for line in bundle.read("events.jsonl").decode().splitlines()
                      if line.strip()]
            app.sessions.append_events(session.id, events)
            frame_map: dict[str, str] = {}
            if manifest.get("frames_included"):
                from io import BytesIO

                from PIL import Image

                for f in json.loads(bundle.read("frames.json")):
                    data = bundle.read(f"frames/{f['path']}")
                    record = app.sessions.add_frame(session.id, Image.open(BytesIO(data)), seq=f["seq"], ts=f["ts"],
                                                    source=f["source"], window_bounds=f.get("window_bounds"),
                                                    media_timestamp=f.get("media_timestamp"), meta=f.get("meta") or {})
                    frame_map[f["id"]] = record.id
            steps = []
            for line in bundle.read("trajectory.jsonl").decode().splitlines():
                if not line.strip():
                    continue
                step = TrajectoryStep.model_validate_json(line)
                steps.append(step.model_copy(update={
                    "id": new_id("step"), "session_id": session.id, "segment_id": None,
                    "frame_before_id": frame_map.get(step.frame_before_id or ""),
                    "frame_after_id": frame_map.get(step.frame_after_id or "")}))
            app.trajectories.replace(session.id, steps)
        app.sessions.finalize(session.id, end_time=original.end_time, outcome=original.outcome or Outcome.UNKNOWN)
        return app.sessions.get(session.id)

    # -- skills ---------------------------------------------------------------------------------------
    def export_skill(self, skill_id: str) -> Path:
        app = self.app
        skill = app.library.get(skill_id)
        path = self.root / "skills" / f"{skill_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "lucius.skill", "version": 1, "exported_at": now(), "skill": skill.model_dump(mode="json"),
            "versions": [{**v, "definition": app.library.definition(skill_id, v["version"]).model_dump(mode="json")}
                         for v in app.library.versions(skill_id)],
            "provenance": app.library.provenance(skill_id)["sources"],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def import_skill(self, path: Path, *, as_new: bool = False) -> str:
        """Import a skill definition. Evidence from elsewhere is not re-counted here: the skill starts as a
        candidate and must earn its status locally."""
        payload = json.loads(Path(path).read_text())
        if payload.get("type") != "lucius.skill":
            raise ValidationError("not a Lucius skill export")
        definition = SkillDefinition.model_validate(payload["versions"][-1]["definition"])
        if self.app.library.exists(definition.skill_id):
            if not as_new:
                raise ConflictError(f"skill {definition.skill_id} already exists", skill_id=definition.skill_id)
            definition.skill_id = f"{definition.skill_id}_imported_{new_id('skill_version')[-6:]}"
        definition.notes = definition.notes + [f"imported from {Path(path).name}; original provenance kept in notes",
                                               f"original sources: {sorted(payload.get('provenance', {}))}"]
        self.app.library.create(definition, created_by="import", change_note="imported skill",
                                status=SkillStatus.CANDIDATE_PATTERN)
        return definition.skill_id
