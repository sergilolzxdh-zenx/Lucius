"""TRAIN X: the practice loop (sections 42, 58).

    select task -> retrieve skills -> execute -> evaluate -> store (success) | record failure,
    recover or request correction (failure) -> repeat -> advance after mastery.

Practice runs are ordinary engine runs (mode ``practice``) against headless or live Blender, so
their evidence updates skills, failure memory and mastery exactly like any other execution.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image
from pydantic import BaseModel, Field

from lucius.errors import NotFoundError, ValidationError
from lucius.events.bus import EventType
from lucius.executor.backend import ExecutionBackend
from lucius.executor.human import HumanChannel
from lucius.ids import new_id
from lucius.ingestion.media import MediaRole
from lucius.practice.curriculum import CURRICULA, REFERENCE_GENERATORS, Curriculum, Stage, resolve_curriculum
from lucius.practice.mastery import MasteryMetrics, check_gate, compute_metrics
from lucius.provenance import DataPolicy, SourceClass
from lucius.storage.db import dumps, loads
from lucius.timeutil import now

if TYPE_CHECKING:
    from lucius.app import Lucius


class AttemptReport(BaseModel):
    run_id: str
    task: str
    params: dict[str, Any]
    verdict: str
    checkpoints_passed: int
    checkpoints_total: int
    takeovers: int
    recoveries: int
    reason_codes: list[str] = Field(default_factory=list)


class PracticeReport(BaseModel):
    curriculum: str
    stage: int
    stage_name: str
    status: str                      # practicing, mastered, needs_demonstration, requires_gui, completed
    attempts: list[AttemptReport] = Field(default_factory=list)
    metrics: MasteryMetrics = Field(default_factory=MasteryMetrics)
    unmet: list[str] = Field(default_factory=list)
    advanced_to: int | None = None
    message: str = ""


class PracticeEngine:
    def __init__(self, app: Lucius) -> None:
        self.app = app

    # -- curriculum bookkeeping -------------------------------------------------------------------
    def curricula(self) -> list[Curriculum]:
        return list(CURRICULA.values())

    def curriculum(self, name: str) -> Curriculum:
        cur = resolve_curriculum(name)
        if cur is None:
            raise NotFoundError(f"unknown curriculum {name!r}", available=sorted(CURRICULA))
        return cur

    def _task_ids(self, cur: Curriculum, stage: Stage) -> list[str]:
        ids = []
        for template in stage.tasks:
            row = self.app.db.query_one("SELECT id FROM practice_tasks WHERE curriculum = ? AND stage = ? AND name = ?",
                                        (cur.name, stage.index, template.name))
            if row is None:
                task_id = new_id("practice_task")
                self.app.db.insert("practice_tasks", {"id": task_id, "curriculum": cur.name, "stage": stage.index,
                                                      "name": template.name,
                                                      "definition": dumps(template.model_dump()), "created_at": now()})
                ids.append(task_id)
            else:
                ids.append(row["id"])
        return ids

    def stage_status(self, cur: Curriculum, stage: Stage) -> dict[str, Any]:
        metrics = compute_metrics(self.app.db, self._task_ids(cur, stage))
        gate = check_gate(metrics, stage.gate)
        override = self.app.db.query_one("SELECT human_override FROM mastery_records WHERE curriculum = ? AND stage = ?",
                                         (cur.name, stage.index))
        human = loads(override["human_override"]) if override and override["human_override"] else None
        if human:
            status = human.get("status", "practicing")
        elif gate.mastered:
            status = "mastered"
        elif metrics.attempts and metrics.needs_demonstration == metrics.attempts:
            status = "needs_demonstration"
        elif stage.requires_gui:
            status = "requires_gui"
        else:
            status = "practicing" if metrics.attempts else "not_started"
        return {"stage": stage.index, "name": stage.name, "description": stage.description, "status": status,
                "metrics": metrics.model_dump(), "unmet": gate.unmet, "mastery_score": gate.mastery_score,
                "human_override": human, "gate": stage.gate.model_dump()}

    def overview(self, name: str) -> dict[str, Any]:
        cur = self.curriculum(name)
        stages = [self.stage_status(cur, s) for s in cur.stages]
        current = next((s["stage"] for s in stages if s["status"] != "mastered"), None)
        return {"curriculum": cur.name, "title": cur.title, "description": cur.description, "current_stage": current,
                "stages": stages}

    def set_override(self, name: str, stage_index: int, status: str | None, *, user_id: str = "local") -> None:
        """Human override of a stage's mastery status (recorded as data, reversible)."""
        cur = self.curriculum(name)
        stage = next((s for s in cur.stages if s.index == stage_index), None)
        if stage is None:
            raise ValidationError(f"stage {stage_index} does not exist")
        value = dumps({"status": status, "by": user_id, "ts": now()}) if status else None
        self.app.db.execute(
            "INSERT INTO mastery_records (id, curriculum, stage, status, metrics, mastery_score, human_override, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(curriculum, stage) DO UPDATE SET human_override = excluded.human_override,"
            " updated_at = excluded.updated_at",
            (new_id("mastery"), cur.name, stage_index, status or "practicing", dumps({}), None, value, now()))

    def _record(self, cur: Curriculum, stage: Stage) -> dict[str, Any]:
        status = self.stage_status(cur, stage)
        self.app.db.execute(
            "INSERT INTO mastery_records (id, curriculum, stage, status, metrics, mastery_score, human_override, updated_at)"
            " VALUES (?,?,?,?,?,?,NULL,?) ON CONFLICT(curriculum, stage) DO UPDATE SET status = excluded.status,"
            " metrics = excluded.metrics, mastery_score = excluded.mastery_score, updated_at = excluded.updated_at",
            (new_id("mastery"), cur.name, stage.index, status["status"], dumps(status["metrics"]),
             status["mastery_score"], now()))
        self.app.bus.publish(EventType.MASTERY_UPDATED, f"{cur.name}:{stage.index}", status=status["status"],
                             mastery_score=status["mastery_score"], unmet=status["unmet"])
        return status

    # -- practice ---------------------------------------------------------------------------------------
    def train(self, name: str, *, attempts: int = 3, stage: int | None = None, backend: ExecutionBackend | None = None,
              seed: int | None = None, human: HumanChannel | None = None) -> PracticeReport:
        cur = self.curriculum(name)
        if stage is None:
            statuses = [self.stage_status(cur, s) for s in cur.stages]
            current = next((s for s in statuses if s["status"] != "mastered"), None)
            if current is None:
                return PracticeReport(curriculum=cur.name, stage=cur.stages[-1].index, stage_name=cur.stages[-1].name,
                                      status="completed", message="every stage is mastered")
            stage = current["stage"]
        stage_def = next((s for s in cur.stages if s.index == stage), None)
        if stage_def is None:
            raise ValidationError(f"stage {stage} does not exist in {cur.name}")
        backend = backend or self.app.backend(prefer="live")
        report = PracticeReport(curriculum=cur.name, stage=stage_def.index, stage_name=stage_def.name, status="practicing")
        if stage_def.requires_gui and not backend.gui_available:
            report.status = "requires_gui"
            report.message = (f"stage '{stage_def.name}' needs an interactive Blender viewport; connect a live Blender "
                              "with the Lucius Bridge add-on to practise it")
            return report
        rng = random.Random(seed)
        task_ids = self._task_ids(cur, stage_def)
        for i in range(attempts):
            template_index = i % len(stage_def.tasks)
            template = stage_def.tasks[template_index]
            values = template.sample(rng)
            text, criteria, task_params = template.instantiate(values)
            references, reference_ids = [], []
            if template.reference in REFERENCE_GENERATORS:
                reference_ids = [self._reference(template.reference, values)]
                references = self.app.ingestion.references.silhouettes(reference_ids, min_iou=0.8)
            run = self.app.engine.run(text, backend, mode="practice", task_params=task_params,
                                      success_criteria=criteria, references=references, reference_ids=reference_ids,
                                      practice_task_id=task_ids[template_index], human=human, reset_scene=True)
            report.attempts.append(AttemptReport(
                run_id=run.run_id, task=text, params=values, verdict=run.verdict,
                checkpoints_passed=run.metrics.get("checkpoints_passed", 0),
                checkpoints_total=run.metrics.get("checkpoints_total", 0), takeovers=run.takeovers,
                recoveries=run.recoveries, reason_codes=run.reason_codes))
            if "needs_demonstration" in run.reason_codes:
                report.status = "needs_demonstration"
                report.message = (f"no learned skill covers '{template.name}' yet: demonstrate it with WATCH ME or "
                                  "teach it from media, then practise again")
                break
        status = self._record(cur, stage_def)
        report.metrics = MasteryMetrics.model_validate(status["metrics"])
        report.unmet = status["unmet"]
        if report.status != "needs_demonstration":
            report.status = status["status"]
        if status["status"] == "mastered":
            nxt = next((s for s in cur.stages if s.index > stage_def.index), None)
            report.advanced_to = nxt.index if nxt else None
            report.message = f"stage '{stage_def.name}' mastered" + (f"; next: {nxt.name}" if nxt else "")
        return report

    def _reference(self, generator: str, values: dict[str, Any], target: str | None = None) -> str:
        """A synthetic reference image for a practice task (stored as media with its provenance)."""
        make, default_target = REFERENCE_GENERATORS[generator]
        image = make(values)
        target = target or default_target
        tmp = Path(self.app.config.media_dir) / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        path = tmp / f"{generator}_{new_id('media')}.png"
        image.save(path)
        policy = DataPolicy(source=SourceClass.SYSTEM_SEEDED, license="generated", reference_only=True,
                            training_allowed=False, export_allowed=True, notes=f"synthetic {generator} reference")
        asset = self.app.ingestion.media.ingest(path, role=MediaRole.REFERENCE, policy=policy)
        path.unlink(missing_ok=True)
        self.app.ingestion.media.update_analysis(asset.id, {"view": "front", "synthetic": values, "target": target})
        with Image.open(self.app.ingestion.media.path(asset)) as img:
            self.app.ingestion.references.measure(asset.id, img.convert("RGB"), "front", target=target)
        return asset.id
