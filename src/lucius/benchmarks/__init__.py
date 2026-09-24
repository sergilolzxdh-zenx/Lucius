"""Benchmarks and A/B experiments (sections 49-50, 93-94).

A benchmark defines a task, its initial state, an optional reference, required checkpoints and
generalisation variants (different dimensions/references). Arms are execution strategies
(no memory, retrieval-only, memory-enhanced, raw-demonstration replay, ...). Results are
compared with bootstrap confidence intervals so improvements are measured, not anecdotal.
"""

from __future__ import annotations

import random
import statistics
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from lucius.errors import NotFoundError
from lucius.executor.backend import ExecutionBackend
from lucius.executor.engine import ArmConfig, RunResult
from lucius.ids import new_id
from lucius.practice.curriculum import REFERENCE_GENERATORS, PracticeTaskTemplate
from lucius.storage.db import dumps, loads
from lucius.timeutil import now

if TYPE_CHECKING:
    from lucius.app import Lucius

ARMS: dict[str, ArmConfig] = {
    "baseline": ArmConfig(name="baseline", retrieval_strategy="none", use_failure_guards=False, use_recovery=False,
                          use_preferences=False),
    "retrieval_only": ArmConfig(name="retrieval_only", retrieval_strategy="vector", use_failure_guards=False,
                                use_recovery=False, use_preferences=False),
    "memory_enhanced": ArmConfig(name="memory_enhanced"),
    "raw_demonstrations": ArmConfig(name="raw_demonstrations", retrieval_strategy="episodes_only",
                                    use_failure_guards=False, use_recovery=False, use_preferences=False),
}
UNAVAILABLE_ARMS = {"trained_policy": "no trained policy is registered (see lucius.training)"}


class BenchmarkDefinition(BaseModel):
    name: str
    description: str
    template: PracticeTaskTemplate
    difficulty: int = 1
    initial_state: str = "empty_scene"
    allowed_methods: list[str] = Field(default_factory=lambda: ["blender_api"])
    evaluation_method: str = "structural+measured_visual"
    variants: list[dict[str, Any]] = Field(default_factory=list)   # generalisation axes (explicit parameter sets)


def seed_benchmarks() -> list[BenchmarkDefinition]:
    from lucius.practice.curriculum import CURRICULA

    stages = {s.name: s for s in CURRICULA["sculpting"].stages}
    blade = stages["Structured hard-surface"].tasks[0]
    ref = stages["Reference matching"].tasks[0]
    return [
        BenchmarkDefinition(name="sized_box", description="Create a precisely sized box", difficulty=1,
                            template=stages["Primitive manipulation"].tasks[0],
                            variants=[{"name": "Prop", "x": 1.0, "y": 2.0, "z": 0.5},
                                      {"name": "Prop", "x": 2.5, "y": 0.6, "z": 1.8}]),
        BenchmarkDefinition(name="blade_blockout", description="Blade blockout at unseen proportions", difficulty=3,
                            template=blade,
                            variants=[{"length": 3.5, "width": 0.22}, {"length": 5.2, "width": 0.3},
                                      {"length": 6.8, "width": 0.42}]),
        BenchmarkDefinition(name="blade_reference_match", description="Match blade reference silhouettes", difficulty=4,
                            template=ref, variants=[{"length": 4.2, "width": 0.25}, {"length": 6.0, "width": 0.4}]),
    ]


def bootstrap_ci(values: list[float], iterations: int = 2000, seed: int = 7) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return round(means[int(0.025 * iterations)], 4), round(means[int(0.975 * iterations) - 1], 4)


class BenchmarkService:
    def __init__(self, app: Lucius) -> None:
        self.app = app
        for definition in seed_benchmarks():
            if not app.db.scalar("SELECT 1 FROM benchmarks WHERE name = ?", (definition.name,)):
                app.db.insert("benchmarks", {"id": new_id("benchmark"), "name": definition.name,
                                             "definition": dumps(definition.model_dump()), "created_at": now()})

    def list(self) -> list[dict[str, Any]]:
        return [{"id": r["id"], "name": r["name"], **loads(r["definition"], {})}
                for r in self.app.db.query("SELECT * FROM benchmarks ORDER BY name")]

    def get(self, name_or_id: str) -> tuple[str, BenchmarkDefinition]:
        row = self.app.db.query_one("SELECT * FROM benchmarks WHERE id = ? OR name = ?", (name_or_id, name_or_id))
        if row is None:
            raise NotFoundError(f"benchmark {name_or_id} not found")
        return row["id"], BenchmarkDefinition.model_validate(loads(row["definition"]))

    def run_one(self, name: str, arm: str, backend: ExecutionBackend, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        benchmark_id, definition = self.get(name)
        config = ARMS[arm]
        results = []
        for variant in definition.variants:
            text, criteria, task_params = definition.template.instantiate(variant)
            references, reference_ids = [], []
            if definition.template.reference in REFERENCE_GENERATORS:
                reference_ids = [self.app.practice._reference(definition.template.reference, variant)]
                references = self.app.ingestion.references.silhouettes(reference_ids, min_iou=0.8)
            if arm == "raw_demonstrations":
                run = self._raw_run(text, backend, task_params, criteria, references, reference_ids, benchmark_id)
            else:
                run = self.app.engine.run(text, backend, mode="benchmark", arm=config, task_params=task_params,
                                          success_criteria=criteria, references=references, reference_ids=reference_ids,
                                          benchmark_id=benchmark_id, reset_scene=True)
            metrics = {**run.metrics, "verdict": run.verdict, "variant": variant}
            self.app.db.insert("benchmark_results", {
                "id": new_id("benchmark_result"), "benchmark_id": benchmark_id, "experiment_id": experiment_id,
                "run_id": run.run_id, "arm": arm, "variant": dumps(variant), "success": int(run.verdict == "success"),
                "metrics": dumps(metrics), "created_at": now()})
            results.append({"run_id": run.run_id, "variant": variant, "verdict": run.verdict, "metrics": run.metrics})
        return results

    def _raw_run(self, text: str, backend: ExecutionBackend, task_params: dict[str, Any], criteria: list,
                 references: list, reference_ids: list[str], benchmark_id: str) -> RunResult:
        """Raw-demonstration arm: the engine executes a plan replayed from the most similar episode."""
        from lucius.retrieval import RetrievalQuery
        from lucius.skills.extract import SkillExtractor, _CandidateBuilder
        from lucius.taxonomy import classify_task

        app = self.app
        retrieval = app.retriever.retrieve(RetrievalQuery(text=text, task_class=classify_task(text)[0],
                                                          strategy="episodes_only"))
        definitions = []
        if retrieval.episodes:
            episode = app.episodes.by_id(retrieval.episodes[0].id)
            session = app.sessions.get(episode.session_id)
            steps = app.trajectories.for_session(session.id)
            segments = app.segments.for_session(session.id)
            object_class, categories = classify_task(session.task_text)
            for unit in SkillExtractor.units(segments):
                name = SkillExtractor._main_object(unit, steps)
                role = name.lower() if name else None
                builder = _CandidateBuilder(session, unit, steps, role, name, object_class, categories)
                candidate = app.extractor._candidate(builder)
                if candidate is not None:
                    definitions.append(candidate)
        return app.engine.run(text, backend, mode="benchmark", arm=ARMS["raw_demonstrations"], task_params=task_params,
                              success_criteria=criteria, references=references, reference_ids=reference_ids,
                              benchmark_id=benchmark_id, reset_scene=True,
                              plan_fn=lambda task: app.planner.plan_from_episode(
                                  task, retrieval, definitions, gui_available=backend.gui_available))

    def experiment(self, name: str, *, arms: list[str], benchmarks: list[str], backend: ExecutionBackend,
                   repeats: int = 1) -> dict[str, Any]:
        experiment_id = new_id("experiment")
        self.app.db.insert("experiments", {"id": experiment_id, "name": name, "arms": dumps(arms),
                                           "benchmark_ids": dumps(benchmarks), "status": "running",
                                           "summary": dumps({}), "created_at": now()})
        unavailable = {a: UNAVAILABLE_ARMS[a] for a in arms if a in UNAVAILABLE_ARMS}
        for _ in range(repeats):
            for arm in [a for a in arms if a in ARMS]:
                for bench in benchmarks:
                    self.run_one(bench, arm, backend, experiment_id=experiment_id)
        summary = self.summary(experiment_id)
        summary["unavailable_arms"] = unavailable
        self.app.db.update("experiments", "id", experiment_id, {"status": "done", "summary": dumps(summary)})
        return {"id": experiment_id, **summary}

    def summary(self, experiment_id: str | None = None, benchmark: str | None = None) -> dict[str, Any]:
        clauses, params = [], []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if benchmark:
            bench_id, _ = self.get(benchmark)
            clauses.append("benchmark_id = ?")
            params.append(bench_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.app.db.query(f"SELECT arm, success, metrics FROM benchmark_results {where}", params)
        arms: dict[str, dict[str, Any]] = {}
        for r in rows:
            entry = arms.setdefault(r["arm"], {"success": [], "checkpoint_rate": [], "takeovers": []})
            m = loads(r["metrics"], {})
            entry["success"].append(float(r["success"]))
            if m.get("checkpoints_total"):
                entry["checkpoint_rate"].append(m["checkpoints_passed"] / m["checkpoints_total"])
            entry["takeovers"].append(float(m.get("takeovers", 0)))
        out: dict[str, Any] = {"arms": {}}
        for arm, v in arms.items():
            out["arms"][arm] = {
                "runs": len(v["success"]), "success_rate": round(statistics.fmean(v["success"]), 4),
                "success_ci95": bootstrap_ci(v["success"]),
                "checkpoint_pass_rate": round(statistics.fmean(v["checkpoint_rate"]), 4) if v["checkpoint_rate"] else None,
                "mean_takeovers": round(statistics.fmean(v["takeovers"]), 4),
            }
        if "baseline" in out["arms"]:
            base = out["arms"]["baseline"]["success_rate"]
            for arm, stats in out["arms"].items():
                stats["delta_vs_baseline"] = round(stats["success_rate"] - base, 4)
        return out
