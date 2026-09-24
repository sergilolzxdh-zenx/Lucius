"""When to train (section 48): evidence-based signals, never an automatic launch."""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel

from lucius.storage.db import Database, loads


class TrainingSignal(BaseModel):
    name: str
    value: float | int | None
    threshold: float | int
    triggered: bool
    evidence: str


class TrainingAdvisor:
    def __init__(self, db: Database) -> None:
        self.db = db

    def assess(self) -> dict[str, Any]:
        signals = [self._retrieval_failures(), self._instability(), self._clean_data(), self._correction_data(),
                   self._taxonomy_stability(), self._plateau()]
        triggered = [s for s in signals if s.triggered]
        data_ready = any(s.name in ("clean_dataset_size", "correction_dataset_size") and s.triggered for s in signals)
        need = any(s.name in ("retrieval_failure_rate", "micro_decision_instability", "benchmark_plateau")
                   and s.triggered for s in signals)
        if data_ready and need:
            recommendation = "consider_targeted_training"
        elif need:
            recommendation = "collect_more_data"
        else:
            recommendation = "not_recommended"
        return {"recommendation": recommendation, "signals": [s.model_dump() for s in signals],
                "triggered": [s.name for s in triggered],
                "note": "Advisory only. Training is never launched automatically."}

    def _retrieval_failures(self) -> TrainingSignal:
        runs = self.db.query("SELECT plan FROM runs WHERE plan IS NOT NULL")
        missing = sum(1 for r in runs if "needs_demonstration" in (loads(r["plan"], {}) or {}).get("reason_codes", []))
        rate = missing / len(runs) if runs else None
        return TrainingSignal(name="retrieval_failure_rate", value=None if rate is None else round(rate, 3),
                              threshold=0.3, triggered=bool(rate is not None and len(runs) >= 10 and rate > 0.3),
                              evidence=f"{missing}/{len(runs)} runs found no applicable skill")

    def _instability(self) -> TrainingSignal:
        rows = self.db.query("SELECT skill_id, outcome FROM skill_examples WHERE role IN ('execution','validation')")
        per_skill: dict[str, list[int]] = {}
        for r in rows:
            per_skill.setdefault(r["skill_id"], []).append(int(r["outcome"] == "success"))
        variances = [statistics.pvariance(v) for v in per_skill.values() if len(v) >= 5]
        value = round(max(variances), 3) if variances else None
        return TrainingSignal(name="micro_decision_instability", value=value, threshold=0.2,
                              triggered=bool(value is not None and value > 0.2),
                              evidence=f"max success variance across {len(variances)} frequently used skills")

    def _clean_data(self) -> TrainingSignal:
        count = self.db.scalar("SELECT COUNT(*) FROM dataset_samples WHERE quality_score >= 0.8") or 0
        return TrainingSignal(name="clean_dataset_size", value=count, threshold=500, triggered=count >= 500,
                              evidence=f"{count} high-quality samples in built datasets")

    def _correction_data(self) -> TrainingSignal:
        count = self.db.scalar("SELECT COUNT(*) FROM corrections WHERE correction_steps != '[]'") or 0
        return TrainingSignal(name="correction_dataset_size", value=count, threshold=100, triggered=count >= 100,
                              evidence=f"{count} human corrections with recorded steps")

    def _taxonomy_stability(self) -> TrainingSignal:
        total = self.db.scalar("SELECT COUNT(*) FROM taxonomy_terms") or 0
        recent = self.db.scalar("SELECT COUNT(*) FROM taxonomy_terms WHERE source != 'system_seeded' AND created_at >"
                                " (SELECT COALESCE(MAX(created_at), 0) - 30*86400 FROM taxonomy_terms)") or 0
        rate = recent / total if total else 0.0
        return TrainingSignal(name="taxonomy_stability", value=round(1 - rate, 3), threshold=0.9, triggered=rate < 0.1,
                              evidence=f"{recent} new taxonomy terms in the last 30 days of {total}")

    def _plateau(self) -> TrainingSignal:
        rows = self.db.query("SELECT success FROM benchmark_results WHERE arm = 'memory_enhanced' ORDER BY created_at")
        values = [r["success"] for r in rows]
        if len(values) < 20:
            return TrainingSignal(name="benchmark_plateau", value=None, threshold=0.02, triggered=False,
                                  evidence=f"only {len(values)} benchmark results")
        half = len(values) // 2
        delta = sum(values[half:]) / (len(values) - half) - sum(values[:half]) / half
        return TrainingSignal(name="benchmark_plateau", value=round(delta, 3), threshold=0.02, triggered=abs(delta) < 0.02,
                              evidence="change in memory-enhanced success rate between first and second half")
