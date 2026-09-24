"""Semantic memory: generalised knowledge promoted from repeated experience (section 25).

Statements are mined from *patterns across episodes*, never copied from a single log. Each
statement carries its support and contradiction counts; it stays a ``candidate`` until the
evidence is strong enough (or a human confirms it).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from lucius.confidence import Evidence, score
from lucius.errors import NotFoundError
from lucius.events.bus import EventBus, EventType
from lucius.ids import new_id
from lucius.memory.episodic import Episode
from lucius.memory.failure import FailureRecord
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now

MIN_SUPPORT = 2
VALIDATION_SUPPORT = 3
ORTHO = {"front", "back", "left", "right", "top", "bottom"}

ORDER_PATTERNS = [
    ("primary_blockout", "detail_pass", "establish primary forms before adding detail"),
    ("primary_blockout", "inspection", "inspect the model after the primary blockout"),
    ("scene_setup", "primary_blockout", "set up the scene before blocking out"),
    ("inspection", "detail_pass", "inspect before starting the detail pass"),
]


class SemanticStatement(BaseModel):
    id: str
    statement: str
    pattern_key: str
    category: str
    scope: dict[str, Any] = Field(default_factory=dict)
    source_class: str
    status: str
    support_count: int
    contradiction_count: int
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    created_at: float
    updated_at: float


def _decode(row: Any) -> SemanticStatement:
    return SemanticStatement(
        id=row["id"], statement=row["statement"], pattern_key=row["pattern_key"], category=row["category"],
        scope=loads(row["scope"], {}), source_class=row["source_class"], status=row["status"],
        support_count=row["support_count"], contradiction_count=row["contradiction_count"],
        confidence=row["confidence"], evidence=loads(row["evidence"], []), created_at=row["created_at"],
        updated_at=row["updated_at"])


def _first(phases: list[dict[str, Any]], label: str) -> int | None:
    return next((i for i, p in enumerate(phases) if p["label"] == label), None)


class SemanticMemory:
    def __init__(self, db: Database, bus: EventBus | None = None) -> None:
        self.db = db
        self.bus = bus

    def mine(self, episodes: list[Episode], failures: list[FailureRecord]) -> list[SemanticStatement]:
        """Recompute pattern statistics over all episodes and upsert supported statements."""
        found: dict[str, dict[str, Any]] = {}

        def observe(key: str, statement: str, category: str, scope: dict[str, Any], *, support: str | None = None,
                    contradiction: str | None = None, source: str) -> None:
            entry = found.setdefault(key, {"statement": statement, "category": category, "scope": scope,
                                           "support": [], "contradict": [], "sources": set()})
            if support:
                entry["support"].append(support)
                entry["sources"].add(source)
            if contradiction:
                entry["contradict"].append(contradiction)

        by_class: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            by_class[ep.task_class or "general"].append(ep)
        for task_class, eps in by_class.items():
            scope = {"task_class": task_class}
            noun = task_class.replace("_", " ")
            for ep in eps:
                phases = [p for p in ep.structured.get("phases", []) if p.get("outcome") != "failure"]
                for a, b, text in ORDER_PATTERNS:
                    ia, ib = _first(phases, a), _first(phases, b)
                    if ia is None or ib is None:
                        continue
                    key = f"order:{task_class}:{a}<{b}"
                    statement = f"In {noun} tasks, {text}."
                    if ia < ib:
                        observe(key, statement, "workflow_order", scope, support=ep.id, source=ep.source_class)
                    else:
                        observe(key, statement, "workflow_order", scope, contradiction=ep.id, source=ep.source_class)
                views = {v for p in phases if p["label"] in ("inspection", "verification") for v in p.get("views", [])}
                if len(views & ORTHO) >= 2:
                    observe(f"inspect_views:{task_class}",
                            f"Check {noun} silhouettes from several orthographic views (e.g. front and side).",
                            "inspection", scope, support=ep.id, source=ep.source_class)
                setup_mirror = any(p["label"] == "scene_setup" and "add_modifier" in p.get("actions", {})
                                   for p in phases)
                if setup_mirror:
                    observe(f"symmetry_setup:{task_class}", f"Set up mirror symmetry before shaping {noun} objects.",
                            "workflow_order", scope, support=ep.id, source=ep.source_class)
        for rec in failures:
            if not rec.future_rule or rec.rule_status == "rejected":
                continue
            if rec.rule_status in ("promoted", "confirmed") or rec.occurrence_count >= MIN_SUPPORT:
                key = f"rule:{rec.signature}"
                entry = found.setdefault(key, {"statement": rec.future_rule, "category": "failure_rule",
                                               "scope": {"task_class": rec.task_class, "phase": rec.phase},
                                               "support": [], "contradict": [], "sources": set()})
                entry["support"] += [f"{rec.id}#{i}" for i in range(max(rec.occurrence_count,
                                                                        rec.correction_successes))]
                entry["sources"].add(rec.source_class)
        out = []
        for key, entry in found.items():
            if len(entry["support"]) < MIN_SUPPORT:
                continue
            out.append(self._upsert(key, entry))
        return out

    def _upsert(self, key: str, entry: dict[str, Any]) -> SemanticStatement:
        support, contradict = len(entry["support"]), len(entry["contradict"])
        confidence, _ = score(Evidence(demonstrations=support, demonstration_count=support, contradictions=contradict))
        row = self.db.query_one("SELECT * FROM semantic_memories WHERE pattern_key = ?", (key,))
        t = now()
        source = sorted(entry["sources"])[0] if len(entry["sources"]) == 1 else "derived"
        if row is None:
            status = "validated" if support >= VALIDATION_SUPPORT and contradict / support < 0.2 else "candidate"
            stmt = SemanticStatement(id=new_id("semantic"), statement=entry["statement"], pattern_key=key,
                                     category=entry["category"], scope=entry["scope"], source_class=source,
                                     status=status, support_count=support, contradiction_count=contradict,
                                     confidence=confidence, evidence=sorted(set(entry["support"]))[:100],
                                     created_at=t, updated_at=t)
            self.db.insert("semantic_memories", {
                "id": stmt.id, "statement": stmt.statement, "pattern_key": key, "category": stmt.category,
                "scope": dumps(stmt.scope), "source_class": source, "status": status, "support_count": support,
                "contradiction_count": contradict, "confidence": confidence, "evidence": dumps(stmt.evidence),
                "created_at": t, "updated_at": t})
            if self.bus is not None:
                self.bus.publish(EventType.MEMORY_CREATED, stmt.id, kind="semantic", statement=stmt.statement,
                                 support=support)
            return stmt
        current = _decode(row)
        status = current.status
        if status == "candidate" and support >= VALIDATION_SUPPORT and contradict / support < 0.2:
            status = "validated"
        self.db.update("semantic_memories", "id", current.id, {
            "support_count": support, "contradiction_count": contradict, "confidence": confidence,
            "evidence": dumps(sorted(set(entry["support"]))[:100]), "status": status, "updated_at": t})
        return self.get(current.id)

    def review(self, statement_id: str, *, accept: bool) -> SemanticStatement:
        stmt = self.get(statement_id)
        self.db.update("semantic_memories", "id", statement_id,
                       {"status": "validated" if accept else "rejected", "updated_at": now()})
        return self.get(stmt.id)

    def get(self, statement_id: str) -> SemanticStatement:
        row = self.db.query_one("SELECT * FROM semantic_memories WHERE id = ?", (statement_id,))
        if row is None:
            raise NotFoundError(f"semantic memory {statement_id} not found")
        return _decode(row)

    def list(self, *, include_rejected: bool = False) -> list[SemanticStatement]:
        rows = self.db.query("SELECT * FROM semantic_memories ORDER BY confidence DESC")
        return [_decode(r) for r in rows if include_rejected or r["status"] != "rejected"]
