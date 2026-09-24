"""Hybrid retrieval (sections 29-33).

Skills are ranked by a transparent combination of:

* embedding similarity between the task and the skill description,
* lexical overlap with the skill's triggers and name,
* metadata (object class, categories, requested phases),
* evidence quality (status, confidence, success rate, provenance),
* workflow-preference boosts,

then expanded along the skill graph (``follows``/``requires``/``co_used``). Relevant failure
records, semantic statements and similar episodes are retrieved alongside. Every item carries
its score components and reason codes; every retrieval is persisted for later evaluation.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from lucius.events.bus import EventBus, EventType
from lucius.graph import Graph
from lucius.ids import new_id
from lucius.memory.episodic import EpisodicMemory
from lucius.memory.failure import FailureMemory
from lucius.memory.preferences import WorkflowPreferences
from lucius.memory.semantic import SemanticMemory
from lucius.retrieval.index import VectorIndex
from lucius.skills.library import SkillLibrary
from lucius.skills.schema import Skill, SkillStatus
from lucius.storage.db import Database, dumps
from lucius.timeutil import now

Strategy = Literal["hybrid", "vector", "lexical", "none", "episodes_only"]

STATUS_WEIGHT = {
    SkillStatus.HIGH_CONFIDENCE: 1.0, SkillStatus.VALIDATED: 0.85, SkillStatus.CANDIDATE_SKILL: 0.6,
    SkillStatus.CANDIDATE_PATTERN: 0.4,
}
_WORD = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "the", "of", "and", "to", "with", "for", "in", "on", "make", "model", "create", "another", "me",
        "using", "use", "what", "i", "you", "taught", "it", "one", "please", "do", "from", "this", "that", "like"}


def tokens(text: str) -> set[str]:
    words = {w for w in _WORD.findall(text.lower()) if w not in STOP}
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3}


class RetrievalQuery(BaseModel):
    text: str
    task_class: str | None = None
    categories: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    strategy: Strategy = "hybrid"
    top_k: int = 8
    min_score: float = 0.15
    include_seeded: bool = True


class RetrievedItem(BaseModel):
    kind: str                    # skill, failure, semantic, episode
    id: str
    title: str
    score: float
    components: dict[str, float] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    confidence: float | None = None
    status: str | None = None
    source_class: str | None = None
    provenance: str | None = None


class RetrievalResult(BaseModel):
    id: str
    query: RetrievalQuery
    skills: list[RetrievedItem] = Field(default_factory=list)
    failures: list[RetrievedItem] = Field(default_factory=list)
    semantic: list[RetrievedItem] = Field(default_factory=list)
    episodes: list[RetrievedItem] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)


class HybridRetriever:
    def __init__(self, db: Database, index: VectorIndex, library: SkillLibrary, failures: FailureMemory,
                 semantic: SemanticMemory, episodes: EpisodicMemory, graph: Graph, preferences: WorkflowPreferences,
                 bus: EventBus | None = None) -> None:
        self.db = db
        self.index = index
        self.library = library
        self.failures = failures
        self.semantic = semantic
        self.episodes = episodes
        self.graph = graph
        self.preferences = preferences
        self.bus = bus

    # -- index maintenance ---------------------------------------------------------------------------
    def refresh(self) -> dict[str, dict[str, int]]:
        return {
            "skill": self.index.sync("skill", [(s.id, s.definition.text()) for s in self.library.list()]),
            "failure": self.index.sync("failure", [(f.id, f.text()) for f in self.failures.list(limit=5000)]),
            "semantic": self.index.sync("semantic", [(s.id, s.statement) for s in self.semantic.list()]),
            "episode": self.index.sync("episode", [(e.id, e.text()) for e in self.episodes.list(limit=5000)]),
        }

    # -- retrieval ------------------------------------------------------------------------------------
    def retrieve(self, query: RetrievalQuery, *, run_id: str | None = None) -> RetrievalResult:
        result = RetrievalResult(id=new_id("retrieval"), query=query)
        if query.strategy == "none":
            return self._persist(result, run_id)
        self.refresh()
        qvec = self.index.embed_query(query.text)
        qtok = tokens(query.text)
        prefs = self.preferences.active() if query.strategy == "hybrid" else {}
        result.preferences = prefs
        if query.strategy != "episodes_only":
            result.skills = self._skills(query, qvec, qtok, prefs)
        if query.strategy in ("hybrid", "episodes_only"):
            result.episodes = self._episodes(query, qvec)
        if query.strategy == "hybrid":
            result.failures = self._failures(query, qvec, result.skills)
            result.semantic = self._semantic(query, qvec, qtok)
        return self._persist(result, run_id)

    def _skills(self, query: RetrievalQuery, qvec: Any, qtok: set[str], prefs: dict[str, Any]) -> list[RetrievedItem]:
        vector = self.index.scores("skill", qvec) if query.strategy in ("hybrid", "vector") else {}
        scored: dict[str, RetrievedItem] = {}
        skills = {s.id: s for s in self.library.list()}
        for skill in skills.values():
            if not query.include_seeded and skill.source_class == "system_seeded":
                continue
            item = self._score_skill(skill, query, vector.get(skill.id, 0.0), qtok, prefs)
            if item.score >= query.min_score:
                scored[skill.id] = item
        if query.strategy == "hybrid":
            top = sorted(scored.values(), key=lambda i: i.score, reverse=True)[: query.top_k]
            for item in top:
                for edge in self.graph.outgoing("skill", item.id, {"follows", "requires", "co_used"}):
                    if edge.dst_kind != "skill" or edge.dst_id not in skills:
                        continue
                    neighbour = skills[edge.dst_id]
                    derived = round(item.score * (0.75 if edge.rel == "requires" else 0.6), 4)
                    existing = scored.get(neighbour.id)
                    if existing is None:
                        new = self._score_skill(neighbour, query, vector.get(neighbour.id, 0.0), qtok, prefs)
                        new.score = max(new.score, derived)
                        new.reason_codes.append(f"graph_{edge.rel}:{item.id}")
                        new.components["graph"] = derived
                        scored[neighbour.id] = new
                    elif derived > existing.score * 0.9:
                        existing.score = round(max(existing.score, derived), 4)
                        existing.reason_codes.append(f"graph_{edge.rel}:{item.id}")
                        existing.components["graph"] = derived
        return sorted(scored.values(), key=lambda i: i.score, reverse=True)[: query.top_k]

    def _score_skill(self, skill: Skill, query: RetrievalQuery, vec: float, qtok: set[str],
                     prefs: dict[str, Any]) -> RetrievedItem:
        d = skill.definition
        reasons: list[str] = []
        comp: dict[str, float] = {}
        if query.strategy in ("hybrid", "vector"):
            comp["vector"] = round(max(0.0, vec), 4)
            if vec > 0.25:
                reasons.append(f"semantic_match:{vec:.2f}")
        if query.strategy in ("hybrid", "lexical"):
            stok = tokens(" ".join([d.name, *d.triggers, d.object_role or "", *d.categories]))
            overlap = len(qtok & stok) / max(1, len(qtok))
            comp["lexical"] = round(overlap, 4)
            if overlap:
                reasons.append("lexical_match:" + ",".join(sorted(qtok & stok))[:60])
        meta = 0.0
        if query.strategy == "hybrid":
            if query.task_class and d.object_class:
                if query.task_class == d.object_class:
                    meta += 0.3
                    reasons.append(f"object_class_match:{d.object_class}")
                else:
                    meta -= 0.4
                    reasons.append(f"object_class_mismatch:{d.object_class}")
            shared = set(query.categories) & set(d.categories)
            if shared:
                meta += min(0.2, 0.1 * len(shared))
                reasons.append("category_match:" + ",".join(sorted(shared)))
            if query.phases and {p.name for p in d.phases} & set(query.phases):
                meta += 0.1
                reasons.append("phase_match")
        comp["metadata"] = round(meta, 4)
        base = 0.55 * comp.get("vector", 0.0) + 0.35 * comp.get("lexical", 0.0) + meta
        if query.strategy == "vector":
            base = comp["vector"]
        quality = 0.5
        if query.strategy == "hybrid":
            quality = STATUS_WEIGHT.get(skill.status, 0.3) * (0.5 + 0.5 * skill.confidence)
            reasons.append(f"status:{skill.status.value}")
            rate = skill.success_rate
            if rate is not None and skill.usage_count >= 2:
                base += 0.1 * (rate - 0.5)
                reasons.append(f"success_rate:{rate:.2f}({skill.usage_count})")
            if prefs.get("symmetry_early") and any(a.action_type == "add_modifier" and a.args.get("type") == "MIRROR"
                                                   for p in d.phases if p.name == "setup" for a in p.actions):
                base += 0.03
                reasons.append("preference:symmetry_early")
            for variant in d.variants:
                if variant.preference_key and prefs.get(variant.preference_key):
                    base += 0.03
                    reasons.append(f"preference:{variant.preference_key}")
        comp["quality"] = round(quality, 4)
        score = base * (0.5 + 0.5 * quality)
        sources = sorted(set(skill.origin_sources))
        return RetrievedItem(kind="skill", id=skill.id, title=d.name, score=round(score, 4), components=comp,
                             reason_codes=reasons, confidence=skill.confidence, status=skill.status.value,
                             source_class=skill.source_class,
                             provenance=f"sources: {', '.join(sources)}; "
                                        f"{skill.confidence_breakdown.get('demonstration_count', 0)} demonstration(s),"
                                        f" {skill.success_count}/{skill.usage_count} successful uses")

    def _failures(self, query: RetrievalQuery, qvec: Any, skills: list[RetrievedItem]) -> list[RetrievedItem]:
        vector = self.index.scores("failure", qvec)
        skill_ids = {s.id for s in skills}
        action_types = {a.action_type for s in skills if self.library.exists(s.id)
                        for p in self.library.get(s.id).definition.phases for a in p.actions}
        out = []
        for rec in self.failures.list(task_class=query.task_class, limit=2000):
            if rec.rule_status == "rejected":
                continue
            reasons = []
            linked = rec.skill_id in skill_ids
            if linked:
                reasons.append(f"linked_to_skill:{rec.skill_id}")
            if rec.trigger_action in action_types:
                reasons.append(f"trigger_in_plan:{rec.trigger_action}")
            if query.task_class and rec.task_class == query.task_class:
                reasons.append(f"task_class_match:{rec.task_class}")
            sim = vector.get(rec.id, 0.0)
            if sim > 0.25:
                reasons.append(f"semantic_match:{sim:.2f}")
            if not reasons:
                continue
            relevance = (0.4 if linked else 0.0) + (0.3 if rec.trigger_action in action_types else 0.0) \
                + (0.2 if query.task_class and rec.task_class == query.task_class else 0.0) + 0.3 * max(0.0, sim)
            score = relevance * rec.confidence * rec.retrieval_priority
            reasons += [f"occurrences:{rec.occurrence_count}", f"rule:{rec.rule_status}"]
            out.append(RetrievedItem(kind="failure", id=rec.id, title=rec.future_rule or rec.observed_problem,
                                     score=round(score, 4), components={"relevance": round(relevance, 4),
                                                                        "priority": rec.retrieval_priority},
                                     reason_codes=reasons, confidence=rec.confidence, status=rec.rule_status,
                                     source_class=rec.source_class,
                                     provenance=f"{rec.occurrence_count} occurrence(s); "
                                                f"{rec.correction_successes}/{rec.correction_attempts} corrections worked"))
        return sorted(out, key=lambda i: i.score, reverse=True)[:10]

    def _semantic(self, query: RetrievalQuery, qvec: Any, qtok: set[str]) -> list[RetrievedItem]:
        vector = self.index.scores("semantic", qvec)
        out = []
        for stmt in self.semantic.list():
            in_scope = not query.task_class or stmt.scope.get("task_class") in (None, query.task_class)
            sim = vector.get(stmt.id, 0.0)
            if not in_scope and sim < 0.4:
                continue
            score = (0.5 if in_scope else 0.0) + 0.5 * max(0.0, sim)
            out.append(RetrievedItem(kind="semantic", id=stmt.id, title=stmt.statement, score=round(score * stmt.confidence, 4),
                                     reason_codes=[f"scope:{stmt.scope.get('task_class')}", f"support:{stmt.support_count}"],
                                     confidence=stmt.confidence, status=stmt.status, source_class=stmt.source_class))
        return sorted(out, key=lambda i: i.score, reverse=True)[:8]

    def _episodes(self, query: RetrievalQuery, qvec: Any) -> list[RetrievedItem]:
        vector = self.index.scores("episode", qvec)
        out = []
        for ep in self.episodes.list(limit=2000):
            sim = vector.get(ep.id, 0.0)
            bonus = 0.2 if query.task_class and ep.task_class == query.task_class else 0.0
            score = 0.8 * max(0.0, sim) + bonus
            if score < 0.15:
                continue
            out.append(RetrievedItem(kind="episode", id=ep.id, title=ep.task_text or ep.session_id, score=round(score, 4),
                                     reason_codes=[f"semantic_match:{sim:.2f}"] + (["task_class_match"] if bonus else []),
                                     confidence=ep.confidence, status=ep.outcome, source_class=ep.source_class,
                                     provenance=f"session {ep.session_id}"))
        return sorted(out, key=lambda i: i.score, reverse=True)[:5]

    def _persist(self, result: RetrievalResult, run_id: str | None) -> RetrievalResult:
        payload = {k: [i.model_dump() for i in getattr(result, k)] for k in ("skills", "failures", "semantic", "episodes")}
        self.db.insert("retrieval_records", {
            "id": result.id, "run_id": run_id, "query_text": result.query.text,
            "query_meta": dumps(result.query.model_dump()), "strategy": result.query.strategy,
            "results": dumps(payload), "feedback": dumps({}), "created_at": result.created_at})
        if self.bus is not None:
            self.bus.publish(EventType.RETRIEVAL_COMPLETED, result.id, strategy=result.query.strategy,
                             skills=[i.id for i in result.skills[:5]], failures=[i.id for i in result.failures[:5]])
        return result

    def record_feedback(self, retrieval_id: str, feedback: dict[str, Any]) -> None:
        """Which retrieved items were actually used/useful (for retrieval precision metrics)."""
        self.db.execute("UPDATE retrieval_records SET feedback = ? WHERE id = ?", (dumps(feedback), retrieval_id))
