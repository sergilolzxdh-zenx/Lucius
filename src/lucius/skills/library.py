"""Procedural memory: the versioned skill library.

* Every change creates a new immutable version (lineage kept, rollback = new version copying
  an old definition), so historical behaviour is never silently overwritten.
* Examples link a skill to the exact sessions, segments, frames, media and runs supporting it.
* Status moves along ``candidate_pattern -> candidate_skill -> validated -> high_confidence``
  only on counted evidence (repetition, objective validation, human confirmation,
  cross-instance success), and moves back down when success degrades.
"""

from __future__ import annotations

import re
from typing import Any

from lucius.confidence import Evidence, explain, score
from lucius.errors import ConflictError, NotFoundError, ValidationError
from lucius.events.bus import EventBus, EventType
from lucius.graph import Graph
from lucius.ids import new_id
from lucius.skills.schema import Skill, SkillDefinition, SkillExample, SkillStatus
from lucius.storage.db import Database, dumps, loads
from lucius.timeutil import now

HIGH_CONFIDENCE_MIN_SUCCESSES = 3
HIGH_CONFIDENCE_MIN_INSTANCES = 2
HIGH_CONFIDENCE_MIN_RATE = 0.8
DEMOTION_MIN_USES = 4
DEMOTION_MAX_RATE = 0.5


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80] or "skill"


def _skill(row: Any, definition: SkillDefinition) -> Skill:
    return Skill(
        id=row["id"], name=row["name"], status=SkillStatus(row["status"]), current_version=row["current_version"],
        source_class=row["source_class"], origin_sources=loads(row["origin_sources"], []),
        categories=loads(row["categories"], []), object_class=row["object_class"], confidence=row["confidence"],
        confidence_breakdown=loads(row["confidence_breakdown"], {}), usage_count=row["usage_count"],
        success_count=row["success_count"], failure_count=row["failure_count"],
        human_confirmations=row["human_confirmations"], human_rejections=row["human_rejections"],
        last_used=row["last_used"], parent_skill_id=row["parent_skill_id"], created_at=row["created_at"],
        updated_at=row["updated_at"], definition=definition,
    )


def _example(row: Any) -> SkillExample:
    return SkillExample(
        id=row["id"], skill_id=row["skill_id"], skill_version=row["skill_version"], role=row["role"],
        source_class=row["source_class"], session_id=row["session_id"], segment_ids=loads(row["segment_ids"], []),
        media_asset_id=row["media_asset_id"], run_id=row["run_id"], t_start=row["t_start"], t_end=row["t_end"],
        frame_ids=loads(row["frame_ids"], []), outcome=row["outcome"], evidence_weight=row["evidence_weight"],
        instance_signature=row["instance_signature"], summary=loads(row["summary"], {}), created_at=row["created_at"],
    )


class SkillLibrary:
    def __init__(self, db: Database, graph: Graph, bus: EventBus | None = None) -> None:
        self.db = db
        self.graph = graph
        self.bus = bus

    # -- reads -------------------------------------------------------------------------------------
    def exists(self, skill_id: str) -> bool:
        return self.db.scalar("SELECT 1 FROM skills WHERE id = ?", (skill_id,)) is not None

    def get(self, skill_id: str, version: int | None = None) -> Skill:
        row = self.db.query_one("SELECT * FROM skills WHERE id = ?", (skill_id,))
        if row is None:
            raise NotFoundError(f"skill {skill_id} not found", skill_id=skill_id)
        return _skill(row, self.definition(skill_id, version or row["current_version"]))

    def definition(self, skill_id: str, version: int) -> SkillDefinition:
        row = self.db.query_one("SELECT definition FROM skill_versions WHERE skill_id = ? AND version = ?",
                                (skill_id, version))
        if row is None:
            raise NotFoundError(f"skill {skill_id} v{version} not found", skill_id=skill_id, version=version)
        return SkillDefinition.model_validate(loads(row["definition"]))

    def list(self, *, statuses: set[SkillStatus] | None = None, search: str | None = None,
             include_inactive: bool = False) -> list[Skill]:
        rows = self.db.query("SELECT * FROM skills ORDER BY confidence DESC, updated_at DESC")
        out = []
        for row in rows:
            status = SkillStatus(row["status"])
            if statuses is not None and status not in statuses:
                continue
            if not include_inactive and status in (SkillStatus.DISABLED, SkillStatus.MERGED):
                continue
            skill = _skill(row, self.definition(row["id"], row["current_version"]))
            if search and search.lower() not in skill.definition.text().lower():
                continue
            out.append(skill)
        return out

    def versions(self, skill_id: str) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT id, version, change_note, created_by, parent_version, created_at FROM skill_versions"
                             " WHERE skill_id = ? ORDER BY version", (skill_id,))
        return [dict(r) for r in rows]

    def examples(self, skill_id: str, role: str | None = None) -> list[SkillExample]:
        sql = "SELECT * FROM skill_examples WHERE skill_id = ?" + (" AND role = ?" if role else "") + " ORDER BY created_at"
        rows = self.db.query(sql, (skill_id, role) if role else (skill_id,))
        return [_example(r) for r in rows]

    def provenance(self, skill_id: str) -> dict[str, Any]:
        """Answer 'where did this skill come from?' structurally."""
        skill = self.get(skill_id)
        examples = self.examples(skill_id)
        by_source: dict[str, list[dict[str, Any]]] = {}
        for ex in examples:
            by_source.setdefault(ex.source_class, []).append({
                "example_id": ex.id, "role": ex.role, "session_id": ex.session_id, "segment_ids": ex.segment_ids,
                "media_asset_id": ex.media_asset_id, "run_id": ex.run_id, "t_start": ex.t_start, "t_end": ex.t_end,
                "frame_ids": ex.frame_ids[:6], "outcome": ex.outcome, "weight": ex.evidence_weight,
            })
        return {
            "skill_id": skill_id, "status": skill.status.value, "confidence": skill.confidence,
            "confidence_explanation": explain(skill.confidence_breakdown), "sources": by_source,
            "versions": self.versions(skill_id),
            "graph": self.graph.neighbourhood("skill", skill_id, depth=1),
        }

    # -- writes ------------------------------------------------------------------------------------
    def create(self, definition: SkillDefinition, *, created_by: str, change_note: str,
               status: SkillStatus = SkillStatus.CANDIDATE_PATTERN, parent_skill_id: str | None = None) -> Skill:
        if self.exists(definition.skill_id):
            raise ConflictError(f"skill {definition.skill_id} already exists", skill_id=definition.skill_id)
        t = now()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO skills (id, name, status, current_version, source_class, origin_sources, categories,"
                " object_class, confidence, confidence_breakdown, parent_skill_id, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (definition.skill_id, definition.name, status.value, 1, definition.source_class,
                 dumps([definition.source_class]), dumps(definition.categories), definition.object_class, 0.0,
                 dumps({}), parent_skill_id, t, t))
            conn.execute(
                "INSERT INTO skill_versions (id, skill_id, version, definition, change_note, created_by,"
                " parent_version, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (new_id("skill_version"), definition.skill_id, 1, dumps(definition.model_dump(mode="json")),
                 change_note, created_by, None, t))
        if parent_skill_id:
            self.graph.link(("skill", parent_skill_id), "parent_of", ("skill", definition.skill_id))
        if self.bus is not None:
            self.bus.publish(EventType.SKILL_CANDIDATE_CREATED, definition.skill_id, name=definition.name,
                             source=definition.source_class, status=status.value)
        return self.get(definition.skill_id)

    def new_version(self, skill_id: str, definition: SkillDefinition, *, change_note: str, created_by: str) -> Skill:
        current = self.get(skill_id)
        if definition.skill_id != skill_id:
            raise ValidationError("definition skill_id does not match")
        if definition.model_dump(mode="json") == current.definition.model_dump(mode="json"):
            return current
        version = current.current_version + 1
        t = now()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO skill_versions (id, skill_id, version, definition, change_note, created_by,"
                " parent_version, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (new_id("skill_version"), skill_id, version, dumps(definition.model_dump(mode="json")), change_note,
                 created_by, current.current_version, t))
            conn.execute("UPDATE skills SET current_version = ?, name = ?, categories = ?, object_class = ?,"
                         " updated_at = ? WHERE id = ?",
                         (version, definition.name, dumps(definition.categories), definition.object_class, t, skill_id))
        if self.bus is not None:
            self.bus.publish(EventType.SKILL_VERSIONED, skill_id, version=version, note=change_note, by=created_by)
        return self.get(skill_id)

    def rollback(self, skill_id: str, to_version: int, *, user_id: str = "local") -> Skill:
        old = self.definition(skill_id, to_version)
        skill = self.new_version(skill_id, old, change_note=f"rollback to v{to_version}", created_by="rollback")
        self._log_edit(skill_id, "rollback", {"from": skill.current_version - 1}, {"to_version": to_version}, user_id)
        return skill

    def human_edit(self, skill_id: str, changes: dict[str, Any], *, user_id: str = "local") -> Skill:
        """Apply a user edit (name, purpose, parameters, checkpoints...) as a new version."""
        current = self.get(skill_id)
        data = current.definition.model_dump(mode="json")
        protected = {"skill_id"}
        if protected & set(changes):
            raise ValidationError("skill_id cannot be edited")
        data.update(changes)
        definition = SkillDefinition.model_validate(data)
        skill = self.new_version(skill_id, definition, change_note="human edit: " + ", ".join(sorted(changes)),
                                 created_by="human")
        self._log_edit(skill_id, "edit", {k: current.definition.model_dump(mode="json").get(k) for k in changes},
                       changes, user_id)
        return skill

    def add_example(self, example: SkillExample) -> None:
        identified = example.session_id is not None or example.run_id is not None
        exists = identified and self.db.scalar(
            "SELECT 1 FROM skill_examples WHERE skill_id = ? AND role = ? AND COALESCE(session_id,'') = ?"
            " AND COALESCE(run_id,'') = ? AND segment_ids = ?",
            (example.skill_id, example.role, example.session_id or "", example.run_id or "", dumps(example.segment_ids)))
        if exists:
            return  # idempotent: re-processing a session does not duplicate evidence
        self.db.insert("skill_examples", {
            "id": example.id, "skill_id": example.skill_id, "skill_version": example.skill_version,
            "role": example.role, "source_class": example.source_class, "session_id": example.session_id,
            "segment_ids": dumps(example.segment_ids), "media_asset_id": example.media_asset_id,
            "run_id": example.run_id, "t_start": example.t_start, "t_end": example.t_end,
            "frame_ids": dumps(example.frame_ids), "outcome": example.outcome,
            "evidence_weight": example.evidence_weight, "instance_signature": example.instance_signature,
            "summary": dumps(example.summary), "created_at": example.created_at,
        })
        with self.db.transaction() as conn:
            row = conn.execute("SELECT origin_sources FROM skills WHERE id = ?", (example.skill_id,)).fetchone()
            sources = loads(row["origin_sources"], [])
            if example.source_class not in sources:
                sources.append(example.source_class)
                conn.execute("UPDATE skills SET origin_sources = ? WHERE id = ?", (dumps(sources), example.skill_id))
        if example.session_id:
            self.graph.link(("skill", example.skill_id), "extracted_from" if example.role == "demonstration"
                            else "validated_by", ("session", example.session_id), weight=example.evidence_weight)
        if example.media_asset_id:
            self.graph.link(("skill", example.skill_id), "extracted_from", ("media", example.media_asset_id),
                            weight=example.evidence_weight)
        if self.bus is not None:
            self.bus.publish(EventType.SKILL_EVIDENCE_ADDED, example.skill_id, role=example.role,
                             source=example.source_class, session_id=example.session_id, run_id=example.run_id)
        self.rescore(example.skill_id)

    def record_use(self, skill_id: str, *, success: bool, run_id: str | None, instance_signature: str | None,
                   objective: bool, environment: str, role: str = "execution", session_id: str | None = None,
                   source_class: str = "agent_success", detail: dict[str, Any] | None = None) -> Skill:
        """Outcome of executing a skill. Only objectively evaluated outcomes can validate it."""
        t = now()
        with self.db.transaction() as conn:
            conn.execute("UPDATE skills SET usage_count = usage_count + 1, success_count = success_count + ?,"
                         " failure_count = failure_count + ?, last_used = ?, updated_at = ? WHERE id = ?",
                         (int(success), int(not success), t, t, skill_id))
        skill = self.get(skill_id)
        self.add_example(SkillExample(
            id=new_id("example"), skill_id=skill_id, skill_version=skill.current_version, role=role,
            source_class=source_class if success else "agent_failure", session_id=session_id, run_id=run_id,
            outcome="success" if success else "failure", evidence_weight=0.9 if objective else 0.4,
            instance_signature=instance_signature,
            summary={"objective": objective, "environment": environment, **(detail or {})}, created_at=t,
        ))
        return self.get(skill_id)

    def review(self, skill_id: str, *, accept: bool, user_id: str = "local") -> Skill:
        column = "human_confirmations" if accept else "human_rejections"
        self.db.execute(f"UPDATE skills SET {column} = {column} + 1, updated_at = ? WHERE id = ?", (now(), skill_id))
        self._log_edit(skill_id, "confirm" if accept else "reject", None, None, user_id)
        return self.rescore(skill_id)

    def set_disabled(self, skill_id: str, disabled: bool, *, user_id: str = "local") -> Skill:
        skill = self.get(skill_id)
        if disabled:
            self._set_status(skill, SkillStatus.DISABLED, "disabled by user")
        else:
            self._set_status(skill, SkillStatus.CANDIDATE_PATTERN, "re-enabled by user")
            skill = self.rescore(skill_id)
        self._log_edit(skill_id, "disable" if disabled else "enable", None, None, user_id)
        return self.get(skill_id)

    def merge(self, src_id: str, dst_id: str, *, user_id: str = "local") -> Skill:
        """Fold ``src`` into ``dst``: evidence moves over, ``src`` is kept (status merged) for history."""
        if src_id == dst_id:
            raise ValidationError("cannot merge a skill into itself")
        src, dst = self.get(src_id), self.get(dst_id)
        with self.db.transaction() as conn:
            conn.execute("UPDATE skill_examples SET skill_id = ?, skill_version = ? WHERE skill_id = ?",
                         (dst_id, dst.current_version, src_id))
            conn.execute("UPDATE skills SET usage_count = usage_count + ?, success_count = success_count + ?,"
                         " failure_count = failure_count + ? WHERE id = ?",
                         (src.usage_count, src.success_count, src.failure_count, dst_id))
        self._set_status(src, SkillStatus.MERGED, f"merged into {dst_id}")
        self.graph.link(("skill", src_id), "merged_into", ("skill", dst_id))
        self._log_edit(dst_id, "merge", {"source": src_id}, {"into": dst_id}, user_id)
        return self.rescore(dst_id)

    def split(self, skill_id: str, phase_names: list[str], *, new_name: str, user_id: str = "local") -> Skill:
        """Create a child skill from a subset of phases (the parent is unchanged)."""
        parent = self.get(skill_id)
        phases = [p for p in parent.definition.phases if p.name in phase_names]
        if not phases:
            raise ValidationError("no matching phases to split out")
        checkpoint_ids = {c for p in phases for c in p.checkpoints}
        child = parent.definition.model_copy(deep=True)
        child.skill_id = slugify(new_name)
        if self.exists(child.skill_id):
            raise ConflictError(f"skill {child.skill_id} already exists")
        child.name = new_name
        child.phases = phases
        child.checkpoints = [c for c in child.checkpoints if c.id in checkpoint_ids]
        child.failure_conditions = [f for f in child.failure_conditions if f.phase in phase_names]
        child.variants = [v for v in child.variants if v.phase in phase_names]
        child.source_class = "derived"
        created = self.create(child, created_by="split", change_note=f"split from {skill_id}: {phase_names}",
                              status=SkillStatus.CANDIDATE_PATTERN, parent_skill_id=skill_id)
        for ex in self.examples(skill_id, role="demonstration"):
            copy = ex.model_copy(update={"id": new_id("example"), "skill_id": created.id, "skill_version": 1})
            self.add_example(copy)
        self._log_edit(skill_id, "split", None, {"child": created.id, "phases": phase_names}, user_id)
        return self.get(created.id)

    # -- scoring & promotion ----------------------------------------------------------------------
    def evidence(self, skill_id: str) -> Evidence:
        row = self.db.query_one("SELECT * FROM skills WHERE id = ?", (skill_id,))
        examples = self.examples(skill_id)
        demos = [e for e in examples if e.role == "demonstration"]
        uses = [e for e in examples if e.role in ("execution", "validation")]
        objective_ok = [e for e in uses if e.outcome == "success" and e.summary.get("objective")
                        and e.summary.get("environment") != "simulation"]
        instances = {e.instance_signature for e in demos + objective_ok if e.instance_signature}
        return Evidence(
            demonstrations=sum(e.evidence_weight for e in demos), demonstration_count=len(demos),
            distinct_sources=len({e.source_class for e in demos}), distinct_instances=len(instances),
            successes=row["success_count"], failures=row["failure_count"],
            validations=sum(1 for e in objective_ok if e.role == "validation"),
            human_confirmations=row["human_confirmations"], human_rejections=row["human_rejections"],
        )

    def rescore(self, skill_id: str) -> Skill:
        ev = self.evidence(skill_id)
        confidence, breakdown = score(ev)
        self.db.execute("UPDATE skills SET confidence = ?, confidence_breakdown = ?, updated_at = ? WHERE id = ?",
                        (confidence, dumps(breakdown), now(), skill_id))
        skill = self.get(skill_id)
        target = self._target_status(skill, ev)
        if target is not None and target != skill.status:
            self._set_status(skill, target, _status_reason(target, ev, skill))
        return self.get(skill_id)

    def _target_status(self, skill: Skill, ev: Evidence) -> SkillStatus | None:
        if skill.status in (SkillStatus.DISABLED, SkillStatus.MERGED):
            return None
        examples = self.examples(skill.id)
        objective_success = [e for e in examples if e.role in ("execution", "validation") and e.outcome == "success"
                             and e.summary.get("objective") and e.summary.get("environment") != "simulation"]
        success_instances = {e.instance_signature for e in objective_success if e.instance_signature}
        rate = skill.success_rate
        if (len(objective_success) >= HIGH_CONFIDENCE_MIN_SUCCESSES
                and len(success_instances) >= HIGH_CONFIDENCE_MIN_INSTANCES
                and rate is not None and rate >= HIGH_CONFIDENCE_MIN_RATE
                and skill.human_confirmations >= skill.human_rejections):
            target = SkillStatus.HIGH_CONFIDENCE
        elif objective_success:
            target = SkillStatus.VALIDATED
        elif ev.demonstration_count >= 2 or skill.human_confirmations >= 1 or skill.source_class == "system_seeded":
            # (seeded capabilities are known operations, not one-off observed patterns)
            target = SkillStatus.CANDIDATE_SKILL
        else:
            target = SkillStatus.CANDIDATE_PATTERN
        if skill.usage_count >= DEMOTION_MIN_USES and rate is not None and rate < DEMOTION_MAX_RATE:
            target = min(target, SkillStatus.CANDIDATE_SKILL, key=lambda s: s.rank)
        if skill.human_rejections > skill.human_confirmations + 1:
            target = SkillStatus.CANDIDATE_PATTERN
        return target

    def _set_status(self, skill: Skill, status: SkillStatus, reason: str) -> None:
        self.db.execute("UPDATE skills SET status = ?, updated_at = ? WHERE id = ?", (status.value, now(), skill.id))
        if self.bus is None:
            return
        promoted = status.rank > skill.status.rank
        self.bus.publish(EventType.SKILL_PROMOTED if promoted else EventType.SKILL_DEMOTED, skill.id,
                         from_status=skill.status.value, to_status=status.value, reason=reason)

    def _log_edit(self, skill_id: str, op: str, before: Any, after: Any, user_id: str) -> None:
        self.db.insert("human_edits", {"id": new_id("edit"), "subject_kind": "skill", "subject_id": skill_id, "op": op,
                                       "before": dumps(before), "after": dumps(after), "user_id": user_id,
                                       "created_at": now()})


def _status_reason(status: SkillStatus, ev: Evidence, skill: Skill) -> str:
    if status == SkillStatus.HIGH_CONFIDENCE:
        return f"{skill.success_count}/{skill.usage_count} successful uses across {ev.distinct_instances} instances"
    if status == SkillStatus.VALIDATED:
        return "objectively verified reproduction"
    if status == SkillStatus.CANDIDATE_SKILL:
        return f"{ev.demonstration_count} demonstrations, {skill.human_confirmations} confirmations"
    return "insufficient evidence"
