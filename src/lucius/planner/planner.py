"""Planner: task -> retrieved skills -> parameters -> ordered, guarded, recoverable plan.

The planner composes skills rather than replaying a past session. Ordering comes from the skill
graph (``follows`` edges learned from demonstrations) and phase order; parameters come from the
task, references and qualifiers, falling back to the median of what was demonstrated.
Failure memory contributes *guards*: a known premature action (e.g. bevel before the silhouette
is validated) is only executed after its guard checkpoint passes.
"""

from __future__ import annotations

import re
from typing import Any

from lucius.events.bus import EventBus, EventType
from lucius.graph import Graph
from lucius.ids import new_id
from lucius.memory.failure import FailureMemory, FailureRecord
from lucius.planner.compile import CompileContext, Uncompilable, compile_template
from lucius.planner.model import Guard, Plan, PlanStep, RecoveryPlan, TaskSpec
from lucius.retrieval.retriever import RetrievalResult
from lucius.skills.library import SkillLibrary
from lucius.skills.schema import ParamSpec, Skill, SkillStatus
from lucius.timeutil import now

PHASE_ORDER = {"setup": 0, "reference": 1, "primary_form": 2, "secondary_form": 3, "detail": 4, "inspection": 5,
               "verification": 6}
MIN_SKILL_SCORE = 0.2
GENERIC_DIMENSIONS = ("length", "width", "height", "thickness")


class Planner:
    def __init__(self, library: SkillLibrary, failures: FailureMemory, graph: Graph,
                 bus: EventBus | None = None) -> None:
        self.library = library
        self.failures = failures
        self.graph = graph
        self.bus = bus

    # -- skill selection -----------------------------------------------------------------------------
    def select_skills(self, task: TaskSpec, retrieval: RetrievalResult) -> list[tuple[Skill, list[str]]]:
        chosen: dict[str, tuple[Skill, list[str]]] = {}
        roles: set[str] = set()
        for item in retrieval.skills:
            if not self.library.exists(item.id):
                continue
            skill = self.library.get(item.id)
            if skill.status in (SkillStatus.DISABLED, SkillStatus.MERGED):
                continue
            d = skill.definition
            if task.object_class and d.object_class and d.object_class != task.object_class:
                continue
            if skill.source_class == "system_seeded":
                if any(s.source_class != "system_seeded" for s, _r in chosen.values()):
                    continue  # learned composite skills cover the task; seeds are not needed
                if not _mentions_trigger(task.text, d.triggers):
                    continue  # a generic capability is only used when the task asks for it
                # An explicit trigger mention is the selection evidence for a seed; the retrieval
                # threshold still applies, the stricter learned-skill threshold does not.
            elif item.score < MIN_SKILL_SCORE:
                continue
            role = d.object_role or skill.id
            if role in roles:
                continue
            roles.add(role)
            chosen[skill.id] = (skill, [f"retrieved:{item.score:.2f}", *item.reason_codes[:4]])
        return self._order(list(chosen.values()))

    def _order(self, selected: list[tuple[Skill, list[str]]]) -> list[tuple[Skill, list[str]]]:
        ids = {s.id for s, _r in selected}
        before: dict[str, set[str]] = {i: set() for i in ids}
        for skill_id in ids:
            for edge in self.graph.outgoing("skill", skill_id, {"follows", "requires"}):
                if edge.dst_id in ids:
                    if edge.rel == "follows":
                        before[edge.dst_id].add(skill_id)
                    else:
                        before[skill_id].add(edge.dst_id)
        ordered, remaining = [], {s.id: (s, r) for s, r in selected}
        while remaining:
            ready = [i for i in remaining if not (before[i] & set(remaining))]
            if not ready:  # cycle: fall back to phase order
                ready = list(remaining)
            ready.sort(key=lambda i: (min((PHASE_ORDER.get(p.name, 9) for p in remaining[i][0].definition.phases),
                                          default=9), 0 if _creates_object(remaining[i][0]) else 1))
            ordered.append(remaining.pop(ready[0]))
        return ordered

    def plan_from_episode(self, task: TaskSpec, retrieval: RetrievalResult, definitions: list[Any],
                          *, gui_available: bool = False) -> Plan:
        """Baseline for A/B experiments: replay the procedure of the most similar demonstration with
        the values it used (no generalisation, no task parameters, no failure guards)."""
        plan = Plan(id=new_id("plan"), task=task, retrieval_id=retrieval.id, strategy="episodes_only",
                    created_at=now(), reason_codes=["raw_demonstration_replay"])
        if not definitions:
            plan.unresolved.append({"reason": "no_similar_demonstration", "required": True})
            return plan
        ctx = CompileContext(gui_available=gui_available)
        for definition in definitions:
            params = {p.name: (p.observed_values[-1] if p.observed_values else p.default) for p in definition.parameters}
            params.setdefault("object_name", (definition.object_role or "Object").title())
            for phase in sorted(definition.phases, key=lambda p: PHASE_ORDER.get(p.name, 9)):
                step = PlanStep(id=new_id("step"), phase=phase.name, skill_id=None, skill_name=definition.name,
                                params=params, reason_codes=["replayed_from_episode"])
                for i, template in enumerate(phase.actions):
                    try:
                        step.actions += compile_template(template, params, ctx, source=f"episode/{phase.name}/{i}")
                    except Uncompilable as exc:
                        plan.unresolved.append({"phase": phase.name, "action": template.action_type,
                                                "reason": exc.reason, "required": not template.optional})
                step.checkpoints = [c for c in (definition.checkpoint(cid) for cid in phase.checkpoints) if c]
                plan.steps.append(step)
        plan.confidence = 0.5
        return plan

    # -- parameters -------------------------------------------------------------------------------------
    @staticmethod
    def resolve_params(skill: Skill, task: TaskSpec) -> tuple[dict[str, Any], dict[str, str]]:
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        role = skill.definition.object_role or ""
        for spec in skill.definition.parameters:
            value, source = _param_value(spec, task, role)
            values[spec.name] = value
            sources[spec.name] = source
        if values.get("object_name") is None:
            values["object_name"] = (role or "Object").title()
            sources["object_name"] = "derived"
        return values, sources

    # -- planning -----------------------------------------------------------------------------------------
    def plan(self, task: TaskSpec, retrieval: RetrievalResult, *, preferences: dict[str, Any] | None = None,
             gui_available: bool = False) -> Plan:
        preferences = preferences or {}
        plan = Plan(id=new_id("plan"), task=task, retrieval_id=retrieval.id, strategy=retrieval.query.strategy,
                    created_at=now())
        selected = self.select_skills(task, retrieval)
        if not selected:
            plan.unresolved.append({"reason": "no_applicable_skills",
                                    "detail": "no retrieved skill covers this task; a demonstration is needed"})
            plan.reason_codes.append("needs_demonstration")
            return plan
        failures = self._failures(task, retrieval, [s for s, _r in selected])
        return self._compose(plan, task, selected, failures, preferences, gui_available)

    def plan_skills(self, task: TaskSpec, skills: list[Skill], *, gui_available: bool = False) -> Plan:
        """Plan exactly these skills, without retrieval (validation by reproduction): no other skill joins
        the run and takes credit for its outcome. Failure guards of the skills still apply."""
        plan = Plan(id=new_id("plan"), task=task, retrieval_id=None, strategy="pinned", created_at=now(),
                    reason_codes=["pinned_skills"])
        selected = [(skill, ["pinned"]) for skill in skills]
        action_types = {a.action_type for s in skills for p in s.definition.phases for a in p.actions}
        failures = [f for f in self.failures.relevant(task_class=task.object_class, action_types=action_types,
                                                      skill_ids={s.id for s in skills})
                    if f.skill_id in {s.id for s in skills} or f.trigger_action in action_types]
        return self._compose(plan, task, selected, failures, {}, gui_available)

    def _compose(self, plan: Plan, task: TaskSpec, selected: list[tuple[Skill, list[str]]],
                 failures: list[FailureRecord], preferences: dict[str, Any], gui_available: bool) -> Plan:
        ctx = CompileContext(gui_available=gui_available)
        names_used: set[str] = set()
        current_object = task.params.get("object_name")
        for skill, reasons in selected:
            params, sources = self.resolve_params(skill, task)
            if skill.definition.object_role is None:
                # Generic skills act on the task's object; the first creator names it.
                name = current_object or params["object_name"]
                current_object = name
            else:
                name = params["object_name"] if (len(selected) == 1 or sources.get("object_name") != "task") \
                    else (skill.definition.object_role or "object").title()
                base = name
                while name in names_used:  # two role-specific skills must not fight over one object
                    name = f"{base}_{len(names_used)}"
            params["object_name"] = name
            names_used.add(name)
            steps = self._skill_steps(skill, params, sources, reasons, ctx, plan)
            self._attach_guards(skill, steps, failures, plan)
            plan.steps += steps
        self._apply_preferences(plan, preferences)
        coverage = 1.0 - min(1.0, len([u for u in plan.unresolved if u.get("required")]) / max(1, len(plan.steps)))
        plan.confidence = round(coverage * sum(s.confidence for s in plan.steps) / max(1, len(plan.steps)), 4)
        if self.bus is not None:
            self.bus.publish(EventType.PLAN_CREATED, plan.id, task=task.text, outline=plan.outline(),
                             skills=plan.skill_ids, unresolved=len(plan.unresolved), confidence=plan.confidence)
        return plan

    def _skill_steps(self, skill: Skill, params: dict[str, Any], sources: dict[str, str], reasons: list[str],
                     ctx: CompileContext, plan: Plan) -> list[PlanStep]:
        d = skill.definition
        steps: list[PlanStep] = []
        phases = sorted(d.phases, key=lambda p: PHASE_ORDER.get(p.name, 9))
        for phase in phases:
            step = PlanStep(id=new_id("step"), phase=phase.name, skill_id=skill.id, skill_version=skill.current_version,
                            skill_name=d.name, params=params, confidence=skill.confidence,
                            reason_codes=reasons + [f"param_sources:{','.join(sorted(set(sources.values())))}"])
            for i, template in enumerate(phase.actions):
                source = f"{skill.id}/{phase.name}/{i}"
                if template.optional and template.frequency < 0.5:
                    step.reason_codes.append(f"skipped_rare_optional:{template.action_type}")
                    continue
                try:
                    step.actions += compile_template(template, params, ctx, source=source)
                except Uncompilable as exc:
                    plan.unresolved.append({"skill_id": skill.id, "phase": phase.name, "action": template.action_type,
                                            "reason": exc.reason, "required": not template.optional})
                    if not template.optional:
                        step.confidence *= 0.7
            step.checkpoints = [c for c in (d.checkpoint(cid) for cid in phase.checkpoints) if c is not None]
            step.recovery = self._recovery(skill, phase.name, params, ctx)
            steps.append(step)
        attached = {c.id for s in steps for c in s.checkpoints}
        leftover = [c for c in d.checkpoints if c.id not in attached]
        if leftover and steps:
            steps[-1].checkpoints += leftover
        return steps

    def _recovery(self, skill: Skill, phase: str, params: dict[str, Any], ctx: CompileContext) -> list[RecoveryPlan]:
        out = []
        for rc in skill.definition.recovery_actions:
            condition = next((f for f in skill.definition.failure_conditions if f.id == rc.when), None)
            if condition is not None and condition.phase not in (None, phase):
                continue
            actions = []
            recovery_ctx = CompileContext(mode="OBJECT", gui_available=ctx.gui_available)
            try:
                for i, template in enumerate(rc.actions):
                    actions += compile_template(template, params, recovery_ctx, source=f"{skill.id}/recovery/{rc.id}/{i}")
            except Uncompilable:
                continue
            checkpoints = set(rc.when_checkpoints)
            if condition is not None and condition.guard_checkpoint:
                checkpoints.add(condition.guard_checkpoint)
            out.append(RecoveryPlan(id=rc.id, description=rc.description,
                                    when_failure_ids=condition.failure_ids if condition else [],
                                    when_checkpoints=sorted(checkpoints), actions=actions, source=rc.source))
        return out

    def _failures(self, task: TaskSpec, retrieval: RetrievalResult, skills: list[Skill]) -> list[FailureRecord]:
        ids = {f.id for f in retrieval.failures}
        action_types = {a.action_type for s in skills for p in s.definition.phases for a in p.actions}
        relevant = self.failures.relevant(task_class=task.object_class, action_types=action_types,
                                          skill_ids={s.id for s in skills})
        return [f for f in relevant if f.id in ids or f.skill_id in {s.id for s in skills}
                or f.trigger_action in action_types]

    def _attach_guards(self, skill: Skill, steps: list[PlanStep], failures: list[FailureRecord], plan: Plan) -> None:
        checkpoint_ids = {c.id for c in skill.definition.checkpoints}
        for rec in failures:
            if rec.rule_status == "rejected" or not rec.trigger_action:
                continue
            guard_cp = rec.guard_checkpoint if rec.guard_checkpoint in checkpoint_ids else None
            for step in steps:
                if not any(a.action_type == rec.trigger_action for a in step.actions):
                    continue
                if guard_cp is None:
                    step.reason_codes.append(f"caution:{rec.id}")
                    continue
                step.guards.append(Guard(failure_id=rec.id, trigger_action=rec.trigger_action, checkpoint_id=guard_cp,
                                         rule=rec.future_rule or rec.observed_problem, confidence=rec.confidence))
                step.reason_codes.append(f"guarded_by:{rec.id}")
                plan.reason_codes.append(f"failure_rule_applied:{rec.id}")

    @staticmethod
    def _apply_preferences(plan: Plan, preferences: dict[str, Any]) -> None:
        if preferences.get("bevel_late") or preferences.get("blockout_first"):
            detail = [s for s in plan.steps if s.phase == "detail"]
            if detail and plan.steps[-len(detail):] != detail:
                plan.steps = [s for s in plan.steps if s.phase != "detail"] + detail
                plan.preferences_applied.append("bevel_late" if preferences.get("bevel_late") else "blockout_first")
        if preferences.get("orthographic_checks"):
            for step in plan.steps:
                for cp in step.checkpoints:
                    if cp.check.get("type") == "silhouette" and not cp.required:
                        cp.required = True
                        plan.preferences_applied.append("orthographic_checks")


def _param_value(spec: ParamSpec, task: TaskSpec, role: str) -> tuple[Any, str]:
    """Task value > generic dimension word ("length 5" -> blade_length) > qualified default > default."""
    if spec.name in task.params:
        return task.params[spec.name], task.param_sources.get(spec.name, "task")
    for word in GENERIC_DIMENSIONS:
        if word in task.params and word in spec.name:
            return task.params[word], task.param_sources.get(word, "task")
    value = spec.default
    if isinstance(value, (int, float)) and not isinstance(value, bool) and spec.unit == "m":
        factors = [f for word, f in task.qualifiers.items() if word == "scale" or word in spec.name]
        if factors:
            for factor in factors:
                value = value * factor
            return round(value, 4), "qualifier"
    return value, "skill_default" if spec.observed_values else "declared_default"


def _creates_object(skill: Skill) -> bool:
    return any(a.action_type == "add_primitive" for p in skill.definition.phases for a in p.actions)


def _mentions_trigger(text: str, triggers: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(t.lower())}s?\b", lowered) for t in triggers)
