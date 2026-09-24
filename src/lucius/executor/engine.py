"""The execution engine: OBSERVE -> PLAN -> EXECUTE -> VERIFY -> (RECOVER | HUMAN_TAKEOVER) -> outcome.

Each run is recorded as a session of its own (agent actions, state, takeovers), so the agent's
own experience flows through the same learning pipeline as human demonstrations. Outcomes feed
back into skill evidence and failure memory:

* checkpoint failures become (or strengthen) failure records,
* guards derived from failure memory that prevented a known failure count as correction
  successes for that rule (the path by which rules get promoted),
* human takeovers produce correction records linked to the agent state that required them.

A run is only ``success`` when the evaluator's verdict is success -- finishing the action
sequence is never enough.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from lucius.config import SafetyConfig
from lucius.corrections import CorrectionStore
from lucius.errors import ActionRejected, BlenderBridgeError, LuciusError
from lucius.evaluation.checks import ReferenceSilhouette
from lucius.evaluation.evaluator import EvaluationReport, Evaluator, verdict_for
from lucius.events.bus import EventBus, EventType
from lucius.executor.backend import ActionResult, ExecutionBackend
from lucius.executor.human import HumanChannel, TakeoverRequest
from lucius.executor.safety import ActionValidator
from lucius.executor.state_machine import ExecState, RunStateMachine
from lucius.ids import new_id
from lucius.logging_setup import get_logger
from lucius.memory.failure import FailureEvidence, FailureMemory, FailureObservation
from lucius.memory.preferences import WorkflowPreferences
from lucius.planner.model import Plan, PlanAction, PlanStep, TaskSpec
from lucius.planner.planner import Planner
from lucius.planner.task import parse_task
from lucius.provenance import DataPolicy, SourceClass
from lucius.retrieval.retriever import HybridRetriever, RetrievalQuery
from lucius.sessions.models import Actor, CapturedEvent, EventKind, Outcome, SessionKind, SessionStatus
from lucius.sessions.store import SessionStore
from lucius.skills.library import SkillLibrary
from lucius.skills.schema import Checkpoint
from lucius.storage.db import Database, dumps
from lucius.timeutil import now
from lucius.trajectory import vocabulary as vocab

log = get_logger("executor")

MODE_TO_KIND = {"execute": SessionKind.AGENT_EXECUTION, "practice": SessionKind.PRACTICE,
                "benchmark": SessionKind.BENCHMARK, "validation": SessionKind.VALIDATION}


class ArmConfig(BaseModel):
    """Experiment arm / execution strategy (section 50)."""

    name: str = "memory_enhanced"
    retrieval_strategy: str = "hybrid"
    use_failure_guards: bool = True
    use_recovery: bool = True
    use_preferences: bool = True
    max_recovery_attempts: int = 1


class StepReport(BaseModel):
    step_id: str
    phase: str
    skill_id: str | None
    actions_ok: int = 0
    actions_failed: int = 0
    actions_skipped: list[str] = Field(default_factory=list)
    guards_applied: list[str] = Field(default_factory=list)
    verdict: str = "pending"
    recovered: bool = False
    takeover: bool = False
    failures: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str
    session_id: str
    status: str
    verdict: str
    plan: Plan | None = None
    final_report: EvaluationReport | None = None
    steps: list[StepReport] = Field(default_factory=list)
    takeovers: int = 0
    recoveries: int = 0
    guard_blocks: int = 0
    failure_ids: list[str] = Field(default_factory=list)
    correction_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _RunContext:
    run_id: str
    session_id: str
    sm: RunStateMachine
    validator: ActionValidator
    references: list[ReferenceSilhouette]
    task: TaskSpec
    mode: str = "execute"
    arm: str = "memory_enhanced"
    criteria: list[Checkpoint] = field(default_factory=list)
    seq: int = 0
    passed_checkpoints: set[str] = field(default_factory=set)
    result: RunResult | None = None


class ExecutionEngine:
    def __init__(self, *, db: Database, sessions: SessionStore, library: SkillLibrary, failures: FailureMemory,
                 retriever: HybridRetriever, planner: Planner, evaluator: Evaluator, corrections: CorrectionStore,
                 preferences: WorkflowPreferences, safety: SafetyConfig, bus: EventBus | None = None,
                 user_id: str = "local", on_session_complete: Callable[[str], None] | None = None) -> None:
        self.db = db
        self.sessions = sessions
        self.library = library
        self.failures = failures
        self.retriever = retriever
        self.planner = planner
        self.evaluator = evaluator
        self.corrections = corrections
        self.preferences = preferences
        self.safety = safety
        self.bus = bus
        self.user_id = user_id
        self.on_session_complete = on_session_complete

    # -- event recording ----------------------------------------------------------------------------
    def _event(self, ctx: _RunContext, kind: EventKind, actor: Actor, payload: dict[str, Any]) -> None:
        self.sessions.append_events(ctx.session_id, [CapturedEvent(seq=ctx.seq, ts=now(), kind=kind, actor=actor,
                                                                   blender_active=True, payload=payload)])
        ctx.seq += 1

    def _state_event(self, ctx: _RunContext, backend: ExecutionBackend, reason: str) -> dict[str, Any] | None:
        try:
            state = backend.observe()
        except BlenderBridgeError:
            return None
        self._event(ctx, EventKind.BLENDER_STATE, Actor.SYSTEM, {**state.model_dump(), "reason": reason})
        return state.compact()

    def record_human_action(self, run_id: str, backend: ExecutionBackend, action: PlanAction) -> ActionResult:
        """Execute and record an action performed *by the human* during a takeover (e.g. via the UI console)."""
        row = self.db.query_one("SELECT session_id FROM runs WHERE id = ?", (run_id,))
        session_id = row["session_id"]
        seq = self.sessions.next_seq(session_id)
        result = backend.execute(action)
        self.sessions.append_events(session_id, [CapturedEvent(
            seq=seq, ts=now(), kind=EventKind.AGENT_ACTION, actor=Actor.HUMAN, blender_active=True,
            payload={"action": {"layer": action.layer, "name": action.name, "args": action.args},
                     "result": result.result if result.ok else {"error": result.error}})])
        state = backend.observe()
        self.sessions.append_events(session_id, [CapturedEvent(
            seq=seq + 1, ts=now(), kind=EventKind.BLENDER_STATE, actor=Actor.SYSTEM, blender_active=True,
            payload={**state.model_dump(), "reason": "human_action"})])
        return result

    # -- main entry --------------------------------------------------------------------------------
    def run(self, task_text: str, backend: ExecutionBackend, *, mode: str = "execute",
            arm: ArmConfig | None = None, task_params: dict[str, Any] | None = None,
            references: list[ReferenceSilhouette] | None = None, reference_ids: list[str] | None = None,
            human: HumanChannel | None = None, practice_task_id: str | None = None,
            benchmark_id: str | None = None, reset_scene: bool = False,
            success_criteria: list[Checkpoint] | None = None,
            plan_fn: Callable[[TaskSpec], Plan] | None = None) -> RunResult:
        """Run a task. ``success_criteria`` are task-level checkpoints (practice/benchmark definitions)
        evaluated alongside the skills' own checkpoints; all of them must pass for success.
        ``plan_fn`` replaces retrieval+planning (used by the raw-demonstration experiment arm)."""
        arm = arm or ArmConfig()
        task = parse_task(task_text, reference_ids)
        if task_params:
            task.params.update(task_params)
            task.param_sources.update({k: "task" for k in task_params})
        session = self.sessions.create(
            user_id=self.user_id, kind=MODE_TO_KIND.get(mode, SessionKind.AGENT_EXECUTION),
            policy=DataPolicy(source=SourceClass.AGENT_SUCCESS, license="agent_generated"), task_text=task_text,
            task_class=task.object_class, reference_ids=reference_ids or [], status=SessionStatus.RECORDING,
            meta={"arm": arm.model_dump(), "backend": backend.environment})
        run_id = new_id("run")
        self.db.insert("runs", {
            "id": run_id, "session_id": session.id, "task_text": task_text, "task_class": task.object_class,
            "mode": mode, "state": ExecState.IDLE.value, "status": "running", "backend": backend.name,
            "environment": backend.environment, "arm": arm.name, "practice_task_id": practice_task_id,
            "benchmark_id": benchmark_id, "params": dumps(task.params), "metrics": dumps({}), "started_at": now()})
        ctx = _RunContext(run_id=run_id, session_id=session.id, sm=RunStateMachine(self.db, run_id, self.bus),
                          validator=ActionValidator(self.safety, bridge_actions=backend.bridge_actions,
                                                    gui_only=backend.gui_only, background=backend.background),
                          references=references or [], task=task, mode=mode, arm=arm.name,
                          criteria=list(success_criteria or []))
        ctx.result = RunResult(run_id=run_id, session_id=session.id, status="running", verdict="pending")
        try:
            self._run(ctx, backend, arm, human, reset_scene, plan_fn)
        except LuciusError as exc:
            log.exception("run %s aborted", run_id)
            ctx.result.reason_codes.append(f"aborted:{exc.code}")
            if not ctx.sm.terminal:
                ctx.sm.transition(ExecState.FAILURE, "run_aborted", evidence=[exc.message])
            ctx.result.status = ctx.result.verdict = "failure"
        self._finish(ctx, backend)
        return ctx.result

    def _run(self, ctx: _RunContext, backend: ExecutionBackend, arm: ArmConfig, human: HumanChannel | None,
             reset_scene: bool, plan_fn: Callable[[TaskSpec], Plan] | None = None) -> None:
        result = ctx.result
        assert result is not None
        ctx.sm.transition(ExecState.OBSERVE, "run_started", context={"arm": arm.name})
        if reset_scene:
            backend.reset()
        self._state_event(ctx, backend, "initial")
        ctx.sm.transition(ExecState.PLAN, "state_observed")
        if plan_fn is not None:
            plan = plan_fn(ctx.task)
        else:
            retrieval = self.retriever.retrieve(RetrievalQuery(text=ctx.task.text, task_class=ctx.task.object_class,
                                                               categories=ctx.task.categories,
                                                               strategy=arm.retrieval_strategy), run_id=ctx.run_id)
            prefs = self.preferences.active() if arm.use_preferences else {}
            plan = self.planner.plan(ctx.task, retrieval, preferences=prefs, gui_available=backend.gui_available)
        if not arm.use_failure_guards:
            for step in plan.steps:
                step.guards = []
        if not arm.use_recovery:
            for step in plan.steps:
                step.recovery = []
        result.plan = plan
        self.db.execute("UPDATE runs SET plan = ?, retrieval_id = ? WHERE id = ?",
                        (plan.model_dump_json(), plan.retrieval_id, ctx.run_id))
        if not plan.steps:
            result.reason_codes.append("needs_demonstration")
            ctx.sm.transition(ExecState.FAILURE, "no_applicable_skills",
                              evidence=[u.get("detail", "") for u in plan.unresolved])
            result.status = result.verdict = "failure"
            return
        for i, step in enumerate(plan.steps):
            report = StepReport(step_id=step.id, phase=step.phase, skill_id=step.skill_id)
            result.steps.append(report)
            ctx.sm.transition(ExecState.EXECUTE, "step_started", context={"phase": step.phase, "skill": step.skill_id,
                                                                          "index": i})
            snapshot = f"{ctx.run_id}_{i}"
            backend.snapshot(snapshot)
            deferred = self._execute_actions(ctx, backend, step, step.actions, report)
            ctx.sm.transition(ExecState.VERIFY, "step_actions_done",
                              evidence=[f"{report.actions_ok} ok, {report.actions_failed} failed"])
            verified = self._verify_step(ctx, backend, step, report)
            if not verified:
                verified = self._recover(ctx, backend, step, report, snapshot, arm, human)
                if not verified:
                    ctx.sm.transition(ExecState.FAILURE, "step_unrecoverable", context={"phase": step.phase})
                    self._final_verdict(ctx, backend, plan, forced_failure=True)
                    return
            if deferred:
                ctx.sm.transition(ExecState.EXECUTE, "deferred_guarded_actions",
                                  evidence=[a.name for a in deferred])
                self._execute_actions(ctx, backend, step, deferred, report, allow_defer=False)
                ctx.sm.transition(ExecState.VERIFY, "deferred_actions_done")
                if not self._verify_step(ctx, backend, step, report):
                    if not self._recover(ctx, backend, step, report, snapshot, arm, human):
                        ctx.sm.transition(ExecState.FAILURE, "step_unrecoverable", context={"phase": step.phase})
                        self._final_verdict(ctx, backend, plan, forced_failure=True)
                        return
        self._final_verdict(ctx, backend, plan)

    # -- execution ----------------------------------------------------------------------------------
    def _execute_actions(self, ctx: _RunContext, backend: ExecutionBackend, step: PlanStep, actions: list[PlanAction],
                         report: StepReport, allow_defer: bool = True) -> list[PlanAction]:
        deferred: list[PlanAction] = []
        index = 0
        while index < len(actions):
            action = actions[index]
            index += 1
            guard = next((g for g in step.guards if g.trigger_action == action.action_type), None)
            if guard is not None and guard.checkpoint_id not in ctx.passed_checkpoints:
                if allow_defer:
                    # Hold the guarded action (and the selection feeding it) until its guard passes.
                    held = [action]
                    if deferred == [] and index >= 2 and actions[index - 2].action_type in ("select_all", "select_elements"):
                        held.insert(0, actions[index - 2])
                    deferred += held
                    report.guards_applied.append(guard.failure_id)
                    ctx.result.guard_blocks += 1
                    self._event(ctx, EventKind.MARKER, Actor.AGENT, {"label": "guard_deferred", "failure_id": guard.failure_id,
                                                                     "action": action.name, "rule": guard.rule})
                    continue
                report.actions_skipped.append(f"{action.name}:guard_not_satisfied")
                continue
            try:
                ctx.validator.validate(action)
            except ActionRejected as exc:
                report.actions_skipped.append(f"{action.name}:{exc.code}")
                if self.bus is not None:
                    self.bus.publish(EventType.ACTION_REJECTED, ctx.run_id, action=action.name, reason=exc.message)
                if not action.optional:
                    report.actions_failed += 1
                continue
            outcome = backend.execute(action)
            self._event(ctx, EventKind.AGENT_ACTION, Actor.AGENT, {
                "action": {"layer": action.layer, "name": action.name, "args": action.args,
                           "action_type": action.action_type, "source": action.source},
                "result": outcome.result if outcome.ok else {"error": outcome.error}})
            if self.bus is not None:
                self.bus.publish(EventType.ACTION_EXECUTED, ctx.run_id, action=action.name, ok=outcome.ok,
                                 layer=action.layer, phase=step.phase)
            if outcome.ok:
                report.actions_ok += 1
            elif action.optional:
                report.actions_skipped.append(f"{action.name}:optional_failed")
            else:
                report.actions_failed += 1
        self._state_event(ctx, backend, f"after_{step.phase}")
        return deferred

    def _verify_step(self, ctx: _RunContext, backend: ExecutionBackend, step: PlanStep, report: StepReport) -> bool:
        structure = backend.structure()
        evaluation = self.evaluator.evaluate(
            subject_kind="run_step", subject_id=step.id, checkpoints=step.checkpoints, params=step.params,
            structure=structure, references=ctx.references, execution_ok=report.actions_failed == 0,
            run_id=ctx.run_id)
        ctx.passed_checkpoints |= {r.checkpoint_id for r in evaluation.results if r.passed}
        report.verdict = evaluation.verdict
        if evaluation.verdict == "failure":
            for fid in self._record_step_failure(ctx, step, evaluation):
                if fid not in report.failures:
                    report.failures.append(fid)
            return False
        return True

    def _record_step_failure(self, ctx: _RunContext, step: PlanStep, evaluation: EvaluationReport) -> list[str]:
        ids = []
        trigger = next((a.action_type for a in reversed(step.actions) if vocab.mutates(a.action_type)), None)
        failed = [r for r in evaluation.results if r.required and r.passed is False]
        if not evaluation.execution_ok and not failed:
            record = self.failures.record(FailureObservation(
                task_class=ctx.task.object_class, phase=step.phase, observed_problem="an action failed to execute",
                symptoms=[s for s in ctx.result.steps[-1].actions_skipped][:5], trigger_action=trigger,
                problem_code="execution_error", skill_id=step.skill_id,
                evidence=FailureEvidence(kind="execution_error", source_class=SourceClass.AGENT_FAILURE.value,
                                         run_id=ctx.run_id, session_id=ctx.session_id)))
            ids.append(record.id)
        for r in failed:
            record = self.failures.record(FailureObservation(
                task_class=ctx.task.object_class, phase=step.phase,
                observed_problem=f"checkpoint failed: {r.description}",
                symptoms=[f"{r.reason_code}: {dumps(r.evidence)[:200]}"], trigger_action=trigger,
                problem_code=f"checkpoint:{r.checkpoint_id}", skill_id=step.skill_id,
                evidence=FailureEvidence(kind="checkpoint_failure", source_class=SourceClass.AGENT_FAILURE.value,
                                         run_id=ctx.run_id, session_id=ctx.session_id, detail=r.reason_code)))
            ids.append(record.id)
        ctx.result.failure_ids = list(dict.fromkeys(ctx.result.failure_ids + ids))
        return ids

    # -- recovery & takeover ------------------------------------------------------------------------
    def _recover(self, ctx: _RunContext, backend: ExecutionBackend, step: PlanStep, report: StepReport,
                 snapshot: str, arm: ArmConfig, human: HumanChannel | None) -> bool:
        latest = self.evaluator.for_subject("run_step", step.id)
        failed_checkpoints = {e["checkpoint_id"] for e in latest if e["passed"] == 0 and e["checkpoint_id"]}
        for attempt in range(arm.max_recovery_attempts if arm.use_recovery else 0):
            plans = [r for r in step.recovery if set(r.when_checkpoints) & failed_checkpoints
                     or set(r.when_failure_ids) & set(report.failures)]
            ctx.sm.transition(ExecState.RECOVER, "checkpoint_failed", evidence=sorted(failed_checkpoints),
                              context={"attempt": attempt, "recovery_plans": [r.id for r in plans]})
            backend.restore(snapshot)
            self._event(ctx, EventKind.MARKER, Actor.AGENT, {"label": "recover_restore", "snapshot": snapshot})
            actions = [a for r in plans for a in r.actions if a.name != "restore_snapshot"] or step.actions
            ctx.sm.transition(ExecState.EXECUTE, "recovery_actions", evidence=[a.name for a in actions][:10])
            # Guarded actions stay held in the step's original deferred list; never re-defer here.
            self._execute_actions(ctx, backend, step, actions, report, allow_defer=False)
            ctx.sm.transition(ExecState.VERIFY, "recovery_executed")
            if self._verify_step(ctx, backend, step, report):
                report.recovered = True
                ctx.result.recoveries += 1
                for fid in report.failures:
                    self.failures.record_correction_outcome(fid, success=True, evidence=FailureEvidence(
                        kind="correction_outcome", source_class=SourceClass.AGENT_SUCCESS.value, run_id=ctx.run_id,
                        detail="automatic recovery"))
                return True
        if human is None or not human.available():
            return False
        return self._takeover(ctx, backend, step, report, sorted(failed_checkpoints), human)

    def _takeover(self, ctx: _RunContext, backend: ExecutionBackend, step: PlanStep, report: StepReport,
                  failed_checkpoints: list[str], human: HumanChannel) -> bool:
        ctx.sm.transition(ExecState.HUMAN_TAKEOVER, "outside_reliable_distribution",
                          evidence=[f"failed: {', '.join(failed_checkpoints) or 'execution'}"],
                          context={"phase": step.phase, "skill": step.skill_id})
        report.takeover = True
        ctx.result.takeovers += 1
        before = self._state_event(ctx, backend, "takeover_before")
        agent_context = {"plan_id": ctx.result.plan.id if ctx.result.plan else None, "step_id": step.id,
                         "phase": step.phase, "skill_id": step.skill_id, "failed_checkpoints": failed_checkpoints,
                         "params": step.params}
        start = now()
        self._event(ctx, EventKind.TAKEOVER_START, Actor.HUMAN, {"reason": None, "agent_context": agent_context,
                                                                 "state": before})
        if self.bus is not None:
            self.bus.publish(EventType.HUMAN_TAKEOVER, ctx.run_id, phase=step.phase, skill=step.skill_id,
                             failed=failed_checkpoints)
        request = TakeoverRequest(run_id=ctx.run_id, reason_code="checkpoint_failed_after_recovery",
                                  message=f"{step.skill_name}: {step.phase} did not pass {failed_checkpoints}",
                                  phase=step.phase, skill_id=step.skill_id, failed_checkpoints=failed_checkpoints,
                                  context=agent_context)
        outcome = human.request(request)
        ctx.seq = self.sessions.next_seq(ctx.session_id)  # the human's actions were appended meanwhile
        after = self._state_event(ctx, backend, "takeover_after")
        self._event(ctx, EventKind.TAKEOVER_END, Actor.HUMAN, {"reason": outcome.reason, "state": after})
        if not outcome.resumed:
            ctx.sm.transition(ExecState.FAILURE, "takeover_aborted", evidence=[outcome.note or ""])
            self.corrections.record(kind="takeover", session_id=ctx.session_id, run_id=ctx.run_id,
                                    failure_id=report.failures[0] if report.failures else None, start_time=start,
                                    end_time=now(), before_state=before, after_state=after, agent_context=agent_context,
                                    reason=outcome.reason, outcome="aborted", correction_id=request.id)
            return False
        ctx.sm.transition(ExecState.RESUME, "human_resumed", evidence=[outcome.note or ""])
        if self.bus is not None:
            self.bus.publish(EventType.HUMAN_RESUME, ctx.run_id, reason_given=outcome.reason is not None)
        ctx.sm.transition(ExecState.VERIFY, "verify_after_takeover")
        ok = self._verify_step(ctx, backend, step, report)
        correction = self.corrections.record(
            kind="takeover", session_id=ctx.session_id, run_id=ctx.run_id,
            failure_id=report.failures[0] if report.failures else None, start_time=start, end_time=now(),
            before_state=before, after_state=after, agent_context=agent_context, reason=outcome.reason,
            outcome="success" if ok else "failure", correction_id=request.id)
        ctx.result.correction_ids.append(correction.id)
        for fid in report.failures:
            self.failures.record_correction_outcome(fid, success=ok, evidence=FailureEvidence(
                kind="correction_outcome", source_class=SourceClass.HUMAN_CORRECTION.value, run_id=ctx.run_id,
                session_id=ctx.session_id, detail=f"human takeover {correction.id}"))
        return ok

    # -- final evaluation & learning update ---------------------------------------------------------
    def _final_verdict(self, ctx: _RunContext, backend: ExecutionBackend, plan: Plan,
                       forced_failure: bool = False) -> None:
        """Evaluate every planned checkpoint on the final scene and update skill evidence.

        ``forced_failure``: a step already failed unrecoverably (state machine is terminal); the
        final evaluation still runs for metrics and per-skill credit, but cannot turn into success.
        """
        result = ctx.result
        structure = backend.structure()
        execution_ok = all(s.actions_failed == 0 for s in result.steps)
        per_skill: dict[str, tuple[list[Checkpoint], dict[str, Any]]] = {}
        for step in plan.steps:
            checkpoints, _params = per_skill.setdefault(step.skill_id or "", ([], step.params))
            for cp in step.checkpoints:
                if all(c.id != cp.id for c in checkpoints):
                    checkpoints.append(cp)
        skill_results: dict[str, list[Any]] = {}
        for skill_id, (checkpoints, params) in per_skill.items():
            report = self.evaluator.evaluate(subject_kind="run", subject_id=ctx.run_id, checkpoints=checkpoints,
                                             params=params, structure=structure, references=ctx.references,
                                             execution_ok=execution_ok, run_id=ctx.run_id)
            skill_results[skill_id] = report.results
        all_results = [r for rs in skill_results.values() for r in rs]
        if ctx.criteria:
            task_report = self.evaluator.evaluate(subject_kind="run", subject_id=ctx.run_id, checkpoints=ctx.criteria,
                                                  params={**ctx.task.params}, structure=structure,
                                                  references=ctx.references, execution_ok=execution_ok,
                                                  run_id=ctx.run_id)
            all_results += task_report.results
        verdict, failed, unevaluated, objective = verdict_for(execution_ok and not forced_failure, all_results)
        result.final_report = EvaluationReport(
            id=new_id("evaluation"), subject_kind="run", subject_id=ctx.run_id, run_id=ctx.run_id,
            execution_ok=execution_ok, results=all_results, verdict=verdict, objective_passes=objective,
            required_failed=failed, required_unevaluated=unevaluated,
            summary=f"{verdict}: {sum(1 for r in all_results if r.passed)}/{len(all_results)} checkpoints passed")
        result.verdict = result.status = verdict
        if not ctx.sm.terminal:
            if verdict == "success":
                ctx.sm.transition(ExecState.SUCCESS, "all_required_checkpoints_passed", confidence=plan.confidence,
                                  evidence=[f"{objective} objective checkpoints passed"])
            else:
                ctx.sm.transition(ExecState.FAILURE,
                                  "final_checks_failed" if verdict == "failure" else "success_not_verified",
                                  evidence=failed + unevaluated)
        if verdict == "success":
            # A guard that held back a known premature action in a verified run is evidence the rule works.
            for fid in {g for s in result.steps for g in s.guards_applied}:
                self.failures.record_correction_outcome(fid, success=True, evidence=FailureEvidence(
                    kind="correction_outcome", source_class=SourceClass.AGENT_SUCCESS.value, run_id=ctx.run_id,
                    detail="guard prevented the known failure"))
        for skill_id, results in skill_results.items():
            if not skill_id:
                continue
            params = next(s.params for s in plan.steps if s.skill_id == skill_id)
            required_ok = all(r.passed is True for r in results if r.required)
            objective_ok = any(r.passed and not r.subjective and r.level in (2, 3) for r in results)
            skill_steps_ok = all(s.actions_failed == 0 and s.verdict != "failure"
                                 for s in result.steps if s.skill_id == skill_id)
            success = skill_steps_ok and required_ok and objective_ok
            signature = hashlib.sha1(dumps({k: v for k, v in params.items() if k != "object_name"}).encode()).hexdigest()[:12]
            self.library.record_use(skill_id, success=success, run_id=ctx.run_id, instance_signature=signature,
                                    objective=objective_ok, environment=backend.environment,
                                    role="validation" if ctx.mode == "validation" else "execution",
                                    session_id=ctx.session_id, detail={"verdict": verdict, "arm": ctx.arm})
        self.retriever.record_feedback(plan.retrieval_id or "", {"used_skills": plan.skill_ids, "verdict": verdict})

    def _finish(self, ctx: _RunContext, backend: ExecutionBackend) -> None:
        result = ctx.result
        assert result is not None
        metrics = {
            "steps": len(result.steps), "actions_ok": sum(s.actions_ok for s in result.steps),
            "actions_failed": sum(s.actions_failed for s in result.steps), "takeovers": result.takeovers,
            "recoveries": result.recoveries, "guard_blocks": result.guard_blocks,
            "checkpoints_passed": sum(1 for r in (result.final_report.results if result.final_report else []) if r.passed),
            "checkpoints_total": len(result.final_report.results) if result.final_report else 0,
            "objective_passes": result.final_report.objective_passes if result.final_report else 0,
        }
        result.metrics.update(metrics)
        outcome = {"success": Outcome.SUCCESS, "failure": Outcome.FAILURE}.get(result.verdict, Outcome.EXECUTED_UNVERIFIED)
        self.db.execute("UPDATE runs SET status = ?, metrics = ?, ended_at = ? WHERE id = ?",
                        (result.status, dumps(result.metrics), now(), ctx.run_id))
        source = SourceClass.AGENT_SUCCESS if result.verdict == "success" else SourceClass.AGENT_FAILURE
        session = self.sessions.get(ctx.session_id)
        self.sessions.update(ctx.session_id, source=source, policy=session.policy.model_copy(update={"source": source}))
        self.sessions.finalize(ctx.session_id, outcome=outcome)
        if self.bus is not None:
            self.bus.publish(EventType.TASK_COMPLETED, ctx.run_id, verdict=result.verdict, metrics=metrics,
                             session_id=ctx.session_id)
        if self.on_session_complete is not None:
            try:
                self.on_session_complete(ctx.session_id)
            except Exception:
                log.exception("post-run processing hook failed")
