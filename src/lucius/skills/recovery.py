"""Learning recovery procedures from human takeovers (sections 43, 95).

After an agent run is processed, the human's actions inside each takeover window are the
correction for the agent state that required it. They are attached to the correction record,
and turned into a recovery action on the skill (as a new version) -- so the next plan for the
same skill carries that recovery path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lucius.corrections import CorrectionStore
from lucius.provenance import SourceClass
from lucius.sessions.models import Session
from lucius.skills.extract import Unit, _CandidateBuilder
from lucius.skills.library import SkillLibrary
from lucius.skills.schema import RecoveryAction
from lucius.taxonomy import classify_task
from lucius.trajectory import vocabulary as vocab
from lucius.trajectory.model import TrajectoryStep


@dataclass
class RecoveryLearningResult:
    corrections_updated: list[str] = field(default_factory=list)
    skills_updated: list[str] = field(default_factory=list)


def learn_from_takeovers(session: Session, steps: list[TrajectoryStep], takeovers: list[dict],
                         corrections: CorrectionStore, library: SkillLibrary) -> RecoveryLearningResult:
    result = RecoveryLearningResult()
    run_corrections = {c.start_time: c for c in corrections.list(run_id=None, limit=5000) if c.session_id == session.id}
    object_class, categories = classify_task(session.task_text)
    for takeover in takeovers:
        start, end = takeover.get("start_ts"), takeover.get("end_ts")
        if start is None or end is None:
            continue
        human_steps = [s for s in steps if s.actor == "human" and start <= s.t_start <= end
                       and vocab.mutates(s.action_type) and not s.meta.get("cancelled")]
        correction = min(run_corrections.values(), key=lambda c: abs(c.start_time - start), default=None)
        if correction is None or abs(correction.start_time - start) > 5.0:
            continue
        corrections.attach_steps(correction.id, [s.id for s in human_steps])
        result.corrections_updated.append(correction.id)
        context = takeover.get("agent_context") or correction.agent_context
        skill_id = context.get("skill_id")
        if not skill_id or not human_steps or not library.exists(skill_id) or correction.outcome == "failure":
            continue
        skill = library.get(skill_id)
        definition = skill.definition.model_copy(deep=True)
        builder = _CandidateBuilder(session, Unit(), steps, definition.object_role, context.get("params", {}).get(
            "object_name"), object_class, categories)
        templates = [t for t in (builder.template(s) for s in human_steps) if t is not None]
        for t in templates:
            # A correction applies to *this* state, so it keeps the concrete values the human used.
            t.args = {k: (builder.params[v.strip("{}")].default if isinstance(v, str) and v.startswith("{")
                          and v.strip("{}") in builder.params else v) for k, v in t.args.items()}
        failed = context.get("failed_checkpoints") or []
        recovery_id = f"rc_human_{'_'.join(sorted(failed)) or context.get('phase', 'step')}"
        existing = next((r for r in definition.recovery_actions if r.id == recovery_id), None)
        if existing is not None:
            existing.evidence = sorted(set(existing.evidence) | {correction.id})
            existing.actions = templates or existing.actions
        else:
            definition.recovery_actions.append(RecoveryAction(
                id=recovery_id, description=f"human correction when {', '.join(failed) or 'the step'} failed: "
                                            + ", ".join(s.describe() for s in human_steps),
                when=None, when_checkpoints=sorted(failed), actions=templates,
                source=SourceClass.HUMAN_CORRECTION.value, evidence=[correction.id]))
            for fc in definition.failure_conditions:
                if fc.guard_checkpoint in failed:
                    definition.recovery_actions[-1].when = fc.id
        library.new_version(skill_id, definition, change_note=f"recovery learned from takeover {correction.id}",
                            created_by="dagger")
        library.graph.link(("skill", skill_id), "modified_by", ("correction", correction.id))
        result.skills_updated.append(skill_id)
    return result
