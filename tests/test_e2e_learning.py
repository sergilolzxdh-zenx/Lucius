"""End-to-end learning loop (sections 69 and 95) against real headless Blender.

    record demonstration -> trajectory -> segments -> intents -> skill candidate -> retrieve ->
    execute -> evaluate -> failure -> human takeover -> correction -> recovery rule ->
    next attempt recovers without a human.
"""

from __future__ import annotations

import pytest

from lucius.app import Lucius
from lucius.events import EventType
from lucius.executor import TakeoverOutcome
from lucius.planner.model import PlanAction
from lucius.provenance import DataPolicy
from lucius.sessions import SessionKind
from lucius.skills import SkillStatus
from tests.conftest import HAS_BPY
from tests.fixtures.demos import sword_blockout_demo

pytestmark = [pytest.mark.bpy, pytest.mark.skipif(not HAS_BPY, reason="needs Blender (bpy module or binary)")]


def record(app: Lucius, demo, task: str = "simple sword blockout") -> str:
    """Store a demonstration exactly as the recorder does, then end it (triggers processing)."""
    session = app.sessions.create(user_id=app.config.user_id, kind=SessionKind.LIVE_DEMO,
                                  policy=DataPolicy.for_live_demo(training_consent=True), task_text=task,
                                  meta={"capture_sources": {"blender_bridge": True}})
    app.sessions.append_events(session.id, demo.events)
    app.sessions.finalize(session.id, end_time=demo.events[-1].ts)
    app.bus.publish(EventType.SESSION_ENDED, session.id, kind="live_demo")
    return session.id


def references(app: Lucius, length: float, width: float, **blade_shape: float) -> tuple[list, list[str]]:
    ids = [app.practice._reference("blade", {"length": length, "width": width, **blade_shape}),
           app.practice._reference("box", {"width": 1.6, "height": 0.2}, target="guard")]
    return app.ingestion.references.silhouettes(ids), ids


class ScriptedHuman:
    """Development fixture standing in for a person at the keyboard during a takeover."""

    def __init__(self, app: Lucius, backend, corrections: list[PlanAction] | None) -> None:
        self.app, self.backend, self.corrections = app, backend, corrections
        self.requests = []

    def available(self) -> bool:
        return True

    def request(self, request):
        self.requests.append(request)
        if self.corrections is None:
            raise AssertionError("the agent asked for a human it should no longer need")
        for action in self.corrections:
            result = self.app.engine.record_human_action(request.run_id, self.backend, action)
            assert result.ok, result.error
        return TakeoverOutcome(resumed=True, reason=None)  # no reason given -> none stored


def ring(lo: float, hi: float, factor: float) -> list[PlanAction]:
    return [PlanAction(id=f"h{lo}", layer="blender_api", name="select_elements", action_type="select_elements",
                       args={"object": "Blade", "axis": "z", "min": lo, "max": hi, "space": "normalized"}),
            PlanAction(id=f"s{lo}", layer="blender_api", name="scale_selection", action_type="scale",
                       args={"object": "Blade", "factor": [factor, 1.0, 1.0], "pivot": "median"})]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    instance = Lucius(data_dir=tmp_path_factory.mktemp("e2e") / "data", background_processing=False)
    yield instance
    instance.close()


def test_full_learning_loop(app):
    # 1. WATCH ME: two demonstrations of a sword blockout (one with a premature bevel that gets undone).
    s1 = record(app, sword_blockout_demo(blade_length=6.0, blade_width=0.3, variant="a", t0=1_700_000_000.0))
    s2 = record(app, sword_blockout_demo(blade_length=4.0, blade_width=0.24, variant="b", with_mistake=False,
                                         t0=1_700_100_000.0))
    status = app.pipeline.status(s1)
    assert all(status[stage]["status"] in ("done", "skipped") for stage in status), status
    assert status["refinement"]["status"] == "skipped"  # no model configured: deterministic labels stand
    labels = [s.label for s in app.segments.for_session(s1)]
    assert {"primary_blockout", "inspection", "corrective_pass", "verification"} <= set(labels)
    corrective = next(s for s in app.segments.for_session(s1) if s.label == "corrective_pass")
    assert app.intents.primary(corrective.id).category == "correct"
    blade = app.library.get("hard_surface_blade_blockout")
    assert blade.status == SkillStatus.CANDIDATE_SKILL and blade.source_class == "user_demo"
    bevel_rule = next(f for f in app.failures.list() if f.trigger_action == "bevel")
    assert bevel_rule.future_rule.startswith("Do not add bevel") and bevel_rule.symptoms == ["the tip is too broad"]
    assert app.episodes.get(s1) is not None and app.episodes.get(s2) is not None
    assert app.preferences.get_all()["orthographic_checks"].value is True

    backend = app.headless_backend()

    # 2. "Make another sword, but use what I taught you."
    refs, ref_ids = references(app, 5.0, 0.28)
    run1 = app.engine.run("Make another sword with blade length 5 and blade width 0.28, use what I taught you",
                          backend, references=refs, reference_ids=ref_ids, reset_scene=True)
    assert run1.verdict == "success", run1.final_report.summary
    assert set(run1.plan.skill_ids) == {"hard_surface_blade_blockout", "hard_surface_guard_blockout"}
    record_row = app.db.query_one("SELECT results FROM retrieval_records WHERE run_id = ?", (run1.run_id,))
    assert bevel_rule.id in record_row["results"]  # the relevant failure rule was retrieved
    dims = {o["name"]: o["dimensions"] for o in backend.structure()["objects"]}
    assert dims["Blade"][2] == pytest.approx(5.0, rel=0.02) and dims["Blade"][0] == pytest.approx(0.28, rel=0.02)
    assert app.library.get("hard_surface_blade_blockout").status == SkillStatus.VALIDATED
    transitions = [r["to_state"] for r in app.db.query("SELECT to_state FROM run_transitions WHERE run_id = ?"
                                                       " ORDER BY seq", (run1.run_id,))]
    assert transitions[:3] == ["OBSERVE", "PLAN", "EXECUTE"] and transitions[-1] == "SUCCESS"

    # 3. A target whose blade tapers from low down: the learned procedure fails its silhouette check,
    #    automatic retry fails too, and a human takes over and corrects the taper.
    long_taper = {"tip_start": 0.1, "tip_width": 0.1}
    refs, ref_ids = references(app, 5.0, 0.28, **long_taper)
    human = ScriptedHuman(app, backend, ring(0.30, 0.37, 0.767) + ring(0.63, 0.70, 0.433) + ring(0.95, 1.0, 0.5))
    run2 = app.engine.run("Make another sword with blade length 5 and blade width 0.28", backend,
                          references=refs, reference_ids=ref_ids, human=human, reset_scene=True)
    assert len(human.requests) == 1 and run2.takeovers == 1
    assert human.requests[0].failed_checkpoints == ["blade_secondary_form_silhouette"]
    assert run2.verdict == "success", run2.final_report.summary
    correction = app.corrections.get(run2.correction_ids[0])
    assert correction.reason is None  # the human gave no reason, so none was invented
    assert correction.outcome == "success" and len(correction.correction_steps) == 3
    silhouette_failure = app.failures.get(correction.failure_id)
    assert silhouette_failure.correction_successes == 1 and silhouette_failure.rule_status == "candidate"
    learned = app.library.get("hard_surface_blade_blockout").definition
    recovery = next(r for r in learned.recovery_actions if r.source == "human_correction" and r.when_checkpoints)
    assert recovery.when_checkpoints == ["blade_secondary_form_silhouette"]
    assert [a.selection.kind for a in recovery.actions] == ["region", "region", "region"]
    assert "wx" not in recovery.model_dump_json()  # no screen coordinates were learned

    # 4. Next attempt at a different size with the same kind of target: the learned recovery fixes it
    #    without asking the human, and the repeated successful correction promotes the rule.
    refs, ref_ids = references(app, 6.5, 0.32, **long_taper)
    unneeded = ScriptedHuman(app, backend, None)
    run3 = app.engine.run("Make another sword with blade length 6.5 and blade width 0.32", backend,
                          references=refs, reference_ids=ref_ids, human=unneeded, reset_scene=True)
    assert run3.verdict == "success", run3.final_report.summary
    assert run3.takeovers == 0 and run3.recoveries == 1 and not unneeded.requests
    promoted = app.failures.get(silhouette_failure.id)
    assert promoted.occurrence_count == 2 and promoted.correction_successes == 2
    assert promoted.rule_status == "promoted"
    assert promoted.retrieval_priority > silhouette_failure.retrieval_priority

    # Provenance: the skill traces back to the demonstrations and the correction that modified it.
    prov = app.library.provenance("hard_surface_blade_blockout")
    demo_sessions = {e["session_id"] for e in prov["sources"]["user_demo"]}
    assert demo_sessions == {s1, s2}
    assert any(e["rel"] == "modified_by" for e in prov["graph"]["edges"])
