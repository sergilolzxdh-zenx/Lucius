from __future__ import annotations

import pytest

from lucius.errors import ValidationError
from lucius.memory.failure import FailureEvidence
from lucius.memory.preferences import WorkflowPreferences
from lucius.memory.semantic import SemanticMemory
from lucius.provenance import SourceClass
from lucius.skills import SkillStatus
from lucius.skills.seeds import seed_system_skills
from tests.fixtures.demos import sword_blockout_demo
from tests.helpers import MiniPipeline


@pytest.fixture
def pipe(db, sessions):
    return MiniPipeline(db, sessions)


def test_single_demo_yields_candidate_pattern_with_structure(pipe):
    _s, _b, _segs, result, _ep = pipe.run(sword_blockout_demo())
    assert result.created == ["hard_surface_blade_blockout", "hard_surface_guard_blockout"]
    blade = pipe.library.get("hard_surface_blade_blockout")
    assert blade.status == SkillStatus.CANDIDATE_PATTERN  # one demo is not a general skill
    d = blade.definition
    assert [p.name for p in d.phases][:2] == ["setup", "primary_form"]
    scale_args = [a.args for p in d.phases for a in p.actions if a.action_type == "scale"]
    assert {"axis": "x", "target_size": "{blade_width}"} in scale_args  # parameterised by result, not factor
    assert all("x" not in a or isinstance(a, dict) for a in scale_args)
    assert {c.check["type"] for c in d.checkpoints} >= {"object_exists", "dimensions_match", "symmetry", "silhouette",
                                                         "taper"}
    assert d.failure_conditions and d.failure_conditions[0].trigger_action == "bevel"
    assert d.recovery_actions[0].actions[0].action_type == "restore"
    # no screen coordinates anywhere in the procedure
    assert "wx" not in d.model_dump_json() and '"x": 800' not in d.model_dump_json()


def test_generalization_separates_invariants_and_parameters(pipe):
    pipe.run(sword_blockout_demo(blade_length=6.0, blade_width=0.3, variant="a"))
    pipe.run(sword_blockout_demo(blade_length=4.0, blade_width=0.24, variant="b", with_mistake=False))
    blade = pipe.library.get("hard_surface_blade_blockout")
    assert blade.status == SkillStatus.CANDIDATE_SKILL
    params = {p.name: p for p in blade.definition.parameters}
    assert params["blade_length"].range == (4.0, 6.0) and not params["blade_length"].invariant
    assert params["thickness"].invariant
    assert params["blade_loop_cuts"].range == (2.0, 3.0)
    assert blade.current_version >= 2
    history = pipe.library.versions(blade.id)
    assert history[0]["created_by"] == "extractor" and history[-1]["created_by"] == "generalizer"
    prov = pipe.library.provenance(blade.id)
    assert len(prov["sources"]["user_demo"]) == 2
    assert prov["confidence"] == blade.confidence and "demonstration" in prov["confidence_explanation"]


def test_reprocessing_is_idempotent(pipe):
    demo = sword_blockout_demo()
    session, built, segments, _r, _e = pipe.run(demo)
    before = pipe.library.get("hard_surface_blade_blockout")
    pipe.extractor.extract(pipe.sessions.get(session.id), built.steps, segments)
    after = pipe.library.get("hard_surface_blade_blockout")
    assert len(pipe.library.examples(after.id)) == len(pipe.library.examples(before.id))
    assert pipe.failures.list()[0].occurrence_count == 1


def test_versioning_rollback_edit_disable_merge_split(pipe):
    pipe.run(sword_blockout_demo())
    lib = pipe.library
    skill = lib.human_edit("hard_surface_blade_blockout", {"name": "Longsword blade blockout"})
    assert skill.definition.name == "Longsword blade blockout"
    v_edit = skill.current_version
    rolled = lib.rollback(skill.id, 1)
    assert rolled.current_version == v_edit + 1 and rolled.definition.name == "Blade blockout"
    assert len(lib.versions(skill.id)) == v_edit + 1  # history kept
    with pytest.raises(ValidationError):
        lib.human_edit(skill.id, {"skill_id": "other"})
    child = lib.split(skill.id, ["setup"], new_name="Blade scene setup")
    assert child.parent_skill_id == skill.id and [p.name for p in child.definition.phases] == ["setup"]
    disabled = lib.set_disabled("hard_surface_guard_blockout", True)
    assert disabled.status == SkillStatus.DISABLED
    assert "hard_surface_guard_blockout" not in {s.id for s in lib.list()}
    lib.set_disabled("hard_surface_guard_blockout", False)
    merged = lib.merge("hard_surface_guard_blockout", skill.id)
    assert lib.get("hard_surface_guard_blockout").status == SkillStatus.MERGED
    assert len(lib.examples(merged.id)) == 2


def test_promotion_requires_objective_cross_instance_success(pipe):
    pipe.run(sword_blockout_demo())
    lib = pipe.library
    sid = "hard_surface_blade_blockout"
    lib.record_use(sid, success=True, run_id=None, instance_signature="a", objective=False, environment="blender_headless")
    assert lib.get(sid).status == SkillStatus.CANDIDATE_PATTERN  # unverified success does not validate
    lib.record_use(sid, success=True, run_id=None, instance_signature="a", objective=True, environment="simulation")
    assert lib.get(sid).status == SkillStatus.CANDIDATE_PATTERN  # simulated evidence does not validate
    lib.record_use(sid, success=True, run_id=None, instance_signature="a", objective=True, environment="blender_headless")
    assert lib.get(sid).status == SkillStatus.VALIDATED
    lib.record_use(sid, success=True, run_id=None, instance_signature="a", objective=True, environment="blender_headless")
    lib.record_use(sid, success=True, run_id=None, instance_signature="a", objective=True, environment="blender_headless")
    assert lib.get(sid).status == SkillStatus.VALIDATED  # same instance only: not generalised
    lib.record_use(sid, success=True, run_id=None, instance_signature="b", objective=True, environment="blender_headless")
    skill = lib.get(sid)
    assert skill.status == SkillStatus.HIGH_CONFIDENCE
    for _ in range(6):
        lib.record_use(sid, success=False, run_id=None, instance_signature="c", objective=True,
                       environment="blender_headless")
    assert lib.get(sid).status.rank < SkillStatus.HIGH_CONFIDENCE.rank  # degraded success demotes


def test_failure_memory_repetition_and_promotion(pipe):
    pipe.run(sword_blockout_demo(t0=1_700_000_000.0))
    rec = pipe.failures.list()[0]
    assert rec.rule_status == "candidate" and rec.symptoms == ["the tip is too broad"]
    assert rec.future_rule.startswith("Do not add bevel until")
    pipe.run(sword_blockout_demo(t0=1_700_100_000.0))
    again = pipe.failures.get(rec.id)
    assert again.occurrence_count == 2 and again.retrieval_priority > rec.retrieval_priority
    for success in (True, True):
        again = pipe.failures.record_correction_outcome(rec.id, success=success, evidence=FailureEvidence(
            kind="x", source_class=SourceClass.AGENT_SUCCESS.value, run_id="run_1"))
    assert again.rule_status == "promoted"
    rejected = pipe.failures.review_rule(rec.id, accept=False)
    assert rejected.rule_status == "rejected" and rejected.confidence < again.confidence
    assert pipe.failures.relevant(task_class="blade_weapon", action_types={"bevel"}) == []


def test_semantic_memory_needs_repetition(pipe, db):
    semantic = SemanticMemory(db)
    pipe.run(sword_blockout_demo(t0=1_700_000_000.0))
    assert semantic.mine(pipe.episodes.list(), pipe.failures.list()) == []  # one episode: nothing general
    pipe.run(sword_blockout_demo(t0=1_700_100_000.0))
    statements = {s.pattern_key: s for s in semantic.mine(pipe.episodes.list(), pipe.failures.list())}
    assert "inspect_views:blade_weapon" in statements
    assert statements["inspect_views:blade_weapon"].support_count == 2
    assert any(k.startswith("rule:") for k in statements)
    assert all(s.status == "candidate" for s in statements.values())
    pipe.run(sword_blockout_demo(t0=1_700_200_000.0))
    statements = {s.pattern_key: s for s in semantic.mine(pipe.episodes.list(), pipe.failures.list())}
    assert statements["inspect_views:blade_weapon"].status == "validated"


def test_preferences_learn_from_user_only_and_respect_human(pipe, db):
    prefs = WorkflowPreferences(db, "u")
    pipe.run(sword_blockout_demo(t0=1_700_000_000.0))
    pipe.run(sword_blockout_demo(t0=1_700_100_000.0), source=SourceClass.EXTERNAL_VIDEO)
    learned = prefs.learn(pipe.episodes.list())
    assert learned["orthographic_checks"].value is True and learned["orthographic_checks"].support == 1
    assert learned["bevel_late"].value is True
    prefs.set("orthographic_checks", False)
    pipe.run(sword_blockout_demo(t0=1_700_200_000.0))
    assert prefs.learn(pipe.episodes.list())["orthographic_checks"].value is False  # human setting wins


def test_seeded_skills_are_marked(pipe):
    created = seed_system_skills(pipe.library)
    assert "seed_extrude" in created and seed_system_skills(pipe.library) == []
    seed = pipe.library.get("seed_extrude")
    assert seed.source_class == "system_seeded" and seed.status == SkillStatus.CANDIDATE_SKILL
    assert seed.confidence == pytest.approx(0.5)
