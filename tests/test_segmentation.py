from __future__ import annotations

import pytest

from lucius.errors import ValidationError
from lucius.intent import IntentEngine
from lucius.provenance import DataPolicy
from lucius.segmentation import SegmentEditor, Segmenter, SegmentStore
from lucius.sessions import SessionKind
from lucius.taxonomy import Taxonomy
from lucius.trajectory import TrajectoryBuilder, TrajectoryStore
from tests.fixtures.demos import DemoScript, sword_blockout_demo


@pytest.fixture
def pipeline(db, sessions):
    taxonomy = Taxonomy(db)
    taxonomy.seed()
    trajectories = TrajectoryStore(db)
    segments = SegmentStore(db)
    return {
        "taxonomy": taxonomy, "trajectories": trajectories, "segments": segments,
        "segmenter": Segmenter(segments, trajectories),
        "editor": SegmentEditor(db, segments, trajectories, taxonomy),
        "intents": IntentEngine(db, taxonomy),
    }


def _process(sessions, pipeline, demo, task="simple sword blockout"):
    session = sessions.create(user_id="u", kind=SessionKind.LIVE_DEMO, policy=DataPolicy.for_live_demo(),
                              task_text=task)
    sessions.append_events(session.id, demo.events)
    built = TrajectoryBuilder().build(session.id, demo.events, [], operator_log_available=True)
    pipeline["trajectories"].replace(session.id, built.steps)
    segs = pipeline["segmenter"].segment(session.id, built.steps, [], built.annotations)
    return session, built, segs


def test_sword_demo_segments_into_meaningful_phases(sessions, pipeline):
    _session, built, segs = _process(sessions, pipeline, sword_blockout_demo())
    labels = [s.label for s in segs]
    assert labels == ["scene_setup", "primary_blockout", "inspection", "corrective_pass", "inspection",
                      "primary_blockout", "verification"]
    corrective = segs[3]
    assert corrective.outcome == "corrected"
    assert "bevel too early - the tip is too broad" in corrective.meta["annotations"]
    assert all(s.label_evidence for s in segs)
    # every step is assigned to exactly one segment
    steps = pipeline["trajectories"].for_session(segs[0].session_id)
    assert all(s.segment_id for s in steps)


def test_pause_boundary_and_undo_boundary(sessions, pipeline):
    d = DemoScript()
    d.add_cube("Box").tab("EDIT_MESH").scale("x", 0.5).scale("y", 0.5).wait(8).scale("z", 2).extrude("z", 1)
    _s, _b, segs = _process(sessions, pipeline, d, task="box")
    assert len(segs) == 2 and any(b.code == "pause" for b in segs[1].boundary_reasons)


def test_manual_edits_are_preserved_on_resegmentation(sessions, pipeline):
    session, built, segs = _process(sessions, pipeline, sword_blockout_demo())
    editor = pipeline["editor"]
    merged = editor.merge([segs[1].id, segs[2].id], label="blade_blockout_with_check")
    left, right = editor.split(segs[5].id, segs[5].step_start + 2)
    editor.relabel(right.id, "guard shaping")
    editor.set_outcome(segs[0].id, "success")
    with pytest.raises(ValidationError):
        editor.merge([segs[0].id, segs[3].id])
    again = pipeline["segmenter"].segment(session.id, built.steps, [], built.annotations)
    by_id = {s.id: s for s in again}
    assert by_id[merged.id].label == "blade_blockout_with_check" and by_id[merged.id].locked
    assert by_id[right.id].label == "guard_shaping"
    assert "guard_shaping" in pipeline["taxonomy"].terms("segment_label")
    assert pipeline["taxonomy"].terms("segment_label")["guard_shaping"]["source"] == "human"
    # coverage stays complete and non-overlapping
    covered = sorted(i for s in again for i in range(s.step_start, s.step_end + 1))
    assert covered == list(range(len(built.steps)))
    edits = pipeline["segments"].db.query("SELECT op FROM human_edits")
    assert {r["op"] for r in edits} >= {"merge", "split", "relabel", "set_outcome"}


def test_intents_are_structured_hypotheses(sessions, pipeline):
    _session, built, segs = _process(sessions, pipeline, sword_blockout_demo())
    engine = pipeline["intents"]
    results = {}
    for seg in segs:
        hyps = engine.hypotheses(seg, built.steps[seg.step_start:seg.step_end + 1], "blade_weapon")
        engine.store(seg.id, hyps)
        results[seg.label] = engine.primary(seg.id)
    assert results["inspection"].category == "inspect" and results["inspection"].target == "silhouette"
    assert results["verification"].category == "verify"
    corrective = results["corrective_pass"]
    assert corrective.category == "correct"
    assert corrective.confidence < 1.0
    corrective_all = engine.for_segment(segs[3].id)
    assert any(i.evidence_source == "annotation" and i.target == "tip_profile" for i in corrective_all)
    assert any("undo reverted bevel" in e for i in corrective_all for e in i.evidence)
    blockout = engine.for_segment(segs[1].id)
    assert any(i.target in ("blade_width", "blade_length") for i in blockout)
    human = engine.set_human(segs[1].id, category="create", target="blade_silhouette", scope="Blade")
    engine.store(segs[1].id, engine.hypotheses(segs[1], built.steps[segs[1].step_start:segs[1].step_end + 1],
                                               "blade_weapon"))
    assert engine.primary(segs[1].id).id == human.id
