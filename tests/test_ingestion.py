from __future__ import annotations

import pytest

from lucius.app import Lucius
from lucius.ingestion import DemoStatus, MediaInput, MediaRole, analyze_reference, analyze_transition
from lucius.practice.curriculum import blade_reference
from lucius.provenance import ActionSource, EvidenceKind, SourceClass
from lucius.sessions import SessionKind
from tests.fixtures.media import UIState, draw, tutorial_script, write_video

pytest.importorskip("cv2")
pytestmark = pytest.mark.media


@pytest.fixture
def app(tmp_path):
    instance = Lucius(data_dir=tmp_path / "data", background_processing=False)
    yield instance
    instance.close()


def test_transition_analysis_distinguishes_change_kinds():
    base = UIState(mode="Edit Mode")
    scaled = analyze_transition(draw(base), draw(UIState(mode="Edit Mode", obj_w=0.04)))
    assert scaled.kind == "geometry" and scaled.candidates[0].action_type == "scale"
    orbit = analyze_transition(draw(base), draw(UIState(mode="Edit Mode", camera=0.8)))
    assert orbit.kind == "camera" and orbit.candidates[0].action_type.startswith("viewport")
    header = analyze_transition(draw(UIState()), draw(base))
    assert header.kind == "ui"
    assert len(header.candidates) > 1  # several hypotheses, none forced
    extruded = analyze_transition(draw(UIState(obj_h=0.5)), draw(UIState(obj_h=0.6)))
    assert extruded.candidates[0].action_type == "extrude"
    none = analyze_transition(draw(base), draw(base))
    assert none.kind == "none" and not none.candidates


def test_reference_measurement():
    image = blade_reference({"length": 6.0, "width": 0.3})
    analysis = analyze_reference(image)
    assert analysis["aspect_ratio"] == pytest.approx(20, rel=0.15)
    assert analysis["symmetry"] > 0.9
    assert analysis["tip_width_ratio"] < 0.7


def test_video_demonstration_pipeline(app, tmp_path):
    video = tmp_path / "tutorial_042.mp4"
    write_video(video, tutorial_script())
    ref_path = tmp_path / "sword_ref.png"
    blade_reference({"length": 5.0, "width": 0.3}).save(ref_path)
    demo = app.ingestion.create(title="Blade tutorial", task_text="simple sword blade blockout",
                                inputs=[MediaInput(path=str(video)),
                                        MediaInput(path=str(ref_path), role=MediaRole.REFERENCE, view="front")],
                                instructions=["Add a cube, then scale it along x to make it thin."])
    assert demo.source_class == SourceClass.EXTERNAL_VIDEO.value
    assert not demo.policy.training_allowed and not demo.policy.export_allowed
    result = app.ingestion.process(demo.id, validate=False)
    assert result.status == DemoStatus.READY, result.status_history[-1]
    statuses = [h["status"] for h in result.status_history]
    assert statuses[:3] == ["UPLOADED", "VALIDATING", "EXTRACTING"] and statuses[-1] == "READY"
    session = app.sessions.get(result.session_id)
    assert session.kind == SessionKind.EXTERNAL_MEDIA and session.source == SourceClass.EXTERNAL_VIDEO
    steps = app.trajectories.for_session(session.id)
    assert len(steps) >= 4
    expected = [2.0, 4.0, 6.0, 8.0, 10.0]
    for t in expected:
        assert any(abs((s.media_timestamp or -9) - t) <= 0.8 for s in steps), (t, [s.media_timestamp for s in steps])
    for step in steps:
        assert step.action_source in (ActionSource.INFERRED, ActionSource.MODEL_INFERRED)
        assert step.action_source != ActionSource.OBSERVED  # nothing in a video is directly observed input
        assert step.evidence_kind == EvidenceKind.VISUAL_STATE_TRANSITION
        assert step.frame_before_id and step.frame_after_id and step.candidate_actions
        assert step.meta["media_asset_id"]
    types = [s.action_type for s in steps]
    assert "scale" in types and any(t.startswith("viewport") for t in types)
    unknown = [s for s in steps if s.action_type == "unknown_action"]
    assert all(len(s.candidate_actions) >= 2 for s in unknown)  # ambiguity kept, not resolved by guessing
    frames = app.sessions.frames_for(session.id)
    assert frames and all(f.media_asset_id for f in frames) and all(f.media_timestamp is not None for f in frames)
    assert len(frames) < 60  # sampled, not every video frame
    constraints = app.ingestion.references.for_media(session.reference_ids)
    assert {c.constraint_type for c in constraints} >= {"silhouette", "aspect_ratio", "symmetry", "taper"}
    assert app.ingestion.references.silhouettes(session.reference_ids)
    notes = app.sessions.events(session.id, kinds=["annotation"])
    assert notes and "scale it along x" in notes[0].payload["text"]
    for skill_id in app.pipeline.status(session.id)["skills"]["result"].get("skills", []):
        skill = app.library.get(skill_id)
        assert skill.source_class == SourceClass.EXTERNAL_VIDEO.value
        assert skill.status.value in ("candidate_pattern", "candidate_skill")
        prov = app.library.provenance(skill_id)
        assert "external_video" in prov["sources"]
    # human correction of an inferred action becomes confirmed evidence
    target = next(s for s in steps if s.action_type == "scale")
    app.ingestion.confirm_action(target.id, "scale", {"axis": "x"})
    confirmed = app.trajectories.get(target.id)
    assert confirmed.action_source == ActionSource.HUMAN_CONFIRMED
    assert confirmed.action_payload["original_interpretation"]["source"] == "inferred"


def test_image_pair_and_invalid_media(app, tmp_path):
    before, after = tmp_path / "before.png", tmp_path / "after.png"
    draw(UIState(mode="Edit Mode")).save(before)
    draw(UIState(mode="Edit Mode", obj_w=0.04)).save(after)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    demo = app.ingestion.create(title="pair", task_text="thin box", inputs=[
        MediaInput(path=str(before), role=MediaRole.BEFORE), MediaInput(path=str(after), role=MediaRole.AFTER),
        MediaInput(path=str(bad), role=MediaRole.REFERENCE)])
    result = app.ingestion.process(demo.id, validate=False)
    assert result.status == DemoStatus.READY
    steps = app.trajectories.for_session(result.session_id)
    assert len(steps) == 1 and steps[0].candidate_actions[0].action_type == "scale"
    assert app.db.scalar("SELECT error FROM demonstrations WHERE id = ?", (demo.id,))  # partial failure recorded
    empty = app.ingestion.create(title="nothing", task_text=None, inputs=[MediaInput(path=str(bad))])
    failed = app.ingestion.process(empty.id)
    assert failed.status == DemoStatus.FAILED and failed.status_history[-1]["error"]["stage"] == "VALIDATING"


def test_compressed_blend_files_are_recognised():
    from lucius.ingestion.media import MediaKind, detect_kind

    assert detect_kind("scene.blend", b"BLENDER-v405") == MediaKind.BLENDER_PROJECT
    assert detect_kind("scene.blend", b"\x28\xb5\x2f\xfd\x60\x19") == MediaKind.BLENDER_PROJECT  # zstd (Blender 5)
    assert detect_kind("scene.blend", b"\x1f\x8b\x08\x00") == MediaKind.BLENDER_PROJECT  # gzip (older versions)
    with pytest.raises(Exception, match="unsupported"):
        detect_kind("archive.zst", b"\x28\xb5\x2f\xfd")  # zstd alone is not a Blender project


@pytest.mark.bpy
def test_blend_project_and_written_instructions(app, tmp_path):
    from lucius.blender.headless import HeadlessBlender, headless_available

    if headless_available() is None:
        pytest.skip("needs Blender")
    project = tmp_path / "crate.blend"
    with HeadlessBlender(allowed_save_dirs=[str(tmp_path)]) as bridge:
        bridge.execute("reset_scene", {"keep_camera_light": True})
        bridge.execute("add_primitive", {"kind": "cube", "name": "Crate"})
        bridge.execute("set_dimensions", {"object": "Crate", "dimensions": [1.2, 0.8, 0.6]})
        bridge.execute("save_file", {"path": str(project)})
    demo = app.ingestion.create(title="crate", task_text="wooden crate", instructions=["Add a cube", "Scale it flat"],
                                inputs=[MediaInput(path=str(project), role=MediaRole.PROJECT)])
    assert demo.source_class == SourceClass.EXTERNAL_PROJECT.value
    result = app.ingestion.process(demo.id, validate=False)
    assert result.status == DemoStatus.READY, result.status_history[-1]
    session = app.sessions.get(result.session_id)
    constraints = app.ingestion.references.for_media(session.meta["media"])
    dims = next(c for c in constraints if c.constraint_type == "project_dimensions" and c.target == "crate")
    assert dims.source == "measured" and dims.value["dimensions"] == pytest.approx({"x": 1.2, "y": 0.8, "z": 0.6}, abs=1e-4)
    steps = app.trajectories.for_session(result.session_id)
    assert any(s.evidence_kind == EvidenceKind.TEXT_INSTRUCTION for s in steps)
    assert not session.policy.training_allowed  # imported material is never silently trainable
