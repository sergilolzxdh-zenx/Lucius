from __future__ import annotations

import json

import pytest

from lucius.app import Lucius
from lucius.errors import ConflictError, PolicyViolation
from lucius.events import EventType
from lucius.provenance import DataPolicy
from lucius.sessions import Outcome, SessionKind
from lucius.training import TrainerRegistry, TrainingAdvisor, TrainingJobSpec, write_training_files
from tests.fixtures.demos import sword_blockout_demo


@pytest.fixture
def app(tmp_path):
    instance = Lucius(data_dir=tmp_path / "data", background_processing=False)
    yield instance
    instance.close()


def _record(app, demo, policy):
    session = app.sessions.create(user_id="local", kind=SessionKind.LIVE_DEMO, policy=policy,
                                  task_text="simple sword blockout", start_time=demo.events[0].ts, meta={"capture_sources": {"blender_bridge": True}})
    app.sessions.append_events(session.id, demo.events)
    app.sessions.finalize(session.id, end_time=demo.events[-1].ts, outcome=Outcome.SUCCESS)
    app.bus.publish(EventType.SESSION_ENDED, session.id)
    return session.id


def test_training_dataset_respects_consent_and_exports_provenance(app, tmp_path):
    consented = _record(app, sword_blockout_demo(t0=1_700_000_000.0), DataPolicy.for_live_demo(training_consent=True))
    private = _record(app, sword_blockout_demo(t0=1_700_100_000.0), DataPolicy.for_live_demo())
    result = app.datasets.build("sword demos", purpose="training")
    assert result.samples == 1
    assert private in result.excluded["training_not_allowed"]
    line = json.loads(open(f"{result.path}/samples.jsonl").readline())
    assert line["session_id"] == consented and line["consent"] == "granted"
    assert line["provenance"]["source"] == "user_demo" and line["trajectory_steps"] and line["segments"]
    assert any(f["future_rule"] for f in line["failure_records"])
    files = write_training_files(result.path, tmp_path / "train")
    assert files["behaviour_cloning"] > 0 and files["behaviour_cloning"] == files["sft"]
    pairs = [json.loads(x) for x in open(tmp_path / "train" / "bc_pairs.jsonl")]
    assert all(p["provenance"]["source"] in ("observed", "human_confirmed") for p in pairs)
    assert not any(p["action"]["action_type"] == "bevel" for p in pairs)  # undone work is never a label
    spec = TrainingJobSpec(strategy="sft", dataset_dir=result.path)
    with pytest.raises(Exception) as exc:
        TrainerRegistry().launch(spec)
    assert "no trainer backend" in str(exc.value)
    advice = TrainingAdvisor(app.db).assess()
    assert advice["recommendation"] == "not_recommended" and "never launched" in advice["note"]


def test_quality_validation_detects_problems(app):
    sid = _record(app, sword_blockout_demo(), DataPolicy.for_live_demo(training_consent=True))
    dup = _record(app, sword_blockout_demo(), DataPolicy.for_live_demo(training_consent=True))
    report = app.datasets.validate([sid, dup])
    codes = {i.code for s in report.sessions for i in s.issues}
    assert "duplicate_session" in codes
    assert report.mean_score < 1.0


def test_session_and_skill_bundles_roundtrip(app, tmp_path):
    sid = _record(app, sword_blockout_demo(), DataPolicy.for_live_demo())
    bundle = app.datasets.export_session(sid)
    imported = app.datasets.import_session(bundle)
    assert imported.id != sid and imported.meta["imported_from"] == sid
    assert len(app.sessions.events(imported.id)) == len(app.sessions.events(sid))
    assert len(app.trajectories.for_session(imported.id)) == len(app.trajectories.for_session(sid))
    external = app.sessions.create(user_id="local", kind=SessionKind.EXTERNAL_MEDIA,
                                   policy=DataPolicy.for_external(__import__("lucius.provenance").provenance.SourceClass.EXTERNAL_VIDEO))
    with pytest.raises(PolicyViolation):
        app.datasets.export_session(external.id)
    path = app.datasets.export_skill("hard_surface_blade_blockout")
    with pytest.raises(ConflictError):
        app.datasets.import_skill(path)
    new_id = app.datasets.import_skill(path, as_new=True)
    copy = app.library.get(new_id)
    assert copy.status.value == "candidate_pattern" and copy.definition.phases
