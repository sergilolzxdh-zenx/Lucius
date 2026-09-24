"""HTTP API: security guards, the learning/editing endpoints, and real runs through background jobs."""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from lucius.app import Lucius
from lucius.api import create_app
from tests.conftest import HAS_BPY
from tests.test_e2e_learning import record
from tests.fixtures.demos import sword_blockout_demo

TOKEN = "test-token"
BASE = "http://127.0.0.1"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    instance = Lucius(data_dir=tmp_path_factory.mktemp("api") / "data", background_processing=False)
    record(instance, sword_blockout_demo(blade_length=6.0, blade_width=0.3, variant="a", t0=1_700_000_000.0))
    record(instance, sword_blockout_demo(blade_length=4.0, blade_width=0.24, variant="b", with_mistake=False,
                                         t0=1_700_100_000.0))
    yield instance
    instance.close()


@pytest.fixture(scope="module")
def client(app):
    with TestClient(create_app(app, token=TOKEN), base_url=BASE, headers={"X-Lucius-Token": TOKEN}) as c:
        yield c


def wait_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish")


def test_guards_reject_foreign_hosts_and_missing_token(app, client):
    assert client.get("/api/status", headers={"host": "evil.example"}).status_code == 403
    unauthenticated = TestClient(create_app(app, token=TOKEN), base_url=BASE)
    assert unauthenticated.post("/api/watch/stop", json={}).status_code == 401
    assert unauthenticated.get("/api/status").status_code == 200  # reads need no token
    page = client.get("/")
    assert page.status_code == 200 and TOKEN in page.text and page.headers["x-frame-options"] == "DENY"
    assert client.get("/static/app.js").status_code == 200


def test_status_and_errors_are_structured(client):
    status = client.get("/api/status").json()
    assert status["counts"]["demonstrations"] == 2 and status["takeover"] is None
    missing = client.get("/api/skills/nope")
    assert missing.status_code == 404 and missing.json()["code"] == "not_found" or missing.json()["message"]
    assert client.post("/api/watch/stop", json={}).status_code == 409  # nothing is recording


def test_timeline_and_human_edits(client):
    sessions = client.get("/api/sessions?kind=live_demo").json()
    assert len(sessions) == 2
    sid = sessions[-1]["id"]
    tl = client.get(f"/api/sessions/{sid}/timeline").json()
    assert tl["steps"] and tl["segments"] and any(m["kind"] == "undo" for m in tl["markers"])
    seg = tl["segments"][0]
    edited = client.post(f"/api/segments/{seg['id']}/relabel", json={"label": seg["label"], "title": "Scene setup"}).json()
    assert edited["locked"] and edited["origin"] == "human" and edited["title"] == "Scene setup"
    intent = client.post(f"/api/segments/{seg['id']}/intent", json={"category": "setup", "target": None}).json()
    assert intent["evidence_source"] == "human"
    # Opting a session out of training is immediate and visible.
    policy = client.patch(f"/api/sessions/{sid}/policy", json={"training_consent": False}).json()["policy"]
    assert policy["training_allowed"] is False and policy["consent_status"] == "denied"
    bundle = client.get(f"/api/sessions/{sid}/export")
    assert bundle.status_code == 200 and bundle.content[:2] == b"PK"


def test_skills_memory_failures_retrieval(client):
    skills = client.get("/api/skills").json()
    blade = next(s for s in skills if s["id"] == "hard_surface_blade_blockout")
    detail = client.get(f"/api/skills/{blade['id']}").json()
    assert detail["versions"] and detail["examples"] and detail["provenance"]
    edited = client.patch(f"/api/skills/{blade['id']}", json={"changes": {"notes": ["checked by me"]}}).json()
    assert edited["version"] == blade["version"] + 1
    back = client.post(f"/api/skills/{blade['id']}/rollback", json={"version": blade["version"]}).json()
    assert back["version"] == blade["version"] + 2
    assert client.patch(f"/api/skills/{blade['id']}", json={"changes": {"skill_id": "x"}}).status_code == 422

    failures = client.get("/api/failures").json()
    assert failures and failures[0]["future_rule"]
    fid = failures[0]["id"]
    assert client.get(f"/api/failures/{fid}").json()["failure"]["id"] == fid
    assert client.patch(f"/api/failures/{fid}", json={"changes": {"likely_cause": "bevel before taper"}}).json()[
        "likely_cause"] == "bevel before taper"
    assert client.patch(f"/api/failures/{fid}", json={"changes": {"rule_status": "accepted"}}).status_code == 422

    ret = client.post("/api/retrieval/debug", json={"text": "make another sword blockout"}).json()
    assert ret["skills"][0]["id"].startswith("hard_surface") and ret["skills"][0]["reason_codes"]
    assert client.get("/api/memory/episodes").json()
    graph = client.get("/api/graph").json()
    assert graph["nodes"] and graph["edges"]
    assert client.get("/api/practice").json()[0]["overview"]["stages"]
    assert client.get("/api/benchmarks").json()["unavailable_arms"]["trained_policy"]
    assert client.get("/api/training").json()["advisor"]["recommendation"]


def test_settings_validate_and_persist(app, client):
    assert client.put("/api/settings", json={"recording": {"fps": 500}}).status_code == 422
    ok = client.put("/api/settings", json={"recording": {"fps": 8}}).json()
    assert ok["changed"] == ["recording"] and app.config.recording.fps == 8
    settings = client.get("/api/settings").json()
    assert settings["config"]["recording"]["fps"] == 8 and "ANTHROPIC_API_KEY" in settings["credentials"]


def _png(draw) -> bytes:
    img = Image.new("RGB", (640, 360), (60, 60, 64))
    draw(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_upload_demonstration_is_analysed(client):
    before = _png(lambda d: d.rectangle([280, 120, 360, 280], fill=(200, 200, 200)))
    after = _png(lambda d: d.rectangle([300, 120, 340, 280], fill=(200, 200, 200)))
    res = client.post("/api/demonstrations", data={"title": "narrowing", "roles": '[{"role":"before"},{"role":"after"}]',
                                                   "license": "own work"},
                      files=[("files", ("before.png", before, "image/png")), ("files", ("after.png", after, "image/png"))])
    assert res.status_code == 200, res.text
    demo_id = res.json()["id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        demo = client.get(f"/api/demonstrations/{demo_id}").json()["demonstration"]
        if demo["status"] in ("READY", "FAILED"):
            break
        time.sleep(0.2)
    assert demo["status"] == "READY", demo
    assert demo["policy"]["training_allowed"] is False  # external media are never silently trainable
    steps = client.get(f"/api/sessions/{demo['session_id']}/timeline").json()["steps"]
    assert steps and steps[0]["candidate_actions"]
    confirmed = client.post(f"/api/steps/{steps[0]['id']}/confirm", json={"action_type": "scale"}).json()
    assert confirmed["action_source"] == "human_confirmed"
    assert client.post("/api/demonstrations", data={"title": "empty"}).status_code == 422


@pytest.mark.bpy
@pytest.mark.skipif(not HAS_BPY, reason="needs Blender")
def test_runs_through_jobs_and_human_review(app, client):
    job = client.post("/api/runs", json={"task_text": "make another simple sword blockout with blade length 5",
                                         "backend": "headless", "allow_takeover": False}).json()
    job = wait_job(client, job["id"], timeout=240)
    assert job["status"] == "done", job
    run_id = job["result"]["run_id"]
    run = client.get(f"/api/runs/{run_id}").json()
    # No reference silhouettes were given: measured visual checks cannot be evaluated, so a person decides.
    assert run["run"]["status"] == "needs_human", run["run"]["status"]
    assert any(t["to_state"] == "FAILURE" and t["reason_code"] == "success_not_verified" for t in run["transitions"])
    assert run["run"]["metrics"]["pending_credit"]
    uses_before = {s["id"]: s["usage_count"] for s in client.get("/api/skills").json()}
    review = client.post(f"/api/runs/{run_id}/review", json={"passed": True, "rating": 4}).json()
    assert review["status"] == "success"
    uses_after = {s["id"]: s["usage_count"] for s in client.get("/api/skills").json()}
    assert any(uses_after[k] == uses_before[k] + 1 for k in uses_before)  # deferred credit applied once
    assert client.post(f"/api/runs/{run_id}/review", json={"passed": True}).status_code == 409
    assert client.post("/api/runs", json={"task_text": "x", "backend": "gpu"}).status_code == 422


def test_cli_config_sets_validated_values(tmp_path, capsys):
    from lucius.cli import main

    data = str(tmp_path / "cfg")
    assert main(["--data-dir", data, "config", "--set", "providers.gemini_model=gemini-x",
                 "--set", "providers.requests_per_minute=12", "providers"]) == 0
    assert '"gemini_model": "gemini-x"' in capsys.readouterr().out
    assert main(["--data-dir", data, "config", "--set", "providers.gemini_modle=typo"]) == 2
    assert main(["--data-dir", data, "config", "--set", "providers.llm=unknown-provider"]) == 2
    from lucius.config import load_config

    cfg = load_config(data)
    assert cfg.providers.gemini_model == "gemini-x" and cfg.providers.requests_per_minute == 12
