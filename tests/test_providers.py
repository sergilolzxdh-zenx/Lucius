"""Model-assisted paths without network access.

The Anthropic provider is exercised with a fake SDK client (request shape, refusal, truncation,
error mapping, retries). ``ScriptedLLM``/``ScriptedJudge`` are development fixtures standing in for
a model so that refinement and visual judgement can be tested deterministically.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from lucius.app import Lucius
from lucius.errors import ProviderError, ProviderRefusal, ProviderUnavailable
from lucius.evaluation.evaluator import Evaluator
from lucius.events import EventType
from lucius.providers import CallLog, HashingEmbeddingProvider, ImageInput, JudgeResult, LoggedLLM, ModelResult, Providers
from lucius.provenance import DataPolicy
from lucius.sessions import Outcome, SessionKind
from lucius.skills.schema import Checkpoint
from tests.fixtures.demos import sword_blockout_demo

anthropic = pytest.importorskip("anthropic")


# -- Anthropic provider with a fake client ----------------------------------------------------------

class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _response(text: str, stop: str = "end_turn"):
    return SimpleNamespace(stop_reason=stop, model="claude-opus-5", stop_details=SimpleNamespace(category="cyber"),
                           content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=12, output_tokens=5))


def _provider(responses):
    from lucius.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider("claude-opus-5", effort="high")
    fake = FakeMessages(responses)
    provider.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    return provider, fake


def _status_error(cls, status: int):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx.Response(status, request=request), body=None)


def test_anthropic_request_shape_and_parsing():
    provider, fake = _provider([_response('{"ok": true}')])
    image = ImageInput.from_image(Image.new("RGB", (32, 32), (10, 20, 30)), label="frame 1:")
    result = provider.complete_json(purpose="t", system="sys", prompt="do it", schema={"type": "object"}, images=[image])
    assert result.data == {"ok": True} and result.usage == {"input_tokens": 12, "output_tokens": 5}
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5" and call["system"] == "sys"
    assert call["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": "high"}
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    blocks = call["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image", "text"] and blocks[1]["source"]["type"] == "base64"


def test_anthropic_refusal_truncation_and_errors():
    provider, _ = _provider([_response("", stop="refusal"), _response('{"a":', stop="max_tokens"),
                             _response("not json"), _status_error(anthropic.RateLimitError, 429),
                             _status_error(anthropic.AuthenticationError, 401)])
    kwargs = {"purpose": "t", "system": "s", "prompt": "p", "schema": {}}
    with pytest.raises(ProviderRefusal) as refusal:
        provider.complete_json(**kwargs)
    assert refusal.value.details["category"] == "cyber"
    with pytest.raises(ProviderError, match="truncated"):
        provider.complete_json(**kwargs)
    with pytest.raises(ProviderError, match="invalid JSON"):
        provider.complete_json(**kwargs)
    with pytest.raises(ProviderError) as limited:
        provider.complete_json(**kwargs)
    assert limited.value.details["transient"] is True
    with pytest.raises(ProviderUnavailable):
        provider.complete_json(**kwargs)


def test_logged_llm_retries_transient_errors_and_accounts_calls(db, monkeypatch):
    monkeypatch.setattr("lucius.providers.base.time.sleep", lambda _s: None)
    provider, fake = _provider([_status_error(anthropic.InternalServerError, 529), _response('{"x": 1}')])
    logged = LoggedLLM(provider, CallLog(db), retries=1)
    assert logged.complete_json(purpose="p", system="s", prompt="q", schema={}).data == {"x": 1}
    assert len(fake.calls) == 2
    rows = db.query("SELECT status FROM model_calls ORDER BY created_at")
    assert [r["status"] for r in rows] == ["error", "ok"]


# -- refinement and judgement with scripted fixtures ----------------------------------------------------

class ScriptedLLM:
    """Development fixture: relabels the first segment it is shown."""

    name, model, supports_images = "scripted", "scripted-1", True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, *, purpose, system, prompt, schema, images=(), max_tokens=8000):
        self.prompts.append(prompt)
        first = int(re.findall(r"^Segment (\d+) ", prompt, re.M)[0])
        return ModelResult(data={"segments": [{"idx": first, "label": "reference_check", "title": "Checking the reference",
                                               "confidence": 1.0, "reason_codes": ["views_compared"],
                                               "new_label_description": "comparing the model with a reference"}]},
                           provider=self.name, model=self.model)


class ScriptedJudge:
    """Development fixture for the visual judge."""

    name = "scripted-judge"

    def __init__(self, verdict: bool | None) -> None:
        self.verdict = verdict
        self.calls = 0

    def judge(self, *, criterion, images, context):
        self.calls += 1
        return JudgeResult(passed=self.verdict, score=0.8 if self.verdict else 0.2, confidence=0.7,
                           reason_codes=["looks_tapered"], observations=["tip narrows"], provider="scripted",
                           model="judge-1")


def test_model_refinement_is_marked_discounted_and_respects_human_edits(tmp_path):
    llm = ScriptedLLM()
    app = Lucius(data_dir=tmp_path / "data", background_processing=False,
                 providers=Providers(llm=llm, embeddings=HashingEmbeddingProvider(256)))
    try:
        demo = sword_blockout_demo()
        session = app.sessions.create(user_id="local", kind=SessionKind.LIVE_DEMO, policy=DataPolicy.for_live_demo(),
                                      task_text="simple sword blockout", start_time=demo.events[0].ts)
        app.sessions.append_events(session.id, demo.events)
        app.sessions.finalize(session.id, end_time=demo.events[-1].ts, outcome=Outcome.SUCCESS)
        app.bus.publish(EventType.SESSION_ENDED, session.id)
        segments = app.segments.for_session(session.id)
        relabelled = next(s for s in segments if s.origin == "model")
        assert relabelled.label == "reference_check" and relabelled.label_confidence == pytest.approx(0.85)
        assert relabelled.meta["deterministic_label"]["label"] != "reference_check"
        assert app.taxonomy.terms("segment_label")["reference_check"]["source"] == "model"

        human = app.segment_editor.relabel(relabelled.id, "scene_setup", "Setup (by me)")
        app.pipeline.process(session.id, force=True, stages=["refinement"])
        assert f"Segment {human.idx} " not in llm.prompts[-1]  # locked segments are never sent
        kept = app.segments.get(human.id)
        assert kept.label == "scene_setup" and kept.origin == "human"

        # The agent's own runs are never sent for relabelling (they cost most of a daily quota once).
        calls = len(llm.prompts)
        run = app.sessions.create(user_id="local", kind=SessionKind.VALIDATION, policy=DataPolicy.for_live_demo(),
                                  task_text="simple sword blockout", start_time=demo.events[0].ts)
        app.sessions.append_events(run.id, demo.events)
        app.sessions.finalize(run.id, end_time=demo.events[-1].ts, outcome=Outcome.SUCCESS)
        app.pipeline.process(run.id)
        assert len(llm.prompts) == calls and app.pipeline.status(run.id)["refinement"]["status"] == "skipped"
    finally:
        app.close()


def _structure() -> dict:
    tri = [-0.1, 0.0, 0.1, 0.0, 0.0, 3.0]
    return {"objects": [{"name": "Blade", "dimensions": [0.2, 0.1, 3.0], "silhouettes": {"front": tri, "side": tri}}],
            "mesh_object_count": 1}


def test_visual_judgement_is_subjective_never_objective(db):
    silhouette = Checkpoint(id="shape", description="blade tapers to a tip", level=3, method="visual_measured",
                            check={"type": "silhouette", "object": "Blade", "views": ["front"]})
    exists = Checkpoint(id="exists", description="blade exists", check={"type": "object_exists", "object": "Blade"})

    judge = ScriptedJudge(True)
    ev = Evaluator(db, Providers(embeddings=HashingEmbeddingProvider(64), evaluation=judge))
    only_model = ev.evaluate(subject_kind="test", subject_id="a", checkpoints=[silhouette], params={},
                             structure=_structure())
    assert judge.calls == 1 and only_model.results[0].subjective and only_model.verdict == "subjective_pass"
    with_structure = ev.evaluate(subject_kind="test", subject_id="b", checkpoints=[silhouette, exists], params={},
                                 structure=_structure())
    assert with_structure.verdict == "success" and with_structure.objective_passes == 1

    unsure = Evaluator(db, Providers(embeddings=HashingEmbeddingProvider(64), evaluation=ScriptedJudge(None)))
    assert unsure.evaluate(subject_kind="test", subject_id="c", checkpoints=[silhouette, exists], params={},
                           structure=_structure()).verdict == "needs_human"
    no_model = Evaluator(db, Providers(embeddings=HashingEmbeddingProvider(64)))
    assert no_model.evaluate(subject_kind="test", subject_id="d", checkpoints=[silhouette, exists], params={},
                             structure=_structure()).verdict == "needs_human"
