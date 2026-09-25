"""Gemini provider without network access: a fake client returning real SDK response objects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from lucius.config import ProviderConfig
from lucius.errors import ProviderError, ProviderRefusal, ProviderUnavailable
from lucius.providers import CallLog, ImageInput, LoggedLLM, build_providers
from lucius.segmentation.refine import SCHEMA as SEGMENT_SCHEMA

genai_types = pytest.importorskip("google.genai.types")
from google.genai import errors as genai_errors  # noqa: E402

from lucius.providers.base import RateLimiter  # noqa: E402
from lucius.providers.gemini_provider import GeminiProvider, list_gemini_models, probe_gemini_models  # noqa: E402


def _response(text: str | None, finish: str = "STOP", *, block: str | None = None, thought: str | None = None):
    parts = []
    if thought:
        parts.append(genai_types.Part(text=thought, thought=True))
    if text is not None:
        parts.append(genai_types.Part(text=text))
    return genai_types.GenerateContentResponse(
        candidates=[] if block else [genai_types.Candidate(content=genai_types.Content(role="model", parts=parts),
                                                           finish_reason=genai_types.FinishReason(finish))],
        prompt_feedback=genai_types.GenerateContentResponsePromptFeedback(block_reason=block) if block else None,
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(prompt_token_count=20, candidates_token_count=6,
                                                                        thoughts_token_count=4),
        model_version="gemini-test-001")


class FakeModels:
    def __init__(self, responses=(), models=()):
        self.responses = list(responses)
        self.models = list(models)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def list(self, config=None):
        return iter(self.models)


def _provider(responses, **kwargs):
    fake = FakeModels(responses)
    return GeminiProvider("gemini-test", client=SimpleNamespace(models=fake), **kwargs), fake


def test_request_shape_schema_and_parsing():
    provider, fake = _provider([_response('{"segments": []}', thought="internal planning")], thinking_level="low")
    image = ImageInput.from_image(Image.new("RGB", (40, 30), (1, 2, 3)), label="Segment 2 frame:")
    result = provider.complete_json(purpose="segment_labeling", system="label phases", prompt="segments...",
                                    schema=SEGMENT_SCHEMA, images=[image], max_tokens=6000)
    assert result.data == {"segments": []}  # thought parts are never used as output
    assert result.model == "gemini-test-001" and result.usage == {"input_tokens": 20, "output_tokens": 10}
    call = fake.calls[0]
    config = call["config"]
    assert call["model"] == "gemini-test" and config.response_mime_type == "application/json"
    assert config.response_json_schema == SEGMENT_SCHEMA and config.response_schema is None
    assert config.system_instruction == "label phases" and config.max_output_tokens == 6000
    assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW
    assert config.automatic_function_calling.disable is True  # Lucius never passes tools
    parts = call["contents"][0].parts
    assert parts[0].text == "Segment 2 frame:" and parts[1].inline_data.mime_type == image.media_type
    assert parts[1].inline_data.data == image.data and parts[2].text == "segments..."


def _client_error(code: int, status: str):
    return genai_errors.ClientError(code, {"error": {"code": code, "message": status, "status": status}})


def test_refusals_truncation_and_error_mapping():
    provider, _ = _provider([
        _response(None, block="SAFETY"), _response("", finish="SAFETY"), _response('{"a":', finish="MAX_TOKENS"),
        _response("not json"), _client_error(429, "RESOURCE_EXHAUSTED"), _client_error(403, "PERMISSION_DENIED"),
        _client_error(404, "NOT_FOUND"), genai_errors.ServerError(503, {"error": {"code": 503, "message": "busy"}}),
    ])
    kw = {"purpose": "t", "system": "s", "prompt": "p", "schema": {"type": "object"}}
    with pytest.raises(ProviderRefusal) as blocked:
        provider.complete_json(**kw)
    assert blocked.value.details["category"] == "SAFETY"
    with pytest.raises(ProviderRefusal):
        provider.complete_json(**kw)
    with pytest.raises(ProviderError, match="truncated"):
        provider.complete_json(**kw)
    with pytest.raises(ProviderError, match="invalid JSON"):
        provider.complete_json(**kw)
    with pytest.raises(ProviderError) as limited:
        provider.complete_json(**kw)
    assert limited.value.details["transient"] is True
    with pytest.raises(ProviderUnavailable):
        provider.complete_json(**kw)
    with pytest.raises(ProviderError) as missing:
        provider.complete_json(**kw)
    assert missing.value.details["transient"] is False
    with pytest.raises(ProviderError) as busy:
        provider.complete_json(**kw)
    assert busy.value.details["transient"] is True


def test_logged_retries_transient_gemini_errors(db, monkeypatch):
    monkeypatch.setattr("lucius.providers.base.time.sleep", lambda _s: None)
    provider, fake = _provider([_client_error(429, "RESOURCE_EXHAUSTED"), _response('{"ok": 1}')])
    assert LoggedLLM(provider, CallLog(db), retries=2).complete_json(purpose="p", system="s", prompt="q",
                                                                    schema={}).data == {"ok": 1}
    assert len(fake.calls) == 2
    assert [r["status"] for r in db.query("SELECT status FROM model_calls ORDER BY created_at")] == ["error", "ok"]


def test_model_is_never_guessed_and_missing_key_is_reported(db, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    providers, notes = build_providers(ProviderConfig(llm="gemini", vlm="gemini", evaluation="gemini"), db)
    assert not providers.has("llm") and not providers.has("evaluation")
    assert "gemini_model" in notes["gemini"]
    providers, notes = build_providers(ProviderConfig(llm="gemini", gemini_model="gemini-test"), db)
    assert not providers.has("llm") and "API key" in notes["gemini"]
    with pytest.raises(ProviderUnavailable):
        list_gemini_models()


def test_configured_gemini_serves_every_role(db, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    providers, notes = build_providers(ProviderConfig(llm="gemini", vlm="gemini", evaluation="gemini",
                                                      gemini_model="gemini-test"), db)
    assert not notes and providers.available()["llm"] == "gemini:gemini-test"
    assert providers.vlm is providers.llm and providers.evaluation.name == "judge:gemini:gemini-test"


def test_model_listing_keeps_generation_models():
    models = [genai_types.Model(name="models/gemini-a", display_name="A", supported_actions=["generateContent"]),
              genai_types.Model(name="models/embed-b", display_name="B", supported_actions=["embedContent"])]
    listed = list_gemini_models(SimpleNamespace(models=FakeModels(models=models)))
    assert [m["id"] for m in listed] == ["gemini-a"]


def _quota_error(message: str, quota_ids: list[str], retry: str | None = None):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaMetric": "m", "quotaId": q} for q in quota_ids]}]
    if retry:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry})
    return genai_errors.ClientError(429, {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED",
                                                    "details": details}})


def test_quota_errors_distinguish_zero_daily_and_per_minute_limits():
    """Shapes taken from live free-tier responses: only a per-minute limit is worth retrying."""
    provider, _ = _provider([
        _quota_error("Quota exceeded for metric: free_tier_requests, limit: 0, model: gemini-pro",
                     ["GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                      "GenerateRequestsPerDayPerProjectPerModel-FreeTier"], "1s"),
        _quota_error("Quota exceeded for metric: free_tier_requests, limit: 500",
                     ["GenerateRequestsPerDayPerProjectPerModel-FreeTier"], "3600s"),
        _quota_error("Quota exceeded for metric: free_tier_requests, limit: 15",
                     ["GenerateRequestsPerMinutePerProjectPerModel-FreeTier"], "23.5s"),
    ])
    kw = {"purpose": "t", "system": "s", "prompt": "p", "schema": {"type": "object"}}
    with pytest.raises(ProviderUnavailable, match="no quota"):
        provider.complete_json(**kw)
    with pytest.raises(ProviderError, match="daily quota") as daily:
        provider.complete_json(**kw)
    assert daily.value.details["transient"] is False
    with pytest.raises(ProviderError, match="rate limited") as minute:
        provider.complete_json(**kw)
    assert minute.value.details["transient"] is True and minute.value.details["retry_after_s"] == 23.5


def test_logged_retry_waits_as_long_as_the_server_asks(db, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("lucius.providers.base.time.sleep", slept.append)
    provider, _ = _provider([_quota_error("limit: 15", ["GenerateRequestsPerMinute-FreeTier"], "23.5s"),
                             _quota_error("limit: 15", ["GenerateRequestsPerMinute-FreeTier"], "600s"),
                             _response('{"ok": 1}')])
    LoggedLLM(provider, CallLog(db), retries=2).complete_json(purpose="p", system="s", prompt="q", schema={})
    assert slept == [23.5, 90.0]  # the server's delay, capped so a run never stalls for long

    slept.clear()
    bare = genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                    "message": "Resource has been exhausted (e.g. check quota)."}})
    provider, _ = _provider([bare, bare, _response('{"ok": 1}')])
    LoggedLLM(provider, CallLog(db), retries=2).complete_json(purpose="p", system="s", prompt="q", schema={})
    assert slept == [15.0, 30.0]  # no delay given: long enough for a per-minute window to pass


def test_rate_limiter_spaces_requests(monkeypatch):
    clock = [100.0]
    slept: list[float] = []
    monkeypatch.setattr("lucius.providers.base.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("lucius.providers.base.time.sleep", slept.append)
    limiter = RateLimiter(per_minute=30)
    for _ in range(3):
        limiter.wait()
    assert slept == [2.0, 4.0]
    RateLimiter(None).wait()  # unlimited never sleeps
    assert len(slept) == 2


def test_probe_reports_which_listed_models_lucius_can_use():
    class ProbeModels(FakeModels):
        def generate_content(self, *, model, contents, config):
            self.calls.append(model)
            assert config.response_json_schema and any(p.inline_data for p in contents[0].parts)
            outcomes = {
                "good": _response('{"colour": "red"}'),
                "retired": _client_error(404, "NOT_FOUND"),
                "pro": _quota_error("limit: 0", ["GenerateRequestsPerDayPerProjectPerModel-FreeTier"]),
                "tts": _client_error(400, "INVALID_ARGUMENT"),
                "busy": genai_errors.ServerError(503, {"error": {"code": 503, "message": "high demand"}}),
            }
            outcome = outcomes[model]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    fake = ProbeModels()
    result = {r["id"]: r for r in probe_gemini_models(["good", "retired", "pro", "tts", "busy"],
                                                      client=SimpleNamespace(models=fake))}
    assert result["good"] == {"id": "good", "usable": True, "status": "ok"}
    assert {k: v["status"] for k, v in result.items() if k != "good"} == {
        "retired": "retired_or_missing", "pro": "no_quota", "tts": "unsupported_request",
        "busy": "temporarily_unavailable"}
    assert sorted(fake.calls) == ["busy", "good", "pro", "retired", "tts"]


def test_overloaded_model_falls_back_and_is_skipped_for_a_while(monkeypatch):
    busy = genai_errors.ServerError(503, {"error": {"code": 503, "message": "high demand"}})
    provider, fake = _provider([busy, _response('{"n": 1}'), _response('{"n": 2}'), busy, busy],
                               fallback_models=["gemini-backup", "gemini-last"])
    kw = {"purpose": "t", "system": "s", "prompt": "p", "schema": {"type": "object"}}
    assert provider.complete_json(**kw).data == {"n": 1}
    assert [c["model"] for c in fake.calls] == ["gemini-test", "gemini-backup"]
    # The overloaded model is not asked again right away: the backup answers directly.
    assert provider.complete_json(**kw).data == {"n": 2}
    assert fake.calls[-1]["model"] == "gemini-backup"
    # When every model is overloaded the last error surfaces (transient: a caller may retry later).
    with pytest.raises(ProviderError) as exhausted:
        provider.complete_json(**kw)
    assert exhausted.value.details["transient"] is True
    assert [c["model"] for c in fake.calls[-2:]] == ["gemini-backup", "gemini-last"]
    # After the cool-down the configured model is tried first again.
    monkeypatch.setattr("lucius.providers.gemini_provider.time.monotonic", lambda: 10**9)
    fake.responses.append(_response('{"n": 3}'))
    assert provider.complete_json(**kw).data == {"n": 3} and fake.calls[-1]["model"] == "gemini-test"
