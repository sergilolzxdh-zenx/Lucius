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

from lucius.providers.gemini_provider import GeminiProvider, list_gemini_models  # noqa: E402


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
