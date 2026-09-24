"""Gemini via the official Google GenAI SDK (LLM, VLM and -- through ``VisionJudge`` -- visual judge).

The SDK reads the key from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``); Lucius never stores it.
Outputs are constrained with ``response_json_schema``. The model is never guessed: it must be set
in ``providers.gemini_model`` (``lucius models --provider gemini`` lists what the key can use).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from lucius.errors import ProviderError, ProviderRefusal, ProviderUnavailable
from lucius.providers.base import ImageInput, ModelResult

# Finish reasons that mean the model declined or was stopped by a content filter.
REFUSAL_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                   "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION"}


class GeminiProvider:
    name = "gemini"
    supports_images = True

    def __init__(self, model: str | None, *, thinking_level: str | None = None, timeout_s: float = 600.0,
                 client: Any = None) -> None:
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise ProviderUnavailable("the google-genai SDK is not installed (pip install lucius[gemini])") from exc
        if not model:
            raise ProviderUnavailable("no Gemini model configured: set providers.gemini_model "
                                      "(`lucius models --provider gemini` lists the models your key can use)")
        self._types = types
        self._errors = errors
        if client is None:
            try:
                client = genai.Client(http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
            except ValueError as exc:  # raised when no key is found in the environment
                raise ProviderUnavailable(f"Gemini client unavailable: {exc}") from exc
        self.client = client
        self.model = model
        self.thinking_level = thinking_level

    def complete_json(self, *, purpose: str, system: str, prompt: str, schema: dict[str, Any],
                      images: Sequence[ImageInput] = (), max_tokens: int = 8000) -> ModelResult:
        types = self._types
        parts = []
        for image in images:
            if image.label:
                parts.append(types.Part.from_text(text=image.label))
            parts.append(types.Part.from_bytes(data=image.data, mime_type=image.media_type))
        parts.append(types.Part.from_text(text=prompt))
        config = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json", response_json_schema=schema,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level.upper())
            if self.thinking_level else None)
        started = time.monotonic()
        response = self._call(purpose, [types.Content(role="user", parts=parts)], config)
        latency = time.monotonic() - started

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise ProviderRefusal("the prompt was blocked", purpose=purpose, category=_name(feedback.block_reason))
        if not response.candidates:
            raise ProviderError("model returned no candidates", purpose=purpose, transient=True)
        candidate = response.candidates[0]
        reason = _name(candidate.finish_reason)
        if reason in REFUSAL_REASONS:
            raise ProviderRefusal("the model declined the request", purpose=purpose, category=reason)
        if reason == "MAX_TOKENS":
            raise ProviderError("model output truncated at max_output_tokens", purpose=purpose, transient=False)
        content_parts = (candidate.content.parts if candidate.content and candidate.content.parts else [])
        text = "".join(p.text for p in content_parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        if not text:
            raise ProviderError("model returned no text", purpose=purpose, transient=False, finish_reason=reason)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError("model returned invalid JSON", purpose=purpose, transient=False) from exc
        usage = response.usage_metadata
        usage_dict = {"input_tokens": getattr(usage, "prompt_token_count", None) or 0,
                      "output_tokens": (getattr(usage, "candidates_token_count", None) or 0)
                      + (getattr(usage, "thoughts_token_count", None) or 0)}
        return ModelResult(data=data, provider=self.name, model=response.model_version or self.model, usage=usage_dict,
                           latency_s=latency)

    def _call(self, purpose: str, contents: list[Any], config: Any) -> Any:
        errors = self._errors
        try:
            return self.client.models.generate_content(model=self.model, contents=contents, config=config)
        except errors.ClientError as exc:
            code = getattr(exc, "code", None)
            if code in (401, 403):
                raise ProviderUnavailable(f"Gemini credentials rejected: {exc}", purpose=purpose) from exc
            if code == 404:
                raise ProviderError(f"model {self.model!r} not found: {exc}", purpose=purpose, transient=False) from exc
            if code == 429:
                raise ProviderError(f"rate limited: {exc}", purpose=purpose, transient=True) from exc
            raise ProviderError(f"bad request ({code}): {exc}", purpose=purpose, transient=False) from exc
        except errors.ServerError as exc:
            raise ProviderError(f"API error {getattr(exc, 'code', '')}: {exc}", purpose=purpose, transient=True) from exc
        except errors.APIError as exc:
            raise ProviderError(f"API error {getattr(exc, 'code', '')}: {exc}", purpose=purpose, transient=False) from exc
        except OSError as exc:
            raise ProviderError(f"connection error: {exc}", purpose=purpose, transient=True) from exc
        except Exception as exc:
            if type(exc).__module__.startswith("httpx"):
                raise ProviderError(f"connection error: {exc}", purpose=purpose, transient=True) from exc
            raise

    def list_models(self) -> list[dict[str, Any]]:
        """Models this key can use for content generation."""
        return _list_models(self.client)


def list_gemini_models(client: Any = None) -> list[dict[str, Any]]:
    """Models available to the key in the environment (no model needs to be configured)."""
    if client is None:
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderUnavailable("the google-genai SDK is not installed (pip install lucius[gemini])") from exc
        try:
            client = genai.Client()
        except ValueError as exc:
            raise ProviderUnavailable(f"Gemini client unavailable: {exc}") from exc
    return _list_models(client)


def _list_models(client: Any) -> list[dict[str, Any]]:
    out = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        out.append({"id": (model.name or "").removeprefix("models/"), "display_name": model.display_name,
                    "input_token_limit": model.input_token_limit, "output_token_limit": model.output_token_limit})
    return out


def _name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)
