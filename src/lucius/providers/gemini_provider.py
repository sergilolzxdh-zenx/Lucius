"""Gemini via the official Google GenAI SDK (LLM, VLM and -- through ``VisionJudge`` -- visual judge).

The SDK reads the key from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``); Lucius never stores it.
Outputs are constrained with ``response_json_schema``. The model is never guessed: it must be set
in ``providers.gemini_model`` (``lucius models --provider gemini`` lists what the key can use).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
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
            # Lucius never passes tools; the SDK otherwise enables automatic function calling on every request.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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
                raise _quota_error(exc, self.model, purpose) from exc
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


def _quota_error(exc: Any, model: str, purpose: str) -> ProviderError:
    """Classify a 429: a zero quota or an exhausted daily quota cannot succeed on retry; a per-minute limit can.

    The API names the violated quotas (``QuotaFailure``) and says when to retry (``RetryInfo``).
    """
    body = exc.details if isinstance(getattr(exc, "details", None), dict) else {}
    error = body.get("error", {}) if isinstance(body.get("error"), dict) else {}
    message = str(error.get("message") or exc)
    quota_ids, retry_after = [], None
    for detail in error.get("details") or []:
        if not isinstance(detail, dict):
            continue
        quota_ids += [str(v.get("quotaId", "")) for v in detail.get("violations") or [] if isinstance(v, dict)]
        delay = re.fullmatch(r"([\d.]+)s", str(detail.get("retryDelay", "")))
        if delay:
            retry_after = float(delay.group(1))
    if re.search(r"\blimit: 0\b", message):
        return ProviderUnavailable(f"model {model!r} has no quota on this API key (free tier keys get none for "
                                   "some models): pick another model or enable billing", purpose=purpose,
                                   quota=quota_ids)
    if any("PerDay" in q for q in quota_ids):
        return ProviderError(f"daily quota for {model!r} is exhausted", purpose=purpose, transient=False,
                             quota=quota_ids)
    return ProviderError(f"rate limited: {message.splitlines()[0][:200]}", purpose=purpose, transient=True,
                         retry_after_s=retry_after, quota=quota_ids)


def _client(client: Any) -> Any:
    if client is not None:
        return client
    try:
        from google import genai
    except ImportError as exc:
        raise ProviderUnavailable("the google-genai SDK is not installed (pip install lucius[gemini])") from exc
    try:
        return genai.Client()
    except ValueError as exc:
        raise ProviderUnavailable(f"Gemini client unavailable: {exc}") from exc


def list_gemini_models(client: Any = None) -> list[dict[str, Any]]:
    """Models available to the key in the environment (no model needs to be configured)."""
    return _list_models(_client(client))


PROBE_SCHEMA = {"type": "object", "properties": {"colour": {"type": "string"}}, "required": ["colour"],
                "additionalProperties": False}


def probe_gemini_models(model_ids: Iterable[str], client: Any = None, workers: int = 8) -> list[dict[str, Any]]:
    """Send each model one minimal request of the kind Lucius makes (an image + a JSON schema).

    Listing is not enough: the API lists retired models, models whose free-tier quota is zero, and
    audio/image-output models that cannot answer in JSON. Each probe costs a few hundred tokens.
    """
    import io

    from PIL import Image

    provider = GeminiProvider("probe", client=_client(client))
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(buffer, format="JPEG")
    image = ImageInput(buffer.getvalue(), "image/jpeg")

    def probe(model_id: str) -> dict[str, Any]:
        provider_for = GeminiProvider(model_id, client=provider.client)
        try:
            provider_for.complete_json(purpose="model_probe", system="Answer in JSON.", schema=PROBE_SCHEMA,
                                       prompt="Which colour fills this image?", images=[image], max_tokens=2000)
            return {"id": model_id, "usable": True, "status": "ok"}
        except ProviderError as exc:
            message = exc.message
            status = ("no_quota" if isinstance(exc, ProviderUnavailable) and "quota" in message
                      else "retired_or_missing" if "not found" in message
                      else "daily_quota_exhausted" if "daily quota" in message
                      else "rate_limited" if message.startswith("rate limited")
                      else "temporarily_unavailable" if exc.details.get("transient")
                      else "unsupported_request")
            return {"id": model_id, "usable": False, "status": status, "detail": message[:300]}

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(probe, list(model_ids)))


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
