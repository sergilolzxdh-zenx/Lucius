"""Claude via the official Anthropic SDK (LLM, VLM and visual judge).

Credentials are resolved by the SDK itself (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or
an ``ant auth login`` profile); Lucius never stores them. Outputs are constrained with
``output_config.format`` (JSON schema). Server-side refusal fallbacks are enabled by default.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from lucius.errors import ProviderError, ProviderRefusal, ProviderUnavailable
from lucius.providers.base import ImageInput, JudgeResult, LLMProvider, ModelResult

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"
    supports_images = True

    def __init__(self, model: str = "claude-opus-5", *, effort: str | None = None, server_fallbacks: bool = True,
                 max_retries: int = 2, timeout_s: float = 600.0) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderUnavailable("the anthropic SDK is not installed (pip install lucius[anthropic])") from exc
        self._sdk = anthropic
        try:
            self.client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)
        except anthropic.AnthropicError as exc:
            raise ProviderUnavailable(f"Anthropic client unavailable: {exc}") from exc
        self.model = model
        self.effort = effort
        self.server_fallbacks = server_fallbacks

    def complete_json(self, *, purpose: str, system: str, prompt: str, schema: dict[str, Any],
                      images: Sequence[ImageInput] = (), max_tokens: int = 8000) -> ModelResult:
        sdk = self._sdk
        content: list[dict[str, Any]] = []
        for image in images:
            if image.label:
                content.append({"type": "text", "text": image.label})
            content.append({"type": "image", "source": {"type": "base64", "media_type": image.media_type,
                                                        "data": image.b64()}})
        content.append({"type": "text", "text": prompt})
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": content}], "output_config": output_config,
        }
        if self.server_fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        started = time.monotonic()
        try:
            response = self.client.beta.messages.create(**kwargs)
        except (sdk.AuthenticationError, sdk.PermissionDeniedError) as exc:
            raise ProviderUnavailable(f"Anthropic credentials rejected: {exc}", purpose=purpose) from exc
        except sdk.NotFoundError as exc:
            raise ProviderError(f"model or endpoint not found: {exc}", purpose=purpose, transient=False) from exc
        except sdk.RateLimitError as exc:
            raise ProviderError(f"rate limited: {exc}", purpose=purpose, transient=True) from exc
        except sdk.BadRequestError as exc:
            raise ProviderError(f"bad request: {exc}", purpose=purpose, transient=False) from exc
        except sdk.APIStatusError as exc:
            raise ProviderError(f"API error {exc.status_code}: {exc}", purpose=purpose,
                                transient=exc.status_code >= 500) from exc
        except sdk.APIConnectionError as exc:
            raise ProviderError(f"connection error: {exc}", purpose=purpose, transient=True) from exc
        latency = time.monotonic() - started
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise ProviderRefusal("the model declined the request", purpose=purpose,
                                  category=getattr(details, "category", None))
        if response.stop_reason == "max_tokens":
            raise ProviderError("model output truncated at max_tokens", purpose=purpose, transient=False)
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise ProviderError("model returned no text block", purpose=purpose, transient=False)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError("model returned invalid JSON", purpose=purpose, transient=False) from exc
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return ModelResult(data=data, provider=self.name, model=response.model, usage=usage, latency_s=latency)


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail", "cannot_determine"]},
        "score": {"type": "number"},
        "confidence": {"type": "number"},
        "reason_codes": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "score", "confidence", "reason_codes", "observations"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You evaluate 3D modelling results in Blender against one explicit criterion. Judge only what is "
    "visible in the images. If the images do not show enough to decide, answer cannot_determine. "
    "score is 0..1 agreement with the criterion; confidence is 0..1 certainty of your verdict. "
    "reason_codes are short snake_case codes; observations are short factual statements."
)


class VisionJudge:
    """Visual checkpoint judge built on any image-capable LLM provider (results are subjective)."""

    def __init__(self, llm: LLMProvider) -> None:
        if not llm.supports_images:
            raise ProviderUnavailable("the configured model does not accept images")
        self.llm = llm
        self.name = f"judge:{llm.name}:{llm.model}"

    def judge(self, *, criterion: str, images: Sequence[ImageInput], context: str) -> JudgeResult:
        result = self.llm.complete_json(
            purpose="visual_checkpoint", system=JUDGE_SYSTEM,
            prompt=f"Criterion: {criterion}\nContext: {context}\nReturn the JSON verdict.",
            schema=JUDGE_SCHEMA, images=images, max_tokens=2000)
        data = result.data
        verdict = data.get("verdict")
        return JudgeResult(
            passed=None if verdict == "cannot_determine" else verdict == "pass",
            score=_clamp(data.get("score")), confidence=_clamp(data.get("confidence")) or 0.0,
            reason_codes=[str(c) for c in data.get("reason_codes", [])][:10],
            observations=[str(o) for o in data.get("observations", [])][:10],
            provider=result.provider, model=result.model,
        )


def _clamp(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
