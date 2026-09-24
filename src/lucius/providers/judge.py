"""Visual checkpoint judge on top of any image-capable LLM provider (results are always subjective)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lucius.errors import ProviderUnavailable
from lucius.providers.base import ImageInput, JudgeResult, LLMProvider


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
