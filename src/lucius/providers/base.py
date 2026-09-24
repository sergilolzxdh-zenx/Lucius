"""Model provider interfaces.

Business logic depends on these protocols only -- never on a provider SDK. Every model call is
structured (JSON-schema constrained output), accounted in ``model_calls``, and failures surface
as :class:`~lucius.errors.ProviderError` so callers can degrade gracefully.

No hidden reasoning is requested or stored: prompts ask for decisions, evidence and confidence.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from lucius.errors import ProviderError, ProviderUnavailable
from lucius.ids import new_id
from lucius.logging_setup import get_logger
from lucius.storage.db import Database
from lucius.timeutil import now

log = get_logger("providers")


@dataclass
class ImageInput:
    """An image for a vision model, downscaled before sending."""

    data: bytes
    media_type: str
    label: str = ""

    @classmethod
    def from_image(cls, image: Image.Image, label: str = "", max_side: int = 1280) -> ImageInput:
        img = image.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return cls(buffer.getvalue(), "image/jpeg", label)

    @classmethod
    def from_path(cls, path: str | Path, label: str = "", max_side: int = 1280) -> ImageInput:
        with Image.open(path) as img:
            return cls.from_image(img, label, max_side)

    def b64(self) -> str:
        return base64.standard_b64encode(self.data).decode("ascii")


@dataclass
class ModelResult:
    data: dict[str, Any]
    provider: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str
    supports_images: bool

    def complete_json(self, *, purpose: str, system: str, prompt: str, schema: dict[str, Any],
                      images: Sequence[ImageInput] = (), max_tokens: int = 8000) -> ModelResult: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class JudgeResult:
    passed: bool | None
    score: float | None
    confidence: float
    reason_codes: list[str]
    observations: list[str]
    provider: str
    model: str


@runtime_checkable
class EvaluationProvider(Protocol):
    name: str

    def judge(self, *, criterion: str, images: Sequence[ImageInput], context: str) -> JudgeResult: ...


class CallLog:
    """Accounts every model call (purpose, tokens, latency, errors) for cost/efficiency review."""

    def __init__(self, db: Database | None) -> None:
        self.db = db

    def record(self, *, provider: str, model: str | None, purpose: str, status: str,
               usage: dict[str, int] | None = None, latency_s: float | None = None, error: str | None = None) -> None:
        if self.db is None:
            return
        usage = usage or {}
        self.db.insert("model_calls", {
            "id": new_id("model_call"), "provider": provider, "model": model, "purpose": purpose, "status": status,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "latency_s": latency_s, "error": error, "created_at": now(),
        })


class RateLimiter:
    """Spaces requests evenly so a per-minute quota is never exceeded (shared by every thread)."""

    def __init__(self, per_minute: int | None) -> None:
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            current = time.monotonic()
            slot = max(current, self._next)
            self._next = slot + self.interval
        if slot > current:
            time.sleep(slot - current)


MAX_RETRY_WAIT_S = 90.0


class LoggedLLM:
    """Wraps an LLM provider with call accounting, rate limiting and bounded retries on transient errors."""

    def __init__(self, inner: LLMProvider, log_: CallLog, retries: int = 1, requests_per_minute: int | None = None) -> None:
        self.inner = inner
        self.log = log_
        self.retries = retries
        self.limiter = RateLimiter(requests_per_minute)
        self.name = inner.name
        self.model = inner.model
        self.supports_images = inner.supports_images

    def complete_json(self, *, purpose: str, system: str, prompt: str, schema: dict[str, Any],
                      images: Sequence[ImageInput] = (), max_tokens: int = 8000) -> ModelResult:
        attempt = 0
        while True:
            self.limiter.wait()
            started = time.monotonic()
            try:
                result = self.inner.complete_json(purpose=purpose, system=system, prompt=prompt, schema=schema,
                                                  images=images, max_tokens=max_tokens)
            except ProviderError as exc:
                transient = bool(exc.details.get("transient"))
                self.log.record(provider=self.name, model=self.model, purpose=purpose, status="error",
                                latency_s=time.monotonic() - started, error=f"{exc.code}: {exc.message}")
                if transient and attempt < self.retries:
                    attempt += 1
                    # A rate-limit response says how long to wait; waiting less just fails again.
                    retry_after = float(exc.details.get("retry_after_s") or 0.0)
                    time.sleep(min(MAX_RETRY_WAIT_S, max(min(8.0, 2.0 ** attempt), retry_after)))
                    continue
                raise
            self.log.record(provider=self.name, model=result.model, purpose=purpose, status="ok", usage=result.usage,
                            latency_s=result.latency_s)
            return result


class Providers:
    """The set of configured providers. Missing capabilities raise ProviderUnavailable on use."""

    def __init__(self, *, llm: LLMProvider | None = None, vlm: LLMProvider | None = None,
                 embeddings: EmbeddingProvider, evaluation: EvaluationProvider | None = None) -> None:
        self._llm = llm
        self._vlm = vlm
        self.embeddings = embeddings
        self._evaluation = evaluation

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            raise ProviderUnavailable("no LLM provider configured")
        return self._llm

    @property
    def vlm(self) -> LLMProvider:
        if self._vlm is None:
            raise ProviderUnavailable("no vision-language provider configured")
        return self._vlm

    @property
    def evaluation(self) -> EvaluationProvider:
        if self._evaluation is None:
            raise ProviderUnavailable("no evaluation provider configured")
        return self._evaluation

    def available(self) -> dict[str, str | None]:
        return {
            "llm": f"{self._llm.name}:{self._llm.model}" if self._llm else None,
            "vlm": f"{self._vlm.name}:{self._vlm.model}" if self._vlm else None,
            "embeddings": self.embeddings.name,
            "evaluation": self._evaluation.name if self._evaluation else None,
        }

    def has(self, capability: str) -> bool:
        return {"llm": self._llm, "vlm": self._vlm, "evaluation": self._evaluation}.get(capability) is not None
