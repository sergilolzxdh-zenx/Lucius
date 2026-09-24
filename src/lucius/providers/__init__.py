"""Model providers (LLM, VLM, embeddings, evaluation) behind provider-neutral interfaces."""

from __future__ import annotations

from lucius.config import ProviderConfig
from lucius.errors import ProviderUnavailable
from lucius.logging_setup import get_logger
from lucius.providers.base import (
    CallLog,
    EmbeddingProvider,
    EvaluationProvider,
    ImageInput,
    JudgeResult,
    LLMProvider,
    LoggedLLM,
    ModelResult,
    Providers,
)
from lucius.providers.embeddings import HashingEmbeddingProvider, SentenceTransformersEmbeddingProvider
from lucius.storage.db import Database

log = get_logger("providers")


def build_providers(config: ProviderConfig, db: Database | None) -> tuple[Providers, dict[str, str]]:
    """Instantiate configured providers. Returns the providers and why any are unavailable."""
    notes: dict[str, str] = {}
    call_log = CallLog(db)
    llm = vlm = evaluation = None
    if "anthropic" in (config.llm, config.vlm, config.evaluation):
        try:
            from lucius.providers.anthropic_provider import AnthropicProvider, VisionJudge

            base = LoggedLLM(AnthropicProvider(config.anthropic_model, effort=config.anthropic_effort,
                                               server_fallbacks=config.anthropic_server_fallbacks,
                                               max_retries=config.max_retries), call_log)
            llm = base if config.llm == "anthropic" else None
            vlm = base if config.vlm == "anthropic" else None
            evaluation = VisionJudge(base) if config.evaluation == "anthropic" else None
        except ProviderUnavailable as exc:
            notes["anthropic"] = exc.message
            log.warning("anthropic provider unavailable: %s", exc.message)
    embeddings: EmbeddingProvider
    if config.embeddings == "sentence-transformers":
        try:
            embeddings = SentenceTransformersEmbeddingProvider(config.sentence_transformers_model)
        except ProviderUnavailable as exc:
            notes["embeddings"] = f"{exc.message}; using hashing embeddings"
            embeddings = HashingEmbeddingProvider(config.hashing_dim)
    else:
        embeddings = HashingEmbeddingProvider(config.hashing_dim)
    return Providers(llm=llm, vlm=vlm, embeddings=embeddings, evaluation=evaluation), notes


__all__ = [
    "CallLog", "EmbeddingProvider", "EvaluationProvider", "HashingEmbeddingProvider", "ImageInput", "JudgeResult",
    "LLMProvider", "LoggedLLM", "ModelResult", "Providers", "build_providers",
]
