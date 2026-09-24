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
    VideoInput,
)
from lucius.providers.embeddings import HashingEmbeddingProvider, SentenceTransformersEmbeddingProvider
from lucius.storage.db import Database

log = get_logger("providers")


def build_providers(config: ProviderConfig, db: Database | None) -> tuple[Providers, dict[str, str]]:
    """Instantiate configured providers. Returns the providers and why any are unavailable."""
    notes: dict[str, str] = {}
    call_log = CallLog(db)
    bases: dict[str, LoggedLLM | None] = {}

    def base(name: str) -> LoggedLLM | None:
        """One logged client per provider, shared by the roles that use it."""
        if name not in bases:
            try:
                if name == "anthropic":
                    from lucius.providers.anthropic_provider import AnthropicProvider

                    inner: LLMProvider = AnthropicProvider(
                        config.anthropic_model, effort=config.anthropic_effort,
                        server_fallbacks=config.anthropic_server_fallbacks, max_retries=config.max_retries)
                else:
                    from lucius.providers.gemini_provider import GeminiProvider

                    inner = GeminiProvider(config.gemini_model, thinking_level=config.gemini_thinking_level)
                # The Anthropic SDK retries internally; the Gemini SDK does not by default.
                bases[name] = LoggedLLM(inner, call_log, retries=config.max_retries if name == "gemini" else 1,
                                        requests_per_minute=config.requests_per_minute)
            except ProviderUnavailable as exc:
                notes[name] = exc.message
                log.warning("%s provider unavailable: %s", name, exc.message)
                bases[name] = None
        return bases[name]

    llm = base(config.llm) if config.llm != "none" else None
    vlm = base(config.vlm) if config.vlm != "none" else None
    evaluation = None
    if config.evaluation != "none":
        from lucius.providers.judge import VisionJudge

        judge_base = base(config.evaluation)
        evaluation = VisionJudge(judge_base) if judge_base is not None else None
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
    "LLMProvider", "LoggedLLM", "ModelResult", "Providers", "VideoInput", "build_providers",
]
