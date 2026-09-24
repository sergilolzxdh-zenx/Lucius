"""Embedding providers.

``HashingEmbeddingProvider`` is the offline default: a deterministic feature-hashing encoder over
word unigrams, bigrams and character trigrams. It is a *lexical* representation -- it matches
shared vocabulary and morphology ("blade"/"blades", "bevel staging"), not paraphrase meaning.
For semantic similarity configure ``sentence-transformers`` (runs locally) instead; the retrieval
layer is agnostic and caches vectors per provider name.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

import numpy as np

from lucius.errors import ProviderUnavailable

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbeddingProvider:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-v1-{dim}"

    def _index(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _features(self, text: str) -> dict[str, float]:
        words = _WORD.findall(text.lower().replace("_", " "))
        feats: dict[str, float] = {}

        def add(key: str, weight: float) -> None:
            feats[key] = feats.get(key, 0.0) + weight

        for word in words:
            add(f"w:{word}", 1.0)
            padded = f"<{word}>"
            for i in range(len(padded) - 2):
                add(f"c:{padded[i:i + 3]}", 0.25)
        for a, b in zip(words, words[1:]):
            add(f"b:{a}_{b}", 0.7)
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                index, sign = self._index(feature)
                out[row, index] += sign * (1.0 + math.log(count)) if count >= 1 else sign * count
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformersEmbeddingProvider:
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ProviderUnavailable("sentence-transformers is not installed") from exc
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)
