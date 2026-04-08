from __future__ import annotations

import math

from meeting_pipeline.cache_utils import LruCache
from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.embeddings.ollama_client import OllamaClient

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
EXPECTED_EMBEDDING_DIM = 768
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder:
    def __init__(
        self,
        client: OllamaClient | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        runtime_settings = settings or get_settings()
        self._client = client or OllamaClient.from_settings(runtime_settings)
        configured_model = (model_name or runtime_settings.ollama_model).strip()
        self._model_name = configured_model or DEFAULT_EMBEDDING_MODEL
        self._cache_enabled = runtime_settings.enable_rag_caching
        self._query_cache = LruCache[tuple[str, str], list[float]](
            runtime_settings.query_embedding_cache_size
        )
        self.last_cache_hit = False

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_document(self, text: str, *, use_cache: bool = True) -> list[float]:
        return self._embed_prefixed(DOCUMENT_PREFIX, text, use_cache=use_cache)

    def embed_query(self, text: str, *, use_cache: bool = True) -> list[float]:
        return self._embed_prefixed(QUERY_PREFIX, text, use_cache=use_cache)

    def _embed_prefixed(self, prefix: str, text: str, *, use_cache: bool = True) -> list[float]:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("text must be a non-empty string")

        cache_key = (prefix, normalized)
        should_use_cache = use_cache and self._cache_enabled
        if should_use_cache:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self.last_cache_hit = True
                return cached

        self.last_cache_hit = False

        embedding = self._client.embed(model=self._model_name, text=f"{prefix}{normalized}")
        validated = _validate_embedding(embedding)
        if should_use_cache:
            self._query_cache.set(cache_key, validated)
        return validated


def _validate_embedding(values: list[float]) -> list[float]:
    if len(values) != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIM}, got {len(values)}"
        )

    cleaned: list[float] = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError("embedding contains a non-numeric value")
        if not isinstance(item, (int, float)):
            raise ValueError("embedding contains a non-numeric value")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError("embedding contains a non-finite value")
        cleaned.append(numeric)

    return cleaned


def embed_document(
    text: str,
    *,
    client: OllamaClient | None = None,
    model_name: str | None = None,
    settings: Settings | None = None,
) -> list[float]:
    return Embedder(client=client, model_name=model_name, settings=settings).embed_document(text)


def embed_query(
    text: str,
    *,
    client: OllamaClient | None = None,
    model_name: str | None = None,
    settings: Settings | None = None,
) -> list[float]:
    return Embedder(client=client, model_name=model_name, settings=settings).embed_query(text)
