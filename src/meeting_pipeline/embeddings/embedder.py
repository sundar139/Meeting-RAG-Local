from __future__ import annotations

import math

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

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_document(self, text: str) -> list[float]:
        return self._embed_prefixed(DOCUMENT_PREFIX, text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_prefixed(QUERY_PREFIX, text)

    def _embed_prefixed(self, prefix: str, text: str) -> list[float]:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("text must be a non-empty string")

        embedding = self._client.embed(model=self._model_name, text=f"{prefix}{normalized}")
        return _validate_embedding(embedding)


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
