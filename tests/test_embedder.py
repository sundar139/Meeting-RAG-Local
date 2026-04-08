from __future__ import annotations

import pytest

from meeting_pipeline.config import Settings
from meeting_pipeline.embeddings.embedder import Embedder


class RecordingClient:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding
        self.calls: list[tuple[str, str]] = []

    def embed(self, model: str, text: str) -> list[float]:
        self.calls.append((model, text))
        return self.embedding


def test_embedder_document_and_query_prefixes() -> None:
    client = RecordingClient(embedding=[0.1] * 768)
    embedder = Embedder(client=client, model_name="nomic-test")

    doc_embedding = embedder.embed_document("hello")
    query_embedding = embedder.embed_query("hello")

    assert len(doc_embedding) == 768
    assert len(query_embedding) == 768
    assert client.calls[0] == ("nomic-test", "search_document: hello")
    assert client.calls[1] == ("nomic-test", "search_query: hello")


def test_embedder_rejects_empty_text() -> None:
    client = RecordingClient(embedding=[0.1] * 768)
    embedder = Embedder(client=client, model_name="nomic-test")

    with pytest.raises(ValueError, match="non-empty"):
        embedder.embed_document("   ")


def test_embedder_validates_embedding_dimension() -> None:
    client = RecordingClient(embedding=[0.1] * 10)
    embedder = Embedder(client=client, model_name="nomic-test")

    with pytest.raises(ValueError, match="dimension mismatch"):
        embedder.embed_document("hello")


def test_embedder_caches_identical_query_embeddings() -> None:
    client = RecordingClient(embedding=[0.1] * 768)
    settings = Settings(_env_file=None, enable_rag_caching=True, query_embedding_cache_size=8)
    embedder = Embedder(client=client, model_name="nomic-test", settings=settings)

    first = embedder.embed_query("where are the decisions")
    second = embedder.embed_query("where are the decisions")

    assert len(first) == 768
    assert len(second) == 768
    assert len(client.calls) == 1
    assert embedder.last_cache_hit is True
