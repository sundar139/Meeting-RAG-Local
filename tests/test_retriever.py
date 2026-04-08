from __future__ import annotations

import pytest

from meeting_pipeline.db.pgvector_search import SimilarChunkResult
from meeting_pipeline.rag.models import QueryRewriteResult
from meeting_pipeline.rag.retriever import Retriever


class FakeRewriter:
    def __init__(self) -> None:
        self.last_question: str | None = None
        self.last_context: list[str] | None = None

    def rewrite(
        self,
        latest_user_question: str,
        conversation_context: list[str] | None = None,
    ) -> QueryRewriteResult:
        self.last_question = latest_user_question
        self.last_context = conversation_context
        return QueryRewriteResult(
            original_query=latest_user_question,
            rewritten_query="standalone rewritten query",
            used_fallback=False,
        )


class FakeEmbedder:
    def __init__(self) -> None:
        self.last_text: str | None = None

    def embed_query(self, text: str) -> list[float]:
        self.last_text = text
        return [0.1] * 768


class FakeSearcher:
    def __init__(self) -> None:
        self.last_meeting_id: str | None = None
        self.last_embedding: list[float] | None = None
        self.last_top_k: int | None = None

    def search_similar_chunks(
        self,
        meeting_id: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[SimilarChunkResult]:
        self.last_meeting_id = meeting_id
        self.last_embedding = query_embedding
        self.last_top_k = top_k
        return [
            SimilarChunkResult(
                chunk_id=1,
                meeting_id=meeting_id,
                speaker_label="SPEAKER_01",
                start_time=12.0,
                end_time=18.2,
                content="We agreed to ship by Friday.",
                similarity=0.92,
            )
        ]


def test_retriever_orchestrates_rewrite_embed_and_search() -> None:
    fake_rewriter = FakeRewriter()
    fake_embedder = FakeEmbedder()
    fake_searcher = FakeSearcher()

    retriever = Retriever(
        searcher=fake_searcher,
        query_rewriter=fake_rewriter,
        embedder=fake_embedder,
    )

    bundle = retriever.retrieve(
        meeting_id="m1",
        user_query=" What did we decide? ",
        conversation_context=["Talked about launch timeline"],
        top_k=3,
    )

    assert fake_rewriter.last_question == "What did we decide?"
    assert fake_rewriter.last_context == ["Talked about launch timeline"]
    assert fake_embedder.last_text == "standalone rewritten query"
    assert fake_searcher.last_meeting_id == "m1"
    assert fake_searcher.last_top_k == 3

    assert bundle.meeting_id == "m1"
    assert bundle.user_query == "What did we decide?"
    assert bundle.rewritten_query == "standalone rewritten query"
    assert bundle.top_k_used == 3
    assert len(bundle.results) == 1
    assert bundle.results[0].speaker_label == "SPEAKER_01"
    assert bundle.results[0].similarity == 0.92


def test_retriever_returns_empty_bundle_when_no_hits() -> None:
    class EmptySearcher(FakeSearcher):
        def search_similar_chunks(
            self,
            meeting_id: str,
            query_embedding: list[float],
            top_k: int = 10,
        ) -> list[SimilarChunkResult]:
            _ = query_embedding
            _ = top_k
            self.last_meeting_id = meeting_id
            return []

    retriever = Retriever(
        searcher=EmptySearcher(),
        query_rewriter=FakeRewriter(),
        embedder=FakeEmbedder(),
    )
    bundle = retriever.retrieve(meeting_id="m1", user_query="status")

    assert bundle.results == []


def test_retriever_validates_inputs() -> None:
    retriever = Retriever(
        searcher=FakeSearcher(),
        query_rewriter=FakeRewriter(),
        embedder=FakeEmbedder(),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(meeting_id="", user_query="status")

    with pytest.raises(ValueError):
        retriever.retrieve(meeting_id="m1", user_query="   ")

    with pytest.raises(ValueError):
        retriever.retrieve(meeting_id="m1", user_query="status", top_k=0)
