from __future__ import annotations

import pytest

from meeting_pipeline.config import Settings
from meeting_pipeline.db.pgvector_search import SimilarChunkResult
from meeting_pipeline.rag.models import (
    ConversationState,
    QueryRewriteResult,
    RetrievalBundle,
    RetrievedChunk,
)
from meeting_pipeline.rag.retriever import RetrievalPolicy, Retriever


class FakeRewriter:
    def __init__(self, rewritten_query: str = "standalone rewritten query") -> None:
        self._rewritten_query = rewritten_query
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
            rewritten_query=self._rewritten_query,
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
    assert bundle.retrieval_mode == "default_factoid"
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


def test_retriever_uses_cached_context_for_meta_questions() -> None:
    class MetaRewriter(FakeRewriter):
        def rewrite(
            self,
            latest_user_question: str,
            conversation_context: list[str] | None = None,
        ) -> QueryRewriteResult:
            _ = conversation_context
            return QueryRewriteResult(
                original_query=latest_user_question,
                rewritten_query=latest_user_question,
                used_fallback=True,
                question_relation="meta_chat_scope",
            )

    fake_searcher = FakeSearcher()
    retriever = Retriever(
        searcher=fake_searcher,
        query_rewriter=MetaRewriter(),
        embedder=FakeEmbedder(),
    )

    cached_bundle = RetrievalBundle(
        meeting_id="m1",
        user_query="What did we decide?",
        rewritten_query="meeting decisions",
        top_k_used=1,
        results=[
            RetrievedChunk(
                chunk_id=99,
                meeting_id="m1",
                speaker_label="SPEAKER_00",
                start_time=10.0,
                end_time=12.0,
                content="Cached context.",
                similarity=0.8,
            )
        ],
    )
    conversation_state = ConversationState(latest_bundle=cached_bundle)

    bundle = retriever.retrieve(
        meeting_id="m1",
        user_query="Which parts were uncertain?",
        conversation_state=conversation_state,
    )

    assert bundle.used_cached_context is True
    assert bundle.retrieval_mode == "meta_or_confidence"
    assert fake_searcher.last_meeting_id is None


def test_retriever_applies_speaker_filter_when_requested() -> None:
    class MultiSearcher(FakeSearcher):
        def search_similar_chunks(
            self,
            meeting_id: str,
            query_embedding: list[float],
            top_k: int = 10,
        ) -> list[SimilarChunkResult]:
            _ = query_embedding
            self.last_meeting_id = meeting_id
            self.last_top_k = top_k
            return [
                SimilarChunkResult(
                    chunk_id=1,
                    meeting_id=meeting_id,
                    speaker_label="SPEAKER_01",
                    start_time=5.0,
                    end_time=9.0,
                    content="one",
                    similarity=0.9,
                ),
                SimilarChunkResult(
                    chunk_id=2,
                    meeting_id=meeting_id,
                    speaker_label="SPEAKER_00",
                    start_time=10.0,
                    end_time=13.0,
                    content="two",
                    similarity=0.85,
                ),
            ]

    retriever = Retriever(
        searcher=MultiSearcher(),
        query_rewriter=FakeRewriter(rewritten_query="What did SPEAKER_00 discuss?"),
        embedder=FakeEmbedder(),
    )

    bundle = retriever.retrieve(meeting_id="m1", user_query="What did SPEAKER_00 discuss?")

    assert bundle.retrieval_mode == "speaker_specific"
    assert bundle.speaker_filter == "SPEAKER_00"
    assert len(bundle.results) == 1
    assert bundle.results[0].speaker_label == "SPEAKER_00"


def test_retriever_uses_broad_summary_policy_top_k_when_not_overridden() -> None:
    class BroadRewriter(FakeRewriter):
        def rewrite(
            self,
            latest_user_question: str,
            conversation_context: list[str] | None = None,
        ) -> QueryRewriteResult:
            _ = conversation_context
            return QueryRewriteResult(
                original_query=latest_user_question,
                rewritten_query="Summarize the whole meeting",
                used_fallback=False,
            )

    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=BroadRewriter(),
        embedder=FakeEmbedder(),
    )

    bundle = retriever.retrieve(meeting_id="m1", user_query="Summarize the whole meeting")

    assert bundle.retrieval_mode == "broad_summary"
    assert bundle.top_k_used == 14
    assert searcher.last_top_k == 28


def test_retriever_uses_configured_policy_defaults() -> None:
    settings = Settings(
        _env_file=None,
        default_factoid_top_k=3,
        speaker_specific_top_k=2,
        action_items_or_decisions_top_k=9,
        broad_summary_top_k=11,
        meta_or_confidence_top_k=4,
        broad_summary_max_candidates=15,
    )
    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=FakeRewriter(rewritten_query="what action items were assigned"),
        embedder=FakeEmbedder(),
        settings=settings,
    )

    bundle = retriever.retrieve(meeting_id="m1", user_query="What action items were assigned?")

    assert bundle.retrieval_mode == "action_items_or_decisions"
    assert bundle.top_k_used == 9
    assert searcher.last_top_k == 9


def test_retriever_applies_custom_broad_summary_candidate_cap() -> None:
    class BroadRewriter(FakeRewriter):
        def rewrite(
            self,
            latest_user_question: str,
            conversation_context: list[str] | None = None,
        ) -> QueryRewriteResult:
            _ = latest_user_question
            _ = conversation_context
            return QueryRewriteResult(
                original_query="Summarize the whole meeting",
                rewritten_query="Summarize the whole meeting",
                used_fallback=False,
            )

    searcher = FakeSearcher()
    policy = RetrievalPolicy(
        default_factoid_top_k=5,
        speaker_specific_top_k=4,
        action_items_or_decisions_top_k=8,
        broad_summary_top_k=7,
        meta_or_confidence_top_k=6,
        broad_summary_max_candidates=11,
    )
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=BroadRewriter(),
        embedder=FakeEmbedder(),
        policy=policy,
    )

    bundle = retriever.retrieve(meeting_id="m1", user_query="Summarize the whole meeting")

    assert bundle.top_k_used == 7
    assert searcher.last_top_k == 11


def test_retriever_exposes_timing_metadata() -> None:
    retriever = Retriever(
        searcher=FakeSearcher(),
        query_rewriter=FakeRewriter(),
        embedder=FakeEmbedder(),
    )

    bundle = retriever.retrieve(meeting_id="m1", user_query="What did we decide?")
    timings = bundle.service_metadata.get("timings_ms")

    assert isinstance(timings, dict)
    assert "query_rewrite" in timings
    assert "embedding_query_prep" in timings
    assert "retrieval" in timings
    assert "retrieval_total" in timings
