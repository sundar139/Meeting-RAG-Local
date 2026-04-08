from __future__ import annotations

import pytest

from meeting_pipeline.config import Settings
from meeting_pipeline.db.pgvector_search import SimilarChunkResult
from meeting_pipeline.rag.models import (
    ConversationState,
    ConversationTurnState,
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
        *,
        use_cache: bool = True,
        fast_mode: bool = False,
    ) -> QueryRewriteResult:
        _ = use_cache
        _ = fast_mode
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
        self.last_cache_hit = False

    def embed_query(self, text: str, *, use_cache: bool = True) -> list[float]:
        _ = use_cache
        self.last_text = text
        self.last_cache_hit = False
        return [0.1] * 768


class FakeSearcher:
    def __init__(self) -> None:
        self.last_meeting_id: str | None = None
        self.last_embedding: list[float] | None = None
        self.last_top_k: int | None = None
        self.last_speaker_label: str | None = None
        self.call_count = 0

    def search_similar_chunks(
        self,
        meeting_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        speaker_label: str | None = None,
    ) -> list[SimilarChunkResult]:
        self.call_count += 1
        self.last_meeting_id = meeting_id
        self.last_embedding = query_embedding
        self.last_top_k = top_k
        self.last_speaker_label = speaker_label
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
            speaker_label: str | None = None,
        ) -> list[SimilarChunkResult]:
            _ = query_embedding
            _ = top_k
            _ = speaker_label
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
            *,
            use_cache: bool = True,
            fast_mode: bool = False,
        ) -> QueryRewriteResult:
            _ = conversation_context
            _ = use_cache
            _ = fast_mode
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
            speaker_label: str | None = None,
        ) -> list[SimilarChunkResult]:
            _ = query_embedding
            self.last_meeting_id = meeting_id
            self.last_top_k = top_k
            rows = [
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
            if speaker_label is None:
                return rows
            return [row for row in rows if row.speaker_label == speaker_label]

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
            *,
            use_cache: bool = True,
            fast_mode: bool = False,
        ) -> QueryRewriteResult:
            _ = conversation_context
            _ = use_cache
            _ = fast_mode
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
    assert searcher.last_top_k == 56


def test_retriever_uses_configured_policy_defaults() -> None:
    settings = Settings.model_validate(
        {
            "default_factoid_top_k": 3,
            "speaker_specific_top_k": 2,
            "action_items_or_decisions_top_k": 9,
            "broad_summary_top_k": 11,
            "meta_or_confidence_top_k": 4,
            "broad_summary_max_candidates": 15,
        }
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
            *,
            use_cache: bool = True,
            fast_mode: bool = False,
        ) -> QueryRewriteResult:
            _ = latest_user_question
            _ = conversation_context
            _ = use_cache
            _ = fast_mode
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
    assert searcher.last_top_k == 28


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
    assert "query_embedding" in timings
    assert "embedding_query_prep" in timings
    assert "postgres_retrieval" in timings
    assert "retrieval" in timings
    assert "retrieval_total" in timings


def test_retriever_uses_retrieval_bundle_cache_for_identical_requests() -> None:
    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=FakeRewriter(),
        embedder=FakeEmbedder(),
    )

    first = retriever.retrieve(meeting_id="m1", user_query="What did we decide?")
    second = retriever.retrieve(meeting_id="m1", user_query="What did we decide?")

    assert first.service_metadata.get("cache") is not None
    assert second.service_metadata.get("cache") is not None
    second_cache = second.service_metadata["cache"]
    assert isinstance(second_cache, dict)
    assert second_cache.get("retrieval_bundle") is True
    assert searcher.call_count == 1


def test_retriever_fast_mode_caps_policy_top_k_when_not_overridden() -> None:
    settings = Settings.model_validate(
        {
            "default_factoid_top_k": 7,
            "fast_mode_policy_top_k_cap": 3,
        }
    )
    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=FakeRewriter(),
        embedder=FakeEmbedder(),
        settings=settings,
    )

    bundle = retriever.retrieve(meeting_id="m1", user_query="status", fast_mode=True)

    assert bundle.top_k_used == 3
    assert searcher.last_top_k == 3


def test_retriever_routes_broad_topic_questions_to_broad_summary() -> None:
    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=FakeRewriter(rewritten_query="What concerns or risks were raised?"),
        embedder=FakeEmbedder(),
    )

    bundle = retriever.retrieve(meeting_id="m1", user_query="What concerns or risks were raised?")

    assert bundle.retrieval_mode == "broad_summary"
    assert searcher.last_top_k is not None
    assert searcher.last_top_k > bundle.top_k_used


def test_retriever_routes_speaker_topic_comparison_to_broad_summary() -> None:
    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=FakeRewriter(
            rewritten_query="Which speaker talked the most about planning or decision-making?"
        ),
        embedder=FakeEmbedder(),
    )

    bundle = retriever.retrieve(
        meeting_id="m1",
        user_query="Which speaker talked the most about planning or decision-making?",
    )

    assert bundle.retrieval_mode == "broad_summary"


def test_retriever_broad_summary_dedupes_overlapping_duplicate_chunks() -> None:
    class DuplicateHeavySearcher(FakeSearcher):
        def search_similar_chunks(
            self,
            meeting_id: str,
            query_embedding: list[float],
            top_k: int = 10,
            speaker_label: str | None = None,
        ) -> list[SimilarChunkResult]:
            _ = query_embedding
            _ = top_k
            _ = speaker_label
            return [
                SimilarChunkResult(
                    chunk_id=1,
                    meeting_id=meeting_id,
                    speaker_label="SPEAKER_00",
                    start_time=0.0,
                    end_time=40.0,
                    content="Window planning and launch decisions",
                    similarity=0.95,
                    chunk_key="chunk-a",
                ),
                SimilarChunkResult(
                    chunk_id=2,
                    meeting_id=meeting_id,
                    speaker_label="SPEAKER_00",
                    start_time=0.0,
                    end_time=40.0,
                    content="Window planning and launch decisions",
                    similarity=0.94,
                    chunk_key="chunk-a",
                ),
                SimilarChunkResult(
                    chunk_id=3,
                    meeting_id=meeting_id,
                    speaker_label="SPEAKER_00",
                    start_time=5.0,
                    end_time=45.0,
                    content="Window planning and launch decisions",
                    similarity=0.93,
                    chunk_key="chunk-b",
                ),
                SimilarChunkResult(
                    chunk_id=4,
                    meeting_id=meeting_id,
                    speaker_label="SPEAKER_01",
                    start_time=120.0,
                    end_time=150.0,
                    content="Risk discussion and mitigation planning",
                    similarity=0.91,
                    chunk_key="chunk-c",
                ),
            ]

    retriever = Retriever(
        searcher=DuplicateHeavySearcher(),
        query_rewriter=FakeRewriter(rewritten_query="Summarize the whole meeting"),
        embedder=FakeEmbedder(),
    )

    bundle = retriever.retrieve(
        meeting_id="m1",
        user_query="Summarize the whole meeting",
        top_k=3,
    )

    assert bundle.retrieval_mode == "broad_summary"
    assert [item.chunk_id for item in bundle.results] == [1, 4]


def test_retriever_routes_disagreement_questions_to_broad_summary() -> None:
    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=FakeRewriter(rewritten_query="Did the speakers disagree on anything?"),
        embedder=FakeEmbedder(),
    )

    bundle = retriever.retrieve(
        meeting_id="m1",
        user_query="Did the speakers disagree on anything?",
    )

    assert bundle.retrieval_mode == "broad_summary"


def test_retriever_routes_meeting_scoped_decisions_to_broad_summary() -> None:
    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=FakeRewriter(rewritten_query="What decisions were made in this meeting?"),
        embedder=FakeEmbedder(),
    )

    bundle = retriever.retrieve(
        meeting_id="m1",
        user_query="What decisions were made in this meeting?",
    )

    assert bundle.retrieval_mode == "broad_summary"


def test_retriever_meta_question_uses_recent_state_without_topical_retrieval() -> None:
    searcher = FakeSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=FakeRewriter(rewritten_query="Which of these answers are low confidence?"),
        embedder=FakeEmbedder(),
    )

    conversation_state = ConversationState(
        recent_turns=[
            ConversationTurnState(
                question="What decisions were made?",
                rewritten_query="meeting decisions",
                retrieval_mode="broad_summary",
                answer_summary="Some decisions were found.",
                insufficient_context=True,
            )
        ]
    )

    bundle = retriever.retrieve(
        meeting_id="m1",
        user_query="Which of these answers are low confidence?",
        conversation_state=conversation_state,
    )

    assert bundle.retrieval_mode == "meta_or_confidence"
    assert bundle.used_cached_context is True
    assert bundle.results == []
    assert searcher.call_count == 0

    routing = bundle.service_metadata.get("routing")
    assert isinstance(routing, dict)
    assert routing.get("meta_scope") == "recent_conversation"
