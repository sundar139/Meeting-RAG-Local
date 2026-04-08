from __future__ import annotations

from meeting_pipeline.app.app import _build_conversation_context
from meeting_pipeline.db.pgvector_search import SimilarChunkResult
from meeting_pipeline.rag.models import (
    ConversationState,
    ConversationTurnState,
    GroundedAnswerResult,
    QueryRewriteResult,
)
from meeting_pipeline.rag.retriever import RetrievalPolicy, Retriever


class ScriptedRewriter:
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
        normalized = " ".join(latest_user_question.split())
        lower = normalized.lower()

        if "previous answer" in lower or "uncertain" in lower:
            return QueryRewriteResult(
                original_query=normalized,
                rewritten_query=normalized,
                used_fallback=True,
                question_relation="meta_chat_scope",
            )

        if "summarize" in lower or "overall" in lower:
            return QueryRewriteResult(
                original_query=normalized,
                rewritten_query="Summarize the whole meeting",
                used_fallback=False,
            )

        return QueryRewriteResult(
            original_query=normalized,
            rewritten_query=normalized,
            used_fallback=False,
        )


class FakeEmbedder:
    def __init__(self) -> None:
        self.last_cache_hit = False

    def embed_query(self, text: str, *, use_cache: bool = True) -> list[float]:
        _ = text
        _ = use_cache
        self.last_cache_hit = False
        return [0.1, 0.2, 0.3]


class ScenarioSearcher:
    def __init__(self) -> None:
        self.call_count = 0

    def search_similar_chunks(
        self,
        meeting_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        speaker_label: str | None = None,
    ) -> list[SimilarChunkResult]:
        _ = query_embedding
        _ = top_k
        self.call_count += 1
        rows = [
            SimilarChunkResult(
                chunk_id=1,
                meeting_id=meeting_id,
                speaker_label="SPEAKER_00",
                start_time=10.0,
                end_time=20.0,
                content="SPEAKER_00 proposed the timeline.",
                similarity=0.95,
            ),
            SimilarChunkResult(
                chunk_id=2,
                meeting_id=meeting_id,
                speaker_label="SPEAKER_01",
                start_time=25.0,
                end_time=35.0,
                content="SPEAKER_01 accepted the timeline.",
                similarity=0.92,
            ),
            SimilarChunkResult(
                chunk_id=3,
                meeting_id=meeting_id,
                speaker_label="SPEAKER_00",
                start_time=50.0,
                end_time=62.0,
                content="SPEAKER_00 volunteered to own follow-up.",
                similarity=0.9,
            ),
        ]
        if speaker_label is None:
            return rows
        return [row for row in rows if row.speaker_label == speaker_label]


def _build_answer(bundle_mode: str, question: str, rewritten: str) -> GroundedAnswerResult:
    return GroundedAnswerResult(
        meeting_id="m1",
        question=question,
        rewritten_query=rewritten,
        sections={
            "Summary": f"Handled with mode: {bundle_mode}",
            "Key Points": "Test answer",
            "Decisions": "Test decision",
            "Action Items": "Test action",
            "Uncertainties / Missing Evidence": "None",
        },
        raw_answer="raw",
        insufficient_context=False,
    )


def test_multiturn_flow_preserves_intent_across_routing_modes() -> None:
    searcher = ScenarioSearcher()
    retriever = Retriever(
        searcher=searcher,
        query_rewriter=ScriptedRewriter(),
        embedder=FakeEmbedder(),
        policy=RetrievalPolicy(
            default_factoid_top_k=3,
            speaker_specific_top_k=2,
            action_items_or_decisions_top_k=3,
            broad_summary_top_k=3,
            meta_or_confidence_top_k=2,
            broad_summary_max_candidates=4,
        ),
    )

    chat_messages: list[dict[str, str]] = []
    recent_turns: list[ConversationTurnState] = []
    conversation_state: ConversationState | None = None

    question_1 = "Summarize the whole meeting"
    bundle_1 = retriever.retrieve(
        meeting_id="m1",
        user_query=question_1,
        conversation_context=_build_conversation_context(chat_messages),
        conversation_state=conversation_state,
    )
    answer_1 = _build_answer(bundle_1.retrieval_mode, question_1, bundle_1.rewritten_query)
    assert bundle_1.retrieval_mode == "broad_summary"
    assert bundle_1.used_cached_context is False
    assert searcher.call_count == 1

    chat_messages.extend(
        [
            {"role": "user", "content": question_1},
            {"role": "assistant", "content": answer_1.sections["Summary"]},
        ]
    )
    recent_turns.append(
        ConversationTurnState(
            question=question_1,
            rewritten_query=bundle_1.rewritten_query,
            retrieval_mode=bundle_1.retrieval_mode,
            answer_summary=answer_1.sections["Summary"],
            insufficient_context=answer_1.insufficient_context,
        )
    )
    conversation_state = ConversationState(
        latest_bundle=bundle_1,
        latest_answer=answer_1,
        recent_turns=recent_turns,
    )

    question_2 = "What did SPEAKER_00 say about ownership?"
    bundle_2 = retriever.retrieve(
        meeting_id="m1",
        user_query=question_2,
        conversation_context=_build_conversation_context(chat_messages),
        conversation_state=conversation_state,
    )
    answer_2 = _build_answer(bundle_2.retrieval_mode, question_2, bundle_2.rewritten_query)
    assert bundle_2.retrieval_mode == "speaker_specific"
    assert bundle_2.speaker_filter == "SPEAKER_00"
    assert all(item.speaker_label == "SPEAKER_00" for item in bundle_2.results)
    assert searcher.call_count == 2

    chat_messages.extend(
        [
            {"role": "user", "content": question_2},
            {"role": "assistant", "content": answer_2.sections["Summary"]},
        ]
    )
    recent_turns.append(
        ConversationTurnState(
            question=question_2,
            rewritten_query=bundle_2.rewritten_query,
            retrieval_mode=bundle_2.retrieval_mode,
            answer_summary=answer_2.sections["Summary"],
            insufficient_context=answer_2.insufficient_context,
        )
    )
    conversation_state = ConversationState(
        latest_bundle=bundle_2,
        latest_answer=answer_2,
        recent_turns=recent_turns,
    )

    question_3 = "Which parts of your previous answer are uncertain?"
    bundle_3 = retriever.retrieve(
        meeting_id="m1",
        user_query=question_3,
        conversation_context=_build_conversation_context(chat_messages),
        conversation_state=conversation_state,
    )

    assert bundle_3.retrieval_mode == "meta_or_confidence"
    assert bundle_3.used_cached_context is True
    assert searcher.call_count == 2
    assert bundle_3.results == bundle_2.results[: bundle_3.top_k_used]
