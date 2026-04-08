from __future__ import annotations

from meeting_pipeline.app.app import (
    _apply_meeting_selection,
    _apply_transcript_filters,
    _run_rag_services,
    _select_default_meeting,
    _user_facing_error_message,
)
from meeting_pipeline.db.connection import DatabaseConnectionError
from meeting_pipeline.db.repository import TranscriptChunk
from meeting_pipeline.embeddings.ollama_client import OllamaUnavailableError
from meeting_pipeline.rag.models import (
    ConversationState,
    GroundedAnswerResult,
    RetrievalBundle,
)


def test_select_default_meeting_handles_empty_state() -> None:
    assert _select_default_meeting(None, []) is None


def test_apply_meeting_selection_resets_chat_state() -> None:
    state = {
        "selected_meeting": "ES2002a",
        "chat_history": [{"question": "q"}],
        "chat_messages": [{"role": "user", "content": "q"}],
        "latest_rewritten_query": "old",
        "latest_evidence_bundle": object(),
        "latest_answer": object(),
        "pending_question": "old",
    }

    changed = _apply_meeting_selection(state, "ES2002b")

    assert changed is True
    assert state["selected_meeting"] == "ES2002b"
    assert state["chat_history"] == []
    assert state["chat_messages"] == []
    assert state["latest_rewritten_query"] == ""
    assert state["latest_evidence_bundle"] is None
    assert state["latest_answer"] is None
    assert state["pending_question"] is None


def test_apply_transcript_filters_by_speaker_and_text() -> None:
    chunks = [
        TranscriptChunk(
            chunk_id=1,
            meeting_id="m1",
            speaker_label="SPEAKER_00",
            start_time=0.0,
            end_time=1.0,
            content="Launch timeline discussed",
        ),
        TranscriptChunk(
            chunk_id=2,
            meeting_id="m1",
            speaker_label="SPEAKER_01",
            start_time=2.0,
            end_time=3.0,
            content="Action items assigned",
        ),
    ]

    filtered = _apply_transcript_filters(
        chunks,
        speaker_filter="SPEAKER_01",
        text_filter="action",
    )

    assert [item.chunk_id for item in filtered] == [2]


def test_run_rag_services_handles_insufficient_context_flow() -> None:
    class FakeRetriever:
        def retrieve(
            self,
            meeting_id: str,
            user_query: str,
            *,
            conversation_context: list[str] | None = None,
            top_k: int | None = None,
            conversation_state: ConversationState | None = None,
        ) -> RetrievalBundle:
            _ = meeting_id
            _ = user_query
            _ = conversation_context
            _ = top_k
            _ = conversation_state
            return RetrievalBundle(
                meeting_id="m1",
                user_query="What decisions were made?",
                rewritten_query="meeting decisions",
                top_k_used=5,
                results=[],
                retrieval_mode="default_factoid",
            )

    class FakeAnswerGenerator:
        def generate(
            self,
            *,
            user_question: str,
            meeting_id: str,
            retrieved_evidence: list[object],
            rewritten_query: str,
            conversation_context: list[str] | None = None,
            retrieval_mode: str = "default_factoid",
            recent_state: ConversationState | None = None,
        ) -> GroundedAnswerResult:
            _ = user_question
            _ = meeting_id
            _ = retrieved_evidence
            _ = rewritten_query
            _ = conversation_context
            _ = retrieval_mode
            _ = recent_state
            return GroundedAnswerResult(
                meeting_id="m1",
                question="What decisions were made?",
                rewritten_query="meeting decisions",
                sections={"Summary": "Insufficient context to answer."},
                raw_answer="Insufficient context to answer.",
                insufficient_context=True,
            )

    bundle, answer = _run_rag_services(
        meeting_id="m1",
        user_question="What decisions were made?",
        top_k=5,
        conversation_context=[],
        retriever=FakeRetriever(),
        answer_generator=FakeAnswerGenerator(),
        conversation_state=None,
    )

    assert bundle.results == []
    assert answer.insufficient_context is True
    timings = answer.service_metadata.get("timings_ms")
    assert isinstance(timings, dict)
    assert "total_request" in timings


def test_user_facing_error_message_maps_known_service_errors() -> None:
    assert "PostgreSQL" in _user_facing_error_message(DatabaseConnectionError("db down"))
    assert "Ollama" in _user_facing_error_message(OllamaUnavailableError("offline"))
