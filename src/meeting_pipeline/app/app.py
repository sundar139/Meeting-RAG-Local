from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol, cast

from meeting_pipeline.app import components
from meeting_pipeline.app.components import page_title
from meeting_pipeline.config import get_settings
from meeting_pipeline.db.connection import DatabaseConnectionError, connection_scope
from meeting_pipeline.db.pgvector_search import PgVectorSearcher
from meeting_pipeline.db.repository import (
    MeetingOverview,
    TranscriptChunk,
    TranscriptChunkRepository,
)
from meeting_pipeline.embeddings.ollama_client import (
    OllamaMalformedResponseError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from meeting_pipeline.rag.answer_generator import AnswerGenerator
from meeting_pipeline.rag.models import (
    ConversationState,
    ConversationTurnState,
    GroundedAnswerResult,
    RetrievalBundle,
    RetrievalMode,
    RetrievedChunk,
)
from meeting_pipeline.rag.retriever import Retriever
from meeting_pipeline.timing import elapsed_ms, now


class RetrieverProtocol(Protocol):
    def retrieve(
        self,
        meeting_id: str,
        user_query: str,
        *,
        conversation_context: list[str] | None = None,
        top_k: int | None = None,
        conversation_state: ConversationState | None = None,
    ) -> RetrievalBundle: ...


class AnswerGeneratorProtocol(Protocol):
    def generate(
        self,
        *,
        user_question: str,
        meeting_id: str,
        retrieved_evidence: Sequence[RetrievedChunk],
        rewritten_query: str,
        conversation_context: Sequence[str] | None = None,
        retrieval_mode: RetrievalMode = "default_factoid",
        recent_state: ConversationState | None = None,
    ) -> GroundedAnswerResult: ...


class SessionStateProtocol(Protocol):
    def __contains__(self, key: object) -> bool: ...

    def __getitem__(self, key: str | int) -> Any: ...

    def __setitem__(self, key: str | int, value: Any) -> None: ...

    def get(self, key: str | int, default: Any = None) -> Any: ...

    def pop(self, key: str | int, default: Any = None) -> Any: ...


def _ensure_session_state(state: SessionStateProtocol) -> None:
    defaults: dict[str, Any] = {
        "selected_meeting": None,
        "chat_history": [],
        "chat_messages": [],
        "latest_rewritten_query": "",
        "latest_evidence_bundle": None,
        "latest_answer": None,
        "recent_turn_states": [],
        "pending_question": None,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def _reset_chat_state(state: SessionStateProtocol) -> None:
    state["chat_history"] = []
    state["chat_messages"] = []
    state["latest_rewritten_query"] = ""
    state["latest_evidence_bundle"] = None
    state["latest_answer"] = None
    state["recent_turn_states"] = []
    state["pending_question"] = None


def _apply_meeting_selection(state: SessionStateProtocol, meeting_id: str) -> bool:
    current = state.get("selected_meeting")
    if current == meeting_id:
        return False

    state["selected_meeting"] = meeting_id
    _reset_chat_state(state)
    return True


def _select_default_meeting(
    current_meeting: str | None,
    available_meeting_ids: Sequence[str],
) -> str | None:
    if not available_meeting_ids:
        return None
    if current_meeting and current_meeting in available_meeting_ids:
        return current_meeting
    return available_meeting_ids[0]


def _build_conversation_context(
    chat_messages: Sequence[dict[str, str]],
    *,
    max_items: int = 8,
) -> list[str]:
    recent = chat_messages[-max_items:]
    return [f"{item['role']}: {item['content']}" for item in recent]


def _apply_transcript_filters(
    chunks: Sequence[TranscriptChunk],
    *,
    speaker_filter: str,
    text_filter: str,
) -> list[TranscriptChunk]:
    normalized_query = text_filter.strip().lower()

    filtered = list(chunks)
    if speaker_filter != "All speakers":
        filtered = [item for item in filtered if item.speaker_label == speaker_filter]

    if normalized_query:
        filtered = [item for item in filtered if normalized_query in item.content.lower()]

    return filtered


def _build_assistant_message(answer: GroundedAnswerResult) -> str:
    summary = answer.sections.get("Summary", "").strip()
    if summary:
        return summary
    if answer.insufficient_context:
        return (
            "Supporting evidence was limited for this question. "
            "Try broadening scope, asking for a whole-meeting summary, or increasing top-k."
        )
    return "Answer generated from retrieved evidence."


def _user_facing_error_message(error: Exception) -> str:
    if isinstance(error, DatabaseConnectionError):
        return (
            "Database unavailable. Confirm PostgreSQL is running and "
            "POSTGRES_* values are correct."
        )
    if isinstance(error, OllamaUnavailableError):
        return "Ollama unavailable. Start `ollama serve` and verify OLLAMA_HOST."
    if isinstance(error, OllamaModelNotFoundError):
        return "Configured Ollama model was not found. Pull OLLAMA_MODEL and OLLAMA_CHAT_MODEL."
    if isinstance(error, OllamaMalformedResponseError):
        return "Ollama returned an invalid response. Retry once or switch models."
    return "The requested operation failed. Please retry or check local services."


def _extract_timing_map(metadata: dict[str, object]) -> dict[str, float]:
    raw = metadata.get("timings_ms")
    if not isinstance(raw, dict):
        return {}

    parsed: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            parsed[key] = float(value)
    return parsed


def _run_rag_services(
    *,
    meeting_id: str,
    user_question: str,
    top_k: int | None,
    conversation_context: list[str],
    retriever: RetrieverProtocol,
    answer_generator: AnswerGeneratorProtocol,
    conversation_state: ConversationState | None = None,
) -> tuple[RetrievalBundle, GroundedAnswerResult]:
    request_started_at = now()
    bundle = retriever.retrieve(
        meeting_id=meeting_id,
        user_query=user_question,
        conversation_context=conversation_context,
        top_k=top_k,
        conversation_state=conversation_state,
    )
    answer = answer_generator.generate(
        user_question=user_question,
        meeting_id=meeting_id,
        rewritten_query=bundle.rewritten_query,
        retrieved_evidence=bundle.results,
        conversation_context=conversation_context,
        retrieval_mode=bundle.retrieval_mode,
        recent_state=conversation_state,
    )
    merged_timings = _extract_timing_map(bundle.service_metadata)
    merged_timings.update(_extract_timing_map(answer.service_metadata))
    merged_timings["total_request"] = elapsed_ms(request_started_at)

    service_metadata = dict(answer.service_metadata)
    service_metadata["timings_ms"] = merged_timings
    answer = replace(answer, service_metadata=service_metadata)
    return bundle, answer


def _load_meeting_ids() -> list[str]:
    with connection_scope(
        application_name="meeting_pipeline:streamlit_list_meetings"
    ) as connection:
        repository = TranscriptChunkRepository(connection)
        return repository.list_meeting_ids()


def _load_meeting_data(
    meeting_id: str,
) -> tuple[MeetingOverview, list[str], list[TranscriptChunk]]:
    with connection_scope(application_name="meeting_pipeline:streamlit_meeting_data") as connection:
        repository = TranscriptChunkRepository(connection)
        overview = repository.get_meeting_overview(meeting_id)
        speakers = repository.get_distinct_speaker_labels(meeting_id)
        chunks = repository.get_chunks_by_meeting(meeting_id)
    return overview, speakers, chunks


def _execute_chat_turn(
    *,
    meeting_id: str,
    user_question: str,
    top_k: int | None,
    conversation_context: list[str],
    conversation_state: ConversationState | None,
) -> tuple[RetrievalBundle, GroundedAnswerResult]:
    with connection_scope(application_name="meeting_pipeline:streamlit_retrieve") as connection:
        searcher = PgVectorSearcher(connection)
        retriever = Retriever(searcher=searcher)
        answer_generator = AnswerGenerator()
        return _run_rag_services(
            meeting_id=meeting_id,
            user_question=user_question,
            top_k=top_k,
            conversation_context=conversation_context,
            retriever=retriever,
            answer_generator=answer_generator,
            conversation_state=conversation_state,
        )


def main() -> None:
    import streamlit as st

    settings = get_settings()
    st.set_page_config(page_title=page_title(settings.app_name), layout="wide")
    st.title(page_title(settings.app_name))
    st.caption("Local-first transcript browser and grounded QA over ingested meetings.")

    state = cast(SessionStateProtocol, st.session_state)
    _ensure_session_state(state)

    st.sidebar.header("Controls")
    debug_mode = st.sidebar.checkbox("Show technical errors", value=False)
    override_top_k = st.sidebar.checkbox("Override adaptive top-k", value=False)
    top_k_override: int | None = None
    if override_top_k:
        top_k_override = st.sidebar.slider("Evidence top-k", min_value=1, max_value=20, value=8)

    try:
        with st.spinner("Loading meetings..."):
            available_meetings = _load_meeting_ids()
    except Exception as exc:
        components.render_warning(_user_facing_error_message(exc))
        if debug_mode:
            st.exception(exc)
        return

    if not available_meetings:
        components.render_empty_state(
            "No meetings available",
            "No ingested meeting data found. Run embedding ingestion for at least one meeting, "
            "then reload.",
        )
        return

    current = _select_default_meeting(state["selected_meeting"], available_meetings)
    default_index = available_meetings.index(current) if current else 0
    selected_meeting = st.sidebar.selectbox(
        "Meeting ID",
        options=available_meetings,
        index=default_index,
    )
    st.sidebar.caption(f"Meetings available: {len(available_meetings)}")
    _apply_meeting_selection(state, selected_meeting)

    if st.sidebar.button("Clear chat", use_container_width=True):
        _reset_chat_state(state)

    st.sidebar.subheader("Suggested questions")
    suggested_questions = [
        "What decisions were made?",
        "What action items are mentioned?",
        "What did SPEAKER_00 discuss?",
    ]
    for idx, suggested in enumerate(suggested_questions):
        if st.sidebar.button(suggested, key=f"suggested_{idx}", use_container_width=True):
            state["pending_question"] = suggested

    try:
        with st.spinner("Loading transcript and meeting details..."):
            overview, speakers, chunks = _load_meeting_data(selected_meeting)
    except Exception as exc:
        components.render_warning(_user_facing_error_message(exc))
        if debug_mode:
            st.exception(exc)
        return

    components.render_meeting_header(selected_meeting, overview)
    recent_questions = [
        str(turn["question"]) for turn in state["chat_history"] if "question" in turn
    ]
    components.render_meeting_insights(overview, speakers, recent_questions)

    transcript_col, chat_col = st.columns([1.2, 1.0])

    with transcript_col:
        st.markdown("### Transcript")
        speaker_options = ["All speakers", *speakers]
        speaker_filter = st.selectbox("Speaker", options=speaker_options)
        text_filter = st.text_input("Text filter", placeholder="Filter transcript content")
        filtered_chunks = _apply_transcript_filters(
            chunks,
            speaker_filter=speaker_filter,
            text_filter=text_filter,
        )
        st.caption(f"Showing {len(filtered_chunks)} of {len(chunks)} transcript chunks")
        components.render_transcript_rows(filtered_chunks)

    with chat_col:
        st.markdown("### Grounded Chat")

        for turn in state["chat_history"]:
            question = str(turn["question"])
            rewritten_query = str(turn["rewritten_query"])
            answer = turn["answer"]
            evidence = turn["evidence"]
            turn_top_k = int(turn.get("top_k_used", 0) or 0)
            turn_service_metadata = cast(dict[str, object], turn.get("service_metadata") or {})

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                components.render_response_diagnostics(
                    retrieval_mode=str(turn.get("retrieval_mode", "default_factoid")),
                    top_k_used=max(turn_top_k, 1),
                    used_cached_context=bool(turn.get("used_cached_context", False)),
                    insufficient_context=bool(answer.insufficient_context),
                    service_metadata=turn_service_metadata,
                    show_latency=debug_mode,
                )
                st.caption(f"Rewritten query: {rewritten_query}")
                components.render_answer_sections(answer)
                with st.expander("Evidence"):
                    components.render_evidence_panel(evidence)

        prompt_from_input = st.chat_input("Ask a question about this meeting")
        prompt_from_button = state.pop("pending_question", None)
        question = prompt_from_input or prompt_from_button

        if question:
            with st.chat_message("user"):
                st.write(question)

            try:
                with st.spinner("Retrieving evidence and generating grounded answer..."):
                    context = _build_conversation_context(state["chat_messages"])
                    conversation_state = ConversationState(
                        latest_bundle=state.get("latest_evidence_bundle"),
                        latest_answer=state.get("latest_answer"),
                        recent_turns=state.get("recent_turn_states") or [],
                    )
                    bundle, answer = _execute_chat_turn(
                        meeting_id=selected_meeting,
                        user_question=question,
                        top_k=top_k_override,
                        conversation_context=context,
                        conversation_state=conversation_state,
                    )
            except Exception as exc:
                components.render_warning(_user_facing_error_message(exc))
                if debug_mode:
                    st.exception(exc)
                return

            state["latest_rewritten_query"] = bundle.rewritten_query
            state["latest_evidence_bundle"] = bundle
            state["latest_answer"] = answer

            assistant_message = _build_assistant_message(answer)
            state["chat_messages"].append({"role": "user", "content": question})
            state["chat_messages"].append({"role": "assistant", "content": assistant_message})
            state["recent_turn_states"].append(
                ConversationTurnState(
                    question=question,
                    rewritten_query=bundle.rewritten_query,
                    retrieval_mode=bundle.retrieval_mode,
                    answer_summary=answer.sections.get("Summary", ""),
                    insufficient_context=answer.insufficient_context,
                )
            )
            if len(state["recent_turn_states"]) > 10:
                state["recent_turn_states"] = state["recent_turn_states"][-10:]
            state["chat_history"].append(
                {
                    "question": question,
                    "rewritten_query": bundle.rewritten_query,
                    "answer": answer,
                    "evidence": bundle.results,
                    "retrieval_mode": bundle.retrieval_mode,
                    "top_k_used": bundle.top_k_used,
                    "used_cached_context": bundle.used_cached_context,
                    "service_metadata": answer.service_metadata,
                }
            )

            with st.chat_message("assistant"):
                components.render_response_diagnostics(
                    retrieval_mode=bundle.retrieval_mode,
                    top_k_used=bundle.top_k_used,
                    used_cached_context=bundle.used_cached_context,
                    insufficient_context=answer.insufficient_context,
                    service_metadata=answer.service_metadata,
                    show_latency=debug_mode,
                )
                st.caption(f"Rewritten query: {bundle.rewritten_query}")
                components.render_answer_sections(answer)
                with st.expander("Evidence", expanded=answer.insufficient_context):
                    components.render_evidence_panel(bundle.results)


if __name__ == "__main__":
    main()
