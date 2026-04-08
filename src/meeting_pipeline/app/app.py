from __future__ import annotations

from collections.abc import Sequence
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
from meeting_pipeline.rag.models import GroundedAnswerResult, RetrievalBundle
from meeting_pipeline.rag.retriever import Retriever


class RetrieverProtocol(Protocol):
    def retrieve(
        self,
        meeting_id: str,
        user_query: str,
        *,
        conversation_context: list[str] | None = None,
        top_k: int = 5,
    ) -> RetrievalBundle: ...


class AnswerGeneratorProtocol(Protocol):
    def generate(
        self,
        *,
        user_question: str,
        meeting_id: str,
        retrieved_evidence: Sequence[Any],
        rewritten_query: str,
        conversation_context: Sequence[str] | None = None,
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
            "Retrieved evidence was not sufficient to provide a grounded answer. "
            "Try refining the question or increasing evidence top-k."
        )
    return "Answer generated from retrieved evidence."


def _user_facing_error_message(error: Exception) -> str:
    if isinstance(error, DatabaseConnectionError):
        return "PostgreSQL is unavailable. Verify DB settings and ensure the server is running."
    if isinstance(error, OllamaUnavailableError):
        return "Ollama is unavailable. Start Ollama and verify OLLAMA_HOST."
    if isinstance(error, OllamaModelNotFoundError):
        return "Required Ollama model is missing. Pull the configured embedding/chat models."
    if isinstance(error, OllamaMalformedResponseError):
        return "Ollama returned malformed output. Try again or switch models."
    return "The requested operation failed. Please retry or check local services."


def _run_rag_services(
    *,
    meeting_id: str,
    user_question: str,
    top_k: int,
    conversation_context: list[str],
    retriever: RetrieverProtocol,
    answer_generator: AnswerGeneratorProtocol,
) -> tuple[RetrievalBundle, GroundedAnswerResult]:
    bundle = retriever.retrieve(
        meeting_id=meeting_id,
        user_query=user_question,
        conversation_context=conversation_context,
        top_k=top_k,
    )
    answer = answer_generator.generate(
        user_question=user_question,
        meeting_id=meeting_id,
        rewritten_query=bundle.rewritten_query,
        retrieved_evidence=bundle.results,
        conversation_context=conversation_context,
    )
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
    top_k: int,
    conversation_context: list[str],
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
    top_k = st.sidebar.slider("Evidence top-k", min_value=1, max_value=10, value=5)

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
            "No ingested meeting data was found. Run the ingestion step, then reload this app.",
        )
        return

    current = _select_default_meeting(state["selected_meeting"], available_meetings)
    default_index = available_meetings.index(current) if current else 0
    selected_meeting = st.sidebar.selectbox(
        "Meeting ID",
        options=available_meetings,
        index=default_index,
    )
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

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
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
                    bundle, answer = _execute_chat_turn(
                        meeting_id=selected_meeting,
                        user_question=question,
                        top_k=top_k,
                        conversation_context=context,
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
            state["chat_history"].append(
                {
                    "question": question,
                    "rewritten_query": bundle.rewritten_query,
                    "answer": answer,
                    "evidence": bundle.results,
                }
            )

            with st.chat_message("assistant"):
                st.caption(f"Rewritten query: {bundle.rewritten_query}")
                components.render_answer_sections(answer)
                with st.expander("Evidence", expanded=answer.insufficient_context):
                    components.render_evidence_panel(bundle.results)


if __name__ == "__main__":
    main()
