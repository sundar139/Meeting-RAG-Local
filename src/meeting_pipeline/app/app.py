from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Protocol, cast

from meeting_pipeline.app import components
from meeting_pipeline.app.components import page_title
from meeting_pipeline.cache_utils import LruCache
from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.db.connection import DatabaseConnectionError, connection_scope
from meeting_pipeline.db.pgvector_search import PgVectorSearcher
from meeting_pipeline.db.repository import (
    MeetingOverview,
    TranscriptChunk,
    TranscriptChunkRepository,
)
from meeting_pipeline.embeddings.embedder import Embedder
from meeting_pipeline.embeddings.ollama_client import (
    OllamaClient,
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
from meeting_pipeline.rag.query_rewriter import QueryRewriter
from meeting_pipeline.rag.retriever import RetrievalCacheKey, Retriever
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
        use_cache: bool = True,
        fast_mode: bool = False,
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
        use_cache: bool = True,
        fast_mode: bool = False,
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
        "_cached_meeting_ids": None,
        "_cached_meeting_data": {},
        "_shared_ollama_client": None,
        "_shared_query_rewriter": None,
        "_shared_embedder": None,
        "_shared_answer_generator": None,
        "_shared_retrieval_cache": None,
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


def _reset_meeting_cache(state: SessionStateProtocol) -> None:
    state["_cached_meeting_ids"] = None
    state["_cached_meeting_data"] = {}


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


def _format_rewritten_query_caption(
    *,
    question: str,
    rewritten_query: str,
    service_metadata: dict[str, object] | None,
) -> str:
    if service_metadata is not None:
        rewrite_meta = service_metadata.get("rewrite")
        if isinstance(rewrite_meta, dict) and rewrite_meta.get("used_fallback") is True:
            fallback_reason = rewrite_meta.get("fallback_reason")
            reason_label = (
                str(fallback_reason).replace("_", " ")
                if isinstance(fallback_reason, str) and fallback_reason
                else "safety fallback"
            )
            return f"Rewritten query (fallback to original; {reason_label}): {question}"
    return f"Rewritten query: {rewritten_query}"


def _meeting_availability_hint(available_meetings: Sequence[str]) -> str | None:
    if len(available_meetings) != 1:
        return None
    return (
        "Only one ingested meeting is available. For multi-meeting demos, run readiness "
        "checks and batch ingestion, then refresh meeting metadata."
    )


def _user_facing_error_message(error: Exception) -> str:
    if isinstance(error, DatabaseConnectionError):
        return (
            "PostgreSQL database unavailable. Confirm PostgreSQL is running and "
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


def _extract_cache_map(metadata: dict[str, object]) -> dict[str, bool]:
    raw = metadata.get("cache")
    if not isinstance(raw, dict):
        return {}

    parsed: dict[str, bool] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, bool):
            parsed[key] = value
    return parsed


def _get_shared_services(
    state: SessionStateProtocol,
    *,
    settings: Settings,
) -> tuple[QueryRewriter, Embedder, AnswerGenerator, LruCache[RetrievalCacheKey, RetrievalBundle]]:
    shared_client = state.get("_shared_ollama_client")
    if not isinstance(shared_client, OllamaClient):
        shared_client = OllamaClient.from_settings(settings)
        state["_shared_ollama_client"] = shared_client

    rewriter = state.get("_shared_query_rewriter")
    if not isinstance(rewriter, QueryRewriter):
        rewriter = QueryRewriter(client=shared_client, settings=settings)
        state["_shared_query_rewriter"] = rewriter

    embedder = state.get("_shared_embedder")
    if not isinstance(embedder, Embedder):
        embedder = Embedder(client=shared_client, settings=settings)
        state["_shared_embedder"] = embedder

    answer_generator = state.get("_shared_answer_generator")
    if not isinstance(answer_generator, AnswerGenerator):
        answer_generator = AnswerGenerator(client=shared_client, settings=settings)
        state["_shared_answer_generator"] = answer_generator

    retrieval_cache = state.get("_shared_retrieval_cache")
    if not isinstance(retrieval_cache, LruCache):
        retrieval_cache = LruCache[RetrievalCacheKey, RetrievalBundle](
            settings.retrieval_bundle_cache_size
        )
        state["_shared_retrieval_cache"] = retrieval_cache

    return rewriter, embedder, answer_generator, retrieval_cache


def _run_rag_services(
    *,
    meeting_id: str,
    user_question: str,
    top_k: int | None,
    conversation_context: list[str],
    retriever: RetrieverProtocol,
    answer_generator: AnswerGeneratorProtocol,
    conversation_state: ConversationState | None = None,
    use_cache: bool = True,
    fast_mode: bool = False,
    progress_reporter: Callable[[str], None] | None = None,
) -> tuple[RetrievalBundle, GroundedAnswerResult]:
    request_started_at = now()
    if progress_reporter is not None:
        progress_reporter("Rewriting, embedding, and retrieving evidence")
    bundle = retriever.retrieve(
        meeting_id=meeting_id,
        user_query=user_question,
        conversation_context=conversation_context,
        top_k=top_k,
        conversation_state=conversation_state,
        use_cache=use_cache,
        fast_mode=fast_mode,
    )
    if progress_reporter is not None:
        progress_reporter("Generating grounded answer")
    answer = answer_generator.generate(
        user_question=user_question,
        meeting_id=meeting_id,
        rewritten_query=bundle.rewritten_query,
        retrieved_evidence=bundle.results,
        conversation_context=conversation_context,
        retrieval_mode=bundle.retrieval_mode,
        recent_state=conversation_state,
        use_cache=use_cache,
        fast_mode=fast_mode,
    )
    merged_timings = _extract_timing_map(bundle.service_metadata)
    merged_timings.update(_extract_timing_map(answer.service_metadata))
    merged_timings["total_request"] = elapsed_ms(request_started_at)

    merged_cache = _extract_cache_map(bundle.service_metadata)
    merged_cache.update(_extract_cache_map(answer.service_metadata))

    service_metadata = dict(answer.service_metadata)
    service_metadata["timings_ms"] = merged_timings
    service_metadata["cache"] = merged_cache
    service_metadata["fast_mode"] = bool(
        service_metadata.get("fast_mode") or bundle.service_metadata.get("fast_mode")
    )
    bundle_routing = bundle.service_metadata.get("routing")
    if isinstance(bundle_routing, dict):
        service_metadata["routing"] = bundle_routing

    bundle_rewrite = bundle.service_metadata.get("rewrite")
    if isinstance(bundle_rewrite, dict):
        service_metadata["rewrite"] = bundle_rewrite

    answer = replace(answer, service_metadata=service_metadata)

    if progress_reporter is not None:
        progress_reporter("Completed")
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


def _load_meeting_ids_cached(
    state: SessionStateProtocol,
    *,
    force_refresh: bool = False,
) -> list[str]:
    if not force_refresh:
        cached = state.get("_cached_meeting_ids")
        if isinstance(cached, list):
            return [str(item) for item in cached]

    loaded = _load_meeting_ids()
    state["_cached_meeting_ids"] = loaded
    return loaded


def _load_meeting_data_cached(
    state: SessionStateProtocol,
    meeting_id: str,
    *,
    force_refresh: bool = False,
) -> tuple[MeetingOverview, list[str], list[TranscriptChunk]]:
    cache = state.get("_cached_meeting_data")
    if not isinstance(cache, dict):
        cache = {}

    if not force_refresh and meeting_id in cache:
        cached_value = cache[meeting_id]
        if isinstance(cached_value, tuple) and len(cached_value) == 3:
            return cast(tuple[MeetingOverview, list[str], list[TranscriptChunk]], cached_value)

    loaded = _load_meeting_data(meeting_id)
    cache[meeting_id] = loaded
    state["_cached_meeting_data"] = cache
    return loaded


def _execute_chat_turn(
    *,
    meeting_id: str,
    user_question: str,
    top_k: int | None,
    conversation_context: list[str],
    conversation_state: ConversationState | None,
    query_rewriter: QueryRewriter,
    embedder: Embedder,
    answer_generator: AnswerGenerator,
    retrieval_cache: LruCache[RetrievalCacheKey, RetrievalBundle],
    settings: Settings,
    use_cache: bool,
    fast_mode: bool,
    progress_reporter: Callable[[str], None] | None = None,
) -> tuple[RetrievalBundle, GroundedAnswerResult]:
    with connection_scope(application_name="meeting_pipeline:streamlit_retrieve") as connection:
        searcher = PgVectorSearcher(connection)
        retriever = Retriever(
            searcher=searcher,
            query_rewriter=query_rewriter,
            embedder=embedder,
            settings=settings,
            retrieval_cache=retrieval_cache,
        )
        return _run_rag_services(
            meeting_id=meeting_id,
            user_question=user_question,
            top_k=top_k,
            conversation_context=conversation_context,
            retriever=retriever,
            answer_generator=answer_generator,
            conversation_state=conversation_state,
            use_cache=use_cache,
            fast_mode=fast_mode,
            progress_reporter=progress_reporter,
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
    fast_mode = st.sidebar.checkbox(
        "Fast mode (lower latency)",
        value=settings.enable_fast_mode,
        help="Uses conservative latency optimizations for demos while keeping grounded answers.",
    )
    override_top_k = st.sidebar.checkbox("Override adaptive top-k", value=False)
    top_k_override: int | None = None
    if override_top_k:
        top_k_override = st.sidebar.slider("Evidence top-k", min_value=1, max_value=20, value=8)
    refresh_meeting_cache = st.sidebar.button("Refresh meeting metadata", use_container_width=True)
    if refresh_meeting_cache:
        _reset_meeting_cache(state)

    try:
        with st.spinner("Loading meetings..."):
            available_meetings = _load_meeting_ids_cached(state)
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

    meeting_hint = _meeting_availability_hint(available_meetings)
    if meeting_hint:
        st.sidebar.info(meeting_hint)

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

    st.sidebar.subheader("Whole-meeting summary")
    if st.sidebar.button("Whole-meeting summary", use_container_width=True):
        state["pending_question"] = "Summarize the whole meeting with grounded evidence."
    if st.sidebar.button("Whole-meeting summary (5 bullets)", use_container_width=True):
        state["pending_question"] = "Summarize the whole meeting in 5 bullet points."

    try:
        with st.spinner("Loading transcript and meeting details..."):
            overview, speakers, chunks = _load_meeting_data_cached(state, selected_meeting)
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
                    confidence_tier=str(turn.get("confidence_tier", answer.confidence_tier)),
                    service_metadata=turn_service_metadata,
                    show_latency=debug_mode,
                )
                st.caption(
                    _format_rewritten_query_caption(
                        question=question,
                        rewritten_query=rewritten_query,
                        service_metadata=turn_service_metadata,
                    )
                )
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
                stage_placeholder = st.empty()

                def _report_stage(stage: str) -> None:
                    stage_placeholder.caption(f"Stage: {stage}")

                _report_stage("Initializing services")
                query_rewriter, embedder, answer_generator, retrieval_cache = _get_shared_services(
                    state,
                    settings=settings,
                )

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
                    query_rewriter=query_rewriter,
                    embedder=embedder,
                    answer_generator=answer_generator,
                    retrieval_cache=retrieval_cache,
                    settings=settings,
                    use_cache=not (debug_mode or override_top_k),
                    fast_mode=fast_mode,
                    progress_reporter=_report_stage,
                )
                stage_placeholder.empty()
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
                    confidence_tier=answer.confidence_tier,
                    evidence_count=len(bundle.results),
                    uncertainty_notes=answer.sections.get("Uncertainties / Missing Evidence", ""),
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
                    "confidence_tier": answer.confidence_tier,
                    "service_metadata": answer.service_metadata,
                }
            )

            with st.chat_message("assistant"):
                components.render_response_diagnostics(
                    retrieval_mode=bundle.retrieval_mode,
                    top_k_used=bundle.top_k_used,
                    used_cached_context=bundle.used_cached_context,
                    insufficient_context=answer.insufficient_context,
                    confidence_tier=answer.confidence_tier,
                    service_metadata=answer.service_metadata,
                    show_latency=debug_mode,
                )
                st.caption(
                    _format_rewritten_query_caption(
                        question=question,
                        rewritten_query=bundle.rewritten_query,
                        service_metadata=answer.service_metadata,
                    )
                )
                components.render_answer_sections(answer)
                with st.expander("Evidence", expanded=answer.insufficient_context):
                    components.render_evidence_panel(bundle.results)


if __name__ == "__main__":
    main()
