from __future__ import annotations

from collections.abc import Mapping, Sequence

from meeting_pipeline.db.repository import MeetingOverview, TranscriptChunk
from meeting_pipeline.rag.models import GroundedAnswerResult, RetrievedChunk


def page_title(app_name: str) -> str:
    return f"{app_name} - Meeting Analysis"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"

    total_seconds = max(0, int(round(value)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_time_range(start_time: float | None, end_time: float | None) -> str:
    return f"{format_seconds(start_time)} - {format_seconds(end_time)}"


def content_excerpt(text: str, max_chars: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def render_warning(message: str, *, hint: str | None = None) -> None:
    import streamlit as st

    if hint:
        st.warning(f"{message}\n\n{hint}")
        return
    st.warning(message)


def render_info(message: str, *, hint: str | None = None) -> None:
    import streamlit as st

    if hint:
        st.info(f"{message}\n\n{hint}")
        return
    st.info(message)


def render_empty_state(title: str, body: str) -> None:
    import streamlit as st

    st.info(f"{title}\n\n{body}")


def render_meeting_header(meeting_id: str, overview: MeetingOverview) -> None:
    import streamlit as st

    st.subheader(f"Meeting {meeting_id}")
    st.caption(
        "Chunks: "
        f"{overview.chunk_count} | "
        f"Speakers: {overview.distinct_speaker_count} | "
        f"Span: {format_time_range(overview.start_time_min, overview.end_time_max)}"
    )


def render_transcript_rows(chunks: Sequence[TranscriptChunk]) -> None:
    import streamlit as st

    if not chunks:
        render_empty_state(
            "No transcript rows found",
            "Try removing filters or ingesting transcript chunks for this meeting.",
        )
        return

    rows = [
        {
            "Chunk": chunk.chunk_id,
            "Speaker": chunk.speaker_label,
            "Time": format_time_range(chunk.start_time, chunk.end_time),
            "Content": chunk.content,
        }
        for chunk in chunks
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True, height=420)


def render_evidence_panel(evidence: Sequence[RetrievedChunk]) -> None:
    import streamlit as st

    if not evidence:
        render_empty_state(
            "No retrieved evidence",
            "No supporting chunks were found for this question. "
            "Try rephrasing the question or increasing top-k.",
        )
        return

    st.caption(f"Retrieved evidence: {len(evidence)} chunks")
    for item in evidence:
        label = (
            f"chunk_id={item.chunk_id} | {item.speaker_label} | "
            f"{format_time_range(item.start_time, item.end_time)} | "
            f"similarity={item.similarity:.3f}"
        )
        with st.expander(label):
            st.write(content_excerpt(item.content, max_chars=500))


def _extract_timing_map(service_metadata: Mapping[str, object] | None) -> dict[str, float]:
    if service_metadata is None:
        return {}

    raw_timings = service_metadata.get("timings_ms")
    if not isinstance(raw_timings, dict):
        return {}

    parsed: dict[str, float] = {}
    for key, value in raw_timings.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            parsed[key] = float(value)
    return parsed


def _format_latency_summary(service_metadata: Mapping[str, object] | None) -> str | None:
    timings = _extract_timing_map(service_metadata)
    if not timings:
        return None

    label_candidates = [
        (("query_rewrite",), "rewrite"),
        (("query_embedding", "embedding_query_prep"), "embed"),
        (("postgres_retrieval", "retrieval"), "retrieve"),
        (("answer_generation",), "answer"),
        (("total_request", "retrieval_total"), "total"),
    ]
    parts: list[str] = []
    for candidates, short in label_candidates:
        for key in candidates:
            if key in timings:
                parts.append(f"{short} {timings[key]:.1f} ms")
                break
    return " | ".join(parts) if parts else None


def _format_cache_summary(service_metadata: Mapping[str, object] | None) -> str | None:
    if service_metadata is None:
        return None

    raw_cache = service_metadata.get("cache")
    if not isinstance(raw_cache, dict):
        return None

    labels = [
        ("query_rewrite", "rewrite"),
        ("query_embedding", "embed"),
        ("retrieval_bundle", "retrieve"),
        ("answer_generation", "answer"),
    ]
    parts: list[str] = []
    for key, short in labels:
        value = raw_cache.get(key)
        if isinstance(value, bool):
            parts.append(f"{short} {'hit' if value else 'miss'}")

    return " | ".join(parts) if parts else None


def _format_routing_summary(service_metadata: Mapping[str, object] | None) -> str | None:
    if service_metadata is None:
        return None

    routing = service_metadata.get("routing")
    if not isinstance(routing, dict):
        return None

    parts: list[str] = []
    retrieval_mode = routing.get("retrieval_mode")
    if isinstance(retrieval_mode, str) and retrieval_mode.strip():
        parts.append(f"mode {retrieval_mode}")

    question_relation = routing.get("question_relation")
    if isinstance(question_relation, str) and question_relation.strip():
        parts.append(f"relation {question_relation}")

    meta_scope = routing.get("meta_scope")
    if isinstance(meta_scope, str) and meta_scope.strip():
        parts.append(f"meta_scope {meta_scope}")

    return " | ".join(parts) if parts else None


def _badge(label: str, value: str, *, tone: str = "neutral") -> str:
    color_map = {
        "neutral": ("#6B7280", "#F9FAFB", "#374151"),
        "good": ("#047857", "#ECFDF5", "#065F46"),
        "warn": ("#B45309", "#FFFBEB", "#92400E"),
        "info": ("#1D4ED8", "#EFF6FF", "#1E40AF"),
    }
    border, background, text = color_map.get(tone, color_map["neutral"])
    return (
        f'<span style="display:inline-block;margin:0 0.35rem 0.35rem 0;'
        f"padding:0.18rem 0.5rem;border:1px solid {border};border-radius:999px;"
        f'background:{background};color:{text};font-size:0.78rem;font-weight:600;">'
        f"{label}: {value}</span>"
    )


def render_response_diagnostics(
    *,
    retrieval_mode: str,
    top_k_used: int,
    used_cached_context: bool,
    insufficient_context: bool,
    confidence_tier: str | None = None,
    service_metadata: Mapping[str, object] | None = None,
    show_latency: bool = False,
) -> None:
    import streamlit as st

    normalized_tier = (confidence_tier or "").strip().lower()
    if not normalized_tier:
        normalized_tier = "insufficient_evidence" if insufficient_context else "grounded"

    if normalized_tier == "insufficient_evidence":
        confidence_text = "Insufficient evidence"
        confidence_tone = "warn"
    elif normalized_tier == "partial_limited_evidence":
        confidence_text = "Partial / limited"
        confidence_tone = "info"
    else:
        confidence_text = "Grounded"
        confidence_tone = "good"

    badges = [
        _badge("Mode", retrieval_mode.replace("_", " "), tone="info"),
        _badge("Top-k", str(top_k_used)),
        _badge("Context", "Cached" if used_cached_context else "Fresh"),
        _badge("Confidence", confidence_text, tone=confidence_tone),
    ]
    st.markdown("<div>" + "".join(badges) + "</div>", unsafe_allow_html=True)

    if show_latency:
        routing_summary = _format_routing_summary(service_metadata)
        if routing_summary:
            st.markdown(
                (
                    '<div style="font-size:0.82rem;color:#6B7280;margin-top:-0.1rem;'
                    'margin-bottom:0.35rem;">'
                    f"Routing: {routing_summary}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        latency_summary = _format_latency_summary(service_metadata)
        if latency_summary:
            st.markdown(
                (
                    '<div style="font-size:0.82rem;color:#6B7280;margin-top:-0.1rem;'
                    'margin-bottom:0.45rem;">'
                    f"Latency: {latency_summary}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        cache_summary = _format_cache_summary(service_metadata)
        if cache_summary:
            st.markdown(
                (
                    '<div style="font-size:0.82rem;color:#6B7280;margin-top:-0.2rem;'
                    'margin-bottom:0.45rem;">'
                    f"Cache: {cache_summary}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_answer_sections(answer: GroundedAnswerResult) -> None:
    import streamlit as st

    if answer.confidence_tier == "insufficient_evidence" or answer.insufficient_context:
        render_warning(
            "Cannot answer confidently from retrieved evidence.",
            hint=(
                "Try a broader question, ask for a whole-meeting summary, "
                "or increase adaptive top-k."
            ),
        )
    elif answer.confidence_tier == "partial_limited_evidence":
        render_info(
            "Answer is partially grounded with limited evidence coverage.",
            hint="Treat this as directional and ask a narrower follow-up if needed.",
        )

    section_order = [
        "Summary",
        "Key Points",
        "Decisions",
        "Action Items",
        "Uncertainties / Missing Evidence",
    ]
    for title in section_order:
        value = answer.sections.get(title)
        if value:
            st.markdown(f"**{title}**")
            st.write(value)


def render_meeting_insights(
    overview: MeetingOverview,
    speaker_labels: Sequence[str],
    recent_questions: Sequence[str],
) -> None:
    import streamlit as st

    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks", str(overview.chunk_count))
    col2.metric("Speakers", str(overview.distinct_speaker_count))
    col3.metric("Time span", format_time_range(overview.start_time_min, overview.end_time_max))

    if speaker_labels:
        st.caption("Speakers: " + ", ".join(speaker_labels))
    if recent_questions:
        with st.expander("Recent questions"):
            for question in recent_questions[-5:]:
                st.write(f"- {question}")
