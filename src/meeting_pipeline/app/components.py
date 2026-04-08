from __future__ import annotations

from collections.abc import Sequence

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


def render_warning(message: str) -> None:
    import streamlit as st

    st.warning(message)


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


def render_answer_sections(answer: GroundedAnswerResult) -> None:
    import streamlit as st

    if answer.insufficient_context:
        render_warning(
            "The answer is flagged as low-confidence because retrieved evidence was limited."
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
