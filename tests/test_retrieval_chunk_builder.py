from __future__ import annotations

import pytest

from meeting_pipeline.audio.retrieval_chunk_builder import build_retrieval_chunks
from meeting_pipeline.schemas.transcript import SpeakerTurn


def _turn(
    *,
    meeting_id: str,
    speaker_label: str,
    start_time: float,
    end_time: float,
    text: str,
) -> SpeakerTurn:
    return SpeakerTurn(
        meeting_id=meeting_id,
        speaker_label=speaker_label,
        start_time=start_time,
        end_time=end_time,
        text=text,
    )


def test_build_retrieval_chunks_creates_windowed_chunks_with_overlap() -> None:
    turns = [
        _turn(
            meeting_id="m1",
            speaker_label="SPEAKER_00",
            start_time=0.0,
            end_time=12.0,
            text="Kickoff and objectives.",
        ),
        _turn(
            meeting_id="m1",
            speaker_label="SPEAKER_01",
            start_time=12.5,
            end_time=26.0,
            text="Risk discussion and mitigation options.",
        ),
        _turn(
            meeting_id="m1",
            speaker_label="SPEAKER_00",
            start_time=27.0,
            end_time=38.0,
            text="Decision recap and next steps.",
        ),
    ]

    chunks = build_retrieval_chunks(
        meeting_id="m1",
        turns=turns,
        window_seconds=30.0,
        overlap_seconds=15.0,
    )

    assert len(chunks) >= 2
    assert len({chunk.chunk_key for chunk in chunks}) == len(chunks)
    assert all("Speaker coverage:" in chunk.content for chunk in chunks)
    assert any("[SPEAKER_00" in chunk.content for chunk in chunks)


def test_build_retrieval_chunks_dedupes_identical_overlap_windows() -> None:
    turns = [
        _turn(
            meeting_id="m1",
            speaker_label="SPEAKER_00",
            start_time=0.0,
            end_time=120.0,
            text="Long uninterrupted speaking segment.",
        )
    ]

    chunks = build_retrieval_chunks(
        meeting_id="m1",
        turns=turns,
        window_seconds=60.0,
        overlap_seconds=30.0,
    )

    assert len(chunks) == 1


def test_build_retrieval_chunks_rejects_invalid_overlap() -> None:
    turns = [
        _turn(
            meeting_id="m1",
            speaker_label="SPEAKER_00",
            start_time=0.0,
            end_time=5.0,
            text="hello",
        )
    ]

    with pytest.raises(ValueError, match="overlap_seconds"):
        build_retrieval_chunks(
            meeting_id="m1",
            turns=turns,
            window_seconds=20.0,
            overlap_seconds=20.0,
        )


def test_build_retrieval_chunks_primary_speaker_uses_window_coverage() -> None:
    turns = [
        _turn(
            meeting_id="m1",
            speaker_label="SPEAKER_00",
            start_time=0.0,
            end_time=20.0,
            text="Context from speaker zero.",
        ),
        _turn(
            meeting_id="m1",
            speaker_label="SPEAKER_01",
            start_time=20.0,
            end_time=70.0,
            text="Planning and decision discussion from speaker one.",
        ),
        _turn(
            meeting_id="m1",
            speaker_label="SPEAKER_00",
            start_time=70.0,
            end_time=80.0,
            text="Short follow-up from speaker zero.",
        ),
    ]

    chunks = build_retrieval_chunks(
        meeting_id="m1",
        turns=turns,
        window_seconds=90.0,
        overlap_seconds=10.0,
    )

    assert len(chunks) == 1
    assert chunks[0].speaker_label == "SPEAKER_01"
    assert "SPEAKER_00" in chunks[0].content
    assert "SPEAKER_01" in chunks[0].content
