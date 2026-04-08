from __future__ import annotations

from meeting_pipeline.audio.attribution import (
    attribute_words,
    compute_overlap,
    select_speaker_label,
)
from meeting_pipeline.schemas.diarization import DiarizationSegment
from meeting_pipeline.schemas.transcript import WordToken


def test_compute_overlap_basic_and_no_overlap() -> None:
    assert compute_overlap(0.0, 1.0, 0.5, 1.5) == 0.5
    assert compute_overlap(0.0, 0.5, 0.6, 1.0) == 0.0


def test_select_speaker_label_highest_overlap() -> None:
    diarization = [
        DiarizationSegment(speaker="spk_0", start=0.0, end=0.4),
        DiarizationSegment(speaker="spk_1", start=0.4, end=1.0),
    ]

    speaker = select_speaker_label(
        word_start=0.3,
        word_end=0.9,
        diarization_segments=diarization,
    )

    assert speaker == "spk_1"


def test_select_speaker_label_defaults_on_gap() -> None:
    diarization = [DiarizationSegment(speaker="spk_0", start=2.0, end=3.0)]

    speaker = select_speaker_label(
        word_start=0.0,
        word_end=1.0,
        diarization_segments=diarization,
        default_label="unknown",
    )

    assert speaker == "unknown"


def test_select_speaker_label_tie_break_is_deterministic() -> None:
    diarization = [
        DiarizationSegment(speaker="SPEAKER_10", start=0.0, end=1.0),
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
    ]

    speaker = select_speaker_label(
        word_start=0.2,
        word_end=0.8,
        diarization_segments=diarization,
    )

    assert speaker == "SPEAKER_00"


def test_attribute_words_assigns_labels_and_preserves_order() -> None:
    words = [
        WordToken(speaker_id="unknown", start_time=0.0, end_time=0.2, text="hello"),
        WordToken(speaker_id="unknown", start_time=0.3, end_time=0.5, text="team"),
    ]
    diarization = [
        DiarizationSegment(speaker="A", start=0.0, end=0.25),
        DiarizationSegment(speaker="B", start=0.25, end=1.0),
    ]

    result = attribute_words(words=words, diarization_segments=diarization)

    assert [item.speaker_id for item in result] == ["A", "B"]
    assert [item.text for item in result] == ["hello", "team"]


def test_attribute_words_handles_empty_input() -> None:
    result = attribute_words(words=[], diarization_segments=[])
    assert result == []
