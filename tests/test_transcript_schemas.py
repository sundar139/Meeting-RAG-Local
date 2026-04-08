from __future__ import annotations

import pytest
from pydantic import ValidationError

from meeting_pipeline.schemas.diarization import DiarizationSegment
from meeting_pipeline.schemas.transcript import AlignedTranscript, SpeakerTurn, WordToken


def test_word_token_normalizes_text_and_validates_time_range() -> None:
    token = WordToken(
        speaker_id=" A ",
        start_time=1.0,
        end_time=1.2,
        text="  hello   world  ",
    )

    assert token.speaker_id == "A"
    assert token.text == "hello world"

    with pytest.raises(ValidationError):
        WordToken(
            speaker_id="A",
            start_time=2.0,
            end_time=1.0,
            text="bad",
        )


def test_speaker_turn_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        SpeakerTurn(
            meeting_id="ES2002a",
            speaker_label="A",
            start_time=0.0,
            end_time=0.1,
            text="   ",
        )


def test_aligned_transcript_requires_non_empty_meeting_id() -> None:
    with pytest.raises(ValidationError):
        AlignedTranscript(meeting_id=" ", words=[])


def test_diarization_segment_supports_legacy_aliases() -> None:
    segment = DiarizationSegment.model_validate(
        {
            "speaker": "spk_0",
            "start": 0.0,
            "end": 1.0,
        }
    )

    assert segment.speaker_label == "spk_0"
    assert segment.start_time == 0.0
    assert segment.end_time == 1.0
    assert segment.speaker == "spk_0"
