from __future__ import annotations

import pytest

from meeting_pipeline.audio.turn_builder import build_speaker_turns
from meeting_pipeline.schemas.transcript import WordToken


def test_build_speaker_turns_merges_same_speaker_sequence() -> None:
    words = [
        WordToken(speaker_id="spk_0", start_time=0.0, end_time=0.2, text="hello"),
        WordToken(speaker_id="spk_0", start_time=0.21, end_time=0.4, text="world"),
    ]

    turns = build_speaker_turns(meeting_id="ES2002a", words=words, max_gap_seconds=1.0)

    assert len(turns) == 1
    assert turns[0].speaker_label == "spk_0"
    assert turns[0].text == "hello world"


def test_build_speaker_turns_splits_on_speaker_change() -> None:
    words = [
        WordToken(speaker_id="A", start_time=0.0, end_time=0.2, text="hi"),
        WordToken(speaker_id="B", start_time=0.21, end_time=0.3, text="there"),
    ]

    turns = build_speaker_turns(meeting_id="ES2002a", words=words)

    assert len(turns) == 2
    assert [turn.speaker_label for turn in turns] == ["A", "B"]


def test_build_speaker_turns_splits_on_long_gap() -> None:
    words = [
        WordToken(speaker_id="A", start_time=0.0, end_time=0.2, text="first"),
        WordToken(speaker_id="A", start_time=2.0, end_time=2.2, text="second"),
    ]

    turns = build_speaker_turns(meeting_id="ES2002a", words=words, max_gap_seconds=0.5)

    assert len(turns) == 2


def test_build_speaker_turns_punctuation_attaches_without_extra_space() -> None:
    words = [
        WordToken(speaker_id="A", start_time=0.0, end_time=0.1, text="hello"),
        WordToken(speaker_id="A", start_time=0.11, end_time=0.12, text=","),
        WordToken(speaker_id="A", start_time=0.13, end_time=0.2, text="world"),
    ]

    turns = build_speaker_turns(meeting_id="ES2002a", words=words)

    assert len(turns) == 1
    assert turns[0].text == "hello, world"


def test_build_speaker_turns_orders_out_of_order_words() -> None:
    words = [
        WordToken(speaker_id="A", start_time=1.0, end_time=1.1, text="late"),
        WordToken(speaker_id="A", start_time=0.0, end_time=0.1, text="early"),
    ]

    turns = build_speaker_turns(meeting_id="ES2002a", words=words)

    assert turns[0].text == "early late"


def test_build_speaker_turns_rejects_bad_gap_threshold() -> None:
    with pytest.raises(ValueError, match="max_gap_seconds"):
        build_speaker_turns(meeting_id="ES2002a", words=[], max_gap_seconds=-1.0)
