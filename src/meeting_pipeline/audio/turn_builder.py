from __future__ import annotations

import math
import string
from collections.abc import Sequence

from meeting_pipeline.schemas.transcript import SpeakerTurn, TranscriptSegment, WordToken

_PUNCTUATION_ONLY = set(string.punctuation)


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _is_usable_word(word: WordToken) -> bool:
    if not math.isfinite(word.start_time) or not math.isfinite(word.end_time):
        return False
    if word.end_time < word.start_time:
        return False
    if not _normalize_text(word.text):
        return False
    return True


def _append_token_text(current_text: str, token_text: str) -> str:
    normalized_token = _normalize_text(token_text)
    if not normalized_token:
        return current_text

    if not current_text:
        return normalized_token

    if set(normalized_token).issubset(_PUNCTUATION_ONLY):
        return f"{current_text}{normalized_token}"

    return f"{current_text} {normalized_token}".strip()


def build_speaker_turns(
    meeting_id: str,
    words: Sequence[WordToken],
    max_gap_seconds: float = 1.0,
    default_label: str = "unknown",
) -> list[SpeakerTurn]:
    if not meeting_id.strip():
        raise ValueError("meeting_id must be a non-empty string")
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be greater than or equal to 0")
    if not words:
        return []

    usable_words = [word for word in words if _is_usable_word(word)]
    if not usable_words:
        return []

    ordered_words = sorted(
        usable_words,
        key=lambda item: (item.start_time, item.end_time, item.speaker_id, item.text),
    )

    turns: list[SpeakerTurn] = []
    for word in ordered_words:
        speaker_label = word.speaker_id.strip() if word.speaker_id.strip() else default_label
        token_text = _normalize_text(word.text)
        if not token_text:
            continue

        if not turns:
            turns.append(
                SpeakerTurn(
                    meeting_id=meeting_id,
                    speaker_label=speaker_label,
                    start_time=word.start_time,
                    end_time=word.end_time,
                    text=token_text,
                )
            )
            continue

        previous = turns[-1]
        gap = word.start_time - previous.end_time
        same_speaker = previous.speaker_label == speaker_label
        close_enough = gap <= max_gap_seconds

        if same_speaker and close_enough:
            turns[-1] = SpeakerTurn(
                meeting_id=previous.meeting_id,
                speaker_label=previous.speaker_label,
                start_time=previous.start_time,
                end_time=max(previous.end_time, word.end_time),
                text=_append_token_text(previous.text, token_text),
            )
            continue

        turns.append(
            SpeakerTurn(
                meeting_id=meeting_id,
                speaker_label=speaker_label,
                start_time=word.start_time,
                end_time=word.end_time,
                text=token_text,
            )
        )

    return turns


class TurnBuilder:
    def __init__(self, max_gap_seconds: float = 0.5) -> None:
        self.max_gap_seconds = max_gap_seconds

    def build(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if not segments:
            return []

        words: list[WordToken] = []
        for segment in segments:
            speaker_label = segment.speaker or "unknown"
            words.append(
                WordToken(
                    speaker_id=speaker_label,
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text,
                )
            )

        built_turns = build_speaker_turns(
            meeting_id="legacy",
            words=words,
            max_gap_seconds=self.max_gap_seconds,
            default_label="unknown",
        )

        return [
            TranscriptSegment(
                speaker=turn.speaker_label,
                text=turn.text,
                start=turn.start_time,
                end=turn.end_time,
            )
            for turn in built_turns
        ]
