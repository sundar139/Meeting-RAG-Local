from __future__ import annotations

import math
from collections.abc import Sequence

from meeting_pipeline.schemas.diarization import DiarizationSegment
from meeting_pipeline.schemas.transcript import TranscriptSegment, WordToken


def compute_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if not all(math.isfinite(value) for value in (start_a, end_a, start_b, end_b)):
        return 0.0
    if end_a < start_a or end_b < start_b:
        return 0.0
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def select_speaker_label(
    word_start: float,
    word_end: float,
    diarization_segments: Sequence[DiarizationSegment],
    default_label: str = "unknown",
) -> str:
    if not diarization_segments:
        return default_label

    candidates: list[tuple[float, float, str]] = []
    for segment in diarization_segments:
        overlap = compute_overlap(word_start, word_end, segment.start_time, segment.end_time)
        if overlap <= 0.0:
            continue

        # Tie-breaking: larger overlap, then earlier segment start, then lexical speaker label.
        candidates.append((overlap, segment.start_time, segment.speaker_label))

    if not candidates:
        return default_label

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][2]


def attribute_words(
    words: Sequence[WordToken],
    diarization_segments: Sequence[DiarizationSegment],
    default_label: str = "unknown",
) -> list[WordToken]:
    if not words:
        return []

    attributed: list[WordToken] = []
    for word in words:
        speaker_label = select_speaker_label(
            word_start=word.start_time,
            word_end=word.end_time,
            diarization_segments=diarization_segments,
            default_label=default_label,
        )
        attributed.append(word.model_copy(update={"speaker_id": speaker_label}))

    return attributed


def attribute_speakers(
    transcript_segments: list[TranscriptSegment], diarization_segments: list[DiarizationSegment]
) -> list[TranscriptSegment]:
    attributed: list[TranscriptSegment] = []

    for segment in transcript_segments:
        best_speaker: str | None = None
        best_overlap = 0.0

        for diar in diarization_segments:
            score = compute_overlap(segment.start, segment.end, diar.start, diar.end)
            if score > best_overlap:
                best_overlap = score
                best_speaker = diar.speaker

        attributed.append(segment.model_copy(update={"speaker": best_speaker}))

    return attributed
