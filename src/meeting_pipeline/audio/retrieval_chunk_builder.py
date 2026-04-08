from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha1

from meeting_pipeline.schemas.transcript import SpeakerTurn


@dataclass(frozen=True)
class RetrievalChunkWindow:
    chunk_key: str
    meeting_id: str
    speaker_label: str
    start_time: float
    end_time: float
    content: str
    source_turn_count: int


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _validate_windowing(
    window_seconds: float,
    overlap_seconds: float,
) -> tuple[float, float, float]:
    normalized_window = float(window_seconds)
    normalized_overlap = float(overlap_seconds)

    if normalized_window <= 0:
        raise ValueError("window_seconds must be greater than 0")
    if normalized_overlap < 0:
        raise ValueError("overlap_seconds must be greater than or equal to 0")
    if normalized_overlap >= normalized_window:
        raise ValueError("overlap_seconds must be less than window_seconds")

    return normalized_window, normalized_overlap, normalized_window - normalized_overlap


def _turn_signature(turn: SpeakerTurn) -> str:
    normalized_text = _normalize_text(turn.text)
    text_hash = sha1(normalized_text.encode("utf-8")).hexdigest()[:12]
    return f"{turn.speaker_label}|" f"{turn.start_time:.3f}|" f"{turn.end_time:.3f}|" f"{text_hash}"


def _overlaps_window(turn: SpeakerTurn, window_start: float, window_end: float) -> bool:
    return turn.end_time > window_start and turn.start_time < window_end


def _build_window_content(
    *,
    window_start: float,
    window_end: float,
    turns: Sequence[SpeakerTurn],
) -> tuple[str, str]:
    speaker_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"turns": 0.0, "seconds": 0.0})
    evidence_lines: list[str] = []

    for turn in turns:
        clipped_start = max(window_start, turn.start_time)
        clipped_end = min(window_end, turn.end_time)
        clipped_duration = max(0.0, clipped_end - clipped_start)

        stats = speaker_stats[turn.speaker_label]
        stats["turns"] += 1.0
        stats["seconds"] += clipped_duration

        evidence_lines.append(
            f"[{turn.speaker_label} {turn.start_time:.2f}-{turn.end_time:.2f}] {turn.text}"
        )

    if not speaker_stats or not evidence_lines:
        raise ValueError("window has no usable speaker statistics")

    ordered_speakers = sorted(
        speaker_stats.items(),
        key=lambda item: (-item[1]["seconds"], -item[1]["turns"], item[0]),
    )
    primary_speaker = ordered_speakers[0][0]

    summary_parts = [
        f"{speaker} ({int(stats['turns'])} turns, {stats['seconds']:.1f}s)"
        for speaker, stats in ordered_speakers
    ]
    speaker_summary = "; ".join(summary_parts)

    content = (
        f"Window {window_start:.2f}-{window_end:.2f}. "
        f"Speaker coverage: {speaker_summary}.\n"
        f"Evidence:\n{chr(10).join(evidence_lines)}"
    )
    return primary_speaker, content


def build_retrieval_chunks(
    *,
    meeting_id: str,
    turns: Sequence[SpeakerTurn],
    window_seconds: float,
    overlap_seconds: float,
) -> list[RetrievalChunkWindow]:
    normalized_meeting_id = meeting_id.strip()
    if not normalized_meeting_id:
        raise ValueError("meeting_id must be a non-empty string")

    window, overlap, step = _validate_windowing(window_seconds, overlap_seconds)
    if not turns:
        return []

    ordered_turns = sorted(
        turns,
        key=lambda turn: (turn.start_time, turn.end_time, turn.speaker_label, turn.text),
    )

    validated_turns: list[SpeakerTurn] = []
    for turn in ordered_turns:
        if turn.meeting_id != normalized_meeting_id:
            raise ValueError(
                "All turns must match meeting_id for retrieval chunking "
                f"({turn.meeting_id} != {normalized_meeting_id})"
            )
        normalized_text = _normalize_text(turn.text)
        if not normalized_text:
            continue
        validated_turns.append(turn.model_copy(update={"text": normalized_text}))

    if not validated_turns:
        return []

    timeline_start = min(turn.start_time for turn in validated_turns)
    timeline_end = max(turn.end_time for turn in validated_turns)

    chunk_windows: list[RetrievalChunkWindow] = []
    seen_keys: set[str] = set()

    window_start = timeline_start
    while window_start <= timeline_end:
        window_end = window_start + window
        window_turns = [
            turn for turn in validated_turns if _overlaps_window(turn, window_start, window_end)
        ]
        if not window_turns:
            window_start += step
            continue

        signatures = [_turn_signature(turn) for turn in window_turns]
        key_payload = f"{normalized_meeting_id}|{'|'.join(signatures)}"
        chunk_key = sha1(key_payload.encode("utf-8")).hexdigest()[:24]

        if chunk_key in seen_keys:
            window_start += step
            continue

        primary_speaker, content = _build_window_content(
            window_start=window_start,
            window_end=window_end,
            turns=window_turns,
        )

        chunk_windows.append(
            RetrievalChunkWindow(
                chunk_key=chunk_key,
                meeting_id=normalized_meeting_id,
                speaker_label=primary_speaker[:50],
                start_time=min(turn.start_time for turn in window_turns),
                end_time=max(turn.end_time for turn in window_turns),
                content=_normalize_text(content.replace("\n", " \n ")),
                source_turn_count=len(window_turns),
            )
        )
        seen_keys.add(chunk_key)
        window_start += step

    return chunk_windows
