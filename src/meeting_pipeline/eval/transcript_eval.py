from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from meeting_pipeline.eval.metrics import mean, safe_divide
from meeting_pipeline.schemas.transcript import AlignedTranscript, SpeakerTurn


@dataclass(frozen=True)
class TranscriptEvaluationResult:
    reference_meeting_id: str
    prediction_meeting_id: str
    reference_word_count: int
    prediction_turn_count: int
    prediction_word_estimate: int
    average_words_per_turn: float
    word_coverage_ratio: float
    lexical_word_error_rate: float
    timing_invalid_turn_count: int
    overlapping_turn_pair_count: int
    reference_time_span_seconds: float
    prediction_time_span_seconds: float
    speaker_coverage_ratio: float
    reference_speakers: list[str]
    prediction_speakers: list[str]
    missing_reference_speakers: list[str]
    extra_prediction_speakers: list[str]

    def to_summary(self) -> dict[str, object]:
        return asdict(self)


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Evaluation artifact does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation artifact is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation artifact must be a JSON object: {path}")

    return payload


def _load_reference_transcript(path: Path) -> AlignedTranscript:
    payload = _load_json_object(path)
    return AlignedTranscript.model_validate(payload)


def _load_prediction_turns(path: Path) -> tuple[str, list[SpeakerTurn]]:
    payload = _load_json_object(path)
    meeting_id_obj = payload.get("meeting_id")
    meeting_id = str(meeting_id_obj).strip() if isinstance(meeting_id_obj, str) else "unknown"

    turns_obj = payload.get("turns")
    if not isinstance(turns_obj, list):
        raise ValueError("Prediction artifact must contain a 'turns' list")

    turns: list[SpeakerTurn] = []
    for idx, item in enumerate(turns_obj):
        if not isinstance(item, dict):
            raise ValueError(f"Turn at index {idx} is not an object")

        turn_payload = item
        if "meeting_id" not in turn_payload and meeting_id != "unknown":
            turn_payload = {**turn_payload, "meeting_id": meeting_id}

        turns.append(SpeakerTurn.model_validate(turn_payload))

    return meeting_id, turns


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1

    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )

    if not reference:
        return 0.0
    return dist[-1][-1] / len(reference)


def _time_span(start_values: list[float], end_values: list[float]) -> float:
    if not start_values or not end_values:
        return 0.0
    return max(0.0, max(end_values) - min(start_values))


def _count_overlapping_turn_pairs(turns: list[SpeakerTurn]) -> int:
    if len(turns) <= 1:
        return 0

    ordered = sorted(turns, key=lambda item: (item.start_time, item.end_time))
    overlaps = 0
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.start_time < previous.end_time:
            overlaps += 1
    return overlaps


def evaluate_transcript_payloads(
    reference: AlignedTranscript,
    prediction_meeting_id: str,
    prediction_turns: list[SpeakerTurn],
) -> TranscriptEvaluationResult:
    reference_tokens = [token.text.lower() for token in reference.words]
    prediction_tokens = [
        token.lower() for turn in prediction_turns for token in turn.text.split() if token.strip()
    ]

    reference_speakers = sorted({item.speaker_id for item in reference.words})
    prediction_speakers = sorted({item.speaker_label for item in prediction_turns})
    shared_speakers = sorted(set(reference_speakers) & set(prediction_speakers))

    average_words_per_turn = mean([float(len(turn.text.split())) for turn in prediction_turns])

    return TranscriptEvaluationResult(
        reference_meeting_id=reference.meeting_id,
        prediction_meeting_id=prediction_meeting_id,
        reference_word_count=len(reference.words),
        prediction_turn_count=len(prediction_turns),
        prediction_word_estimate=len(prediction_tokens),
        average_words_per_turn=average_words_per_turn,
        word_coverage_ratio=safe_divide(len(prediction_tokens), len(reference_tokens)),
        lexical_word_error_rate=word_error_rate(reference_tokens, prediction_tokens),
        timing_invalid_turn_count=sum(
            1 for turn in prediction_turns if turn.end_time < turn.start_time
        ),
        overlapping_turn_pair_count=_count_overlapping_turn_pairs(prediction_turns),
        reference_time_span_seconds=_time_span(
            [item.start_time for item in reference.words],
            [item.end_time for item in reference.words],
        ),
        prediction_time_span_seconds=_time_span(
            [item.start_time for item in prediction_turns],
            [item.end_time for item in prediction_turns],
        ),
        speaker_coverage_ratio=safe_divide(len(shared_speakers), len(reference_speakers)),
        reference_speakers=reference_speakers,
        prediction_speakers=prediction_speakers,
        missing_reference_speakers=sorted(set(reference_speakers) - set(prediction_speakers)),
        extra_prediction_speakers=sorted(set(prediction_speakers) - set(reference_speakers)),
    )


def evaluate_transcript_files(
    *,
    reference_path: Path,
    prediction_path: Path,
) -> TranscriptEvaluationResult:
    reference = _load_reference_transcript(reference_path)
    prediction_meeting_id, prediction_turns = _load_prediction_turns(prediction_path)
    return evaluate_transcript_payloads(reference, prediction_meeting_id, prediction_turns)
