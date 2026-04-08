from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_pipeline.eval.transcript_eval import evaluate_transcript_files


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_transcript_files_returns_expected_summary(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.json"
    prediction_path = tmp_path / "prediction.json"

    _write_json(
        reference_path,
        {
            "meeting_id": "ES2002a",
            "words": [
                {"speaker_id": "A", "start_time": 0.0, "end_time": 0.2, "text": "hello"},
                {"speaker_id": "B", "start_time": 0.3, "end_time": 0.5, "text": "team"},
            ],
        },
    )
    _write_json(
        prediction_path,
        {
            "meeting_id": "ES2002a",
            "turns": [
                {
                    "meeting_id": "ES2002a",
                    "speaker_label": "A",
                    "start_time": 0.0,
                    "end_time": 0.4,
                    "text": "hello team",
                }
            ],
        },
    )

    result = evaluate_transcript_files(
        reference_path=reference_path,
        prediction_path=prediction_path,
    )

    assert result.reference_meeting_id == "ES2002a"
    assert result.prediction_turn_count == 1
    assert result.reference_word_count == 2
    assert result.word_coverage_ratio == 1.0
    assert result.speaker_coverage_ratio == 0.5


def test_evaluate_transcript_files_rejects_missing_turns_list(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.json"
    prediction_path = tmp_path / "prediction.json"

    _write_json(reference_path, {"meeting_id": "m1", "words": []})
    _write_json(prediction_path, {"meeting_id": "m1"})

    with pytest.raises(ValueError, match="turns"):
        evaluate_transcript_files(
            reference_path=reference_path,
            prediction_path=prediction_path,
        )
