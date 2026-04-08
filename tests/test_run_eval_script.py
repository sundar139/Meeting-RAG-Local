from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_eval import run_evaluation


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_evaluation_supports_transcript_and_retrieval(tmp_path: Path) -> None:
    transcript_reference = tmp_path / "reference.json"
    transcript_prediction = tmp_path / "turns.json"
    retrieval_benchmark = tmp_path / "benchmark.json"
    retrieval_predictions = tmp_path / "predictions.json"

    _write_json(
        transcript_reference,
        {
            "meeting_id": "m1",
            "words": [{"speaker_id": "A", "start_time": 0.0, "end_time": 0.1, "text": "hello"}],
        },
    )
    _write_json(
        transcript_prediction,
        {
            "meeting_id": "m1",
            "turns": [
                {
                    "meeting_id": "m1",
                    "speaker_label": "A",
                    "start_time": 0.0,
                    "end_time": 0.1,
                    "text": "hello",
                }
            ],
        },
    )
    _write_json(
        retrieval_benchmark,
        {
            "items": [
                {
                    "meeting_id": "m1",
                    "question": "What happened?",
                    "expected_chunk_ids": [10],
                }
            ]
        },
    )
    _write_json(
        retrieval_predictions,
        {
            "items": [
                {
                    "meeting_id": "m1",
                    "question": "What happened?",
                    "rewritten_query": "what happened",
                    "retrieved": [{"chunk_id": 10, "speaker_label": "A", "content": "hello"}],
                }
            ]
        },
    )

    summary = run_evaluation(
        transcript_reference_path=transcript_reference,
        transcript_prediction_path=transcript_prediction,
        retrieval_benchmark_path=retrieval_benchmark,
        retrieval_predictions_path=retrieval_predictions,
        retrieval_top_k=5,
    )

    assert "transcript" in summary
    assert "retrieval" in summary


def test_run_evaluation_requires_retrieval_predictions_or_live_mode(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    _write_json(
        benchmark,
        {"items": [{"meeting_id": "m1", "question": "What happened?"}]},
    )

    with pytest.raises(ValueError, match="retrieval_predictions_path"):
        run_evaluation(retrieval_benchmark_path=benchmark)
