from __future__ import annotations

import json
from pathlib import Path

from meeting_pipeline.eval.retrieval_eval import (
    evaluate_retrieval_benchmark,
    load_retrieval_benchmark,
    load_retrieval_predictions,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_retrieval_eval_computes_expected_rates(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    predictions_path = tmp_path / "predictions.json"

    _write_json(
        benchmark_path,
        {
            "items": [
                {
                    "meeting_id": "m1",
                    "question": "What was decided?",
                    "expected_chunk_ids": [101],
                    "expected_speaker_labels": ["SPEAKER_00"],
                },
                {
                    "meeting_id": "m1",
                    "question": "Any blockers?",
                    "expected_hints": ["blocked"],
                },
            ]
        },
    )

    _write_json(
        predictions_path,
        {
            "items": [
                {
                    "meeting_id": "m1",
                    "question": "What was decided?",
                    "rewritten_query": "meeting decisions",
                    "retrieved": [
                        {
                            "chunk_id": 101,
                            "speaker_label": "SPEAKER_00",
                            "content": "We decided to launch Friday.",
                            "similarity": 0.92,
                        }
                    ],
                },
                {
                    "meeting_id": "m1",
                    "question": "Any blockers?",
                    "rewritten_query": "blockers",
                    "retrieved": [],
                },
            ]
        },
    )

    benchmark_items = load_retrieval_benchmark(benchmark_path)
    prediction_items = load_retrieval_predictions(predictions_path)
    result = evaluate_retrieval_benchmark(benchmark_items, prediction_items, top_k=5)

    assert result.query_count == 2
    assert result.recall_at_k == 1.0
    assert result.evidence_hit_rate == 0.5
    assert result.empty_retrieval_rate == 0.5


def test_retrieval_eval_tracks_missing_predictions(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    predictions_path = tmp_path / "predictions.json"

    _write_json(
        benchmark_path,
        {"items": [{"meeting_id": "m1", "question": "Q1", "expected_chunk_ids": [1]}]},
    )
    _write_json(predictions_path, {"items": []})

    result = evaluate_retrieval_benchmark(
        load_retrieval_benchmark(benchmark_path),
        load_retrieval_predictions(predictions_path),
        top_k=3,
    )

    assert result.missing_predictions == 1
    assert result.empty_retrieval_rate == 1.0
