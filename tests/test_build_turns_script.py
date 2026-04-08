from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_turns import build_turns_artifact


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_turns_artifact_happy_path(tmp_path: Path) -> None:
    aligned_path = tmp_path / "aligned.json"
    diarization_path = tmp_path / "diarization.json"
    output_dir = tmp_path / "processed"

    _write_json(
        aligned_path,
        {
            "meeting_id": "ES2002a",
            "words": [
                {
                    "speaker_id": "unknown",
                    "start_time": 0.0,
                    "end_time": 0.3,
                    "text": "hello",
                },
                {
                    "speaker_id": "unknown",
                    "start_time": 0.31,
                    "end_time": 0.6,
                    "text": "team",
                },
            ],
        },
    )
    _write_json(
        diarization_path,
        {
            "meeting_id": "ES2002a",
            "segments": [
                {
                    "speaker_label": "SPEAKER_00",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ],
        },
    )

    output_path = build_turns_artifact(
        meeting_id="ES2002a",
        aligned_path=aligned_path,
        diarization_path=diarization_path,
        output_dir=output_dir,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "ES2002a_turns.json"
    assert payload["meeting_id"] == "ES2002a"
    assert len(payload["turns"]) == 1
    assert payload["turns"][0]["speaker_label"] == "SPEAKER_00"
    assert payload["turns"][0]["text"] == "hello team"


def test_build_turns_artifact_rejects_meeting_id_mismatch(tmp_path: Path) -> None:
    aligned_path = tmp_path / "aligned.json"
    diarization_path = tmp_path / "diarization.json"

    _write_json(aligned_path, {"meeting_id": "ES2002a", "words": []})
    _write_json(
        diarization_path,
        {
            "meeting_id": "ES2002b",
            "segments": [
                {
                    "speaker_label": "SPEAKER_00",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="mismatch"):
        build_turns_artifact(
            meeting_id="ES2002a",
            aligned_path=aligned_path,
            diarization_path=diarization_path,
            output_dir=tmp_path,
        )


def test_build_turns_artifact_rejects_missing_words(tmp_path: Path) -> None:
    aligned_path = tmp_path / "aligned.json"
    diarization_path = tmp_path / "diarization.json"

    _write_json(aligned_path, {"meeting_id": "ES2002a", "words": []})
    _write_json(
        diarization_path,
        {
            "meeting_id": "ES2002a",
            "segments": [
                {
                    "speaker_label": "SPEAKER_00",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="usable words"):
        build_turns_artifact(
            meeting_id="ES2002a",
            aligned_path=aligned_path,
            diarization_path=diarization_path,
            output_dir=tmp_path,
        )
