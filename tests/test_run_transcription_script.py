from __future__ import annotations

import json
from pathlib import Path

from scripts.run_transcription import run_transcription_pipeline


def test_run_transcription_pipeline_writes_expected_artifact(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    output_dir = tmp_path / "interim"

    monkeypatch.setattr(
        "scripts.run_transcription.transcribe_audio",
        lambda **_kwargs: {
            "language": "en",
            "segments": [{"id": 0, "start_time": 0.0, "end_time": 0.5, "text": "hello"}],
        },
    )
    monkeypatch.setattr(
        "scripts.run_transcription.align_transcript",
        lambda **_kwargs: {
            "language": "en",
            "segments": [{"id": 0, "start_time": 0.0, "end_time": 0.5, "text": "hello"}],
            "words": [
                {
                    "speaker_id": "unknown",
                    "start_time": 0.0,
                    "end_time": 0.2,
                    "text": "hello",
                }
            ],
        },
    )

    output_path = run_transcription_pipeline(
        audio_path=audio_path,
        meeting_id="ES2002a",
        output_dir=output_dir,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "ES2002a_aligned.json"
    assert payload["meeting_id"] == "ES2002a"
    assert payload["language"] == "en"
    assert len(payload["segments"]) == 1
    assert len(payload["words"]) == 1
