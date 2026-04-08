from __future__ import annotations

import json
from pathlib import Path

from meeting_pipeline.schemas.diarization import DiarizationSegment
from scripts.run_diarization import run_diarization_pipeline


def test_run_diarization_pipeline_writes_expected_artifact(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    output_dir = tmp_path / "interim"

    monkeypatch.setattr(
        "scripts.run_diarization.run_diarization",
        lambda **_kwargs: [
            DiarizationSegment(speaker_label="SPEAKER_00", start_time=0.0, end_time=1.2),
            DiarizationSegment(speaker_label="SPEAKER_01", start_time=1.3, end_time=2.0),
        ],
    )

    output_path = run_diarization_pipeline(
        audio_path=audio_path,
        meeting_id="ES2002a",
        output_dir=output_dir,
        auth_token="token",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "ES2002a_diarization.json"
    assert payload["meeting_id"] == "ES2002a"
    assert len(payload["segments"]) == 2
