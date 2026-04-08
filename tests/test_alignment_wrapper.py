from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from meeting_pipeline.audio.alignment import align_transcript


def test_align_transcript_returns_normalized_words(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    fake_whisperx = SimpleNamespace(
        load_align_model=lambda **_kwargs: (object(), {"meta": True}),
        load_audio=lambda _path: object(),
        align=lambda *_args, **_kwargs: {
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "hello"},
            ],
            "word_segments": [
                {"start": 0.0, "end": 0.2, "word": "hello"},
            ],
        },
    )

    cleanup_called = {"value": False}
    monkeypatch.setattr("meeting_pipeline.audio.alignment._import_whisperx", lambda: fake_whisperx)
    monkeypatch.setattr(
        "meeting_pipeline.audio.alignment.clear_torch_memory",
        lambda: cleanup_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        "meeting_pipeline.audio.alignment.log_gpu_state",
        lambda *_args, **_kwargs: None,
    )

    result = align_transcript(
        audio_path=audio_path,
        transcription_result={
            "language": "en",
            "segments": [{"start_time": 0.0, "end_time": 0.5, "text": "hello"}],
        },
    )

    assert result["language"] == "en"
    assert result["words"] == [
        {"speaker_id": "unknown", "start_time": 0.0, "end_time": 0.2, "text": "hello"}
    ]
    assert cleanup_called["value"] is True


def test_align_transcript_requires_language(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    with pytest.raises(ValueError, match="language"):
        align_transcript(
            audio_path=audio_path,
            transcription_result={"segments": [{"start_time": 0.0, "end_time": 0.1, "text": "x"}]},
        )
