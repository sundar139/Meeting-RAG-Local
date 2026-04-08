from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from meeting_pipeline.audio.whisperx_runner import TranscriptionConfig, transcribe_audio


def test_transcribe_audio_returns_normalized_payload(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    class FakeModel:
        def transcribe(self, _audio: object, batch_size: int) -> dict[str, object]:
            assert batch_size == 4
            return {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 0.4, "text": "  hello   world "},
                ],
            }

    fake_whisperx = SimpleNamespace(
        load_model=lambda *_args, **_kwargs: FakeModel(),
        load_audio=lambda _path: object(),
    )

    cleanup_called = {"value": False}
    monkeypatch.setattr(
        "meeting_pipeline.audio.whisperx_runner._import_whisperx",
        lambda: fake_whisperx,
    )
    monkeypatch.setattr(
        "meeting_pipeline.audio.whisperx_runner.clear_torch_memory",
        lambda: cleanup_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        "meeting_pipeline.audio.whisperx_runner.log_gpu_state",
        lambda *_args, **_kwargs: None,
    )

    result = transcribe_audio(
        audio_path=audio_path,
        config=TranscriptionConfig(model_name="large-v2", batch_size=4),
    )

    assert result["language"] == "en"
    assert result["segments"] == [
        {"id": 0, "start_time": 0.0, "end_time": 0.4, "text": "hello world"},
    ]
    assert cleanup_called["value"] is True


def test_transcribe_audio_cleans_up_on_failure(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    class FailingModel:
        def transcribe(self, _audio: object, batch_size: int) -> dict[str, object]:
            _ = batch_size
            raise RuntimeError("boom")

    fake_whisperx = SimpleNamespace(
        load_model=lambda *_args, **_kwargs: FailingModel(),
        load_audio=lambda _path: object(),
    )

    cleanup_called = {"value": False}
    monkeypatch.setattr(
        "meeting_pipeline.audio.whisperx_runner._import_whisperx",
        lambda: fake_whisperx,
    )
    monkeypatch.setattr(
        "meeting_pipeline.audio.whisperx_runner.clear_torch_memory",
        lambda: cleanup_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        "meeting_pipeline.audio.whisperx_runner.log_gpu_state",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        transcribe_audio(audio_path=audio_path, config=TranscriptionConfig())

    assert cleanup_called["value"] is True
