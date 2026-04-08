from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from meeting_pipeline.audio.diarization import DiarizationConfig, run_diarization


@dataclass(frozen=True)
class FakeTurn:
    start: float
    end: float


class FakeAnnotation:
    def itertracks(self, yield_label: bool = False):
        assert yield_label is True
        yield FakeTurn(0.0, 1.0), None, "SPEAKER_00"
        yield FakeTurn(1.1, 2.0), None, "SPEAKER_01"


class FakePipeline:
    def __init__(self) -> None:
        self.moved_to: object | None = None

    @classmethod
    def from_pretrained(cls, model_name: str, use_auth_token: str):
        _ = model_name
        assert use_auth_token == "token"
        return cls()

    def to(self, device: object) -> None:
        self.moved_to = device

    def __call__(self, audio_path: str, **_kwargs):
        _ = audio_path
        return FakeAnnotation()


class FakeSecret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


def test_run_diarization_returns_normalized_segments(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    cleanup_called = {"value": False}
    monkeypatch.setattr(
        "meeting_pipeline.audio.diarization._load_pyannote_pipeline_class",
        lambda: FakePipeline,
    )
    monkeypatch.setattr("meeting_pipeline.audio.diarization.get_torch_device", lambda: "cpu")
    monkeypatch.setattr(
        "meeting_pipeline.audio.diarization.clear_torch_memory",
        lambda: cleanup_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        "meeting_pipeline.audio.diarization.log_gpu_state",
        lambda *_args, **_kwargs: None,
    )

    segments = run_diarization(
        audio_path=audio_path,
        config=DiarizationConfig(auth_token="token"),
    )

    assert len(segments) == 2
    assert segments[0].speaker_label == "SPEAKER_00"
    assert cleanup_called["value"] is True


def test_run_diarization_requires_auth_token(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.delenv("PYANNOTE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "meeting_pipeline.audio.diarization.get_settings",
        lambda: SimpleNamespace(
            pyannote_auth_token=FakeSecret(""),
            huggingface_token=FakeSecret(""),
        ),
    )

    with pytest.raises(ValueError, match="token"):
        run_diarization(audio_path=audio_path, config=DiarizationConfig(auth_token=None))
