from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from meeting_pipeline.audio.gpu_utils import clear_torch_memory, get_torch_device, log_gpu_state

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionConfig:
    model_name: str = "large-v2"
    compute_type: str | None = None
    batch_size: int = 8
    device: str | None = None
    language: str | None = None


class WhisperXModel(Protocol):
    def transcribe(self, audio: object, batch_size: int) -> Mapping[str, object]: ...


def _import_whisperx() -> Any:
    try:
        import whisperx

        return whisperx
    except ModuleNotFoundError as exc:
        if exc.name and exc.name != "whisperx":
            raise RuntimeError(
                "whisperx import failed because a required dependency is missing "
                f"('{exc.name}'). Install GPU dependencies with "
                "'uv sync --group dev --extra gpu'."
            ) from exc
        raise RuntimeError(
            "whisperx is not installed. Install GPU dependencies with "
            "'uv sync --group dev --extra gpu'."
        ) from exc


def _first_present(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _as_float(value: object | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_segment(segment: Mapping[str, object], index: int) -> dict[str, object] | None:
    start_time = _as_float(_first_present(segment, "start", "start_time"))
    end_time = _as_float(_first_present(segment, "end", "end_time"))
    text_raw = _first_present(segment, "text")

    if start_time is None or end_time is None:
        return None

    if end_time < start_time:
        return None

    if not isinstance(text_raw, str):
        return None
    text = " ".join(text_raw.split())
    if not text:
        return None

    return {
        "id": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
    }


def _normalize_transcription_result(
    raw_result: Mapping[str, object], fallback_language: str | None
) -> dict[str, object]:
    language_raw = raw_result.get("language")
    language = fallback_language or "unknown"
    if isinstance(language_raw, str) and language_raw.strip():
        language = language_raw.strip()

    raw_segments_obj = raw_result.get("segments")
    if not isinstance(raw_segments_obj, Sequence):
        raise ValueError("WhisperX transcription result is missing a valid 'segments' list")

    normalized_segments: list[dict[str, object]] = []
    for idx, segment in enumerate(raw_segments_obj):
        if not isinstance(segment, Mapping):
            continue
        normalized = _normalize_segment(segment, idx)
        if normalized is not None:
            normalized_segments.append(normalized)

    return {
        "language": language,
        "segments": normalized_segments,
    }


def transcribe_audio(audio_path: Path | str, config: TranscriptionConfig) -> dict[str, object]:
    audio_file = Path(audio_path)
    if not audio_file.exists() or not audio_file.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")

    whisperx = _import_whisperx()
    device = config.device or get_torch_device()
    compute_type = config.compute_type or "int8"

    model: WhisperXModel | None = None
    LOGGER.info(
        "Loading WhisperX model model_name=%s device=%s compute_type=%s",
        config.model_name,
        device,
        compute_type,
    )
    log_gpu_state(LOGGER, context="whisperx_before_load")

    try:
        load_kwargs: dict[str, object] = {
            "compute_type": compute_type,
        }
        if config.language is not None:
            load_kwargs["language"] = config.language

        model = cast(WhisperXModel, whisperx.load_model(config.model_name, device, **load_kwargs))
        LOGGER.info("Transcribing audio file=%s batch_size=%d", audio_file, config.batch_size)

        audio = whisperx.load_audio(str(audio_file))
        raw_result = model.transcribe(audio, batch_size=config.batch_size)

        if not isinstance(raw_result, Mapping):
            raise RuntimeError("WhisperX returned an unexpected transcription payload")

        return _normalize_transcription_result(raw_result, fallback_language=config.language)
    finally:
        LOGGER.info("Cleaning up WhisperX transcription resources")
        model = None
        clear_torch_memory()
        log_gpu_state(LOGGER, context="whisperx_after_cleanup")


class WhisperXRunner:
    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config

    def transcribe(self, audio_path: Path) -> dict[str, object]:
        return transcribe_audio(audio_path=audio_path, config=self.config)
