from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from meeting_pipeline.audio.gpu_utils import clear_torch_memory, get_torch_device, log_gpu_state
from meeting_pipeline.schemas.transcript import WordToken

LOGGER = logging.getLogger(__name__)


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


def _normalize_segments_for_alignment(segments_obj: object) -> list[dict[str, object]]:
    if not isinstance(segments_obj, Sequence):
        raise ValueError("Transcription result is missing 'segments' for alignment")

    normalized: list[dict[str, object]] = []
    for segment in segments_obj:
        if not isinstance(segment, Mapping):
            continue

        start_time = _as_float(_first_present(segment, "start", "start_time"))
        end_time = _as_float(_first_present(segment, "end", "end_time"))
        text_raw = _first_present(segment, "text")

        if start_time is None or end_time is None:
            continue

        if end_time < start_time or not isinstance(text_raw, str):
            continue

        text = " ".join(text_raw.split())
        if not text:
            continue

        normalized.append({"start": start_time, "end": end_time, "text": text})

    if not normalized:
        raise ValueError("No valid transcription segments available for alignment")

    return normalized


def _extract_word_tokens(alignment_payload: Mapping[str, object]) -> list[WordToken]:
    word_entries: list[Mapping[str, object]] = []

    word_segments = alignment_payload.get("word_segments")
    if isinstance(word_segments, Sequence):
        for item in word_segments:
            if isinstance(item, Mapping):
                word_entries.append(item)

    if not word_entries:
        segments_obj = alignment_payload.get("segments")
        if isinstance(segments_obj, Sequence):
            for segment in segments_obj:
                if not isinstance(segment, Mapping):
                    continue
                words_obj = segment.get("words")
                if not isinstance(words_obj, Sequence):
                    continue
                for item in words_obj:
                    if isinstance(item, Mapping):
                        word_entries.append(item)

    tokens: list[WordToken] = []
    for item in word_entries:
        start_time = _as_float(_first_present(item, "start", "start_time"))
        end_time = _as_float(_first_present(item, "end", "end_time"))
        text_raw = _first_present(item, "word", "text")
        speaker_raw = _first_present(item, "speaker", "speaker_id") or "unknown"

        if start_time is None or end_time is None or not isinstance(text_raw, str):
            continue

        try:
            token = WordToken(
                speaker_id=str(speaker_raw),
                start_time=start_time,
                end_time=end_time,
                text=text_raw,
            )
        except (TypeError, ValueError):
            continue

        tokens.append(token)

    return sorted(tokens, key=lambda item: (item.start_time, item.end_time, item.text))


def align_transcript(
    audio_path: Path | str,
    transcription_result: Mapping[str, object],
    device: str | None = None,
) -> dict[str, object]:
    audio_file = Path(audio_path)
    if not audio_file.exists() or not audio_file.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")

    language_obj = transcription_result.get("language")
    if not isinstance(language_obj, str) or not language_obj.strip():
        raise ValueError(
            "Transcription result is missing 'language'. "
            "Ensure transcription runs before alignment."
        )
    language = language_obj.strip()

    segments_for_alignment = _normalize_segments_for_alignment(transcription_result.get("segments"))

    whisperx = _import_whisperx()
    selected_device = device or get_torch_device()
    align_model: object | None = None

    LOGGER.info("Loading WhisperX alignment model language=%s device=%s", language, selected_device)
    log_gpu_state(LOGGER, context="alignment_before_load")

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=selected_device,
        )
        audio = whisperx.load_audio(str(audio_file))
        aligned = whisperx.align(
            segments_for_alignment,
            align_model,
            metadata,
            audio,
            selected_device,
            return_char_alignments=False,
        )

        if not isinstance(aligned, Mapping):
            raise RuntimeError("WhisperX returned an unexpected alignment payload")

        words = _extract_word_tokens(aligned)

        segments_normalized: list[dict[str, object]] = []
        segments_obj = aligned.get("segments")
        if isinstance(segments_obj, Sequence):
            for idx, segment in enumerate(segments_obj):
                if not isinstance(segment, Mapping):
                    continue
                start_time = _as_float(_first_present(segment, "start", "start_time"))
                end_time = _as_float(_first_present(segment, "end", "end_time"))
                text_raw = _first_present(segment, "text")

                if start_time is None or end_time is None:
                    continue

                if end_time < start_time or not isinstance(text_raw, str):
                    continue

                text = " ".join(text_raw.split())
                if not text:
                    continue

                segments_normalized.append(
                    {
                        "id": idx,
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                    }
                )

        payload: dict[str, object] = {
            "language": language,
            "segments": segments_normalized,
            "words": [token.model_dump(mode="json") for token in words],
        }

        meeting_id = transcription_result.get("meeting_id")
        if isinstance(meeting_id, str) and meeting_id.strip():
            payload["meeting_id"] = meeting_id.strip()

        return payload
    finally:
        LOGGER.info("Cleaning up WhisperX alignment resources")
        align_model = None
        clear_torch_memory()
        log_gpu_state(LOGGER, context="alignment_after_cleanup")


class AlignmentService:
    def align(
        self,
        audio_path: Path,
        transcription_result: Mapping[str, object],
        device: str | None = None,
    ) -> dict[str, object]:
        return align_transcript(
            audio_path=audio_path,
            transcription_result=transcription_result,
            device=device,
        )
