from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meeting_pipeline.audio.gpu_utils import clear_torch_memory, get_torch_device, log_gpu_state
from meeting_pipeline.config import get_settings
from meeting_pipeline.schemas.diarization import DiarizationDocument, DiarizationSegment

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiarizationConfig:
    model_name: str = "pyannote/speaker-diarization-3.1"
    auth_token: str | None = None
    device: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


def _load_pyannote_pipeline_class() -> Any:
    try:
        from pyannote.audio import Pipeline

        return Pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Install GPU dependencies with "
            "'uv sync --group dev --extra gpu'."
        ) from exc


def _resolve_auth_token(config: DiarizationConfig) -> str:
    candidates: list[str | None] = [
        config.auth_token,
        os.getenv("PYANNOTE_AUTH_TOKEN"),
        os.getenv("HUGGINGFACE_TOKEN"),
        os.getenv("HF_TOKEN"),
    ]

    try:
        settings = get_settings()
        candidates.extend(
            [
                settings.pyannote_auth_token.get_secret_value(),
                settings.huggingface_token.get_secret_value(),
            ]
        )
    except Exception:
        # Fall back to explicit args and environment variables only.
        pass

    token = next(
        (item.strip() for item in candidates if isinstance(item, str) and item.strip()),
        "",
    )

    if not token:
        raise ValueError(
            "Hugging Face auth token is required for pyannote model loading. "
            "Set PYANNOTE_AUTH_TOKEN/HUGGINGFACE_TOKEN/HF_TOKEN, configure .env, "
            "or pass auth_token in config."
        )
    return token


def run_diarization(audio_path: Path | str, config: DiarizationConfig) -> list[DiarizationSegment]:
    audio_file = Path(audio_path)
    if not audio_file.exists() or not audio_file.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")

    token = _resolve_auth_token(config)
    device = config.device or get_torch_device()
    pipeline: Any | None = None

    LOGGER.info("Loading pyannote pipeline model_name=%s", config.model_name)
    log_gpu_state(LOGGER, context="diarization_before_load")

    try:
        pipeline_class = _load_pyannote_pipeline_class()
        try:
            pipeline = pipeline_class.from_pretrained(
                config.model_name,
                use_auth_token=token,
            )
        except TypeError as exc:
            if "hf_hub_download() got an unexpected keyword argument 'use_auth_token'" not in str(
                exc
            ):
                raise
            pipeline = pipeline_class.from_pretrained(
                config.model_name,
                token=token,
            )

        if pipeline is None:
            raise RuntimeError(
                "Failed to load pyannote pipeline. Ensure the token has access and accept "
                "model terms at https://hf.co/pyannote/speaker-diarization-3.1."
            )

        if device == "cuda":
            try:
                import torch

                pipeline.to(torch.device("cuda"))
            except Exception:
                LOGGER.warning("Could not move pyannote pipeline to CUDA; using default device")

        diarization_kwargs: dict[str, object] = {}
        if config.min_speakers is not None:
            diarization_kwargs["min_speakers"] = config.min_speakers
        if config.max_speakers is not None:
            diarization_kwargs["max_speakers"] = config.max_speakers

        LOGGER.info("Running diarization audio=%s", audio_file)
        diarization_output = pipeline(str(audio_file), **diarization_kwargs) # type: ignore

        segments: list[DiarizationSegment] = []
        for turn, _, speaker_label in diarization_output.itertracks(yield_label=True):
            try:
                segments.append(
                    DiarizationSegment(
                        speaker_label=str(speaker_label),
                        start_time=float(turn.start),
                        end_time=float(turn.end),
                    )
                )
            except ValueError:
                continue

        return sorted(
            segments,
            key=lambda item: (item.start_time, item.end_time, item.speaker_label),
        )
    finally:
        LOGGER.info("Cleaning up pyannote diarization resources")
        pipeline = None
        clear_torch_memory()
        log_gpu_state(LOGGER, context="diarization_after_cleanup")


class DiarizationService:
    def diarize(self, meeting_id: str, audio_path: Path) -> DiarizationDocument:
        config = DiarizationConfig()
        segments = run_diarization(audio_path=audio_path, config=config)
        return DiarizationDocument(
            meeting_id=meeting_id,
            segments=segments,
        )
