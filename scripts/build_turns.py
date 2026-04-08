from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

from meeting_pipeline.audio.attribution import attribute_words
from meeting_pipeline.audio.turn_builder import build_speaker_turns
from meeting_pipeline.schemas.diarization import DiarizationSegment
from meeting_pipeline.schemas.transcript import WordToken

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Turn building pipeline entry point.")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Input artifact does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in artifact: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a top-level object: {path}")

    return payload


def _load_words(payload: dict[str, Any]) -> list[WordToken]:
    words_obj = payload.get("words")
    if not isinstance(words_obj, list):
        raise ValueError("Aligned artifact must contain a 'words' list")

    words: list[WordToken] = []
    for item in words_obj:
        if not isinstance(item, dict):
            continue
        try:
            words.append(WordToken.model_validate(item))
        except ValueError:
            continue

    if not words:
        raise ValueError("Aligned artifact does not contain usable words")

    return words


def _load_diarization_segments(payload: dict[str, Any]) -> list[DiarizationSegment]:
    segments_obj = payload.get("segments")
    if not isinstance(segments_obj, list):
        raise ValueError("Diarization artifact must contain a 'segments' list")

    segments: list[DiarizationSegment] = []
    for item in segments_obj:
        if not isinstance(item, dict):
            continue
        try:
            segments.append(DiarizationSegment.model_validate(item))
        except ValueError:
            continue

    if not segments:
        raise ValueError("Diarization artifact does not contain usable segments")

    return segments


def _assert_meeting_consistency(
    meeting_id: str,
    aligned_payload: dict[str, Any],
    diarization_payload: dict[str, Any],
) -> None:
    aligned_meeting = aligned_payload.get("meeting_id")
    diarization_meeting = diarization_payload.get("meeting_id")

    if (
        isinstance(aligned_meeting, str)
        and aligned_meeting.strip()
        and aligned_meeting != meeting_id
    ):
        raise ValueError(
            "Meeting ID mismatch: aligned artifact has "
            f"'{aligned_meeting}', expected '{meeting_id}'"
        )

    if (
        isinstance(diarization_meeting, str)
        and diarization_meeting.strip()
        and diarization_meeting != meeting_id
    ):
        raise ValueError(
            "Meeting ID mismatch: diarization artifact has "
            f"'{diarization_meeting}', expected '{meeting_id}'"
        )


def build_turns_artifact(
    meeting_id: str,
    aligned_path: Path,
    diarization_path: Path,
    output_dir: Path,
    *,
    max_gap_seconds: float = 1.0,
    default_label: str = "unknown",
) -> Path:
    normalized_meeting_id = meeting_id.strip()
    if not normalized_meeting_id:
        raise ValueError("meeting_id must be a non-empty string")

    aligned_payload = _load_json(aligned_path)
    diarization_payload = _load_json(diarization_path)
    _assert_meeting_consistency(normalized_meeting_id, aligned_payload, diarization_payload)

    words = _load_words(aligned_payload)
    diarization_segments = _load_diarization_segments(diarization_payload)

    attributed_words = attribute_words(
        words=words,
        diarization_segments=diarization_segments,
        default_label=default_label,
    )
    turns = build_speaker_turns(
        meeting_id=normalized_meeting_id,
        words=attributed_words,
        max_gap_seconds=max_gap_seconds,
        default_label=default_label,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{normalized_meeting_id}_turns.json"
    payload = {
        "meeting_id": normalized_meeting_id,
        "turns": [turn.model_dump(mode="json") for turn in turns],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOGGER.info("aligned_words=%d", len(words))
    LOGGER.info("diarization_segments=%d", len(diarization_segments))
    LOGGER.info("attributed_words=%d", len(attributed_words))
    LOGGER.info("speaker_turns=%d", len(turns))
    LOGGER.info("output=%s", output_path)

    return output_path


@app.command()
def main(
    meeting_id: str = typer.Option(..., "--meeting-id"),
    aligned_path: Path = typer.Option(..., "--aligned-path", exists=True, readable=True),
    diarization_path: Path = typer.Option(..., "--diarization-path", exists=True, readable=True),
    output_dir: Path = typer.Option(Path("data/processed"), "--output-dir"),
    max_gap_seconds: float = typer.Option(1.0, "--max-gap-seconds"),
    default_label: str = typer.Option("unknown", "--default-label"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        output_path = build_turns_artifact(
            meeting_id=meeting_id,
            aligned_path=aligned_path,
            diarization_path=diarization_path,
            output_dir=output_dir,
            max_gap_seconds=max_gap_seconds,
            default_label=default_label,
        )
    except Exception as exc:
        LOGGER.error("build_turns failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
