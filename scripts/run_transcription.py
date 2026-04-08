from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from meeting_pipeline.audio.alignment import align_transcript
from meeting_pipeline.audio.whisperx_runner import TranscriptionConfig, transcribe_audio

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Transcription pipeline entry point.")


def run_transcription_pipeline(
    audio_path: Path,
    meeting_id: str,
    output_dir: Path,
    *,
    model_name: str = "large-v2",
    batch_size: int = 8,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
) -> Path:
    normalized_meeting_id = meeting_id.strip()
    if not normalized_meeting_id:
        raise ValueError("meeting_id must be a non-empty string")

    if not audio_path.exists() or not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    config = TranscriptionConfig(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    transcription_result = transcribe_audio(audio_path=audio_path, config=config)
    alignment_result = align_transcript(
        audio_path=audio_path,
        transcription_result=transcription_result,
        device=device,
    )

    language_out = alignment_result.get("language") or transcription_result.get("language")
    if not isinstance(language_out, str) or not language_out:
        language_out = "unknown"

    segments_obj = alignment_result.get("segments")
    words_obj = alignment_result.get("words")
    segments = segments_obj if isinstance(segments_obj, list) else []
    words = words_obj if isinstance(words_obj, list) else []

    payload: dict[str, object] = {
        "meeting_id": normalized_meeting_id,
        "language": language_out,
        "segments": segments,
        "words": words,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{normalized_meeting_id}_aligned.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOGGER.info(
        "transcription_alignment_complete segments=%d words=%d output=%s",
        len(segments),
        len(words),
        output_path,
    )
    return output_path


@app.command()
def main(
    audio_path: Path = typer.Option(..., "--audio-path", exists=True, readable=True),
    meeting_id: str = typer.Option(..., "--meeting-id"),
    output_dir: Path = typer.Option(Path("data/interim"), "--output-dir"),
    model_name: str = typer.Option("large-v2", "--model-name"),
    batch_size: int = typer.Option(8, "--batch-size"),
    device: str | None = typer.Option(None, "--device"),
    compute_type: str | None = typer.Option(None, "--compute-type"),
    language: str | None = typer.Option(None, "--language"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        output_path = run_transcription_pipeline(
            audio_path=audio_path,
            meeting_id=meeting_id,
            output_dir=output_dir,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    except Exception as exc:
        LOGGER.error("transcription pipeline failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
