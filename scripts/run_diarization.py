from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from meeting_pipeline.audio.diarization import DiarizationConfig, run_diarization

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Speaker diarization pipeline entry point.")


def run_diarization_pipeline(
    audio_path: Path,
    meeting_id: str,
    output_dir: Path,
    *,
    model_name: str = "pyannote/speaker-diarization-3.1",
    device: str | None = None,
    auth_token: str | None = None,
) -> Path:
    normalized_meeting_id = meeting_id.strip()
    if not normalized_meeting_id:
        raise ValueError("meeting_id must be a non-empty string")

    if not audio_path.exists() or not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    config = DiarizationConfig(
        model_name=model_name,
        device=device,
        auth_token=auth_token,
    )

    segments = run_diarization(audio_path=audio_path, config=config)

    payload: dict[str, object] = {
        "meeting_id": normalized_meeting_id,
        "segments": [segment.model_dump(mode="json") for segment in segments],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{normalized_meeting_id}_diarization.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    distinct_speakers = {segment.speaker_label for segment in segments}
    LOGGER.info(
        "diarization_complete segments=%d speakers=%d output=%s",
        len(segments),
        len(distinct_speakers),
        output_path,
    )
    return output_path


@app.command()
def main(
    audio_path: Path = typer.Option(..., "--audio-path", exists=True, readable=True),
    meeting_id: str = typer.Option(..., "--meeting-id"),
    output_dir: Path = typer.Option(Path("data/interim"), "--output-dir"),
    model_name: str = typer.Option("pyannote/speaker-diarization-3.1", "--model-name"),
    device: str | None = typer.Option(None, "--device"),
    auth_token: str | None = typer.Option(None, "--auth-token"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        output_path = run_diarization_pipeline(
            audio_path=audio_path,
            meeting_id=meeting_id,
            output_dir=output_dir,
            model_name=model_name,
            device=device,
            auth_token=auth_token,
        )
    except Exception as exc:
        LOGGER.error("diarization pipeline failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
