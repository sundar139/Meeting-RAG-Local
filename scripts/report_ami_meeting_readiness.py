from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import typer

from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.repository import ConnectionProtocol, TranscriptChunkRepository

try:
    from scripts.ingest_many_meetings import _discover_meeting_ids_from_words_xml
except ModuleNotFoundError:  # pragma: no cover
    from ingest_many_meetings import _discover_meeting_ids_from_words_xml

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Report AMI meeting readiness across raw/interim/processed/DB assets.")


def _db_ingested_count(meeting_id: str) -> int | None:
    try:
        with connection_scope(application_name="meeting_pipeline:meeting_readiness") as connection:
            repository = TranscriptChunkRepository(cast(ConnectionProtocol, connection))
            overview = repository.get_meeting_overview(meeting_id)
            return overview.chunk_count
    except Exception:
        return None


def build_readiness_report(
    *,
    meeting_ids: list[str],
    raw_ami_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
) -> list[dict[str, Any]]:
    selected_ids = [item.strip() for item in meeting_ids if item.strip()]
    if not selected_ids:
        selected_ids = _discover_meeting_ids_from_words_xml(raw_ami_dir)

    report: list[dict[str, Any]] = []
    for meeting_id in sorted(set(selected_ids)):
        words_files = sorted(raw_ami_dir.glob(f"{meeting_id}.*.words.xml"))
        audio_file = raw_ami_dir / f"{meeting_id}.Mix-Headset.wav"
        aligned_path = interim_dir / f"{meeting_id}_aligned.json"
        diarization_path = interim_dir / f"{meeting_id}_diarization.json"
        turns_path = processed_dir / f"{meeting_id}_turns.json"
        chunk_count = _db_ingested_count(meeting_id)

        report.append(
            {
                "meeting_id": meeting_id,
                "words_xml_count": len(words_files),
                "audio_exists": audio_file.exists(),
                "aligned_exists": aligned_path.exists(),
                "diarization_exists": diarization_path.exists(),
                "turns_exists": turns_path.exists(),
                "db_chunk_count": chunk_count,
            }
        )

    return report


@app.command()
def main(
    meeting_id: list[str] = typer.Option([], "--meeting-id"),
    raw_ami_dir: Path = typer.Option(Path("data/raw/ami"), "--raw-ami-dir"),
    interim_dir: Path = typer.Option(Path("data/interim"), "--interim-dir"),
    processed_dir: Path = typer.Option(Path("data/processed"), "--processed-dir"),
    only_missing: bool = typer.Option(False, "--only-missing"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = build_readiness_report(
        meeting_ids=list(meeting_id),
        raw_ami_dir=raw_ami_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
    )

    if only_missing:
        report = [
            row
            for row in report
            if (
                row["words_xml_count"] == 0
                or not row["audio_exists"]
                or not row["turns_exists"]
                or not row["aligned_exists"]
                or not row["diarization_exists"]
                or row["db_chunk_count"] in (None, 0)
            )
        ]

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
