from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import typer

from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.repository import ConnectionProtocol, TranscriptChunkRepository

try:
    from scripts.ingest_embeddings import ingest_embeddings
except ModuleNotFoundError:  # pragma: no cover
    from ingest_embeddings import ingest_embeddings

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Batch embedding ingestion across many AMI meeting IDs.")


def _discover_meeting_ids_from_words_xml(raw_ami_dir: Path) -> list[str]:
    if not raw_ami_dir.exists() or not raw_ami_dir.is_dir():
        return []

    discovered: set[str] = set()
    for xml_path in raw_ami_dir.glob("*.words.xml"):
        name = xml_path.name
        prefix = name.split(".", 1)[0].strip()
        if prefix:
            discovered.add(prefix)
    return sorted(discovered)


def _load_meeting_ids_from_file(path: Path) -> list[str]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Meeting list file does not exist: {path}")

    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            ids.append(cleaned)
    return ids


def _meeting_has_existing_chunks(meeting_id: str) -> bool:
    with connection_scope(application_name="meeting_pipeline:batch_ingest_check") as connection:
        repository = TranscriptChunkRepository(cast(ConnectionProtocol, connection))
        overview = repository.get_meeting_overview(meeting_id)
        return overview.chunk_count > 0


def ingest_many_meetings(
    *,
    meeting_ids: list[str],
    raw_ami_dir: Path,
    turns_dir: Path,
    replace_existing: bool,
    skip_existing: bool,
    batch_size: int,
    dry_run: bool,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    selected_ids = [item.strip() for item in meeting_ids if item.strip()]
    if not selected_ids:
        selected_ids = _discover_meeting_ids_from_words_xml(raw_ami_dir)

    deduped_ids: list[str] = []
    seen: set[str] = set()
    for item in selected_ids:
        if item in seen:
            continue
        deduped_ids.append(item)
        seen.add(item)

    processed = 0
    ingested = 0
    skipped_existing = 0
    skipped_missing_turns = 0
    failed = 0

    for meeting_id in deduped_ids:
        processed += 1
        turns_path = turns_dir / f"{meeting_id}_turns.json"
        if not turns_path.exists() or not turns_path.is_file():
            skipped_missing_turns += 1
            LOGGER.warning("Skipping %s: missing turns artifact %s", meeting_id, turns_path)
            continue

        if skip_existing and not replace_existing:
            try:
                if _meeting_has_existing_chunks(meeting_id):
                    skipped_existing += 1
                    LOGGER.info("Skipping %s: chunks already ingested", meeting_id)
                    continue
            except Exception as exc:
                LOGGER.warning(
                    "DB existing-check failed for %s; continuing. reason=%s",
                    meeting_id,
                    exc,
                )

        try:
            ingest_embeddings(
                meeting_id=meeting_id,
                turns_path=turns_path,
                replace_existing=replace_existing,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            ingested += 1
            LOGGER.info("Ingested meeting_id=%s", meeting_id)
        except Exception as exc:
            failed += 1
            LOGGER.error("Failed ingestion meeting_id=%s reason=%s", meeting_id, exc)

    return {
        "processed": processed,
        "ingested": ingested,
        "skipped_existing": skipped_existing,
        "skipped_missing_turns": skipped_missing_turns,
        "failed": failed,
    }


@app.command()
def main(
    meeting_id: list[str] = typer.Option([], "--meeting-id"),
    meeting_list_file: Path | None = typer.Option(None, "--meeting-list-file"),
    raw_ami_dir: Path = typer.Option(Path("data/raw/ami"), "--raw-ami-dir"),
    turns_dir: Path = typer.Option(Path("data/processed"), "--turns-dir"),
    replace_existing: bool = typer.Option(False, "--replace-existing"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing"),
    batch_size: int = typer.Option(16, "--batch-size"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    selected_ids = list(meeting_id)
    if meeting_list_file is not None:
        selected_ids.extend(_load_meeting_ids_from_file(meeting_list_file))

    summary = ingest_many_meetings(
        meeting_ids=selected_ids,
        raw_ami_dir=raw_ami_dir,
        turns_dir=turns_dir,
        replace_existing=replace_existing,
        skip_existing=skip_existing,
        batch_size=batch_size,
        dry_run=dry_run,
    )

    LOGGER.info("processed=%d", summary["processed"])
    LOGGER.info("ingested=%d", summary["ingested"])
    LOGGER.info("skipped_existing=%d", summary["skipped_existing"])
    LOGGER.info("skipped_missing_turns=%d", summary["skipped_missing_turns"])
    LOGGER.info("failed=%d", summary["failed"])


if __name__ == "__main__":
    app()
