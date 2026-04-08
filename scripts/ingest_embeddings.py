from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import typer

from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.repository import TranscriptChunkInsert, TranscriptChunkRepository
from meeting_pipeline.embeddings.embedder import Embedder
from meeting_pipeline.embeddings.ollama_client import OllamaClientError
from meeting_pipeline.schemas.transcript import SpeakerTurn

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Embedding ingestion entry point.")

MAX_EMBED_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.2


def _load_turns_payload(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Turns artifact does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Turns artifact is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Turns artifact must be a JSON object")

    return payload


def _validate_turns(payload: dict[str, Any], meeting_id: str) -> list[SpeakerTurn]:
    top_level_meeting = payload.get("meeting_id")
    if (
        isinstance(top_level_meeting, str)
        and top_level_meeting.strip()
        and top_level_meeting != meeting_id
    ):
        raise ValueError(
            "Meeting ID mismatch: turns artifact has "
            f"'{top_level_meeting}', expected '{meeting_id}'"
        )

    turns_obj = payload.get("turns")
    if not isinstance(turns_obj, list):
        raise ValueError("Turns artifact must contain a 'turns' list")

    validated: list[SpeakerTurn] = []
    for idx, item in enumerate(turns_obj):
        if not isinstance(item, dict):
            raise ValueError(f"Turn at index {idx} is not an object")

        if "meeting_id" not in item:
            item = {**item, "meeting_id": meeting_id}

        turn = SpeakerTurn.model_validate(item)
        if turn.meeting_id != meeting_id:
            raise ValueError(
                f"Meeting ID mismatch in turn index {idx}: '{turn.meeting_id}' != '{meeting_id}'"
            )
        validated.append(turn)

    if not validated:
        raise ValueError("Turns artifact contains no usable turns")

    return validated


def _embed_with_retry(embedder: Embedder, text: str, turn_index: int) -> list[float]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            return embedder.embed_document(text)
        except OllamaClientError as exc:
            last_error = exc
            if attempt == MAX_EMBED_RETRIES:
                break
            LOGGER.warning(
                "Embedding retry for turn=%d attempt=%d reason=%s",
                turn_index,
                attempt,
                exc,
            )
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Embedding failed for turn index {turn_index} after {MAX_EMBED_RETRIES} attempts"
    ) from last_error


def _batched(items: list[SpeakerTurn], batch_size: int) -> list[list[SpeakerTurn]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def ingest_embeddings(
    meeting_id: str,
    turns_path: Path,
    *,
    replace_existing: bool = False,
    batch_size: int = 16,
    dry_run: bool = False,
) -> dict[str, int | str]:
    normalized_meeting_id = meeting_id.strip()
    if not normalized_meeting_id:
        raise ValueError("meeting_id must be a non-empty string")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    turns_payload = _load_turns_payload(turns_path)
    turns = _validate_turns(turns_payload, normalized_meeting_id)

    turns_loaded = len(turns)
    turns_skipped = 0
    embeddings_attempted = 0
    embeddings_succeeded = 0

    dedupe_seen: set[tuple[str, float, float, str]] = set()
    unique_turns: list[SpeakerTurn] = []
    for turn in turns:
        key = (turn.speaker_label, turn.start_time, turn.end_time, turn.text)
        if key in dedupe_seen:
            turns_skipped += 1
            continue
        dedupe_seen.add(key)
        unique_turns.append(turn)

    embedder = Embedder()

    insert_rows: list[TranscriptChunkInsert] = []
    for index, turn in enumerate(unique_turns):
        embeddings_attempted += 1
        embedding = _embed_with_retry(embedder=embedder, text=turn.text, turn_index=index)
        embeddings_succeeded += 1
        insert_rows.append(
            TranscriptChunkInsert(
                meeting_id=normalized_meeting_id,
                speaker_label=turn.speaker_label,
                start_time=turn.start_time,
                end_time=turn.end_time,
                content=turn.text,
                embedding=embedding,
            )
        )

    rows_inserted = 0
    rows_deleted = 0

    if not dry_run:
        with connection_scope(application_name="meeting_pipeline:ingest_embeddings") as connection:
            repository = TranscriptChunkRepository(connection)

            if replace_existing:
                rows_deleted = repository.delete_chunks_for_meeting(normalized_meeting_id)

            for batch in _batched(insert_rows, batch_size=batch_size):
                rows_inserted += repository.insert_transcript_chunks(batch)
    else:
        rows_inserted = len(insert_rows)

    summary: dict[str, int | str] = {
        "meeting_id": normalized_meeting_id,
        "turns_loaded": turns_loaded,
        "turns_skipped": turns_skipped,
        "embeddings_attempted": embeddings_attempted,
        "embeddings_succeeded": embeddings_succeeded,
        "rows_deleted": rows_deleted,
        "rows_inserted": rows_inserted,
    }

    LOGGER.info("meeting_id=%s", normalized_meeting_id)
    LOGGER.info("turns_loaded=%d", turns_loaded)
    LOGGER.info("turns_skipped=%d", turns_skipped)
    LOGGER.info("embeddings_attempted=%d", embeddings_attempted)
    LOGGER.info("embeddings_succeeded=%d", embeddings_succeeded)
    LOGGER.info("rows_deleted=%d", rows_deleted)
    LOGGER.info("rows_inserted=%d", rows_inserted)
    LOGGER.info("dry_run=%s", dry_run)

    return summary


@app.command()
def main(
    meeting_id: str = typer.Option(..., "--meeting-id"),
    turns_path: Path = typer.Option(..., "--turns-path", exists=True, readable=True),
    replace_existing: bool = typer.Option(False, "--replace-existing"),
    batch_size: int = typer.Option(16, "--batch-size"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        ingest_embeddings(
            meeting_id=meeting_id,
            turns_path=turns_path,
            replace_existing=replace_existing,
            batch_size=batch_size,
            dry_run=dry_run,
        )
    except Exception as exc:
        LOGGER.error("embedding ingestion failed: %s", exc)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
