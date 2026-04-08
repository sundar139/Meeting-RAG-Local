from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import typer

from meeting_pipeline.db.connection import connection_scope
from meeting_pipeline.db.repository import ConnectionProtocol, TranscriptChunkRepository

try:
    from scripts.ingest_embeddings import ingest_embeddings
except ModuleNotFoundError:  # pragma: no cover
    from ingest_embeddings import ingest_embeddings

LOGGER = logging.getLogger(__name__)
app = typer.Typer(help="Batch embedding ingestion across many AMI meeting IDs.")

DiscoverySource = Literal["words", "turns", "both"]


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


def _discover_meeting_ids_from_turns_json(turns_dir: Path) -> list[str]:
    if not turns_dir.exists() or not turns_dir.is_dir():
        return []

    discovered: set[str] = set()
    for turns_path in turns_dir.glob("*_turns.json"):
        name = turns_path.name
        if not name.endswith("_turns.json"):
            continue
        meeting_id = name[: -len("_turns.json")].strip()
        if meeting_id:
            discovered.add(meeting_id)
    return sorted(discovered)


def discover_meeting_ids(
    *,
    raw_ami_dir: Path,
    turns_dir: Path,
    discovery_source: DiscoverySource = "both",
) -> list[str]:
    discovered: set[str] = set()

    if discovery_source in ("words", "both"):
        discovered.update(_discover_meeting_ids_from_words_xml(raw_ami_dir))
    if discovery_source in ("turns", "both"):
        discovered.update(_discover_meeting_ids_from_turns_json(turns_dir))

    return sorted(discovered)


def build_ingestion_plan(
    *,
    meeting_ids: list[str],
    raw_ami_dir: Path,
    turns_dir: Path,
    discovery_source: DiscoverySource = "both",
) -> list[dict[str, Any]]:
    selected_ids = [item.strip() for item in meeting_ids if item.strip()]
    if not selected_ids:
        selected_ids = discover_meeting_ids(
            raw_ami_dir=raw_ami_dir,
            turns_dir=turns_dir,
            discovery_source=discovery_source,
        )

    plan: list[dict[str, Any]] = []
    for meeting_id in sorted(set(selected_ids)):
        words_xml_count = len(list(raw_ami_dir.glob(f"{meeting_id}.*.words.xml")))
        turns_path = turns_dir / f"{meeting_id}_turns.json"
        turns_exists = turns_path.exists() and turns_path.is_file()
        plan.append(
            {
                "meeting_id": meeting_id,
                "words_xml_count": words_xml_count,
                "turns_exists": turns_exists,
                "ready_for_ingest": turns_exists,
            }
        )
    return plan


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
    discovery_source: DiscoverySource = "both",
    retrieval_chunk_window_seconds: float | None = None,
    retrieval_chunk_overlap_seconds: float | None = None,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    selected_ids = [item.strip() for item in meeting_ids if item.strip()]
    if not selected_ids:
        selected_ids = discover_meeting_ids(
            raw_ami_dir=raw_ami_dir,
            turns_dir=turns_dir,
            discovery_source=discovery_source,
        )

    deduped_ids: list[str] = []
    seen: set[str] = set()
    for item in selected_ids:
        if item in seen:
            continue
        deduped_ids.append(item)
        seen.add(item)

    if len(deduped_ids) == 1:
        LOGGER.warning(
            "Only one meeting candidate found (%s). "
            "Use 'discover' command to inspect local readiness for additional meetings.",
            deduped_ids[0],
        )

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
            ingest_kwargs: dict[str, object] = {
                "meeting_id": meeting_id,
                "turns_path": turns_path,
                "replace_existing": replace_existing,
                "batch_size": batch_size,
                "dry_run": dry_run,
            }
            if retrieval_chunk_window_seconds is not None:
                ingest_kwargs["retrieval_chunk_window_seconds"] = retrieval_chunk_window_seconds
            if retrieval_chunk_overlap_seconds is not None:
                ingest_kwargs["retrieval_chunk_overlap_seconds"] = retrieval_chunk_overlap_seconds

            ingest_embeddings(**ingest_kwargs)
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


@app.command("discover")
def discover(
    meeting_id: list[str] = typer.Option([], "--meeting-id"),
    raw_ami_dir: Path = typer.Option(Path("data/raw/ami"), "--raw-ami-dir"),
    turns_dir: Path = typer.Option(Path("data/processed"), "--turns-dir"),
    discovery_source: DiscoverySource = typer.Option("both", "--discovery-source"),
) -> None:
    plan = build_ingestion_plan(
        meeting_ids=list(meeting_id),
        raw_ami_dir=raw_ami_dir,
        turns_dir=turns_dir,
        discovery_source=discovery_source,
    )
    ready_count = sum(1 for item in plan if bool(item.get("ready_for_ingest")))
    payload: dict[str, Any] = {
        "discovery_source": discovery_source,
        "meeting_count": len(plan),
        "ready_for_ingest_count": ready_count,
        "meetings": plan,
    }
    typer.echo(json.dumps(payload, indent=2))


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
    discovery_source: DiscoverySource = typer.Option("both", "--discovery-source"),
    retrieval_chunk_window_seconds: float | None = typer.Option(
        None,
        "--retrieval-chunk-window-seconds",
    ),
    retrieval_chunk_overlap_seconds: float | None = typer.Option(
        None,
        "--retrieval-chunk-overlap-seconds",
    ),
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
        discovery_source=discovery_source,
        retrieval_chunk_window_seconds=retrieval_chunk_window_seconds,
        retrieval_chunk_overlap_seconds=retrieval_chunk_overlap_seconds,
    )

    LOGGER.info("processed=%d", summary["processed"])
    LOGGER.info("ingested=%d", summary["ingested"])
    LOGGER.info("skipped_existing=%d", summary["skipped_existing"])
    LOGGER.info("skipped_missing_turns=%d", summary["skipped_missing_turns"])
    LOGGER.info("failed=%d", summary["failed"])


if __name__ == "__main__":
    app()
