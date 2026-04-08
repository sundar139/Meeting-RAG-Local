from __future__ import annotations

from pathlib import Path

from scripts.ingest_many_meetings import (
    _discover_meeting_ids_from_turns_json,
    _discover_meeting_ids_from_words_xml,
    build_ingestion_plan,
    discover_meeting_ids,
    ingest_many_meetings,
)


def test_discover_meeting_ids_from_words_xml(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "EN2002a.A.words.xml").write_text("<xml/>", encoding="utf-8")
    (raw_dir / "EN2002a.B.words.xml").write_text("<xml/>", encoding="utf-8")
    (raw_dir / "ES2004c.A.words.xml").write_text("<xml/>", encoding="utf-8")

    discovered = _discover_meeting_ids_from_words_xml(raw_dir)

    assert discovered == ["EN2002a", "ES2004c"]


def test_discover_meeting_ids_from_turns_json(tmp_path: Path) -> None:
    turns_dir = tmp_path / "processed"
    turns_dir.mkdir(parents=True)
    (turns_dir / "EN2002a_turns.json").write_text("{}", encoding="utf-8")
    (turns_dir / "ES2004c_turns.json").write_text("{}", encoding="utf-8")

    discovered = _discover_meeting_ids_from_turns_json(turns_dir)

    assert discovered == ["EN2002a", "ES2004c"]


def test_discover_meeting_ids_uses_words_and_turns_sources(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    turns_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    turns_dir.mkdir(parents=True)

    (raw_dir / "EN2002a.A.words.xml").write_text("<xml/>", encoding="utf-8")
    (turns_dir / "ES2004c_turns.json").write_text("{}", encoding="utf-8")

    discovered = discover_meeting_ids(
        raw_ami_dir=raw_dir,
        turns_dir=turns_dir,
        discovery_source="both",
    )

    assert discovered == ["EN2002a", "ES2004c"]


def test_ingest_many_meetings_skips_existing_and_missing_turns(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    turns_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    turns_dir.mkdir(parents=True)

    (turns_dir / "m1_turns.json").write_text('{"meeting_id":"m1","turns":[]}', encoding="utf-8")
    (turns_dir / "m2_turns.json").write_text('{"meeting_id":"m2","turns":[]}', encoding="utf-8")

    ingested_ids: list[str] = []

    def fake_ingest_embeddings(
        *,
        meeting_id: str,
        turns_path: Path,
        replace_existing: bool,
        batch_size: int,
        dry_run: bool,
    ) -> dict[str, int | str]:
        _ = turns_path
        _ = replace_existing
        _ = batch_size
        _ = dry_run
        ingested_ids.append(meeting_id)
        return {"rows_inserted": 1}

    monkeypatch.setattr("scripts.ingest_many_meetings.ingest_embeddings", fake_ingest_embeddings)
    monkeypatch.setattr(
        "scripts.ingest_many_meetings._meeting_has_existing_chunks",
        lambda meeting_id: meeting_id == "m1",
    )

    summary = ingest_many_meetings(
        meeting_ids=["m1", "m1", "m2", "m3"],
        raw_ami_dir=raw_dir,
        turns_dir=turns_dir,
        replace_existing=False,
        skip_existing=True,
        batch_size=16,
        dry_run=True,
    )

    assert summary["processed"] == 3
    assert summary["ingested"] == 1
    assert summary["skipped_existing"] == 1
    assert summary["skipped_missing_turns"] == 1
    assert summary["failed"] == 0
    assert ingested_ids == ["m2"]


def test_ingest_many_meetings_replace_existing_ignores_skip_existing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    turns_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    turns_dir.mkdir(parents=True)
    (turns_dir / "m1_turns.json").write_text('{"meeting_id":"m1","turns":[]}', encoding="utf-8")

    ingested_ids: list[str] = []

    def fake_ingest_embeddings(
        *,
        meeting_id: str,
        turns_path: Path,
        replace_existing: bool,
        batch_size: int,
        dry_run: bool,
    ) -> dict[str, int | str]:
        _ = turns_path
        _ = replace_existing
        _ = batch_size
        _ = dry_run
        ingested_ids.append(meeting_id)
        return {"rows_inserted": 1}

    monkeypatch.setattr("scripts.ingest_many_meetings.ingest_embeddings", fake_ingest_embeddings)
    monkeypatch.setattr(
        "scripts.ingest_many_meetings._meeting_has_existing_chunks",
        lambda _id: True,
    )

    summary = ingest_many_meetings(
        meeting_ids=["m1"],
        raw_ami_dir=raw_dir,
        turns_dir=turns_dir,
        replace_existing=True,
        skip_existing=True,
        batch_size=16,
        dry_run=True,
    )

    assert summary["ingested"] == 1
    assert summary["skipped_existing"] == 0
    assert ingested_ids == ["m1"]


def test_build_ingestion_plan_marks_turn_readiness(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    turns_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    turns_dir.mkdir(parents=True)

    (raw_dir / "EN2002a.A.words.xml").write_text("<xml/>", encoding="utf-8")
    (turns_dir / "EN2002a_turns.json").write_text("{}", encoding="utf-8")

    plan = build_ingestion_plan(
        meeting_ids=[],
        raw_ami_dir=raw_dir,
        turns_dir=turns_dir,
        discovery_source="both",
    )

    assert len(plan) == 1
    assert plan[0]["meeting_id"] == "EN2002a"
    assert plan[0]["ready_for_ingest"] is True
