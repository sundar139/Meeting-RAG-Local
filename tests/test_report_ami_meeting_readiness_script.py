from __future__ import annotations

from pathlib import Path

from scripts.report_ami_meeting_readiness import build_readiness_report


def test_build_readiness_report_flags_asset_presence(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    meeting_id = "EN2002a"
    (raw_dir / f"{meeting_id}.A.words.xml").write_text("<xml/>", encoding="utf-8")
    (raw_dir / f"{meeting_id}.Mix-Headset.wav").write_text("wav", encoding="utf-8")
    (interim_dir / f"{meeting_id}_aligned.json").write_text("{}", encoding="utf-8")
    (processed_dir / f"{meeting_id}_turns.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.report_ami_meeting_readiness._db_ingested_count",
        lambda _meeting_id: 42,
    )

    report = build_readiness_report(
        meeting_ids=[meeting_id],
        raw_ami_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
    )

    assert len(report) == 1
    row = report[0]
    assert row["meeting_id"] == meeting_id
    assert row["words_xml_count"] == 1
    assert row["audio_exists"] is True
    assert row["aligned_exists"] is True
    assert row["diarization_exists"] is False
    assert row["turns_exists"] is True
    assert row["db_chunk_count"] == 42


def test_build_readiness_report_discovers_from_turns_when_words_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    meeting_id = "ES2004c"
    (processed_dir / f"{meeting_id}_turns.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.report_ami_meeting_readiness._db_ingested_count",
        lambda _meeting_id: 0,
    )

    report = build_readiness_report(
        meeting_ids=[],
        raw_ami_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
    )

    assert len(report) == 1
    assert report[0]["meeting_id"] == meeting_id
    assert report[0]["turns_exists"] is True
