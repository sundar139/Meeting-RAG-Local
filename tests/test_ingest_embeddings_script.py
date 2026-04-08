from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_pipeline.audio.retrieval_chunk_builder import RetrievalChunkWindow
from meeting_pipeline.embeddings.ollama_client import OllamaClientError
from scripts.ingest_embeddings import ingest_embeddings


def _write_turns(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class FakeEmbedder:
    def __init__(self, failures_before_success: int = 0) -> None:
        self.calls = 0
        self.failures_before_success = failures_before_success

    def embed_document(self, text: str) -> list[float]:
        _ = text
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise OllamaClientError("transient")
        return [0.1] * 768


class FakeRepository:
    def __init__(self, _connection: object) -> None:
        self.deleted = 0
        self.inserted = 0

    def delete_chunks_for_meeting(self, meeting_id: str) -> int:
        _ = meeting_id
        self.deleted += 3
        return 3

    def insert_transcript_chunks(self, chunks: list[object]) -> int:
        self.inserted += len(chunks)
        return len(chunks)


def test_ingest_embeddings_dry_run(tmp_path: Path, monkeypatch) -> None:
    turns_path = tmp_path / "turns.json"
    _write_turns(
        turns_path,
        {
            "meeting_id": "ES2002a",
            "turns": [
                {
                    "meeting_id": "ES2002a",
                    "speaker_label": "SPEAKER_00",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "text": "hello world",
                }
            ],
        },
    )

    monkeypatch.setattr("scripts.ingest_embeddings.Embedder", lambda: FakeEmbedder())

    summary = ingest_embeddings(
        meeting_id="ES2002a",
        turns_path=turns_path,
        dry_run=True,
    )

    assert summary["rows_inserted"] == 1
    assert summary["embeddings_succeeded"] == 1


def test_ingest_embeddings_rejects_meeting_id_mismatch(tmp_path: Path) -> None:
    turns_path = tmp_path / "turns.json"
    _write_turns(turns_path, {"meeting_id": "ES2002b", "turns": []})

    with pytest.raises(ValueError, match="mismatch"):
        ingest_embeddings(meeting_id="ES2002a", turns_path=turns_path, dry_run=True)


def test_ingest_embeddings_replace_existing(tmp_path: Path, monkeypatch) -> None:
    turns_path = tmp_path / "turns.json"
    _write_turns(
        turns_path,
        {
            "meeting_id": "ES2002a",
            "turns": [
                {
                    "meeting_id": "ES2002a",
                    "speaker_label": "SPEAKER_00",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "text": "hello world",
                }
            ],
        },
    )

    fake_repository = FakeRepository(object())
    monkeypatch.setattr("scripts.ingest_embeddings.Embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(
        "scripts.ingest_embeddings.connection_scope", lambda **_kwargs: _FakeConnCtx()
    )
    monkeypatch.setattr(
        "scripts.ingest_embeddings.TranscriptChunkRepository",
        lambda connection: fake_repository,
    )

    summary = ingest_embeddings(
        meeting_id="ES2002a",
        turns_path=turns_path,
        replace_existing=True,
        dry_run=False,
    )

    assert summary["rows_deleted"] == 3
    assert summary["rows_inserted"] == 1


def test_ingest_embeddings_retries_transient_embedding_failure(tmp_path: Path, monkeypatch) -> None:
    turns_path = tmp_path / "turns.json"
    _write_turns(
        turns_path,
        {
            "meeting_id": "ES2002a",
            "turns": [
                {
                    "meeting_id": "ES2002a",
                    "speaker_label": "SPEAKER_00",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "text": "hello world",
                }
            ],
        },
    )

    embedder = FakeEmbedder(failures_before_success=1)
    monkeypatch.setattr("scripts.ingest_embeddings.Embedder", lambda: embedder)
    monkeypatch.setattr("scripts.ingest_embeddings.time.sleep", lambda _seconds: None)

    summary = ingest_embeddings(
        meeting_id="ES2002a",
        turns_path=turns_path,
        dry_run=True,
    )

    assert embedder.calls == 2
    assert summary["embeddings_succeeded"] == 1


class _FakeConnCtx:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type
        _ = exc
        _ = tb
        return None


def test_ingest_embeddings_supports_window_override(tmp_path: Path, monkeypatch) -> None:
    turns_path = tmp_path / "turns.json"
    _write_turns(
        turns_path,
        {
            "meeting_id": "ES2002a",
            "turns": [
                {
                    "meeting_id": "ES2002a",
                    "speaker_label": "SPEAKER_00",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "text": "hello world",
                }
            ],
        },
    )

    captured: dict[str, float] = {}

    def fake_build_retrieval_chunks(*, meeting_id, turns, window_seconds, overlap_seconds):
        _ = meeting_id
        _ = turns
        captured["window"] = float(window_seconds)
        captured["overlap"] = float(overlap_seconds)
        return [
            RetrievalChunkWindow(
                chunk_key="abc123",
                meeting_id="ES2002a",
                speaker_label="SPEAKER_00",
                start_time=0.0,
                end_time=1.0,
                content="[SPEAKER_00 0.00-1.00] hello world",
                source_turn_count=1,
            )
        ]

    monkeypatch.setattr("scripts.ingest_embeddings.Embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(
        "scripts.ingest_embeddings.build_retrieval_chunks",
        fake_build_retrieval_chunks,
    )

    summary = ingest_embeddings(
        meeting_id="ES2002a",
        turns_path=turns_path,
        dry_run=True,
        retrieval_chunk_window_seconds=60.0,
        retrieval_chunk_overlap_seconds=20.0,
    )

    assert summary["rows_inserted"] == 1
    assert captured["window"] == 60.0
    assert captured["overlap"] == 20.0
