from __future__ import annotations

from dataclasses import dataclass

import pytest

from meeting_pipeline.db.repository import (
    TranscriptChunkInsert,
    TranscriptChunkRepository,
)


@dataclass
class RecordingCursor:
    executed: list[tuple[str, tuple[object, ...] | None]]
    executemany_calls: list[tuple[str, list[tuple[object, ...]]]]
    fetchone_result: tuple[object, ...] | None
    fetchall_result: list[tuple[object, ...]]
    rowcount: int
    delete_rowcount: int

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append((query, params))
        if query.startswith("DELETE"):
            self.rowcount = self.delete_rowcount

    def executemany(self, query: str, params_seq: list[tuple[object, ...]]) -> None:
        self.executemany_calls.append((query, params_seq))
        self.rowcount = len(params_seq)

    def fetchone(self) -> tuple[object, ...] | None:
        return self.fetchone_result

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.fetchall_result

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        return None


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> RecordingCursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _make_cursor() -> RecordingCursor:
    return RecordingCursor(
        executed=[],
        executemany_calls=[],
        fetchone_result=(101,),
        fetchall_result=[],
        rowcount=0,
        delete_rowcount=0,
    )


def test_insert_transcript_chunk_executes_parameterized_sql() -> None:
    cursor = _make_cursor()
    conn = RecordingConnection(cursor)
    repository = TranscriptChunkRepository(conn)

    chunk_id = repository.insert_transcript_chunk(
        meeting_id="meeting-123",
        speaker_label="spk_0",
        start_time=0.0,
        end_time=1.2,
        content="hello world",
    )

    assert chunk_id == 101
    assert conn.commits == 1
    assert len(cursor.executed) == 1
    query, params = cursor.executed[0]
    assert "INSERT INTO meeting_transcripts" in query
    assert params is not None
    assert params[0] == "meeting-123"
    assert params[1] == "spk_0"
    assert params[6] is None


def test_insert_transcript_chunk_rejects_bad_content() -> None:
    cursor = _make_cursor()
    repository = TranscriptChunkRepository(RecordingConnection(cursor))

    with pytest.raises(ValueError, match="content"):
        repository.insert_transcript_chunk(
            meeting_id="meeting-123",
            speaker_label="spk_0",
            start_time=0.0,
            end_time=1.0,
            content="   ",
        )


def test_insert_transcript_chunk_rejects_invalid_embedding_size() -> None:
    cursor = _make_cursor()
    repository = TranscriptChunkRepository(RecordingConnection(cursor))

    with pytest.raises(ValueError, match="768"):
        repository.insert_transcript_chunk(
            meeting_id="meeting-123",
            speaker_label="spk_0",
            start_time=0.0,
            end_time=1.0,
            content="ok",
            embedding=[0.1, 0.2],
        )


def test_insert_transcript_chunks_uses_batch_execution() -> None:
    cursor = _make_cursor()
    conn = RecordingConnection(cursor)
    repository = TranscriptChunkRepository(conn)

    count = repository.insert_transcript_chunks(
        [
            TranscriptChunkInsert(
                meeting_id="m1",
                speaker_label="spk_0",
                start_time=0.0,
                end_time=1.0,
                content="hello",
            ),
            TranscriptChunkInsert(
                meeting_id="m1",
                speaker_label="spk_1",
                start_time=1.1,
                end_time=2.0,
                content="world",
            ),
        ]
    )

    assert count == 2
    assert conn.commits == 1
    assert len(cursor.executemany_calls) == 1
    query, values = cursor.executemany_calls[0]
    assert "INSERT INTO meeting_transcripts" in query
    assert len(values) == 2


def test_get_chunks_by_meeting_maps_rows_and_limit() -> None:
    cursor = _make_cursor()
    cursor.fetchall_result = [
        (1, "m1", "spk_0", 0.0, 1.0, "hello", "chunk-a"),
        (2, "m1", "spk_1", 1.1, 2.0, "world", "chunk-b"),
    ]
    repository = TranscriptChunkRepository(RecordingConnection(cursor))

    chunks = repository.get_chunks_by_meeting("m1", limit=1)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == 1
    assert chunks[0].chunk_key == "chunk-a"
    _, params = cursor.executed[0]
    assert params == ("m1", 1)


def test_insert_transcript_chunk_accepts_chunk_key() -> None:
    cursor = _make_cursor()
    conn = RecordingConnection(cursor)
    repository = TranscriptChunkRepository(conn)

    repository.insert_transcript_chunk(
        meeting_id="meeting-123",
        speaker_label="spk_0",
        start_time=0.0,
        end_time=1.2,
        content="hello world",
        chunk_key="stable-key-001",
    )

    _, params = cursor.executed[0]
    assert params is not None
    assert params[6] == "stable-key-001"


def test_delete_chunks_for_meeting_returns_deleted_count() -> None:
    cursor = _make_cursor()
    cursor.delete_rowcount = 4
    conn = RecordingConnection(cursor)
    repository = TranscriptChunkRepository(conn)

    deleted = repository.delete_chunks_for_meeting("meeting-123")

    assert deleted == 4
    assert conn.commits == 1


def test_list_meeting_ids_returns_sorted_ids() -> None:
    cursor = _make_cursor()
    cursor.fetchall_result = [
        ("ES2002a",),
        ("ES2002b",),
    ]
    repository = TranscriptChunkRepository(RecordingConnection(cursor))

    meeting_ids = repository.list_meeting_ids()

    assert meeting_ids == ["ES2002a", "ES2002b"]
    query, params = cursor.executed[0]
    assert "SELECT DISTINCT meeting_id" in query
    assert params == ()


def test_get_distinct_speaker_labels_returns_sorted_labels() -> None:
    cursor = _make_cursor()
    cursor.fetchall_result = [
        ("SPEAKER_00",),
        ("SPEAKER_01",),
    ]
    repository = TranscriptChunkRepository(RecordingConnection(cursor))

    labels = repository.get_distinct_speaker_labels("ES2002a")

    assert labels == ["SPEAKER_00", "SPEAKER_01"]
    _, params = cursor.executed[0]
    assert params == ("ES2002a",)


def test_get_meeting_overview_maps_aggregate_row() -> None:
    cursor = _make_cursor()
    cursor.fetchone_result = (7, 10.0, 98.2, 3)
    repository = TranscriptChunkRepository(RecordingConnection(cursor))

    overview = repository.get_meeting_overview("ES2002a")

    assert overview.meeting_id == "ES2002a"
    assert overview.chunk_count == 7
    assert overview.start_time_min == 10.0
    assert overview.end_time_max == 98.2
    assert overview.distinct_speaker_count == 3
