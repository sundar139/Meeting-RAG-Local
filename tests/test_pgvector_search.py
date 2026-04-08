from __future__ import annotations

from dataclasses import dataclass

import pytest

from meeting_pipeline.db.pgvector_search import PgVectorSearcher


@dataclass
class RecordingCursor:
    executed: list[tuple[str, tuple[object, ...] | None]]
    fetchall_result: list[tuple[object, ...]]

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append((query, params))

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

    def cursor(self) -> RecordingCursor:
        return self._cursor


def test_search_similar_chunks_rejects_invalid_dimension() -> None:
    cursor = RecordingCursor(executed=[], fetchall_result=[])
    searcher = PgVectorSearcher(RecordingConnection(cursor))

    with pytest.raises(ValueError, match="768"):
        searcher.search_similar_chunks(
            meeting_id="m1",
            query_embedding=[0.1, 0.2],
        )


def test_search_similar_chunks_executes_expected_query_and_maps_results() -> None:
    cursor = RecordingCursor(
        executed=[],
        fetchall_result=[
            (1, "m1", "spk_0", 0.0, 1.0, "hello", "chunk-a", 0.95),
        ],
    )
    searcher = PgVectorSearcher(RecordingConnection(cursor))

    query_embedding = [0.1] * 768
    results = searcher.search_similar_chunks("m1", query_embedding, top_k=3)

    assert len(results) == 1
    assert results[0].chunk_id == 1
    assert results[0].chunk_key == "chunk-a"
    assert results[0].similarity == 0.95

    assert len(cursor.executed) == 1
    query, params = cursor.executed[0]
    assert "ORDER BY embedding <=> %s::vector" in query
    assert params is not None
    assert params[1] == "m1"
    assert params[3] == 3
    assert isinstance(params[0], str)
    assert params[0].startswith("[")
    assert params[0].endswith("]")


def test_search_similar_chunks_can_apply_db_side_speaker_filter() -> None:
    cursor = RecordingCursor(
        executed=[],
        fetchall_result=[
            (1, "m1", "SPEAKER_00", 0.0, 1.0, "hello", "chunk-b", 0.95),
        ],
    )
    searcher = PgVectorSearcher(RecordingConnection(cursor))

    results = searcher.search_similar_chunks(
        meeting_id="m1",
        query_embedding=[0.2] * 768,
        top_k=2,
        speaker_label="SPEAKER_00",
    )

    assert len(results) == 1
    assert len(cursor.executed) == 1
    query, params = cursor.executed[0]
    assert "speaker_label = %s" in query
    assert params is not None
    assert params[2] == "SPEAKER_00"
