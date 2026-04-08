from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

EMBEDDING_DIMENSION = 768


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    meeting_id: str
    text: str
    score: float


@dataclass(frozen=True)
class SimilarChunkResult:
    chunk_id: int
    meeting_id: str
    speaker_label: str
    start_time: float
    end_time: float
    content: str
    similarity: float


class CursorProtocol(Protocol):
    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None: ...

    def fetchall(self) -> list[tuple[object, ...]]: ...

    def __enter__(self) -> CursorProtocol: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class ConnectionProtocol(Protocol):
    def cursor(self) -> CursorProtocol: ...


def _validate_meeting_id(meeting_id: str) -> str:
    normalized = meeting_id.strip()
    if not normalized:
        raise ValueError("meeting_id must be a non-empty string")
    if len(normalized) > 255:
        raise ValueError("meeting_id cannot exceed 255 characters")
    return normalized


def _validate_embedding(query_embedding: Sequence[float]) -> list[float]:
    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise ValueError(f"query_embedding must contain exactly {EMBEDDING_DIMENSION} values")

    validated: list[float] = []
    for value in query_embedding:
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("query_embedding must contain only finite numbers")
        validated.append(normalized)
    return validated


def _to_pgvector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.10g}" for value in embedding) + "]"


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"Unexpected boolean value for {field_name}")
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        return int(value)
    raise RuntimeError(f"Unexpected type for {field_name}: {type(value).__name__}")


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Unexpected boolean value for {field_name}")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise RuntimeError(f"Unexpected type for {field_name}: {type(value).__name__}")


class PgVectorSearcher:
    _SEARCH_SQL = (
        "SELECT chunk_id, meeting_id, speaker_label, start_time, end_time, content, "
        "(1 - (embedding <=> %s::vector)) AS similarity "
        "FROM meeting_transcripts "
        "WHERE meeting_id = %s AND embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s"
    )
    _SEARCH_SQL_WITH_SPEAKER = (
        "SELECT chunk_id, meeting_id, speaker_label, start_time, end_time, content, "
        "(1 - (embedding <=> %s::vector)) AS similarity "
        "FROM meeting_transcripts "
        "WHERE meeting_id = %s AND speaker_label = %s AND embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s"
    )

    def __init__(self, connection: ConnectionProtocol) -> None:
        self._connection = connection

    def search_similar_chunks(
        self,
        meeting_id: str,
        query_embedding: Sequence[float],
        top_k: int = 10,
        speaker_label: str | None = None,
    ) -> list[SimilarChunkResult]:
        validated_meeting_id = _validate_meeting_id(meeting_id)
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        normalized_speaker = speaker_label.strip() if isinstance(speaker_label, str) else None
        if normalized_speaker == "":
            normalized_speaker = None

        validated_embedding = _validate_embedding(query_embedding)
        vector_literal = _to_pgvector_literal(validated_embedding)

        with self._connection.cursor() as cursor:
            if normalized_speaker is None:
                cursor.execute(
                    self._SEARCH_SQL,
                    (vector_literal, validated_meeting_id, vector_literal, top_k),
                )
            else:
                cursor.execute(
                    self._SEARCH_SQL_WITH_SPEAKER,
                    (
                        vector_literal,
                        validated_meeting_id,
                        normalized_speaker,
                        vector_literal,
                        top_k,
                    ),
                )
            rows = cursor.fetchall()

        return [
            SimilarChunkResult(
                chunk_id=_coerce_int(row[0], field_name="chunk_id"),
                meeting_id=str(row[1]),
                speaker_label=str(row[2]),
                start_time=_coerce_float(row[3], field_name="start_time"),
                end_time=_coerce_float(row[4], field_name="end_time"),
                content=str(row[5]),
                similarity=_coerce_float(row[6], field_name="similarity"),
            )
            for row in rows
        ]
