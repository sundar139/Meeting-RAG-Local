from __future__ import annotations

import math
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

EMBEDDING_DIMENSION = 768


@dataclass(frozen=True)
class TranscriptChunkInsert:
    meeting_id: str
    speaker_label: str
    start_time: float
    end_time: float
    content: str
    embedding: list[float] | None = None
    chunk_key: str | None = None


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_id: int
    meeting_id: str
    speaker_label: str
    start_time: float
    end_time: float
    content: str
    chunk_key: str | None = None


@dataclass(frozen=True)
class MeetingOverview:
    meeting_id: str
    chunk_count: int
    start_time_min: float | None
    end_time_max: float | None
    distinct_speaker_count: int


class CursorProtocol(Protocol):
    rowcount: int

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None: ...

    def executemany(self, query: str, params_seq: list[tuple[object, ...]]) -> None: ...

    def fetchone(self) -> tuple[object, ...] | None: ...

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

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _require_non_empty(value: str, *, field_name: str, max_length: int | None = None) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    if max_length is not None and len(normalized) > max_length:
        raise ValueError(f"{field_name} cannot exceed {max_length} characters")
    return normalized


def _validate_time_bounds(start_time: float, end_time: float) -> tuple[float, float]:
    start = float(start_time)
    end = float(end_time)

    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("start_time and end_time must be finite numbers")
    if end < start:
        raise ValueError("end_time must be greater than or equal to start_time")

    return start, end


def _validate_embedding(embedding: list[float] | None) -> list[float] | None:
    if embedding is None:
        return None
    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(f"embedding must contain exactly {EMBEDDING_DIMENSION} values")

    cleaned: list[float] = []
    for value in embedding:
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("embedding must contain only finite numbers")
        cleaned.append(normalized)

    return cleaned


def _validate_chunk_key(chunk_key: str | None) -> str | None:
    if chunk_key is None:
        return None

    normalized = chunk_key.strip()
    if not normalized:
        return None
    if len(normalized) > 80:
        raise ValueError("chunk_key cannot exceed 80 characters")
    return normalized


def _to_pgvector_literal(embedding: list[float] | None) -> str | None:
    if embedding is None:
        return None
    joined = ",".join(f"{value:.10g}" for value in embedding)
    return f"[{joined}]"


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


def _coerce_optional_float(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, field_name=field_name)


class TranscriptChunkRepository:
    _INSERT_ONE_SQL = (
        "INSERT INTO meeting_transcripts "
        "(meeting_id, speaker_label, start_time, end_time, content, embedding, chunk_key) "
        "VALUES (%s, %s, %s, %s, %s, %s::vector, %s) "
        "ON CONFLICT (meeting_id, chunk_key) DO UPDATE SET chunk_key = EXCLUDED.chunk_key "
        "RETURNING chunk_id"
    )
    _INSERT_BATCH_SQL = (
        "INSERT INTO meeting_transcripts "
        "(meeting_id, speaker_label, start_time, end_time, content, embedding, chunk_key) "
        "VALUES (%s, %s, %s, %s, %s, %s::vector, %s) "
        "ON CONFLICT (meeting_id, chunk_key) DO NOTHING"
    )

    def __init__(self, connection: ConnectionProtocol) -> None:
        self._connection = connection

    def insert_transcript_chunk(
        self,
        meeting_id: str,
        speaker_label: str,
        start_time: float,
        end_time: float,
        content: str,
        embedding: list[float] | None = None,
        chunk_key: str | None = None,
    ) -> int:
        validated_meeting_id = _require_non_empty(
            meeting_id, field_name="meeting_id", max_length=255
        )
        validated_speaker_label = _require_non_empty(
            speaker_label, field_name="speaker_label", max_length=50
        )
        validated_content = _require_non_empty(content, field_name="content")
        validated_start, validated_end = _validate_time_bounds(start_time, end_time)
        validated_embedding = _validate_embedding(embedding)
        validated_chunk_key = _validate_chunk_key(chunk_key)

        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    self._INSERT_ONE_SQL,
                    (
                        validated_meeting_id,
                        validated_speaker_label,
                        validated_start,
                        validated_end,
                        validated_content,
                        _to_pgvector_literal(validated_embedding),
                        validated_chunk_key,
                    ),
                )
                row = cursor.fetchone()

            self._connection.commit()
        except Exception as exc:
            self._connection.rollback()
            raise RuntimeError("Failed to insert transcript chunk") from exc

        if row is None or len(row) == 0:
            raise RuntimeError("Insert did not return a chunk_id")

        return _coerce_int(row[0], field_name="chunk_id")

    def list_meeting_ids(self, limit: int | None = None) -> list[str]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be a positive integer when provided")

        base_sql = "SELECT DISTINCT meeting_id FROM meeting_transcripts " "ORDER BY meeting_id ASC"

        if limit is None:
            sql = base_sql
            params: tuple[object, ...] = ()
        else:
            sql = f"{base_sql} LIMIT %s"
            params = (limit,)

        with self._connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        return [normalized for row in rows for normalized in [str(row[0]).strip()] if normalized]

    def get_distinct_speaker_labels(self, meeting_id: str) -> list[str]:
        validated_meeting_id = _require_non_empty(
            meeting_id,
            field_name="meeting_id",
            max_length=255,
        )

        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT speaker_label FROM meeting_transcripts "
                "WHERE meeting_id = %s ORDER BY speaker_label ASC",
                (validated_meeting_id,),
            )
            rows = cursor.fetchall()

        return [normalized for row in rows for normalized in [str(row[0]).strip()] if normalized]

    def get_meeting_overview(self, meeting_id: str) -> MeetingOverview:
        validated_meeting_id = _require_non_empty(
            meeting_id,
            field_name="meeting_id",
            max_length=255,
        )

        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*), MIN(start_time), MAX(end_time), "
                "COUNT(DISTINCT speaker_label) "
                "FROM meeting_transcripts WHERE meeting_id = %s",
                (validated_meeting_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return MeetingOverview(
                meeting_id=validated_meeting_id,
                chunk_count=0,
                start_time_min=None,
                end_time_max=None,
                distinct_speaker_count=0,
            )

        chunk_count = _coerce_int(row[0], field_name="chunk_count")
        start_time_min = _coerce_optional_float(row[1], field_name="start_time_min")
        end_time_max = _coerce_optional_float(row[2], field_name="end_time_max")
        distinct_speaker_count = _coerce_int(row[3], field_name="distinct_speaker_count")

        return MeetingOverview(
            meeting_id=validated_meeting_id,
            chunk_count=chunk_count,
            start_time_min=start_time_min,
            end_time_max=end_time_max,
            distinct_speaker_count=distinct_speaker_count,
        )

    def insert_transcript_chunks(self, chunks: list[TranscriptChunkInsert]) -> int:
        if not chunks:
            return 0

        values: list[tuple[object, ...]] = []
        for chunk in chunks:
            validated_meeting_id = _require_non_empty(
                chunk.meeting_id, field_name="meeting_id", max_length=255
            )
            validated_speaker_label = _require_non_empty(
                chunk.speaker_label, field_name="speaker_label", max_length=50
            )
            validated_content = _require_non_empty(chunk.content, field_name="content")
            validated_start, validated_end = _validate_time_bounds(chunk.start_time, chunk.end_time)
            validated_embedding = _validate_embedding(chunk.embedding)
            validated_chunk_key = _validate_chunk_key(chunk.chunk_key)

            values.append(
                (
                    validated_meeting_id,
                    validated_speaker_label,
                    validated_start,
                    validated_end,
                    validated_content,
                    _to_pgvector_literal(validated_embedding),
                    validated_chunk_key,
                )
            )

        try:
            with self._connection.cursor() as cursor:
                cursor.executemany(self._INSERT_BATCH_SQL, values)
                inserted = max(0, int(cursor.rowcount))
            self._connection.commit()
            return inserted
        except Exception as exc:
            self._connection.rollback()
            raise RuntimeError("Failed to insert transcript chunk batch") from exc

    def get_chunks_by_meeting(
        self, meeting_id: str, limit: int | None = None
    ) -> list[TranscriptChunk]:
        validated_meeting_id = _require_non_empty(
            meeting_id,
            field_name="meeting_id",
            max_length=255,
        )

        if limit is not None and limit <= 0:
            raise ValueError("limit must be a positive integer when provided")

        base_sql = (
            "SELECT chunk_id, meeting_id, speaker_label, start_time, end_time, content, chunk_key "
            "FROM meeting_transcripts WHERE meeting_id = %s "
            "ORDER BY start_time ASC, chunk_id ASC"
        )

        if limit is None:
            sql = base_sql
            params: tuple[object, ...] = (validated_meeting_id,)
        else:
            sql = f"{base_sql} LIMIT %s"
            params = (validated_meeting_id, limit)

        with self._connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        return [
            TranscriptChunk(
                chunk_id=_coerce_int(row[0], field_name="chunk_id"),
                meeting_id=str(row[1]),
                speaker_label=str(row[2]),
                start_time=_coerce_float(row[3], field_name="start_time"),
                end_time=_coerce_float(row[4], field_name="end_time"),
                content=str(row[5]),
                chunk_key=str(row[6]) if row[6] is not None else None,
            )
            for row in rows
        ]

    def delete_chunks_for_meeting(self, meeting_id: str) -> int:
        validated_meeting_id = _require_non_empty(
            meeting_id,
            field_name="meeting_id",
            max_length=255,
        )

        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM meeting_transcripts WHERE meeting_id = %s",
                    (validated_meeting_id,),
                )
                deleted = cursor.rowcount
            self._connection.commit()
        except Exception as exc:
            self._connection.rollback()
            raise RuntimeError("Failed to delete transcript chunks") from exc

        return deleted
