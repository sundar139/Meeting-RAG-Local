from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _normalize_text(value: str, *, field_name: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _validate_time_order(start_time: float, end_time: float) -> tuple[float, float]:
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError("time values must be finite numbers")
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time")
    return start_time, end_time


class WordToken(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    speaker_id: str
    start_time: float
    end_time: float
    text: str

    @field_validator("speaker_id")
    @classmethod
    def _validate_speaker_id(cls, value: str) -> str:
        return _validate_non_empty(value, field_name="speaker_id")

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _normalize_text(value, field_name="text")

    @model_validator(mode="after")
    def _validate_times(self) -> WordToken:
        _validate_time_order(self.start_time, self.end_time)
        return self


class SpeakerTurn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    meeting_id: str
    speaker_label: str
    start_time: float
    end_time: float
    text: str

    @field_validator("meeting_id", "speaker_label")
    @classmethod
    def _validate_ids(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "value")
        return _validate_non_empty(value, field_name=field_name)

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _normalize_text(value, field_name="text")

    @model_validator(mode="after")
    def _validate_times(self) -> SpeakerTurn:
        _validate_time_order(self.start_time, self.end_time)
        return self


class AlignedTranscript(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    meeting_id: str
    words: list[WordToken]

    @field_validator("meeting_id")
    @classmethod
    def _validate_meeting_id(cls, value: str) -> str:
        return _validate_non_empty(value, field_name="meeting_id")


class WordSegment(BaseModel):
    """Compatibility schema retained for early scaffold modules."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str
    start: float
    end: float
    confidence: float | None = None

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _normalize_text(value, field_name="text")

    @model_validator(mode="after")
    def _validate_times(self) -> WordSegment:
        _validate_time_order(self.start, self.end)
        return self


class TranscriptSegment(BaseModel):
    """Compatibility schema retained for early scaffold modules."""

    model_config = ConfigDict(str_strip_whitespace=True)

    speaker: str | None = None
    text: str
    start: float
    end: float
    words: list[WordSegment] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _normalize_text(value, field_name="text")

    @field_validator("speaker")
    @classmethod
    def _normalize_speaker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_empty(value, field_name="speaker")

    @model_validator(mode="after")
    def _validate_times(self) -> TranscriptSegment:
        _validate_time_order(self.start, self.end)
        return self


class TranscriptDocument(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    meeting_id: str
    segments: list[TranscriptSegment]

    @field_validator("meeting_id")
    @classmethod
    def _validate_meeting_id(cls, value: str) -> str:
        return _validate_non_empty(value, field_name="meeting_id")
