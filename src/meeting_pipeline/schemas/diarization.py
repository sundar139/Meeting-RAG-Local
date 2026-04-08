from __future__ import annotations

import math

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _validate_time_order(start_time: float, end_time: float) -> tuple[float, float]:
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError("time values must be finite numbers")
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time")
    return start_time, end_time


class DiarizationSegment(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    speaker_label: str = Field(validation_alias=AliasChoices("speaker_label", "speaker"))
    start_time: float = Field(validation_alias=AliasChoices("start_time", "start"))
    end_time: float = Field(validation_alias=AliasChoices("end_time", "end"))

    @field_validator("speaker_label")
    @classmethod
    def _validate_speaker_label(cls, value: str) -> str:
        return _validate_non_empty(value, field_name="speaker_label")

    @model_validator(mode="after")
    def _validate_times(self) -> DiarizationSegment:
        _validate_time_order(self.start_time, self.end_time)
        return self

    @property
    def speaker(self) -> str:
        return self.speaker_label

    @property
    def start(self) -> float:
        return self.start_time

    @property
    def end(self) -> float:
        return self.end_time


class DiarizationDocument(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    meeting_id: str
    segments: list[DiarizationSegment]

    @field_validator("meeting_id")
    @classmethod
    def _validate_meeting_id(cls, value: str) -> str:
        return _validate_non_empty(value, field_name="meeting_id")
