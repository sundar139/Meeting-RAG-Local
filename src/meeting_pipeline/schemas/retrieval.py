from __future__ import annotations

from pydantic import BaseModel


class RetrievalChunk(BaseModel):
    chunk_id: str
    meeting_id: str
    text: str


class RetrievalResult(BaseModel):
    chunk: RetrievalChunk
    score: float
