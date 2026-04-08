from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QueryRelation = Literal[
    "standalone_direct",
    "followup_previous",
    "meta_chat_scope",
]

RetrievalMode = Literal[
    "speaker_specific",
    "action_items_or_decisions",
    "broad_summary",
    "meta_or_confidence",
    "default_factoid",
]

ConfidenceTier = Literal[
    "grounded",
    "partial_limited_evidence",
    "insufficient_evidence",
]


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    used_fallback: bool
    fallback_reason: str | None = None
    question_relation: QueryRelation = "standalone_direct"
    was_lossy: bool = False
    used_cache: bool = False


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    meeting_id: str
    speaker_label: str
    start_time: float
    end_time: float
    content: str
    similarity: float
    chunk_key: str | None = None


@dataclass(frozen=True)
class RetrievalBundle:
    meeting_id: str
    user_query: str
    rewritten_query: str
    top_k_used: int
    results: list[RetrievedChunk]
    retrieval_mode: RetrievalMode = "default_factoid"
    question_relation: QueryRelation = "standalone_direct"
    used_cached_context: bool = False
    speaker_filter: str | None = None
    service_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundedAnswerResult:
    meeting_id: str
    question: str
    rewritten_query: str
    sections: dict[str, str]
    raw_answer: str
    insufficient_context: bool
    confidence_tier: ConfidenceTier = "grounded"
    service_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FormatDirectives:
    bullet_count: int | None = None
    use_table: bool = False
    short_summary: bool = False
    action_items_only: bool = False


@dataclass(frozen=True)
class ConversationTurnState:
    question: str
    rewritten_query: str
    retrieval_mode: RetrievalMode
    answer_summary: str
    insufficient_context: bool
    confidence_tier: ConfidenceTier = "grounded"
    evidence_count: int = 0
    uncertainty_notes: str = ""


@dataclass(frozen=True)
class ConversationState:
    latest_bundle: RetrievalBundle | None = None
    latest_answer: GroundedAnswerResult | None = None
    recent_turns: list[ConversationTurnState] | None = None
