from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Protocol

from meeting_pipeline.cache_utils import LruCache
from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.db.pgvector_search import SimilarChunkResult
from meeting_pipeline.embeddings.embedder import Embedder
from meeting_pipeline.rag.models import (
    ConversationState,
    QueryRewriteResult,
    RetrievalBundle,
    RetrievalMode,
    RetrievedChunk,
)
from meeting_pipeline.rag.query_rewriter import QueryRewriter
from meeting_pipeline.timing import elapsed_ms, now

LOGGER = logging.getLogger(__name__)
_SPEAKER_TOKEN_PATTERN = re.compile(r"\bSPEAKER_\d+\b", re.IGNORECASE)
_SPEAKER_BRACKET_PATTERN = re.compile(r"\[(SPEAKER_\d+)\s", re.IGNORECASE)

RetrievalCacheKey = tuple[str, str, int, RetrievalMode, str | None]


def _is_meta_confidence_question(text: str) -> bool:
    lower_text = text.lower()
    return any(
        phrase in lower_text
        for phrase in (
            "confidence",
            "uncertain",
            "missing evidence",
            "what can you not",
            "which part is unsupported",
            "cannot be answered",
            "could not be answered",
            "low confidence",
            "missing from the evidence",
            "retrieved chunks",
            "supported by evidence",
            "which of these answers",
            "conversation so far",
            "which of my questions",
            "which questions were low confidence",
            "what could not be answered confidently",
            "across these answers",
            "in this chat",
        )
    )


def _is_whole_meeting_scope(text: str) -> bool:
    return bool(
        re.search(
            r"\b(in|across|throughout)\s+(this|the)\s+meeting\b",
            text.lower(),
        )
    )


def _is_broad_summary_question(text: str) -> bool:
    lower_text = text.lower()
    return any(
        phrase in lower_text
        for phrase in (
            "summarize the whole meeting",
            "summary of the meeting",
            "high-level summary",
            "overall summary",
            "what did they discuss",
            "main themes",
            "broad summary",
            "main topics discussed",
            "main topics",
            "topics discussed",
            "concerns or risks",
            "risks were raised",
            "risks raised",
            "did the speakers disagree",
            "disagree on anything",
            "points of disagreement",
            "summarize the meeting",
            "meeting in 5 bullet points",
            "meeting in 5 bullets",
        )
    ) or _is_whole_meeting_scope(lower_text)


def _is_speaker_topic_comparison_question(text: str) -> bool:
    lower_text = text.lower()
    if "speaker" not in lower_text and "who" not in lower_text:
        return False
    return any(
        phrase in lower_text
        for phrase in (
            "which speaker talked the most",
            "which speaker contributed most",
            "who discussed",
            "who talked most",
            "which speaker discussed",
            "contributed most to",
        )
    )


def _token_set_for_similarity(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left.intersection(right))
    union = len(left.union(right))
    if union == 0:
        return 0.0
    return intersection / union


def _time_overlap_ratio(first: SimilarChunkResult, second: SimilarChunkResult) -> float:
    overlap = max(
        0.0,
        min(first.end_time, second.end_time) - max(first.start_time, second.start_time),
    )
    first_span = max(0.0, first.end_time - first.start_time)
    second_span = max(0.0, second.end_time - second.start_time)
    baseline = max(0.001, min(first_span, second_span))
    return overlap / baseline


def _extract_speaker_labels(result: SimilarChunkResult) -> set[str]:
    labels: set[str] = set()
    if result.speaker_label.strip():
        labels.add(result.speaker_label.strip().upper())
    labels.update(match.upper() for match in _SPEAKER_BRACKET_PATTERN.findall(result.content))
    return labels


def _infer_meta_scope(text: str) -> str:
    lower_text = text.lower()
    if any(
        phrase in lower_text
        for phrase in (
            "these answers",
            "recent answers",
            "across prior answers",
            "conversation so far",
            "broader recent conversation",
            "overall confidence",
            "across these answers",
            "in this chat",
            "which of my questions",
        )
    ):
        return "recent_conversation"
    return "latest_turn"


@dataclass(frozen=True)
class RetrievalPolicy:
    default_factoid_top_k: int
    speaker_specific_top_k: int
    action_items_or_decisions_top_k: int
    broad_summary_top_k: int
    meta_or_confidence_top_k: int
    broad_summary_max_candidates: int

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalPolicy:
        return cls(
            default_factoid_top_k=settings.default_factoid_top_k,
            speaker_specific_top_k=settings.speaker_specific_top_k,
            action_items_or_decisions_top_k=settings.action_items_or_decisions_top_k,
            broad_summary_top_k=settings.broad_summary_top_k,
            meta_or_confidence_top_k=settings.meta_or_confidence_top_k,
            broad_summary_max_candidates=max(
                settings.broad_summary_max_candidates,
                settings.broad_summary_top_k,
            ),
        )


class QueryRewriterProtocol(Protocol):
    def rewrite(
        self,
        latest_user_question: str,
        conversation_context: list[str] | None = None,
        *,
        use_cache: bool = True,
        fast_mode: bool = False,
    ) -> QueryRewriteResult: ...


class QueryEmbedderProtocol(Protocol):
    last_cache_hit: bool

    def embed_query(self, text: str, *, use_cache: bool = True) -> list[float]: ...


class VectorSearcherProtocol(Protocol):
    def search_similar_chunks(
        self,
        meeting_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        speaker_label: str | None = None,
    ) -> list[SimilarChunkResult]: ...


class Retriever:
    def __init__(
        self,
        searcher: VectorSearcherProtocol,
        *,
        query_rewriter: QueryRewriterProtocol | None = None,
        embedder: QueryEmbedderProtocol | None = None,
        policy: RetrievalPolicy | None = None,
        settings: Settings | None = None,
        retrieval_cache: LruCache[RetrievalCacheKey, RetrievalBundle] | None = None,
    ) -> None:
        runtime_settings = settings or get_settings()
        self._searcher = searcher
        self._query_rewriter = query_rewriter or QueryRewriter()
        self._embedder = embedder or Embedder()
        self._policy = policy or RetrievalPolicy.from_settings(runtime_settings)
        self._cache_enabled = runtime_settings.enable_rag_caching
        self._fast_mode_top_k_cap = runtime_settings.fast_mode_policy_top_k_cap
        self._retrieval_cache = retrieval_cache or LruCache[RetrievalCacheKey, RetrievalBundle](
            runtime_settings.retrieval_bundle_cache_size
        )

    def _search_similar_chunks(
        self,
        *,
        meeting_id: str,
        query_embedding: list[float],
        top_k: int,
        speaker_filter: str | None,
        use_db_speaker_filter: bool,
    ) -> tuple[list[SimilarChunkResult], bool]:
        if use_db_speaker_filter:
            try:
                return (
                    self._searcher.search_similar_chunks(
                        meeting_id=meeting_id,
                        query_embedding=query_embedding,
                        top_k=top_k,
                        speaker_label=speaker_filter,
                    ),
                    True,
                )
            except TypeError:
                # Backward compatibility for test doubles that do not accept speaker_label.
                pass

        return (
            self._searcher.search_similar_chunks(
                meeting_id=meeting_id,
                query_embedding=query_embedding,
                top_k=top_k,
            ),
            False,
        )

    def _embed_query(self, text: str, *, use_cache: bool) -> list[float]:
        try:
            return self._embedder.embed_query(text, use_cache=use_cache)
        except TypeError:
            # Backward compatibility for test doubles that do not accept use_cache.
            return self._embedder.embed_query(text)

    def _rewrite_query(
        self,
        *,
        latest_user_question: str,
        conversation_context: list[str] | None,
        use_cache: bool,
        fast_mode: bool,
    ) -> QueryRewriteResult:
        try:
            return self._query_rewriter.rewrite(
                latest_user_question=latest_user_question,
                conversation_context=conversation_context,
                use_cache=use_cache,
                fast_mode=fast_mode,
            )
        except TypeError:
            # Backward compatibility for test doubles that do not accept cache/fast-mode kwargs.
            return self._query_rewriter.rewrite(
                latest_user_question=latest_user_question,
                conversation_context=conversation_context,
            )

    def _retrieval_cache_key(
        self,
        *,
        meeting_id: str,
        rewritten_query: str,
        top_k: int,
        retrieval_mode: RetrievalMode,
        speaker_filter: str | None,
    ) -> RetrievalCacheKey:
        return (meeting_id, rewritten_query, top_k, retrieval_mode, speaker_filter)

    def _classify_retrieval_mode(
        self,
        user_query: str,
        rewrite_result: QueryRewriteResult,
    ) -> RetrievalMode:
        lower_query = user_query.lower()

        if rewrite_result.question_relation == "meta_chat_scope" or _is_meta_confidence_question(
            lower_query
        ):
            return "meta_or_confidence"

        if _SPEAKER_TOKEN_PATTERN.search(user_query) or "specific speaker" in lower_query:
            return "speaker_specific"

        if _is_speaker_topic_comparison_question(lower_query):
            return "broad_summary"

        if _is_broad_summary_question(lower_query):
            return "broad_summary"

        if any(
            phrase in lower_query
            for phrase in (
                "action item",
                "next step",
                "follow-up",
                "follow up",
                "owner",
                "deadline",
                "decision",
                "decided",
            )
        ):
            if _is_whole_meeting_scope(lower_query):
                return "broad_summary"
            return "action_items_or_decisions"

        return "default_factoid"

    def _build_service_metadata(
        self,
        *,
        timings_ms: dict[str, float],
        cache: dict[str, bool],
        rewrite_result: QueryRewriteResult,
        retrieval_mode: RetrievalMode,
        fast_mode: bool,
        meta_scope: str | None,
    ) -> dict[str, object]:
        return {
            "timings_ms": timings_ms,
            "cache": cache,
            "fast_mode": fast_mode,
            "rewrite": {
                "used_fallback": rewrite_result.used_fallback,
                "fallback_reason": rewrite_result.fallback_reason,
                "used_cache": rewrite_result.used_cache,
            },
            "routing": {
                "retrieval_mode": retrieval_mode,
                "question_relation": rewrite_result.question_relation,
                "meta_scope": meta_scope,
            },
        }

    def _extract_speaker_filter(self, text: str) -> str | None:
        match = _SPEAKER_TOKEN_PATTERN.search(text)
        return match.group(0).upper() if match else None

    def _top_k_for_mode(self, mode: RetrievalMode) -> int:
        if mode == "speaker_specific":
            return self._policy.speaker_specific_top_k
        if mode == "action_items_or_decisions":
            return self._policy.action_items_or_decisions_top_k
        if mode == "broad_summary":
            return self._policy.broad_summary_top_k
        if mode == "meta_or_confidence":
            return self._policy.meta_or_confidence_top_k
        return self._policy.default_factoid_top_k

    def _diversify_for_broad_summary(
        self,
        results: list[SimilarChunkResult],
        final_top_k: int,
    ) -> list[SimilarChunkResult]:
        if not results:
            return []

        deduped = self._dedupe_overlapping_results(results)
        if len(deduped) <= final_top_k:
            return deduped

        selected: list[SimilarChunkResult] = []
        selected_ids: set[int] = set()
        seen_speakers: set[str] = set()
        seen_windows: set[int] = set()

        # First pass: maximize temporal spread across 3-minute windows.
        for result in deduped:
            time_window = int(result.start_time // 180)
            if time_window in seen_windows:
                continue
            selected.append(result)
            selected_ids.add(result.chunk_id)
            seen_windows.add(time_window)
            seen_speakers.update(_extract_speaker_labels(result))
            if len(selected) >= final_top_k:
                return selected

        # Second pass: improve speaker diversity while preserving chronology spread.
        for result in deduped:
            if result.chunk_id in selected_ids:
                continue
            candidate_speakers = _extract_speaker_labels(result)
            if candidate_speakers and candidate_speakers.issubset(seen_speakers):
                continue
            selected.append(result)
            selected_ids.add(result.chunk_id)
            seen_speakers.update(candidate_speakers)
            if len(selected) >= final_top_k:
                return selected

        # Final pass: fill by retrieval ranking.
        for result in deduped:
            if result.chunk_id in selected_ids:
                continue
            selected.append(result)
            if len(selected) >= final_top_k:
                return selected

        return selected

    def _dedupe_overlapping_results(
        self,
        results: Iterable[SimilarChunkResult],
    ) -> list[SimilarChunkResult]:
        deduped: list[SimilarChunkResult] = []
        for candidate in results:
            if any(self._is_near_duplicate(candidate, existing) for existing in deduped):
                continue
            deduped.append(candidate)
        return deduped

    def _is_near_duplicate(
        self,
        first: SimilarChunkResult,
        second: SimilarChunkResult,
    ) -> bool:
        if first.chunk_key and second.chunk_key and first.chunk_key == second.chunk_key:
            return True

        first_content = " ".join(first.content.lower().split())
        second_content = " ".join(second.content.lower().split())
        if first_content == second_content:
            return True

        overlap_ratio = _time_overlap_ratio(first, second)
        if overlap_ratio < 0.7:
            return False

        similarity = _jaccard_similarity(
            _token_set_for_similarity(first_content),
            _token_set_for_similarity(second_content),
        )
        return similarity >= 0.72

    def retrieve(
        self,
        meeting_id: str,
        user_query: str,
        *,
        conversation_context: list[str] | None = None,
        top_k: int | None = None,
        conversation_state: ConversationState | None = None,
        use_cache: bool = True,
        fast_mode: bool = False,
    ) -> RetrievalBundle:
        request_started_at = now()
        normalized_meeting_id = meeting_id.strip()
        normalized_query = " ".join(user_query.split())

        if not normalized_meeting_id:
            raise ValueError("meeting_id must be a non-empty string")
        if not normalized_query:
            raise ValueError("user_query must be a non-empty string")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        rewrite_started_at = now()
        rewrite_result = self._rewrite_query(
            latest_user_question=normalized_query,
            conversation_context=conversation_context,
            use_cache=use_cache,
            fast_mode=fast_mode,
        )
        rewrite_elapsed_ms = elapsed_ms(rewrite_started_at)

        retrieval_mode = self._classify_retrieval_mode(normalized_query, rewrite_result)
        meta_scope = (
            _infer_meta_scope(normalized_query) if retrieval_mode == "meta_or_confidence" else None
        )
        speaker_filter = self._extract_speaker_filter(rewrite_result.rewritten_query)
        policy_top_k = self._top_k_for_mode(retrieval_mode)
        final_top_k = top_k or policy_top_k
        if fast_mode and top_k is None:
            final_top_k = min(final_top_k, self._fast_mode_top_k_cap)

        should_use_cache = use_cache and self._cache_enabled
        retrieval_cache_key = self._retrieval_cache_key(
            meeting_id=normalized_meeting_id,
            rewritten_query=rewrite_result.rewritten_query,
            top_k=final_top_k,
            retrieval_mode=retrieval_mode,
            speaker_filter=speaker_filter if retrieval_mode == "speaker_specific" else None,
        )

        if (
            retrieval_mode == "meta_or_confidence"
            and conversation_state is not None
            and conversation_state.latest_bundle is not None
            and conversation_state.latest_bundle.meeting_id == normalized_meeting_id
            and conversation_state.latest_bundle.results
        ):
            cached_chunks = conversation_state.latest_bundle.results[:final_top_k]
            LOGGER.info(
                "retrieval_cached_context meeting_id=%s mode=%s top_k=%d",
                normalized_meeting_id,
                retrieval_mode,
                final_top_k,
            )
            timings_ms = {
                "query_rewrite": rewrite_elapsed_ms,
                "query_embedding": 0.0,
                "embedding_query_prep": 0.0,
                "postgres_retrieval": 0.0,
                "retrieval": 0.0,
                "retrieval_total": elapsed_ms(request_started_at),
            }
            return RetrievalBundle(
                meeting_id=normalized_meeting_id,
                user_query=normalized_query,
                rewritten_query=rewrite_result.rewritten_query,
                top_k_used=min(final_top_k, len(cached_chunks)),
                results=cached_chunks,
                retrieval_mode=retrieval_mode,
                question_relation=rewrite_result.question_relation,
                used_cached_context=True,
                speaker_filter=speaker_filter,
                service_metadata=self._build_service_metadata(
                    timings_ms=timings_ms,
                    cache={
                        "query_rewrite": rewrite_result.used_cache,
                        "query_embedding": False,
                        "retrieval_bundle": False,
                    },
                    rewrite_result=rewrite_result,
                    retrieval_mode=retrieval_mode,
                    fast_mode=fast_mode,
                    meta_scope=meta_scope,
                ),
            )

        if (
            retrieval_mode == "meta_or_confidence"
            and conversation_state is not None
            and (conversation_state.latest_answer is not None or conversation_state.recent_turns)
        ):
            timings_ms = {
                "query_rewrite": rewrite_elapsed_ms,
                "query_embedding": 0.0,
                "embedding_query_prep": 0.0,
                "postgres_retrieval": 0.0,
                "retrieval": 0.0,
                "retrieval_total": elapsed_ms(request_started_at),
            }
            return RetrievalBundle(
                meeting_id=normalized_meeting_id,
                user_query=normalized_query,
                rewritten_query=rewrite_result.rewritten_query,
                top_k_used=0,
                results=[],
                retrieval_mode=retrieval_mode,
                question_relation=rewrite_result.question_relation,
                used_cached_context=True,
                speaker_filter=None,
                service_metadata=self._build_service_metadata(
                    timings_ms=timings_ms,
                    cache={
                        "query_rewrite": rewrite_result.used_cache,
                        "query_embedding": False,
                        "retrieval_bundle": False,
                    },
                    rewrite_result=rewrite_result,
                    retrieval_mode=retrieval_mode,
                    fast_mode=fast_mode,
                    meta_scope=meta_scope,
                ),
            )

        if should_use_cache:
            cached_bundle = self._retrieval_cache.get(retrieval_cache_key)
            if cached_bundle is not None:
                timings_ms = {
                    "query_rewrite": rewrite_elapsed_ms,
                    "query_embedding": 0.0,
                    "embedding_query_prep": 0.0,
                    "postgres_retrieval": 0.0,
                    "retrieval": 0.0,
                    "retrieval_total": elapsed_ms(request_started_at),
                }
                metadata = self._build_service_metadata(
                    timings_ms=timings_ms,
                    cache={
                        "query_rewrite": rewrite_result.used_cache,
                        "query_embedding": False,
                        "retrieval_bundle": True,
                    },
                    rewrite_result=rewrite_result,
                    retrieval_mode=retrieval_mode,
                    fast_mode=fast_mode,
                    meta_scope=meta_scope,
                )
                return replace(
                    cached_bundle,
                    user_query=normalized_query,
                    top_k_used=min(final_top_k, len(cached_bundle.results)),
                    service_metadata=metadata,
                )

        LOGGER.info(
            "retrieval_start meeting_id=%s mode=%s top_k=%d fallback_rewrite=%s",
            normalized_meeting_id,
            retrieval_mode,
            final_top_k,
            rewrite_result.used_fallback,
        )

        embed_started_at = now()
        query_embedding = self._embed_query(rewrite_result.rewritten_query, use_cache=use_cache)
        embed_elapsed_ms = elapsed_ms(embed_started_at)
        embed_cache_hit = bool(getattr(self._embedder, "last_cache_hit", False))

        search_top_k = final_top_k
        if retrieval_mode == "broad_summary":
            search_top_k = max(
                final_top_k * 4,
                self._policy.broad_summary_max_candidates,
            )

        retrieval_started_at = now()
        use_db_speaker_filter = retrieval_mode == "speaker_specific" and bool(speaker_filter)
        search_results, db_filter_applied = self._search_similar_chunks(
            meeting_id=normalized_meeting_id,
            query_embedding=query_embedding,
            top_k=search_top_k,
            speaker_filter=speaker_filter,
            use_db_speaker_filter=use_db_speaker_filter,
        )

        if retrieval_mode == "speaker_specific" and speaker_filter and not db_filter_applied:
            search_results = [
                result
                for result in search_results
                if speaker_filter in _extract_speaker_labels(result)
            ]

        if retrieval_mode == "broad_summary":
            search_results = self._diversify_for_broad_summary(search_results, final_top_k)
        elif len(search_results) > final_top_k:
            search_results = search_results[:final_top_k]

        retrieval_elapsed_ms = elapsed_ms(retrieval_started_at)

        chunks = [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                meeting_id=result.meeting_id,
                speaker_label=result.speaker_label,
                start_time=result.start_time,
                end_time=result.end_time,
                content=result.content,
                similarity=result.similarity,
                chunk_key=result.chunk_key,
            )
            for result in search_results
        ]

        timings_ms = {
            "query_rewrite": rewrite_elapsed_ms,
            "query_embedding": embed_elapsed_ms,
            "embedding_query_prep": embed_elapsed_ms,
            "postgres_retrieval": retrieval_elapsed_ms,
            "retrieval": retrieval_elapsed_ms,
            "retrieval_total": elapsed_ms(request_started_at),
        }

        bundle = RetrievalBundle(
            meeting_id=normalized_meeting_id,
            user_query=normalized_query,
            rewritten_query=rewrite_result.rewritten_query,
            top_k_used=final_top_k,
            results=chunks,
            retrieval_mode=retrieval_mode,
            question_relation=rewrite_result.question_relation,
            used_cached_context=False,
            speaker_filter=speaker_filter,
            service_metadata=self._build_service_metadata(
                timings_ms=timings_ms,
                cache={
                    "query_rewrite": rewrite_result.used_cache,
                    "query_embedding": embed_cache_hit,
                    "retrieval_bundle": False,
                },
                rewrite_result=rewrite_result,
                retrieval_mode=retrieval_mode,
                fast_mode=fast_mode,
                meta_scope=meta_scope,
            ),
        )

        if should_use_cache:
            self._retrieval_cache.set(retrieval_cache_key, replace(bundle, service_metadata={}))

        return bundle
