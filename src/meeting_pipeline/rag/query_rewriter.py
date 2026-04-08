from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import replace

from meeting_pipeline.cache_utils import LruCache
from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.embeddings.ollama_client import OllamaClient, OllamaClientError
from meeting_pipeline.rag.models import QueryRelation, QueryRewriteResult

LOGGER = logging.getLogger(__name__)
DEFAULT_CHAT_MODEL = "llama3.2:3b-instruct"
_SPEAKER_TOKEN_PATTERN = re.compile(r"\bSPEAKER_\d+\b", re.IGNORECASE)
_YEAR_OR_NUMBER_PATTERN = re.compile(r"\b\d{2,4}\b")
_REWRITE_PREFIX_PATTERN = re.compile(
    r"^(?:final\s+answer|answer|rewritten\s+query|query)\s*[:\-]\s*",
    re.IGNORECASE,
)

_MIN_REWRITE_CHARS = 8
_MAX_REWRITE_CHARS = 280
_MAX_REWRITE_WORDS = 48


def _is_meta_confidence_question(normalized_question: str) -> bool:
    lower_question = normalized_question.lower()
    meta_phrases = (
        "your previous",
        "you said",
        "earlier answer",
        "confidence",
        "uncertain",
        "missing evidence",
        "what can you not",
        "what parts of your answer",
        "cannot be answered",
        "could not be answered",
        "low confidence",
        "missing from the evidence",
        "retrieved chunks",
        "supported by evidence",
        "unsupported",
        "which of these answers",
        "broader recent conversation",
        "conversation so far",
        "which of my questions",
        "which questions were low confidence",
        "what could not be answered confidently",
        "across these answers",
        "in this chat",
    )
    return any(phrase in lower_question for phrase in meta_phrases)


def _has_reasoning_markers(text: str) -> bool:
    lower_text = text.lower()
    markers = (
        "chain of thought",
        "let's think",
        "let us think",
        "step by step",
        "reasoning:",
        "analysis:",
        "thought process",
        "final answer:",
        "\\boxed{",
        "<think>",
        "</think>",
    )
    return any(marker in lower_text for marker in markers)


def _sanitize_rewrite_output(raw_text: str) -> str | None:
    text = raw_text.strip()
    if not text:
        return None

    if text.startswith("```"):
        stripped = text.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        text = stripped.strip()

    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed_match is not None:
        text = boxed_match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) > 1:
        text = lines[-1]
    else:
        text = lines[0]

    text = _REWRITE_PREFIX_PATTERN.sub("", text).strip()
    text = text.strip("`\"'")

    normalized = " ".join(text.split())
    return normalized or None


def _is_valid_rewrite_output(candidate: str) -> bool:
    if len(candidate) < _MIN_REWRITE_CHARS or len(candidate) > _MAX_REWRITE_CHARS:
        return False

    if len(candidate.split()) > _MAX_REWRITE_WORDS:
        return False

    lower_candidate = candidate.lower()
    if _has_reasoning_markers(candidate):
        return False

    if any(
        phrase in lower_candidate
        for phrase in (
            "this question asks",
            "the user asks",
            "to answer this",
            "i should",
            "we should",
            "here is",
        )
    ):
        return False

    if re.search(r"\b(?:step\s*\d+|first,|second,|third,)\b", lower_candidate):
        return False

    if candidate.count(". ") >= 4:
        return False

    return True


def _infer_question_relation(
    normalized_question: str,
    context_lines: Sequence[str],
) -> QueryRelation:
    if _is_meta_confidence_question(normalized_question):
        return "meta_chat_scope"

    lower_question = normalized_question.lower()

    if context_lines and re.search(
        r"\b(that|those|it|they|them|this|these|earlier)\b",
        lower_question,
    ):
        return "followup_previous"

    return "standalone_direct"


def _contains_format_instruction(text: str) -> bool:
    lower_text = text.lower()
    return any(
        phrase in lower_text
        for phrase in (
            "bullet",
            "table",
            "short summary",
            "concise",
            "just the",
            "only",
            "format as",
        )
    )


def _rewrite_is_lossy(original: str, rewritten: str) -> bool:
    if not rewritten:
        return True

    lower_rewritten = rewritten.lower()

    # Reject collapsed rewrites that drop too much detail for longer prompts.
    if len(original) > 48 and len(rewritten) < int(len(original) * 0.55):
        return True

    if _contains_format_instruction(original) and not _contains_format_instruction(rewritten):
        return True

    original_speakers = set(_SPEAKER_TOKEN_PATTERN.findall(original))
    rewritten_speakers = set(_SPEAKER_TOKEN_PATTERN.findall(rewritten))
    if original_speakers and not original_speakers.issubset(rewritten_speakers):
        return True

    original_numbers = set(_YEAR_OR_NUMBER_PATTERN.findall(original))
    rewritten_numbers = set(_YEAR_OR_NUMBER_PATTERN.findall(rewritten))
    if original_numbers and not original_numbers.issubset(rewritten_numbers):
        return True

    if any(
        generic in lower_rewritten
        for generic in (
            "summarize the conversation",
            "answer the question",
            "provide details",
        )
    ):
        return True

    # Keep direct quotes intact when present.
    quoted_fragments = re.findall(r'"([^"]{3,})"', original)
    if quoted_fragments and not any(
        fragment.lower() in lower_rewritten for fragment in quoted_fragments
    ):
        return True

    return False


def _rewrite_cache_key(
    normalized_question: str,
    context_lines: Sequence[str],
    question_relation: QueryRelation,
) -> tuple[str, tuple[str, ...], QueryRelation]:
    return (normalized_question, tuple(context_lines), question_relation)


def _is_fast_mode_rewrite_skip_candidate(
    normalized_question: str,
    question_relation: QueryRelation,
    context_lines: Sequence[str],
) -> bool:
    if question_relation != "standalone_direct":
        return False
    if context_lines:
        return False
    if len(normalized_question) > 180:
        return False
    if _contains_format_instruction(normalized_question):
        return False
    if _SPEAKER_TOKEN_PATTERN.search(normalized_question):
        return False
    return True


class QueryRewriter:
    def __init__(
        self,
        client: OllamaClient | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        runtime_settings = settings or get_settings()
        self._client = client or OllamaClient.from_settings(runtime_settings)
        configured_model = model_name or runtime_settings.ollama_chat_model
        normalized_model = str(configured_model).strip()
        self._model_name = normalized_model or DEFAULT_CHAT_MODEL
        self._cache_enabled = runtime_settings.enable_rag_caching
        self._fast_mode_skip_rewrite = runtime_settings.fast_mode_skip_query_rewrite
        self._rewrite_cache = LruCache[
            tuple[str, tuple[str, ...], QueryRelation], QueryRewriteResult
        ](runtime_settings.query_rewrite_cache_size)

    def rewrite(
        self,
        latest_user_question: str,
        conversation_context: Sequence[str] | None = None,
        *,
        use_cache: bool = True,
        fast_mode: bool = False,
    ) -> QueryRewriteResult:
        normalized_question = " ".join(latest_user_question.split())
        if not normalized_question:
            raise ValueError("latest_user_question must be a non-empty string")

        context_lines = [
            " ".join(item.split()) for item in conversation_context or [] if item.strip()
        ]
        question_relation = _infer_question_relation(normalized_question, context_lines)
        context_block = "\n".join(f"- {item}" for item in context_lines)

        cache_key = _rewrite_cache_key(normalized_question, context_lines, question_relation)
        should_use_cache = use_cache and self._cache_enabled
        if should_use_cache:
            cached = self._rewrite_cache.get(cache_key)
            if cached is not None:
                return replace(cached, used_cache=True)

        if (
            fast_mode
            and self._fast_mode_skip_rewrite
            and _is_fast_mode_rewrite_skip_candidate(
                normalized_question,
                question_relation,
                context_lines,
            )
        ):
            skip_result = QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_question,
                used_fallback=True,
                fallback_reason="fast_mode_skip",
                question_relation=question_relation,
                was_lossy=False,
                used_cache=False,
            )
            if should_use_cache:
                self._rewrite_cache.set(cache_key, skip_result)
            return skip_result

        if question_relation == "meta_chat_scope":
            meta_result = QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_question,
                used_fallback=True,
                fallback_reason="meta_question",
                question_relation=question_relation,
                was_lossy=False,
                used_cache=False,
            )
            if should_use_cache:
                self._rewrite_cache.set(cache_key, meta_result)
            return meta_result

        system_prompt = (
            "Rewrite the latest user question into a standalone retrieval query for a meeting "
            "transcript search system. Preserve every concrete constraint from the user: speaker "
            "labels, names, numbers, dates, requested output format, scope qualifiers, and action "
            "vs summary intent. Keep the query semantically equivalent and do not broaden or "
            "narrow scope. Do not answer the question. Do not invent facts. Return only "
            "rewritten query text."
        )

        user_prompt = (
            f"Latest question:\n{normalized_question}\n\n"
            f"Detected relation: {question_relation}\n\n"
            f"Recent conversation context (optional):\n{context_block or '- none'}"
        )

        LOGGER.info("Rewriting query context_messages=%d", len(context_lines))
        try:
            rewritten = self._client.chat(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if _has_reasoning_markers(rewritten):
                raise ValueError("rewrite contains reasoning markers")

            sanitized_rewrite = _sanitize_rewrite_output(rewritten)
            if sanitized_rewrite is None:
                raise ValueError("rewrite sanitization failed")

            if not _is_valid_rewrite_output(sanitized_rewrite):
                raise ValueError("rewrite validation failed")

            normalized_rewrite = sanitized_rewrite

            if _rewrite_is_lossy(normalized_question, normalized_rewrite):
                LOGGER.info("Rewrite deemed lossy; preserving original query.")
                lossy_result = QueryRewriteResult(
                    original_query=normalized_question,
                    rewritten_query=normalized_question,
                    used_fallback=True,
                    fallback_reason="lossy_rewrite",
                    question_relation=question_relation,
                    was_lossy=True,
                    used_cache=False,
                )
                if should_use_cache:
                    self._rewrite_cache.set(cache_key, lossy_result)
                return lossy_result

            result = QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_rewrite,
                used_fallback=False,
                fallback_reason=None,
                question_relation=question_relation,
                was_lossy=False,
                used_cache=False,
            )
            if should_use_cache:
                self._rewrite_cache.set(cache_key, result)
            return result
        except (OllamaClientError, ValueError) as exc:
            LOGGER.warning("Query rewriting failed; using fallback query. reason=%s", exc)
            fallback_reason = (
                "rewrite_generation_failed"
                if isinstance(exc, OllamaClientError)
                else "rewrite_output_rejected"
            )
            fallback_result = QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_question,
                used_fallback=True,
                fallback_reason=fallback_reason,
                question_relation=question_relation,
                was_lossy=False,
                used_cache=False,
            )
            if should_use_cache:
                self._rewrite_cache.set(cache_key, fallback_result)
            return fallback_result
