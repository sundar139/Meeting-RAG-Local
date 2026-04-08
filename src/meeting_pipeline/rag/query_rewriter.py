from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.embeddings.ollama_client import OllamaClient, OllamaClientError
from meeting_pipeline.rag.models import QueryRelation, QueryRewriteResult

LOGGER = logging.getLogger(__name__)
DEFAULT_CHAT_MODEL = "llama3.2:3b-instruct"
_SPEAKER_TOKEN_PATTERN = re.compile(r"\bSPEAKER_\d+\b", re.IGNORECASE)
_YEAR_OR_NUMBER_PATTERN = re.compile(r"\b\d{2,4}\b")


def _infer_question_relation(
    normalized_question: str,
    context_lines: Sequence[str],
) -> QueryRelation:
    lower_question = normalized_question.lower()
    if any(
        phrase in lower_question
        for phrase in (
            "your previous",
            "you said",
            "earlier answer",
            "confidence",
            "uncertain",
            "missing evidence",
            "what can you not",
            "what parts of your answer",
        )
    ):
        return "meta_chat_scope"

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

    lower_original = original.lower()
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
    if quoted_fragments and not any(fragment.lower() in lower_rewritten for fragment in quoted_fragments):
        return True

    return False


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

    def rewrite(
        self,
        latest_user_question: str,
        conversation_context: Sequence[str] | None = None,
    ) -> QueryRewriteResult:
        normalized_question = " ".join(latest_user_question.split())
        if not normalized_question:
            raise ValueError("latest_user_question must be a non-empty string")

        context_lines = [
            " ".join(item.split()) for item in conversation_context or [] if item.strip()
        ]
        question_relation = _infer_question_relation(normalized_question, context_lines)
        context_block = "\n".join(f"- {item}" for item in context_lines)

        if question_relation == "meta_chat_scope":
            return QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_question,
                used_fallback=True,
                question_relation=question_relation,
                was_lossy=False,
            )

        system_prompt = (
            "Rewrite the latest user question into a standalone retrieval query for a meeting "
            "transcript search system. Preserve every concrete constraint from the user: speaker "
            "labels, names, numbers, dates, requested output format, scope qualifiers, and action "
            "vs summary intent. Keep the query semantically equivalent and do not broaden or narrow "
            "scope. Do not answer the question. Do not invent facts. Return only rewritten query text."
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
            normalized_rewrite = " ".join(rewritten.split())
            if not normalized_rewrite:
                raise ValueError("empty rewrite")

            if _rewrite_is_lossy(normalized_question, normalized_rewrite):
                LOGGER.info("Rewrite deemed lossy; preserving original query.")
                return QueryRewriteResult(
                    original_query=normalized_question,
                    rewritten_query=normalized_question,
                    used_fallback=True,
                    question_relation=question_relation,
                    was_lossy=True,
                )

            return QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_rewrite,
                used_fallback=False,
                question_relation=question_relation,
                was_lossy=False,
            )
        except (OllamaClientError, ValueError) as exc:
            LOGGER.warning("Query rewriting failed; using fallback query. reason=%s", exc)
            return QueryRewriteResult(
                original_query=normalized_question,
                rewritten_query=normalized_question,
                used_fallback=True,
                question_relation=question_relation,
                was_lossy=False,
            )
