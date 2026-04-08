from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha1

from meeting_pipeline.cache_utils import LruCache
from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.embeddings.ollama_client import OllamaClient, OllamaClientError
from meeting_pipeline.rag.models import (
    ConversationState,
    FormatDirectives,
    GroundedAnswerResult,
    RetrievalMode,
    RetrievedChunk,
)
from meeting_pipeline.timing import elapsed_ms, now

LOGGER = logging.getLogger(__name__)
DEFAULT_CHAT_MODEL = "llama3.2:3b-instruct"
SECTION_ORDER = [
    "Summary",
    "Key Points",
    "Decisions",
    "Action Items",
    "Uncertainties / Missing Evidence",
]

AnswerCacheKey = tuple[str, str, RetrievalMode, tuple[object, ...], bool]


@dataclass(frozen=True)
class EvidenceCompactionPolicy:
    max_chunks: int
    max_total_chars: int
    max_chunk_chars: int


def _format_directives_signature(directives: FormatDirectives) -> tuple[object, ...]:
    return (
        directives.bullet_count,
        directives.use_table,
        directives.short_summary,
        directives.action_items_only,
    )


def _evidence_signature(evidence: Sequence[RetrievedChunk]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            chunk.chunk_id,
            chunk.speaker_label,
            round(chunk.start_time, 3),
            round(chunk.end_time, 3),
            round(chunk.similarity, 6),
            sha1(chunk.content.encode("utf-8")).hexdigest()[:12],
        )
        for chunk in evidence
    )


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3].rstrip() + "..."


def _compact_evidence_for_prompt(
    evidence: Sequence[RetrievedChunk],
    policy: EvidenceCompactionPolicy,
) -> tuple[list[RetrievedChunk], dict[str, int]]:
    ranked = sorted(
        evidence,
        key=lambda chunk: (-chunk.similarity, chunk.start_time, chunk.chunk_id),
    )

    selected: list[RetrievedChunk] = []
    total_chars = 0
    truncated_chunks = 0

    for chunk in ranked:
        if len(selected) >= policy.max_chunks:
            break
        if total_chars >= policy.max_total_chars:
            break

        remaining = policy.max_total_chars - total_chars
        chunk_limit = min(policy.max_chunk_chars, remaining)
        if chunk_limit < 80:
            break

        compact_content = _truncate_text(chunk.content, chunk_limit)
        if compact_content != " ".join(chunk.content.split()):
            truncated_chunks += 1

        selected.append(replace(chunk, content=compact_content))
        total_chars += len(compact_content)

    metadata = {
        "original_chunks": len(evidence),
        "selected_chunks": len(selected),
        "dropped_chunks": max(0, len(evidence) - len(selected)),
        "truncated_chunks": truncated_chunks,
        "total_evidence_chars": total_chars,
    }
    return selected, metadata


class AnswerGenerator:
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
        self._answer_cache = LruCache[AnswerCacheKey, GroundedAnswerResult](
            runtime_settings.answer_cache_size
        )
        self._default_compaction_policy = EvidenceCompactionPolicy(
            max_chunks=runtime_settings.answer_max_evidence_chunks,
            max_total_chars=runtime_settings.answer_max_evidence_chars,
            max_chunk_chars=runtime_settings.answer_max_chunk_chars,
        )
        self._fast_compaction_policy = EvidenceCompactionPolicy(
            max_chunks=runtime_settings.fast_mode_answer_max_evidence_chunks,
            max_total_chars=runtime_settings.fast_mode_answer_max_evidence_chars,
            max_chunk_chars=runtime_settings.fast_mode_answer_max_chunk_chars,
        )

    def generate(
        self,
        *,
        user_question: str,
        meeting_id: str,
        retrieved_evidence: Sequence[RetrievedChunk],
        rewritten_query: str,
        conversation_context: Sequence[str] | None = None,
        retrieval_mode: RetrievalMode = "default_factoid",
        recent_state: ConversationState | None = None,
        use_cache: bool = True,
        fast_mode: bool = False,
    ) -> GroundedAnswerResult:
        generation_started_at = now()
        normalized_question = " ".join(user_question.split())
        normalized_meeting_id = meeting_id.strip()
        normalized_rewrite = " ".join(rewritten_query.split())

        if not normalized_question:
            raise ValueError("user_question must be a non-empty string")
        if not normalized_meeting_id:
            raise ValueError("meeting_id must be a non-empty string")
        if not normalized_rewrite:
            raise ValueError("rewritten_query must be a non-empty string")

        directives = _extract_format_directives(normalized_question)
        if fast_mode and not directives.short_summary:
            directives = replace(directives, short_summary=True)

        compaction_policy = (
            self._fast_compaction_policy if fast_mode else self._default_compaction_policy
        )
        compacted_evidence, compaction_metadata = _compact_evidence_for_prompt(
            retrieved_evidence,
            compaction_policy,
        )
        should_use_cache = use_cache and self._cache_enabled
        cache_key: AnswerCacheKey = (
            normalized_meeting_id,
            normalized_rewrite,
            retrieval_mode,
            _format_directives_signature(directives) + (_evidence_signature(compacted_evidence),),
            fast_mode,
        )

        if should_use_cache:
            cached = self._answer_cache.get(cache_key)
            if cached is not None:
                return _with_answer_timing(
                    replace(cached, question=normalized_question),
                    generation_started_at,
                    cache_metadata={
                        "answer_generation": True,
                    },
                    compaction_metadata=compaction_metadata,
                    fast_mode=fast_mode,
                )

        if not compacted_evidence:
            if retrieval_mode == "meta_or_confidence" and recent_state is not None:
                return _with_answer_timing(
                    _build_meta_answer_from_state(
                        meeting_id=normalized_meeting_id,
                        question=normalized_question,
                        rewritten_query=normalized_rewrite,
                        recent_state=recent_state,
                    ),
                    generation_started_at,
                    cache_metadata={
                        "answer_generation": False,
                    },
                    compaction_metadata=compaction_metadata,
                    fast_mode=fast_mode,
                )

            sections = _build_insufficient_sections(retrieval_mode)
            sections = _apply_format_directives(sections, directives)
            raw_answer = "\n".join(f"{key}: {value}" for key, value in sections.items())
            return _with_answer_timing(
                GroundedAnswerResult(
                    meeting_id=normalized_meeting_id,
                    question=normalized_question,
                    rewritten_query=normalized_rewrite,
                    sections=sections,
                    raw_answer=raw_answer,
                    insufficient_context=True,
                ),
                generation_started_at,
                cache_metadata={
                    "answer_generation": False,
                },
                compaction_metadata=compaction_metadata,
                fast_mode=fast_mode,
            )

        context_lines = [
            f"[chunk_id:{chunk.chunk_id} speaker:{chunk.speaker_label} "
            f"{chunk.start_time:.2f}-{chunk.end_time:.2f}] {chunk.content}"
            for chunk in compacted_evidence
        ]
        context_block = "\n".join(context_lines)

        conversation_lines = [
            " ".join(item.split()) for item in (conversation_context or []) if item.strip()
        ]
        conversation_block = "\n".join(f"- {line}" for line in conversation_lines)
        recent_state_block = _render_recent_state_block(recent_state)
        format_instruction_block = _render_format_instruction_block(directives)

        mode_instruction = {
            "speaker_specific": "Prioritize statements from the requested speaker.",
            "action_items_or_decisions": "Prioritize owners, decisions, and follow-up commitments.",
            "broad_summary": "Synthesize across speakers and across the full meeting timeline.",
            "meta_or_confidence": "Audit confidence and explicitly flag unsupported claims.",
            "default_factoid": "Answer directly with concise evidence-grounded details.",
        }[retrieval_mode]

        system_prompt = (
            "You are a meeting QA assistant. Use only the provided retrieved evidence. "
            "Do not invent facts. If evidence is insufficient, explicitly say so. "
            f"Routing mode: {retrieval_mode}. {mode_instruction} "
            f"Formatting guidance: {format_instruction_block} "
            "Respond in strict JSON with keys: Summary, Key Points, Decisions, Action Items, "
            "Uncertainties / Missing Evidence. Include local citations like "
            "[chunk_id:12 speaker:SPEAKER_01 32.4-48.2] in statements where evidence "
            "supports claims."
        )
        user_prompt = (
            f"Meeting ID: {normalized_meeting_id}\n"
            f"Original question: {normalized_question}\n"
            f"Rewritten query: {normalized_rewrite}\n"
            f"Recent answer state (optional):\n{recent_state_block}\n\n"
            f"Recent conversation context:\n{conversation_block or '- none'}\n\n"
            f"Retrieved evidence:\n{context_block}"
        )

        LOGGER.info("Generating grounded answer evidence_count=%d", len(compacted_evidence))
        try:
            raw_answer = self._client.chat(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OllamaClientError as exc:
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

        sections = _parse_structured_sections(raw_answer)
        sections = _apply_format_directives(sections, directives)
        insufficient = _is_insufficient_answer(sections)

        result = GroundedAnswerResult(
            meeting_id=normalized_meeting_id,
            question=normalized_question,
            rewritten_query=normalized_rewrite,
            sections=sections,
            raw_answer=raw_answer,
            insufficient_context=insufficient,
        )
        if should_use_cache:
            self._answer_cache.set(cache_key, replace(result, service_metadata={}))

        return _with_answer_timing(
            result,
            generation_started_at,
            cache_metadata={
                "answer_generation": False,
            },
            compaction_metadata=compaction_metadata,
            fast_mode=fast_mode,
        )


def _parse_structured_sections(raw_answer: str) -> dict[str, str]:
    text = raw_answer.strip()
    payload: object

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {
            "Summary": text or "No answer content returned.",
            "Key Points": "No structured key points parsed.",
            "Decisions": "No structured decisions parsed.",
            "Action Items": "No structured action items parsed.",
            "Uncertainties / Missing Evidence": "Response was not structured JSON.",
        }

    if not isinstance(payload, dict):
        return {
            "Summary": "No structured summary returned.",
            "Key Points": "No structured key points returned.",
            "Decisions": "No structured decisions returned.",
            "Action Items": "No structured action items returned.",
            "Uncertainties / Missing Evidence": "Model response JSON was not an object.",
        }

    sections: dict[str, str] = {}
    for key in SECTION_ORDER:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            sections[key] = value.strip()
        else:
            sections[key] = "Insufficient evidence for this section."

    return sections


def _is_insufficient_answer(sections: dict[str, str]) -> bool:
    summary = sections.get("Summary", "").lower()
    missing = sections.get("Uncertainties / Missing Evidence", "").lower()
    return "insufficient" in summary or "insufficient" in missing or "no evidence" in missing


def _build_insufficient_sections(retrieval_mode: RetrievalMode) -> dict[str, str]:
    return {
        "Summary": "Retrieved evidence is insufficient to answer this question reliably.",
        "Key Points": "No supporting evidence was retrieved from transcript chunks.",
        "Decisions": "No decision evidence was found in retrieved context.",
        "Action Items": "No action-item evidence was found in retrieved context.",
        "Uncertainties / Missing Evidence": (
            "Try broadening the question, adding context, or increasing retrieval top-k. "
            f"Current retrieval mode: {retrieval_mode}."
        ),
    }


def _extract_format_directives(question: str) -> FormatDirectives:
    lower_question = question.lower()
    bullet_count_match = re.search(r"\b(\d{1,2})\s+bullet", lower_question)
    bullet_count = int(bullet_count_match.group(1)) if bullet_count_match else None

    return FormatDirectives(
        bullet_count=bullet_count,
        use_table="table" in lower_question,
        short_summary=any(
            phrase in lower_question
            for phrase in ("short summary", "brief summary", "concise summary", "in 2 sentences")
        ),
        action_items_only=any(
            phrase in lower_question
            for phrase in (
                "action items only",
                "only action items",
                "just action items",
            )
        ),
    )


def _render_format_instruction_block(directives: FormatDirectives) -> str:
    instructions: list[str] = ["Keep defaults when no explicit format request exists."]
    if directives.bullet_count is not None:
        instructions.append(f"Use exactly {directives.bullet_count} bullets where applicable.")
    if directives.use_table:
        instructions.append("Provide key points as a markdown table.")
    if directives.short_summary:
        instructions.append("Keep summary to at most two concise sentences.")
    if directives.action_items_only:
        instructions.append("Prioritize Action Items content over other sections.")
    return " ".join(instructions)


def _render_recent_state_block(recent_state: ConversationState | None) -> str:
    if recent_state is None or recent_state.latest_answer is None:
        return "- none"

    latest_answer = recent_state.latest_answer
    latest_bundle = recent_state.latest_bundle
    cached_count = len(latest_bundle.results) if latest_bundle is not None else 0
    summary = latest_answer.sections.get("Summary", "")
    uncertainty = latest_answer.sections.get("Uncertainties / Missing Evidence", "")
    return (
        f"- Prior answer summary: {summary}\n"
        f"- Prior uncertainty notes: {uncertainty}\n"
        f"- Prior retrieval chunks: {cached_count}"
    )


def _build_meta_answer_from_state(
    *,
    meeting_id: str,
    question: str,
    rewritten_query: str,
    recent_state: ConversationState,
) -> GroundedAnswerResult:
    latest_answer = recent_state.latest_answer
    latest_bundle = recent_state.latest_bundle
    if latest_answer is None:
        sections = _build_insufficient_sections("meta_or_confidence")
        raw_answer = "\n".join(f"{key}: {value}" for key, value in sections.items())
        return GroundedAnswerResult(
            meeting_id=meeting_id,
            question=question,
            rewritten_query=rewritten_query,
            sections=sections,
            raw_answer=raw_answer,
            insufficient_context=True,
        )

    evidence_preview = "No cached evidence was available from the previous turn."
    if latest_bundle is not None and latest_bundle.results:
        preview_chunks = latest_bundle.results[:2]
        evidence_preview = " ".join(
            (
                f"[chunk_id:{chunk.chunk_id} speaker:{chunk.speaker_label} "
                f"{chunk.start_time:.2f}-{chunk.end_time:.2f}]"
            )
            for chunk in preview_chunks
        )

    uncertainty_text = latest_answer.sections.get(
        "Uncertainties / Missing Evidence",
        "Uncertainty details were not captured in the prior turn.",
    )
    sections = {
        "Summary": (
            "Confidence review based on the most recent answer context. "
            "No new retrieval evidence was added for this meta question."
        ),
        "Key Points": (
            f"Prior answer summary: {latest_answer.sections.get('Summary', 'Not available.')}"
        ),
        "Decisions": latest_answer.sections.get(
            "Decisions", "No prior decision details available for confidence review."
        ),
        "Action Items": latest_answer.sections.get(
            "Action Items", "No prior action-item details available for confidence review."
        ),
        "Uncertainties / Missing Evidence": (
            f"Prior uncertainty notes: {uncertainty_text} Evidence preview: {evidence_preview}"
        ),
    }
    raw_answer = "\n".join(f"{key}: {value}" for key, value in sections.items())
    return GroundedAnswerResult(
        meeting_id=meeting_id,
        question=question,
        rewritten_query=rewritten_query,
        sections=sections,
        raw_answer=raw_answer,
        insufficient_context=latest_answer.insufficient_context,
    )


def _apply_format_directives(
    sections: dict[str, str],
    directives: FormatDirectives,
) -> dict[str, str]:
    updated = dict(sections)

    if directives.action_items_only:
        updated["Summary"] = "User requested action items only."
        updated["Key Points"] = "User requested action items only."
        updated["Decisions"] = "User requested action items only."

    if directives.short_summary:
        sentences = re.split(r"(?<=[.!?])\s+", updated.get("Summary", "").strip())
        updated["Summary"] = " ".join(sentences[:2]).strip() or updated.get("Summary", "")

    if directives.bullet_count is not None:
        updated["Summary"] = _as_bullets(updated.get("Summary", ""), directives.bullet_count)
        updated["Key Points"] = _as_bullets(updated.get("Key Points", ""), directives.bullet_count)

    if directives.use_table:
        updated["Key Points"] = _as_two_column_table(updated.get("Key Points", ""))

    return updated


def _as_bullets(text: str, bullet_count: int) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "- No evidence available."

    raw_items = [
        item.strip(" -") for item in re.split(r"\n|;|(?<=[.!?])\s+", cleaned) if item.strip()
    ]
    if not raw_items:
        raw_items = [cleaned]

    if len(raw_items) < bullet_count:
        raw_items.extend([raw_items[-1]] * (bullet_count - len(raw_items)))

    selected = raw_items[:bullet_count]
    return "\n".join(f"- {item}" for item in selected)


def _as_two_column_table(text: str) -> str:
    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    if not lines:
        lines = ["No key points extracted."]

    rows = ["| Topic | Evidence |", "|---|---|"]
    for line in lines[:8]:
        rows.append(f"| Key Point | {line} |")
    return "\n".join(rows)


def _with_answer_timing(
    result: GroundedAnswerResult,
    started_at: float,
    *,
    cache_metadata: dict[str, bool],
    compaction_metadata: dict[str, int],
    fast_mode: bool,
) -> GroundedAnswerResult:
    metadata = dict(result.service_metadata)
    timings = _extract_numeric_timings(metadata.get("timings_ms"))
    timings["answer_generation"] = elapsed_ms(started_at)
    metadata["timings_ms"] = timings
    existing_cache = metadata.get("cache")
    merged_cache = dict(existing_cache) if isinstance(existing_cache, dict) else {}
    merged_cache.update(cache_metadata)
    metadata["cache"] = merged_cache
    metadata["compaction"] = compaction_metadata
    metadata["fast_mode"] = fast_mode
    return replace(result, service_metadata=metadata)


def _extract_numeric_timings(raw_timings: object) -> dict[str, float]:
    if not isinstance(raw_timings, dict):
        return {}

    timings: dict[str, float] = {}
    for key, value in raw_timings.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            timings[key] = float(value)
    return timings
