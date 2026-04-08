from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from meeting_pipeline.config import Settings, get_settings
from meeting_pipeline.embeddings.ollama_client import OllamaClient, OllamaClientError
from meeting_pipeline.rag.models import GroundedAnswerResult, RetrievedChunk

LOGGER = logging.getLogger(__name__)
DEFAULT_CHAT_MODEL = "llama3.2:3b-instruct"
SECTION_ORDER = [
    "Summary",
    "Key Points",
    "Decisions",
    "Action Items",
    "Uncertainties / Missing Evidence",
]


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

    def generate(
        self,
        *,
        user_question: str,
        meeting_id: str,
        retrieved_evidence: Sequence[RetrievedChunk],
        rewritten_query: str,
        conversation_context: Sequence[str] | None = None,
    ) -> GroundedAnswerResult:
        normalized_question = " ".join(user_question.split())
        normalized_meeting_id = meeting_id.strip()
        normalized_rewrite = " ".join(rewritten_query.split())

        if not normalized_question:
            raise ValueError("user_question must be a non-empty string")
        if not normalized_meeting_id:
            raise ValueError("meeting_id must be a non-empty string")
        if not normalized_rewrite:
            raise ValueError("rewritten_query must be a non-empty string")

        if not retrieved_evidence:
            sections = {
                "Summary": "Retrieved evidence is insufficient to answer this question reliably.",
                "Key Points": "No supporting evidence was retrieved from transcript chunks.",
                "Decisions": "No decision evidence was found in retrieved context.",
                "Action Items": "No action-item evidence was found in retrieved context.",
                "Uncertainties / Missing Evidence": (
                    "Try broadening the question, adding context, or increasing retrieval top-k."
                ),
            }
            raw_answer = "\n".join(f"{key}: {value}" for key, value in sections.items())
            return GroundedAnswerResult(
                meeting_id=normalized_meeting_id,
                question=normalized_question,
                rewritten_query=normalized_rewrite,
                sections=sections,
                raw_answer=raw_answer,
                insufficient_context=True,
            )

        context_lines = [
            f"[chunk_id:{chunk.chunk_id} speaker:{chunk.speaker_label} "
            f"{chunk.start_time:.2f}-{chunk.end_time:.2f}] {chunk.content}"
            for chunk in retrieved_evidence
        ]
        context_block = "\n".join(context_lines)

        conversation_lines = [
            " ".join(item.split()) for item in (conversation_context or []) if item.strip()
        ]
        conversation_block = "\n".join(f"- {line}" for line in conversation_lines)

        system_prompt = (
            "You are a meeting QA assistant. Use only the provided retrieved evidence. "
            "Do not invent facts. If evidence is insufficient, explicitly say so. "
            "Respond in strict JSON with keys: Summary, Key Points, Decisions, Action Items, "
            "Uncertainties / Missing Evidence. Include local citations like "
            "[chunk_id:12 speaker:SPEAKER_01 32.4-48.2] in statements where evidence "
            "supports claims."
        )
        user_prompt = (
            f"Meeting ID: {normalized_meeting_id}\n"
            f"Original question: {normalized_question}\n"
            f"Rewritten query: {normalized_rewrite}\n"
            f"Recent conversation context:\n{conversation_block or '- none'}\n\n"
            f"Retrieved evidence:\n{context_block}"
        )

        LOGGER.info("Generating grounded answer evidence_count=%d", len(retrieved_evidence))
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
        insufficient = _is_insufficient_answer(sections)

        return GroundedAnswerResult(
            meeting_id=normalized_meeting_id,
            question=normalized_question,
            rewritten_query=normalized_rewrite,
            sections=sections,
            raw_answer=raw_answer,
            insufficient_context=insufficient,
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
