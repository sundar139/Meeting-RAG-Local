from __future__ import annotations

import pytest

from meeting_pipeline.config import Settings
from meeting_pipeline.embeddings.ollama_client import OllamaClientError
from meeting_pipeline.rag.answer_generator import AnswerGenerator
from meeting_pipeline.rag.models import (
    ConversationState,
    ConversationTurnState,
    GroundedAnswerResult,
    RetrievalBundle,
    RetrievedChunk,
)


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0
        self.last_model: str | None = None
        self.last_messages: list[dict[str, str]] | None = None

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        self.last_model = model
        self.last_messages = messages
        return self.response


class RaisingChatClient:
    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        _ = model
        _ = messages
        raise OllamaClientError("offline")


def _sample_evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=7,
            meeting_id="m1",
            speaker_label="SPEAKER_01",
            start_time=10.5,
            end_time=16.2,
            content="We decided to launch Friday and Priya will own QA.",
            similarity=0.93,
        )
    ]


def test_answer_generator_returns_structured_sections() -> None:
    response = (
        '{"Summary":"Launch confirmed [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Key Points":"Friday launch [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Decisions":"Ship Friday [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Action Items":"Priya owns QA [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Uncertainties / Missing Evidence":"None noted."}'
    )
    client = FakeChatClient("".join(response))
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="What did we decide?",
        meeting_id="m1",
        rewritten_query="meeting decisions",
        retrieved_evidence=_sample_evidence(),
    )

    assert result.insufficient_context is False
    assert result.sections["Decisions"].startswith("Ship Friday")
    timings = result.service_metadata.get("timings_ms")
    assert isinstance(timings, dict)
    assert "answer_generation" in timings
    assert client.call_count == 1
    assert client.last_model == "llama-test"
    assert client.last_messages is not None
    assert "Retrieved evidence" in client.last_messages[1]["content"]


def test_answer_generator_handles_empty_evidence_without_model_call() -> None:
    client = FakeChatClient('{"Summary":"unused"}')
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="Any action items?",
        meeting_id="m1",
        rewritten_query="action items",
        retrieved_evidence=[],
    )

    assert result.insufficient_context is True
    assert "No supporting evidence" in result.sections["Key Points"]
    assert client.call_count == 0


def test_answer_generator_respects_bullet_format_request() -> None:
    client = FakeChatClient(
        '{"Summary":"One sentence. Two sentence. Three sentence.",'
        '"Key Points":"A. B. C.",'
        '"Decisions":"Ship Friday",'
        '"Action Items":"Priya owns QA",'
        '"Uncertainties / Missing Evidence":"None"}'
    )
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="Give 2 bullet points for the summary",
        meeting_id="m1",
        rewritten_query="summary",
        retrieved_evidence=_sample_evidence(),
    )

    assert result.sections["Summary"].count("- ") == 2


def test_answer_generator_can_build_meta_answer_from_recent_state() -> None:
    client = FakeChatClient('{"Summary":"unused"}')
    generator = AnswerGenerator(client=client, model_name="llama-test")

    recent_answer = GroundedAnswerResult(
        meeting_id="m1",
        question="What decisions were made?",
        rewritten_query="meeting decisions",
        sections={
            "Summary": "They committed to Friday launch.",
            "Key Points": "Launch and QA alignment.",
            "Decisions": "Ship Friday.",
            "Action Items": "Priya to own QA.",
            "Uncertainties / Missing Evidence": "Need confirmation from SPEAKER_02.",
        },
        raw_answer="raw",
        insufficient_context=True,
    )
    recent_bundle = RetrievalBundle(
        meeting_id="m1",
        user_query="decisions",
        rewritten_query="decisions",
        top_k_used=1,
        results=_sample_evidence(),
    )

    result = generator.generate(
        user_question="Which parts of your prior answer were uncertain?",
        meeting_id="m1",
        rewritten_query="confidence review",
        retrieved_evidence=[],
        retrieval_mode="meta_or_confidence",
        recent_state=ConversationState(
            latest_bundle=recent_bundle,
            latest_answer=recent_answer,
        ),
    )

    assert "Confidence review" in result.sections["Summary"]
    assert "Prior uncertainty notes" in result.sections["Uncertainties / Missing Evidence"]
    assert client.call_count == 0


def test_answer_generator_marks_insufficient_when_model_says_so() -> None:
    client = FakeChatClient(
        '{"Summary":"Insufficient context to answer.",'
        '"Key Points":"Insufficient evidence for this section.",'
        '"Decisions":"Insufficient evidence for this section.",'
        '"Action Items":"Insufficient evidence for this section.",'
        '"Uncertainties / Missing Evidence":"Insufficient evidence in retrieved chunks."}'
    )
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="What are blockers?",
        meeting_id="m1",
        rewritten_query="blockers",
        retrieved_evidence=_sample_evidence(),
    )

    assert result.insufficient_context is True


def test_answer_generator_wraps_chat_errors() -> None:
    generator = AnswerGenerator(client=RaisingChatClient(), model_name="llama-test")

    with pytest.raises(RuntimeError, match="Answer generation failed"):
        generator.generate(
            user_question="What did we decide?",
            meeting_id="m1",
            rewritten_query="decisions",
            retrieved_evidence=_sample_evidence(),
        )


def test_answer_generator_compacts_evidence_and_uses_cache() -> None:
    client = FakeChatClient(
        '{"Summary":"Summary [chunk_id:7 speaker:SPEAKER_01 10.5-16.2]",'
        '"Key Points":"Point",'
        '"Decisions":"Decision",'
        '"Action Items":"Action",'
        '"Uncertainties / Missing Evidence":"None"}'
    )
    settings = Settings(
        _env_file=None,
        enable_rag_caching=True,
        answer_cache_size=8,
        answer_max_evidence_chunks=1,
        answer_max_evidence_chars=500,
        answer_max_chunk_chars=90,
    )
    generator = AnswerGenerator(client=client, model_name="llama-test", settings=settings)
    long_evidence = [
        RetrievedChunk(
            chunk_id=7,
            meeting_id="m1",
            speaker_label="SPEAKER_01",
            start_time=10.5,
            end_time=16.2,
            content="Long evidence " * 50,
            similarity=0.95,
        ),
        RetrievedChunk(
            chunk_id=8,
            meeting_id="m1",
            speaker_label="SPEAKER_00",
            start_time=20.0,
            end_time=25.0,
            content="Secondary evidence " * 40,
            similarity=0.82,
        ),
    ]

    first = generator.generate(
        user_question="What did we decide?",
        meeting_id="m1",
        rewritten_query="meeting decisions",
        retrieved_evidence=long_evidence,
    )
    second = generator.generate(
        user_question="What did we decide?",
        meeting_id="m1",
        rewritten_query="meeting decisions",
        retrieved_evidence=long_evidence,
    )

    assert client.call_count == 1
    assert client.last_messages is not None
    assert "chunk_id:8" not in client.last_messages[1]["content"]

    first_compaction = first.service_metadata.get("compaction")
    assert isinstance(first_compaction, dict)
    assert first_compaction.get("selected_chunks") == 1
    assert first_compaction.get("dropped_chunks") == 1

    second_cache = second.service_metadata.get("cache")
    assert isinstance(second_cache, dict)
    assert second_cache.get("answer_generation") is True


def test_answer_generator_fast_mode_forces_short_summary_default() -> None:
    client = FakeChatClient(
        '{"Summary":"Sentence one. Sentence two. Sentence three.",'
        '"Key Points":"A",'
        '"Decisions":"B",'
        '"Action Items":"C",'
        '"Uncertainties / Missing Evidence":"None"}'
    )
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="Give me the meeting summary",
        meeting_id="m1",
        rewritten_query="meeting summary",
        retrieved_evidence=_sample_evidence(),
        fast_mode=True,
    )

    assert "Sentence three" not in result.sections["Summary"]
    assert result.service_metadata.get("fast_mode") is True


def test_answer_generator_builds_recent_conversation_meta_review() -> None:
    client = FakeChatClient('{"Summary":"unused"}')
    generator = AnswerGenerator(client=client, model_name="llama-test")

    recent_state = ConversationState(
        recent_turns=[
            ConversationTurnState(
                question="What decisions were made?",
                rewritten_query="meeting decisions",
                retrieval_mode="broad_summary",
                answer_summary="Decision summary.",
                insufficient_context=False,
            ),
            ConversationTurnState(
                question="What risks were raised?",
                rewritten_query="meeting risks",
                retrieval_mode="broad_summary",
                answer_summary="Risk summary.",
                insufficient_context=True,
            ),
        ]
    )

    result = generator.generate(
        user_question="Which of these answers are low confidence across the conversation so far?",
        meeting_id="m1",
        rewritten_query="meta confidence review",
        retrieved_evidence=[],
        retrieval_mode="meta_or_confidence",
        recent_state=recent_state,
    )

    assert "recent conversation state" in result.sections["Summary"]
    assert "Low-confidence recent questions" in result.sections["Uncertainties / Missing Evidence"]
    assert result.insufficient_context is True
    assert client.call_count == 0


def test_answer_generator_broad_summary_reports_insufficient_evidence_cleanly() -> None:
    client = FakeChatClient('{"Summary":"unused"}')
    generator = AnswerGenerator(client=client, model_name="llama-test")

    result = generator.generate(
        user_question="Summarize the meeting in 5 bullet points",
        meeting_id="m1",
        rewritten_query="summarize meeting in 5 bullet points",
        retrieved_evidence=[],
        retrieval_mode="broad_summary",
    )

    assert result.insufficient_context is True
    assert result.sections["Summary"].count("- ") == 5
    assert (
        "Current retrieval mode: broad_summary"
        in result.sections["Uncertainties / Missing Evidence"]
    )
