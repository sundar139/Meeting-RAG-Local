from __future__ import annotations

import logging
from typing import Any

from meeting_pipeline.config import Settings, get_settings

LOGGER = logging.getLogger(__name__)


class OllamaClientError(RuntimeError):
    """Base error for Ollama embedding operations."""


class OllamaUnavailableError(OllamaClientError):
    """Raised when Ollama server is unreachable or client initialization fails."""


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when requested model is not available in the Ollama runtime."""


class OllamaMalformedResponseError(OllamaClientError):
    """Raised when Ollama responds with an unexpected payload."""


def _coerce_payload_object(payload: Any, *, context: str) -> dict[str, object]:
    if isinstance(payload, dict):
        return payload

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    dict_method = getattr(payload, "dict", None)
    if callable(dict_method):
        dumped = dict_method()
        if isinstance(dumped, dict):
            return dumped

    raise OllamaMalformedResponseError(f"Ollama {context} response payload is not an object")


def _normalize_embedding(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None

    embedding: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if not isinstance(item, (int, float)):
            return None
        embedding.append(float(item))

    return embedding


def _extract_embeddings(payload: Any) -> list[list[float]]:
    payload_obj = _coerce_payload_object(payload, context="embedding")

    if "embeddings" in payload_obj:
        value = payload_obj["embeddings"]
        if not isinstance(value, list):
            raise OllamaMalformedResponseError("Ollama response field 'embeddings' is not a list")

        results: list[list[float]] = []
        for item in value:
            embedding = _normalize_embedding(item)
            if embedding is None:
                raise OllamaMalformedResponseError(
                    "Ollama response contains a non-numeric embedding"
                )
            results.append(embedding)

        if not results:
            raise OllamaMalformedResponseError("Ollama response returned no embeddings")
        return results

    if "embedding" in payload_obj:
        embedding = _normalize_embedding(payload_obj["embedding"])
        if embedding is None:
            raise OllamaMalformedResponseError("Ollama response field 'embedding' is invalid")
        return [embedding]

    raise OllamaMalformedResponseError("Ollama response must contain 'embedding' or 'embeddings'")


def _extract_chat_content(payload: Any) -> str:
    payload_obj = _coerce_payload_object(payload, context="chat")

    message_obj = payload_obj.get("message")
    if isinstance(message_obj, dict):
        content = message_obj.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    content_obj = payload_obj.get("response")
    if isinstance(content_obj, str) and content_obj.strip():
        return content_obj.strip()

    raise OllamaMalformedResponseError("Ollama chat response does not contain message content")


class OllamaClient:
    def __init__(self, host: str) -> None:
        self.host = host.rstrip("/")
        self._client = self._build_client()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> OllamaClient:
        runtime_settings = settings or get_settings()
        return cls(host=runtime_settings.ollama_host)

    def _build_client(self) -> Any:
        try:
            import ollama
        except ModuleNotFoundError as exc:
            raise OllamaUnavailableError(
                "The ollama package is not installed. Install with "
                "'uv sync --group dev --extra services'."
            ) from exc

        try:
            if hasattr(ollama, "Client"):
                return ollama.Client(host=self.host)
            return ollama
        except Exception as exc:
            raise OllamaUnavailableError(
                "Failed to initialize Ollama client. Verify OLLAMA_HOST and local server status."
            ) from exc

    def _embed_request(self, model: str, texts: list[str]) -> list[list[float]]:
        LOGGER.info("Requesting embeddings count=%d model=%s", len(texts), model)

        try:
            if hasattr(self._client, "embed"):
                payload = self._client.embed(
                    model=model,
                    input=texts if len(texts) > 1 else texts[0],
                )
            elif hasattr(self._client, "embeddings"):
                if len(texts) != 1:
                    raise OllamaClientError(
                        "Installed Ollama client only supports single-text embeddings per request"
                    )
                payload = self._client.embeddings(model=model, prompt=texts[0])
            else:
                raise OllamaClientError("Installed Ollama client exposes no embedding method")
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message and "model" in message:
                raise OllamaModelNotFoundError(
                    f"Ollama model '{model}' was not found. Pull it with 'ollama pull {model}'."
                ) from exc
            if any(
                token in message for token in ("connection", "refused", "unreachable", "timed out")
            ):
                raise OllamaUnavailableError(
                    "Could not reach Ollama server. Ensure Ollama is running "
                    "and OLLAMA_HOST is correct."
                ) from exc
            raise OllamaClientError(f"Ollama embedding request failed: {exc}") from exc

        return _extract_embeddings(payload)

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        if not messages:
            raise ValueError("messages must contain at least one item")

        LOGGER.info("Requesting chat completion message_count=%d model=%s", len(messages), model)

        try:
            if hasattr(self._client, "chat"):
                payload = self._client.chat(model=model, messages=messages)
            else:
                raise OllamaClientError("Installed Ollama client exposes no chat method")
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message and "model" in message:
                raise OllamaModelNotFoundError(
                    f"Ollama model '{model}' was not found. Pull it with 'ollama pull {model}'."
                ) from exc
            if any(
                token in message for token in ("connection", "refused", "unreachable", "timed out")
            ):
                raise OllamaUnavailableError(
                    "Could not reach Ollama server. Ensure Ollama is running "
                    "and OLLAMA_HOST is correct."
                ) from exc
            raise OllamaClientError(f"Ollama chat request failed: {exc}") from exc

        return _extract_chat_content(payload)

    def embed(self, model: str, text: str) -> list[float]:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("text must be a non-empty string")

        embeddings = self._embed_request(model=model, texts=[normalized])
        return embeddings[0]

    def embed_many(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must contain at least one item")

        normalized_texts = [" ".join(text.split()) for text in texts]
        if any(not item for item in normalized_texts):
            raise ValueError("all texts must be non-empty after normalization")

        return self._embed_request(model=model, texts=normalized_texts)
