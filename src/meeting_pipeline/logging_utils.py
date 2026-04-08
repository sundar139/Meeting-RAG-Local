from __future__ import annotations

import json
import logging
from collections.abc import Mapping, MutableMapping
from datetime import datetime, timezone
from typing import Any

from rich.logging import RichHandler


class JsonFormatter(logging.Formatter):
    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._reserved:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class StructuredAdapter(logging.LoggerAdapter[logging.Logger]):
    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        provided_extra = kwargs.get("extra")
        merged_extra: dict[str, Any] = {}
        if isinstance(self.extra, Mapping):
            merged_extra.update(dict(self.extra))
        if isinstance(provided_extra, Mapping):
            merged_extra.update(dict(provided_extra))
        kwargs["extra"] = merged_extra
        return msg, kwargs


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    root_logger.handlers.clear()

    if json_logs:
        handler: logging.Handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = RichHandler(rich_tracebacks=True, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.addHandler(handler)


def get_logger(name: str, **context: Any) -> StructuredAdapter:
    return StructuredAdapter(logging.getLogger(name), context)
