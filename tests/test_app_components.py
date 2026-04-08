from __future__ import annotations

from meeting_pipeline.app.components import content_excerpt, format_seconds, format_time_range


def test_format_seconds_and_range() -> None:
    assert format_seconds(65.0) == "00:01:05"
    assert format_seconds(None) == "n/a"
    assert format_time_range(5.0, 75.0) == "00:00:05 - 00:01:15"


def test_content_excerpt_truncates_long_text() -> None:
    text = "word " * 100
    excerpt = content_excerpt(text, max_chars=50)

    assert len(excerpt) <= 50
    assert excerpt.endswith("...")
