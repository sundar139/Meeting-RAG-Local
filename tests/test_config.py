from __future__ import annotations

import pytest
from pydantic import ValidationError

from meeting_pipeline.config import Settings


def test_settings_include_safe_retrieval_policy_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.default_factoid_top_k == 5
    assert settings.speaker_specific_top_k == 4
    assert settings.action_items_or_decisions_top_k == 8
    assert settings.broad_summary_top_k == 14
    assert settings.meta_or_confidence_top_k == 6
    assert settings.broad_summary_max_candidates == 28


def test_settings_parse_retrieval_policy_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_FACTOID_TOP_K", "6")
    monkeypatch.setenv("SPEAKER_SPECIFIC_TOP_K", "3")
    monkeypatch.setenv("ACTION_ITEMS_OR_DECISIONS_TOP_K", "10")
    monkeypatch.setenv("BROAD_SUMMARY_TOP_K", "12")
    monkeypatch.setenv("META_OR_CONFIDENCE_TOP_K", "5")
    monkeypatch.setenv("BROAD_SUMMARY_MAX_CANDIDATES", "20")

    settings = Settings(_env_file=None)

    assert settings.default_factoid_top_k == 6
    assert settings.speaker_specific_top_k == 3
    assert settings.action_items_or_decisions_top_k == 10
    assert settings.broad_summary_top_k == 12
    assert settings.meta_or_confidence_top_k == 5
    assert settings.broad_summary_max_candidates == 20


def test_settings_reject_non_positive_retrieval_policy_values() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, default_factoid_top_k=0)
