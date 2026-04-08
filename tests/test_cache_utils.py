from __future__ import annotations

import pytest

from meeting_pipeline.cache_utils import LruCache


def test_lru_cache_eviction_and_recency() -> None:
    cache = LruCache[str, int](max_size=2)

    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1

    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_lru_cache_requires_positive_size() -> None:
    with pytest.raises(ValueError):
        LruCache[str, int](max_size=0)
