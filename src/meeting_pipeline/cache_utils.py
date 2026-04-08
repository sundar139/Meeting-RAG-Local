from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LruCache(Generic[K, V]):
    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        self._max_size = max_size
        self._store: OrderedDict[K, V] = OrderedDict()

    @property
    def max_size(self) -> int:
        return self._max_size

    def get(self, key: K) -> V | None:
        value = self._store.get(key)
        if value is None:
            return None

        self._store.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        self._store[key] = value
        self._store.move_to_end(key)

        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
