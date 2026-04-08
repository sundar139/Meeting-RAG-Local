from __future__ import annotations

import logging

from meeting_pipeline.audio import gpu_utils


class FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.empty_cache_called = False
        self.synchronize_called = False

    def is_available(self) -> bool:
        return self._available

    def empty_cache(self) -> None:
        self.empty_cache_called = True

    def synchronize(self) -> None:
        self.synchronize_called = True

    def device_count(self) -> int:
        return 1 if self._available else 0

    def current_device(self) -> int:
        return 0

    def get_device_name(self, _index: int) -> str:
        return "Fake GPU"


class FakeTorch:
    def __init__(self, cuda_available: bool = True) -> None:
        self.cuda = FakeCuda(available=cuda_available)


def test_get_torch_device_cpu_when_torch_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(gpu_utils, "_get_torch_module", lambda: None)
    assert gpu_utils.get_torch_device() == "cpu"


def test_get_torch_device_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(gpu_utils, "_get_torch_module", lambda: FakeTorch(cuda_available=True))
    assert gpu_utils.get_torch_device() == "cuda"


def test_clear_torch_memory_calls_cache_clear(monkeypatch) -> None:
    fake_torch = FakeTorch(cuda_available=True)
    gc_called = {"value": False}

    def fake_collect() -> None:
        gc_called["value"] = True

    monkeypatch.setattr(gpu_utils, "_get_torch_module", lambda: fake_torch)
    monkeypatch.setattr(gpu_utils.gc, "collect", fake_collect)

    gpu_utils.clear_torch_memory()

    assert gc_called["value"] is True
    assert fake_torch.cuda.empty_cache_called is True
    assert fake_torch.cuda.synchronize_called is True


def test_get_gpu_info_with_cuda(monkeypatch) -> None:
    monkeypatch.setattr(gpu_utils, "_get_torch_module", lambda: FakeTorch(cuda_available=True))
    info = gpu_utils.get_gpu_info()

    assert info["cuda_available"] is True
    assert info["device_count"] == 1
    assert info["device_name"] == "Fake GPU"


def test_log_gpu_state_emits_summary(monkeypatch, caplog) -> None:
    monkeypatch.setattr(gpu_utils, "_get_torch_module", lambda: FakeTorch(cuda_available=False))
    caplog.set_level(logging.INFO)

    gpu_utils.log_gpu_state(logging.getLogger("gpu-test"), context="unit-test")

    assert "gpu_state context=unit-test" in caplog.text
