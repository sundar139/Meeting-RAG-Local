from __future__ import annotations

import gc
import logging
from typing import Any


def _get_torch_module() -> Any | None:
    try:
        import torch

        return torch
    except Exception:
        return None


def get_torch_device() -> str:
    torch = _get_torch_module()
    if torch is None:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return "cpu"

    return "cpu"


def clear_torch_memory() -> None:
    gc.collect()

    torch = _get_torch_module()
    if torch is None:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                # synchronize may fail in constrained runtimes; cache clear still helps.
                pass
    except Exception:
        # Cleanup should never crash callers.
        return


def get_gpu_info() -> dict[str, object]:
    info: dict[str, object] = {
        "torch_available": False,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
    }

    torch = _get_torch_module()
    if torch is None:
        return info

    info["torch_available"] = True

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        return info

    info["cuda_available"] = cuda_available
    if not cuda_available:
        return info

    try:
        device_count = int(torch.cuda.device_count())
    except Exception:
        device_count = 0

    info["device_count"] = device_count

    if device_count <= 0:
        return info

    try:
        current_index = int(torch.cuda.current_device())
        device_name = str(torch.cuda.get_device_name(current_index))
    except Exception:
        device_name = None

    info["device_name"] = device_name
    return info


def log_gpu_state(logger: logging.Logger, context: str) -> None:
    info = get_gpu_info()
    logger.info(
        "gpu_state context=%s cuda_available=%s device_count=%s device_name=%s",
        context,
        info["cuda_available"],
        info["device_count"],
        info["device_name"] or "n/a",
    )
