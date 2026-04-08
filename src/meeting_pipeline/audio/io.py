from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def read_audio_file(path: Path) -> tuple[np.ndarray[Any, Any], int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path)
    return audio, sample_rate


def write_audio_file(path: Path, samples: np.ndarray[Any, Any], sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate)
