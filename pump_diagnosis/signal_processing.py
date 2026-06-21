from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy import signal

from config import PipelineConfig


def preprocess_run(samples: np.ndarray, config: PipelineConfig) -> np.ndarray:
    centered = np.asarray(samples, dtype=np.float64) - np.mean(samples)
    resampled = signal.resample_poly(centered, up=config.resample_up, down=config.resample_down)
    sos = signal.butter(
        N=config.filter_order,
        Wn=(config.bandpass_low, config.bandpass_high),
        btype="bandpass",
        fs=config.processed_fs,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, resampled)
    return np.asarray(filtered, dtype=np.float64)


def iter_windows(
    samples: np.ndarray,
    window_size: int,
    step_size: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    for start in range(0, samples.shape[0] - window_size + 1, step_size):
        end = start + window_size
        yield start, end, samples[start:end]
