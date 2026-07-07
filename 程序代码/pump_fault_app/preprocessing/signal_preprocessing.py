from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from pump_diagnosis.inference_contract import FORMAL_V2_CONTRACT
from pump_fault_app.domain.records import PreprocessedSignalRecord, RawSignalRecord


def preprocess_raw_signal(record: RawSignalRecord) -> PreprocessedSignalRecord:
    contract = FORMAL_V2_CONTRACT.signal
    processing_log: list[str] = []

    samples = np.asarray(record.samples, dtype=np.float64)
    centered = samples - np.mean(samples)
    processing_log.append("demean_before_resample")

    if record.sampling_rate_hz == contract.target_sampling_rate:
        resampled = centered
        was_resampled = False
    else:
        ratio = Fraction(contract.target_sampling_rate, record.sampling_rate_hz).limit_denominator(1000)
        resampled = signal.resample_poly(centered, up=ratio.numerator, down=ratio.denominator)
        was_resampled = True
        processing_log.append("resample_to_formal_rate")

    sos = signal.butter(
        N=4,
        Wn=(contract.filter_low_hz, contract.filter_high_hz),
        btype="bandpass",
        fs=contract.target_sampling_rate,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, resampled)
    processing_log.append("bandpass_filter")
    processed = np.asarray(filtered, dtype=np.float64) - np.mean(filtered)
    processing_log.append("demean_after_filter")

    if not np.isfinite(processed).all():
        raise ValueError("preprocessed signal contains non-finite values")

    return PreprocessedSignalRecord(
        source_file=record.source_file,
        original_sample_count=record.sample_count,
        processed_sample_count=int(processed.shape[0]),
        original_sampling_rate_hz=record.sampling_rate_hz,
        target_sampling_rate_hz=contract.target_sampling_rate,
        was_resampled=was_resampled,
        dc_removed_before_filter=True,
        dc_removed_after_filter=True,
        filter_applied=True,
        processed_samples=processed,
        processing_log=tuple(processing_log),
        rpm=record.rpm,
        device_id=record.device_id,
        measurement_position=record.measurement_position,
    )
