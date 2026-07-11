from __future__ import annotations

import numpy as np

from pump_fault_app.domain.formal_contract import FORMAL_V2_CONTRACT
from pump_fault_app.domain.records import RawSignalRecord, SignalQualityReport


_NEAR_ZERO_SPAN = 1e-8
_LARGE_ZERO_RATIO = 0.5
_CLIPPING_RATIO_WARN = 0.01
_ABRUPT_JUMP_RATIO_WARN = 0.0002


def assess_signal_quality(record: RawSignalRecord) -> SignalQualityReport:
    samples = np.asarray(record.samples, dtype=np.float64)
    finite_samples = samples[np.isfinite(samples)]

    nan_count = int(np.isnan(samples).sum())
    inf_count = int(np.isinf(samples).sum())
    zero_ratio = float(np.mean(np.isclose(samples, 0.0, atol=1e-12))) if samples.size else 0.0
    is_all_zero = bool(finite_samples.size > 0 and np.all(np.isclose(finite_samples, 0.0, atol=1e-12)))
    is_constant = bool(finite_samples.size > 0 and np.allclose(finite_samples, finite_samples[0], atol=1e-12, rtol=0.0))
    amplitude_span = float(np.ptp(finite_samples)) if finite_samples.size else 0.0
    clipped_sample_ratio = _clipped_ratio(finite_samples)
    abrupt_jump_ratio = _abrupt_jump_ratio(finite_samples)

    warnings: list[str] = []
    rejection_reasons: list[str] = []

    if nan_count > 0:
        rejection_reasons.append("signal contains NaN values")
    if inf_count > 0:
        rejection_reasons.append("signal contains infinite values")
    if record.sampling_rate_hz < FORMAL_V2_CONTRACT.signal.filter_high_hz * 2:
        rejection_reasons.append("sampling rate is too low for 5000 Hz analysis")
    if record.sample_count < FORMAL_V2_CONTRACT.signal.window_size:
        rejection_reasons.append("signal length is shorter than one formal window")
    if is_all_zero:
        rejection_reasons.append("signal is all zeros")
    elif is_constant:
        rejection_reasons.append("signal is constant")

    if amplitude_span <= _NEAR_ZERO_SPAN and not rejection_reasons:
        warnings.append("signal amplitude is near zero")
    if zero_ratio > _LARGE_ZERO_RATIO and not is_all_zero:
        warnings.append("signal contains a large proportion of zeros")
    if clipped_sample_ratio > _CLIPPING_RATIO_WARN:
        warnings.append("signal may contain clipping")
    if abrupt_jump_ratio > _ABRUPT_JUMP_RATIO_WARN:
        warnings.append("signal contains abrupt jumps")

    quality_level = "fail" if rejection_reasons else ("warn" if warnings else "pass")
    return SignalQualityReport(
        allow_diagnosis=not rejection_reasons,
        quality_level=quality_level,
        nan_count=nan_count,
        inf_count=inf_count,
        zero_ratio=zero_ratio,
        is_all_zero=is_all_zero,
        is_constant=is_constant,
        sample_count=record.sample_count,
        duration_seconds=record.duration_seconds,
        sampling_rate_hz=record.sampling_rate_hz,
        amplitude_span=amplitude_span,
        clipped_sample_ratio=clipped_sample_ratio,
        abrupt_jump_ratio=abrupt_jump_ratio,
        warnings=tuple(warnings),
        rejection_reasons=tuple(rejection_reasons),
    )


def _clipped_ratio(samples: np.ndarray) -> float:
    if samples.size < 3:
        return 0.0
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        return 0.0
    near_peak = np.isclose(np.abs(samples), peak, rtol=0.0, atol=max(peak * 1e-6, 1e-12))
    return float(np.mean(near_peak))


def _abrupt_jump_ratio(samples: np.ndarray) -> float:
    if samples.size < 3:
        return 0.0
    diffs = np.abs(np.diff(samples))
    median_diff = float(np.median(diffs))
    if median_diff <= 1e-12:
        return 0.0
    jump_mask = diffs > (median_diff * 10.0)
    return float(np.mean(jump_mask))
