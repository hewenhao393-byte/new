from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pump_fault_app.domain.labels import assert_valid_label


@dataclass(frozen=True)
class WindowSlice:
    source_file: Path
    window_id: str
    start_index: int
    end_index: int
    sample_rate_hz: int


@dataclass(frozen=True)
class SampleMetadata:
    label: str | None = None
    rpm: float | None = None
    device_id: str | None = None
    channel: int | None = None

    def __post_init__(self) -> None:
        if self.label is not None:
            assert_valid_label(self.label)


@dataclass(frozen=True)
class FeatureVector:
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    metadata: SampleMetadata

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.values):
            raise ValueError("feature name and value counts must match")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = asdict(self.metadata)
        payload["feature_map"] = dict(zip(self.feature_names, self.values))
        return payload


@dataclass(frozen=True)
class DiagnosisResult:
    predicted_label: str
    confidence: float
    feature_vector: FeatureVector

    def __post_init__(self) -> None:
        assert_valid_label(self.predicted_label)


@dataclass(frozen=True)
class RawSignalRecord:
    file_name: str
    source_file: Path
    samples: np.ndarray
    sample_count: int
    sampling_rate_hz: int
    duration_seconds: float
    rpm: float
    device_id: str | None = None
    measurement_position: str | None = None
    selected_signal_column: str | None = None
    time_column: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("raw signal record must contain at least one sample")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling rate must be positive")
        if self.samples.shape[0] != self.sample_count:
            raise ValueError("sample_count does not match samples length")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_file"] = str(self.source_file)
        payload["samples"] = self.samples.tolist()
        return payload


@dataclass(frozen=True)
class SignalQualityReport:
    allow_diagnosis: bool
    quality_level: str
    nan_count: int
    inf_count: int
    zero_ratio: float
    is_all_zero: bool
    is_constant: bool
    sample_count: int
    duration_seconds: float
    sampling_rate_hz: int
    amplitude_span: float
    clipped_sample_ratio: float
    abrupt_jump_ratio: float
    warnings: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessedSignalRecord:
    source_file: Path
    original_sample_count: int
    processed_sample_count: int
    original_sampling_rate_hz: int
    target_sampling_rate_hz: int
    was_resampled: bool
    dc_removed_before_filter: bool
    dc_removed_after_filter: bool
    filter_applied: bool
    processed_samples: np.ndarray
    processing_log: tuple[str, ...] = ()
    rpm: float | None = None
    device_id: str | None = None
    measurement_position: str | None = None

    def __post_init__(self) -> None:
        if self.processed_sample_count <= 0:
            raise ValueError("processed signal must contain at least one sample")
        if self.processed_samples.shape[0] != self.processed_sample_count:
            raise ValueError("processed_sample_count does not match processed_samples length")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_file"] = str(self.source_file)
        payload["processed_samples"] = self.processed_samples.tolist()
        return payload


@dataclass(frozen=True)
class SignalWindow:
    window_index: int
    start_index: int
    end_index: int
    start_time_seconds: float
    end_time_seconds: float
    sample_rate_hz: int
    samples: np.ndarray
    rpm: float | None = None
    device_id: str | None = None
    measurement_position: str | None = None

    def __post_init__(self) -> None:
        if self.end_index <= self.start_index:
            raise ValueError("window end index must be greater than start index")
        if self.samples.shape[0] != self.end_index - self.start_index:
            raise ValueError("window sample length does not match index range")


@dataclass(frozen=True)
class WindowingResult:
    source_file: Path
    window_size: int
    step_size: int
    overlap_ratio: float
    window_duration_seconds: float
    window_count: int
    windows: tuple[SignalWindow, ...]

    def __post_init__(self) -> None:
        if self.window_count != len(self.windows):
            raise ValueError("window_count does not match windows length")


@dataclass(frozen=True)
class WindowPredictionResult:
    predicted_label: str
    confidence: float
    label_probabilities: dict[str, float]
    feature_vector: FeatureVector

    def __post_init__(self) -> None:
        assert_valid_label(self.predicted_label)


@dataclass(frozen=True)
class RecordPredictionResult:
    source_file: Path
    predicted_label: str
    confidence: float
    label_probabilities: dict[str, float]
    window_count: int
    window_predictions: tuple[WindowPredictionResult, ...]

    def __post_init__(self) -> None:
        assert_valid_label(self.predicted_label)
        if self.window_count <= 0:
            raise ValueError("record prediction must contain at least one window")
        if self.window_count != len(self.window_predictions):
            raise ValueError("window_count does not match window_predictions length")


@dataclass(frozen=True)
class TimeDomainSeries:
    time_s: tuple[float, ...]
    amplitude: tuple[float, ...]
    sampling_rate_hz: int
    point_count: int
    downsampled_for_display: bool

    def __post_init__(self) -> None:
        if len(self.time_s) != len(self.amplitude):
            raise ValueError("time axis and amplitude lengths must match")
        if self.point_count != len(self.time_s):
            raise ValueError("point_count does not match time axis length")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling rate must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpectrumSeries:
    frequency_hz: tuple[float, ...]
    amplitude: tuple[float, ...]
    frequency_min_hz: float
    frequency_max_hz: float
    resolution_hz: float

    def __post_init__(self) -> None:
        if len(self.frequency_hz) != len(self.amplitude):
            raise ValueError("frequency axis and amplitude lengths must match")
        if self.frequency_max_hz < self.frequency_min_hz:
            raise ValueError("frequency_max_hz must be greater than or equal to frequency_min_hz")
        if self.resolution_hz < 0:
            raise ValueError("resolution_hz must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WaveletPacketEnergySeries:
    band_labels: tuple[str, ...]
    band_start_hz: tuple[float, ...]
    band_end_hz: tuple[float, ...]
    energy_ratio: tuple[float, ...]
    wavelet: str
    decomposition_level: int

    def __post_init__(self) -> None:
        counts = {len(self.band_labels), len(self.band_start_hz), len(self.band_end_hz), len(self.energy_ratio)}
        if len(counts) != 1:
            raise ValueError("wavelet packet band metadata lengths must match")
        if self.decomposition_level <= 0:
            raise ValueError("decomposition_level must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosisVisualizationData:
    time_domain: TimeDomainSeries | None
    frequency_spectrum: SpectrumSeries | None
    envelope_spectrum: SpectrumSeries | None
    wavelet_packet_energy: WaveletPacketEnergySeries | None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_domain": None if self.time_domain is None else self.time_domain.as_dict(),
            "frequency_spectrum": None if self.frequency_spectrum is None else self.frequency_spectrum.as_dict(),
            "envelope_spectrum": None if self.envelope_spectrum is None else self.envelope_spectrum.as_dict(),
            "wavelet_packet_energy": None if self.wavelet_packet_energy is None else self.wavelet_packet_energy.as_dict(),
            "warnings": list(self.warnings),
        }
