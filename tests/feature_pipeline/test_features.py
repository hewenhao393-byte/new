import numpy as np
import pytest
from feature_pipeline.config import FEATURE_NAMES
from feature_pipeline.features import BAND_RATIO_NAMES, OutOfBandOrderError, extract_features, order_band

def test_exact_finite_feature_contract():
    t=np.arange(4800)/12000
    values=extract_features(np.sin(2*np.pi*25*t)+.2*np.sin(2*np.pi*50*t),1500)
    assert tuple(values)==FEATURE_NAMES
    assert len(values)==43 and np.isfinite(list(values.values())).all()

def test_order_band_clips_and_rejects_center():
    assert order_band(6.17)==pytest.approx((5.0,8.67))
    with pytest.raises(OutOfBandOrderError): order_band(4.99)

def test_energy_partitions_and_wavelet_ratios_sum_to_one():
    values=extract_features(np.random.default_rng(2026).normal(size=4800),740)
    assert abs(sum(values[n] for n in BAND_RATIO_NAMES)-1)<1e-6
    assert abs(sum(values[f"wp_energy_ratio_{i}"] for i in range(8))-1)<1e-8

def test_time_features_match_hand_calculation():
    x=np.tile(np.array([-2.,-1.,1.,2.]),1200)
    v=extract_features(x,1480)
    rms=np.sqrt(np.mean(x*x))
    assert v["rms"]==pytest.approx(rms)
    assert v["shape_factor"]==pytest.approx(rms/np.mean(np.abs(x)))
