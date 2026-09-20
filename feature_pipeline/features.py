from collections import OrderedDict
import numpy as np
import pywt
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew
from .config import CONFIG, FEATURE_NAMES

BAND_RATIO_NAMES=("band_energy_5_300_ratio","band_energy_300_1000_ratio","band_energy_1000_3000_ratio","band_energy_3000_5000_ratio")
class OutOfBandOrderError(ValueError): pass

def order_band(center):
    if not CONFIG.bandpass_low <= center <= CONFIG.bandpass_high:
        raise OutOfBandOrderError(f"order center {center} outside 5-5000 Hz")
    return max(center-CONFIG.order_half_width,CONFIG.bandpass_low),min(center+CONFIG.order_half_width,CONFIG.bandpass_high)

def _spectrum(x,low=5.,high=5000.):
    freq=np.fft.rfftfreq(len(x),1/CONFIG.target_fs)
    power=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    mask=(freq>=low)&(freq<=high)
    return freq[mask],power[mask]

def _entropy(power):
    p=power/(power.sum()+CONFIG.eps)
    return float(-np.sum(p*np.log(p+CONFIG.eps))/np.log(len(p))) if len(p)>1 else 0.

def _band_energy(freq,power,low,high,include_high=True):
    mask=(freq>=low)&((freq<=high) if include_high else (freq<high))
    return float(power[mask].sum())

def extract_features(segment,rpm):
    x=np.asarray(segment,float)
    if x.shape!=(CONFIG.window_size,) or not np.isfinite(x).all(): raise ValueError("window must contain 4800 finite samples")
    eps=CONFIG.eps; out=OrderedDict()
    absx=np.abs(x); rms=float(np.sqrt(np.mean(x*x))); peak=float(absx.max()); mean_abs=float(absx.mean())
    out.update(rms=rms,std=float(np.std(x)),peak_to_peak=float(np.ptp(x)),skewness=float(skew(x,bias=False)),kurtosis=float(kurtosis(x,fisher=False,bias=False)),crest_factor=peak/(rms+eps),impulse_factor=peak/(mean_abs+eps),clearance_factor=peak/(float(np.mean(np.sqrt(absx)))**2+eps),shape_factor=rms/(mean_abs+eps))
    freq,power=_spectrum(x); total=float(power.sum()); f1=float(rpm)/60
    energies={order:_band_energy(freq,power,*order_band(order*f1)) for order in (.5,1,1.5,2,2.5,3,4,5)}
    h=sum(energies[o] for o in (1,2,3,4,5))
    out.update(rot_1x_energy_ratio=energies[1]/(total+eps),rot_2x_energy_ratio=energies[2]/(total+eps),rot_3x_energy_ratio=energies[3]/(total+eps),rot_2x_1x_ratio=energies[2]/(energies[1]+eps),rot_3x_1x_ratio=energies[3]/(energies[1]+eps),harmonic_energy_ratio_1x_5x=h/(total+eps),harmonic_energy_ratio_3x_5x=sum(energies[o] for o in (3,4,5))/(h+eps),rot_2x_harmonic_ratio=energies[2]/(h+eps),rot_05x_1x_ratio=energies[.5]/(energies[1]+eps),noninteger_harmonic_energy_ratio=sum(energies[o] for o in (.5,1.5,2.5))/(total+eps))
    centroid=float(np.sum(freq*power)/(total+eps))
    out.update(spectral_entropy=_entropy(power),spectral_flatness=float(np.exp(np.mean(np.log(power+eps)))/(np.mean(power)+eps)),spectral_centroid=centroid,spectral_bandwidth=float(np.sqrt(np.sum(((freq-centroid)**2)*power)/(total+eps))))
    e1=_band_energy(freq,power,5,300,False); e2=_band_energy(freq,power,300,1000,False); e3=_band_energy(freq,power,1000,3000,False); e4=_band_energy(freq,power,3000,5000,True); partition=e1+e2+e3+e4
    out.update(band_energy_5_300_ratio=e1/(partition+eps),band_energy_300_1000_ratio=e2/(partition+eps),band_energy_1000_3000_ratio=e3/(partition+eps),band_energy_3000_5000_ratio=e4/(partition+eps),high_low_energy_ratio=(e3+e4)/(e1+eps))
    nodes=pywt.WaveletPacket(x,CONFIG.wavelet,maxlevel=CONFIG.wavelet_level).get_level(CONFIG.wavelet_level,order="natural")
    we=np.array([np.sum(np.square(n.data)) for n in nodes],float); wp=we/(we.sum()+eps)
    for i,v in enumerate(wp): out[f"wp_energy_ratio_{i}"]=float(v)
    out["wp_energy_entropy"]=float(-np.sum(wp*np.log(wp+eps))/np.log(8))
    sos=butter(4,[1000,5000],btype="bandpass",fs=CONFIG.target_fs,output="sos")
    env=np.abs(hilbert(sosfiltfilt(sos,x))); env=env-env.mean(); ef,ep=_spectrum(env,5,500); et=float(ep.sum())
    peaks,_=find_peaks(ep,prominence=(float(ep.max())*.05 if len(ep) else 0),distance=max(1,int(np.ceil(5/(CONFIG.target_fs/CONFIG.window_size)))))
    peaks=peaks[np.argsort(ep[peaks])[-5:]] if len(peaks)>5 else peaks
    peak_e=[_band_energy(ef,ep,max(5,ef[p]-2.5),min(500,ef[p]+2.5)) for p in peaks]
    pes=sum(peak_e)
    out.update(env_kurtosis=float(kurtosis(env,fisher=False,bias=False)),env_crest_factor=float(np.max(np.abs(env))/(np.sqrt(np.mean(env*env))+eps)),env_spectral_entropy=_entropy(ep),env_peak_energy_ratio=pes/(et+eps),env_peak_concentration=(max(peak_e)/(pes+eps) if peak_e else 0.),env_peak_count=float(len(peaks)))
    if tuple(out)!=FEATURE_NAMES: raise RuntimeError("feature order mismatch")
    if not np.isfinite(list(out.values())).all(): raise ValueError("non-finite feature")
    return out
