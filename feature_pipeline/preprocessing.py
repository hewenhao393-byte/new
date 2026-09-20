from dataclasses import dataclass
import numpy as np
from scipy.signal import butter,resample_poly,sosfiltfilt
from .config import CONFIG

@dataclass(frozen=True)
class QualityResult:
    quality_flag:str
    quality_reason:str
    rejection_reasons:tuple

def assess_quality(x):
    x=np.asarray(x,float); reasons=[]; warnings=[]
    if len(x)<CONFIG.original_fs: reasons.append("too_short")
    if not np.isfinite(x).all(): reasons.append("non_finite")
    if x.size and np.all(x==0): reasons.append("all_zero")
    if x.size>1 and np.max(np.abs(np.diff(x)))>50*np.std(x): warnings.append("possible_discontinuity")
    flag="reject" if reasons else ("warn" if warnings else "pass")
    return QualityResult(flag,";".join(reasons+warnings),tuple(reasons))

def preprocess_record(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all(): raise ValueError("non-finite signal")
    y=resample_poly(x-np.mean(x),3,5)
    sos=butter(4,[CONFIG.bandpass_low,CONFIG.bandpass_high],btype="bandpass",fs=CONFIG.target_fs,output="sos")
    return sosfiltfilt(sos,y)
