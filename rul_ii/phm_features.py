# -----------------
# File: phm_features.py
# -----------------
import numpy as np
import pandas as pd
from scipy.signal import welch

FS = 25600.0  # Hz per spec
NYQ = FS / 2.0

# Frequency bands (Hz) — tweak later if needed
BANDS = [
    (100, 1000),
    (1000, 5000),
    (5000, 10000),
    (10000, 15000),
    (15000, 20000),
    (20000, 24000),
]

NUM_COLS = [
    "Acceleration X (g)",
    "Acceleration Y (g)",
    "Acceleration Z (g)",
    "AE (V)",
]


def _safe_series(x: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").astype(float).to_numpy()
    return np.nan_to_num(arr, copy=False)


def _basic_stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "rms": 0.0,
            "ptp": 0.0,
            "q10": 0.0,
            "q50": 0.0,
            "q90": 0.0,
            "skew": 0.0,
            "kurt": 0.0,
        }
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x * x)))
    ptp = float(np.ptp(x))
    q10, q50, q90 = np.quantile(x, [0.1, 0.5, 0.9])
    # robust skew/kurt (fallback if std ~ 0)
    if std > 1e-12:
        z = (x - mean) / std
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "ptp": ptp,
        "q10": float(q10),
        "q50": float(q50),
        "q90": float(q90),
        "skew": skew,
        "kurt": kurt,
    }


def _band_powers(x: np.ndarray) -> dict:
    if x.size == 0:
        return {f"p_{lo}_{hi}": 0.0 for (lo, hi) in BANDS} | {"spec_cent": 0.0, "spec_bw": 0.0}
    # Welch PSD
    f, Pxx = welch(x, fs=FS, nperseg=4096, noverlap=2048, detrend="constant")
    Pxx = np.maximum(Pxx, 0.0)
    total = np.sum(Pxx) + 1e-12
    # centroid & bandwidth
    spec_cent = float(np.sum(f * Pxx) / total)
    spec_bw = float(np.sqrt(np.sum(((f - spec_cent) ** 2) * Pxx) / total))
    out = {"spec_cent": spec_cent, "spec_bw": spec_bw}
    for (lo, hi) in BANDS:
        m = (f >= lo) & (f < hi)
        out[f"p_{lo}_{hi}"] = float(np.sum(Pxx[m]))
    return out


def compute_cut_features(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a single cut's sensor frame.
    Returns one-row DataFrame with named columns.
    """
    feats = {}
    for col in NUM_COLS:
        if col not in sensor_df.columns:
            continue
        x = _safe_series(sensor_df[col])
        base = _basic_stats(x)
        bp = _band_powers(x)
        for k, v in base.items():
            feats[f"{col}__{k}"] = v
        for k, v in bp.items():
            feats[f"{col}__{k}"] = v
    return pd.DataFrame([feats])


def monotone_cummax(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    out = np.maximum.accumulate(y)
    out = np.clip(out, 0.0, None)
    return out


