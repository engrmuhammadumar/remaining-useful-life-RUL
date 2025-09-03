# similarity.py
import numpy as np

def saw_kernel(n:int):
    k = np.zeros(n, dtype=float)
    for t in range(n):
        k[t] = 1.0 - (2.0*t)/(n-1)
    return k

def derivative_convolution(x: np.ndarray, n:int):
    # Use np.convolve to avoid SciPy dependency; 'same' length via padding behavior.
    k = saw_kernel(n)
    return np.convolve(x, k, mode="same")

def gaussian_weights(distances: np.ndarray):
    if len(distances) == 0:
        return distances
    sigma = distances.std(ddof=1) if len(distances) > 1 else 1.0
    if sigma == 0:
        sigma = 1.0
    w = np.exp(-(distances**2)/(2.0*(sigma**2)))
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w)/len(w)

def segment_reference(traj: np.ndarray, step:int, test_len:int):
    segs = []
    N = len(traj)
    for start in range(0, max(1, N - test_len), step):
        end = min(N, start + test_len)
        segs.append((start, traj[start:end]))
    return segs

def predict_rul_for_test(test_t: np.ndarray, test_hi: np.ndarray,
                         references: list, fs_hz:int,
                         saw_n:int, ref_step:int):
    """
    references: list of dicts with keys {'pipe','sensor','t','hi'}
    Returns predicted RUL (seconds).
    """
    test_dc = derivative_convolution(test_hi.astype(float), saw_n)
    distances, ref_ruls = [], []

    test_len = len(test_dc)
    for ref in references:
        ref_t, ref_hi = ref["t"], ref["hi"]
        ref_dc = derivative_convolution(ref_hi.astype(float), saw_n)
        for start_idx, seg in segment_reference(ref_dc, step=ref_step, test_len=test_len):
            L = min(len(test_dc), len(seg))
            d = np.linalg.norm(test_dc[:L] - seg[:L])
            distances.append(d)
            ref_end_time = ref_t[-1]
            seg_end_time = ref_t[min(len(ref_t)-1, start_idx + test_len - 1)]
            ref_ruls.append(ref_end_time - seg_end_time)

    distances = np.array(distances, dtype=float)
    ref_ruls = np.array(ref_ruls, dtype=float)
    if len(distances) == 0:
        return np.nan
    w = gaussian_weights(distances)
    return float((w * ref_ruls).sum())
