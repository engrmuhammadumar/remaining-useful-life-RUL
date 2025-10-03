# %% [markdown]
# # 04b — Build HI for all pipes & sensors (S4–S7) using cached OS alpha
# Writes one CSV per (pipe,sensor): <PIPE>_S<sensor>_HI.csv

# %%
import os, json, csv, numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# --- Config ---
DATA_DIR = r"D:\Pipeline RUL Data"
FILES = ["B.wfs","C.wfs","D.wfs","E.wfs"]
N_CHANNELS = 8
DTYPE = np.dtype("<i2"); HEADER_BYTES = 2
FS = 1_000_000; CELL = 500
SENSORS = [4,5,6,7]

# CFAR (OS) parameters
T = 50; G = 4; Q = 0.70; PFA = 1e-4
ALPHA_CACHE = "os_cfar_alpha_cache.json"  # should contain empirical alpha

# %%
def iter_cell_power(path, seconds_per_chunk=10.0):
    frame_bytes = N_CHANNELS * DTYPE.itemsize
    cell_bytes = frame_bytes * CELL
    n_cells = int(seconds_per_chunk * FS // CELL)
    with open(path, "rb") as f:
        f.seek(HEADER_BYTES, os.SEEK_SET)
        start_cell = 0
        while True:
            raw = f.read(n_cells * cell_bytes)
            if not raw: break
            arr = np.frombuffer(raw, dtype=DTYPE)
            if arr.size == 0: break
            cells = (arr.size // (N_CHANNELS*CELL))
            if cells == 0: break
            arr = arr[:cells*N_CHANNELS*CELL].reshape(cells, N_CHANNELS, CELL)
            cp = (arr.astype(np.float32)**2).mean(axis=2)  # [cells, ch]
            yield start_cell, cp.T.copy()
            start_cell += cells

def cfar_os_emit(x, L, T, G, alpha, q):
    W = T + G; K = 2*T
    if x.size < 2*W+1: return np.zeros_like(x, bool)
    win = sliding_window_view(x, 2*W+1)         # (N-2W, 2W+1)
    idx = np.arange(2*W+1); mask = (np.abs(idx - W) > G)
    train = win[:, mask]                        # (N-2W, K)
    qi = int(np.floor(q*(K-1)))
    part = np.partition(train, qi, axis=1)
    qv = part[:, qi]
    centers = np.arange(W, len(x)-W)
    det = np.zeros_like(x, bool)
    det[centers] = x[centers] > (alpha * qv)
    return det

def load_cached_alpha(L, T, q, pfa, fallback=None):
    key = f"L={L}|K={2*T}|q={q}|pfa={pfa}"
    try:
        with open(ALPHA_CACHE, "r") as f:
            cache = json.load(f)
        if key in cache:
            a = float(cache[key])
            print(f"[alpha] Using cached OS alpha for {key}: {a:.6f}")
            return a
    except Exception:
        pass
    if fallback is None:
        raise RuntimeError("No cached alpha found and no fallback provided.")
    print("[alpha] No cache found; using fallback alpha", fallback)
    return float(fallback)

def write_hi_for(path, pipe_name, sensor, alpha):
    cells_per_sec = FS // CELL
    sec_counts = []
    total_hits = 0
    for start, block in iter_cell_power(path, seconds_per_chunk=10.0):
        x = block[sensor]
        det = cfar_os_emit(x, CELL, T, G, alpha, Q)
        idx = start + np.arange(x.size)
        secs = idx // cells_per_sec
        uniq = np.unique(secs)
        for u in uniq:
            m = (secs == u)
            hits = int(det[m].sum())
            while len(sec_counts) < u:
                sec_counts.append(0)
            sec_counts.append(hits)
            total_hits += hits
    hi = np.cumsum(np.array(sec_counts, dtype=np.int64))
    out_csv = f"{pipe_name}_S{sensor}_HI.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["second","HI"])
        for i, v in enumerate(hi):
            w.writerow([i, int(v)])
    print(f"[{pipe_name} S{sensor}] seconds={len(hi)} total_hits={total_hits:,} -> {out_csv}")

# %%
L = CELL
alpha = load_cached_alpha(L, T, Q, PFA, fallback=1.209586)  # MC fallback

for fname in FILES:
    path = os.path.join(DATA_DIR, fname)
    assert os.path.exists(path), f"Missing file: {path}"
    pipe = os.path.splitext(os.path.basename(path))[0]  # "B","C","D","E"
    for s in SENSORS:
        write_hi_for(path, pipe, s, alpha)
