# %% [markdown]
# # 04 — HI Construction & Metrics
#
# Build the paper-style HI = cumulative AE-hit count (per second) from CFAR detections.
# Reports monotonicity & trendability and writes a CSV.

# %%
import os, json, csv, numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

# --- Paths and constants ---
DATA_DIR = r"D:\Pipeline RUL Data"
FILE = "B.wfs"
OUT_CSV = "B_S4_HI.csv"   # output HI (per-second cumulative) for this sensor
N_CHANNELS = 8
DTYPE = np.dtype("<i2")
HEADER_BYTES = 2
FS = 1_000_000
CELL = 500
SENSOR = 4

# CFAR params (choose CA or OS)
KIND = "os"      # "ca" or "os"
T = 50
G = 4
PFA = 1e-4
Q = 0.70
ALPHA_CACHE = "os_cfar_alpha_cache.json"

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

def alpha_ca_mc(pfa, L, K, N=300_000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N)/L
    Y = rng.chisquare(df=L*K, size=N)/(L*K)
    R = X / Y
    return float(np.quantile(R, 1.0 - pfa))

def alpha_os_mc(pfa, L, K, q, N=300_000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.chisquare(df=L, size=N)/L
    Y = rng.chisquare(df=L, size=(N, K))/L
    qv = np.quantile(Y, q, axis=1)
    R = X / qv
    return float(np.quantile(R, 1.0 - pfa))

def cfar_ca_emit(x, L, T, G, alpha):
    W = T + G
    k = np.arange(-W, W+1)
    ker = ((np.abs(k) > G) & (np.abs(k) <= G+T)).astype(float)
    s = np.convolve(x, ker, mode="same"); c = np.convolve(np.ones_like(x), ker, mode="same")
    c = np.maximum(c, 1.0); thr = alpha * (s/c)
    valid = (c >= 0.8*(2*T))
    det = (x > thr) & valid
    return det

def cfar_os_emit(x, L, T, G, alpha, q):
    W = T + G; K = 2*T
    if x.size < 2*W+1: return np.zeros_like(x, bool)
    win = sliding_window_view(x, 2*W+1)         # (N-2W, 2W+1)
    idx = np.arange(2*W+1); mask = (np.abs(idx - W) > G)
    train = win[:, mask]                        # (N-2W, K)
    qi = int(np.floor(q*(K-1))); part = np.partition(train, qi, axis=1)
    qv = part[:, qi]
    centers = np.arange(W, len(x)-W)
    thr_core = alpha * qv
    det = np.zeros_like(x, bool)
    det[centers] = x[centers] > thr_core
    return det

# %%
# Build HI per second for SENSOR
path = os.path.join(DATA_DIR, FILE)
cells_per_sec = FS // CELL
L = CELL; K = 2*T

if KIND.lower() == "os":
    a = None
    key = f"L={L}|K={K}|q={Q}|pfa={PFA}"
    try:
        with open(ALPHA_CACHE, "r") as f: cache = json.load(f)
        if key in cache: a = float(cache[key])
    except Exception:
        pass
    if a is None:
        a = alpha_os_mc(PFA, L, K, Q)
        print(f"[OS] No cache found; Monte-Carlo alpha≈{a:.4f}")
else:
    a = alpha_ca_mc(PFA, L, K)

sec_counts = []   # per-second detection counts
for start, block in iter_cell_power(path, seconds_per_chunk=10.0):
    x = block[SENSOR]
    det = cfar_os_emit(x, L, T, G, a, Q) if KIND.lower()=="os" else cfar_ca_emit(x, L, T, G, a)
    idx = start + np.arange(x.size)
    secs = idx // cells_per_sec
    uniq = np.unique(secs)
    for u in uniq:
        m = (secs == u)
        hits = int(det[m].sum())
        while len(sec_counts) < u:  # fill any skipped seconds
            sec_counts.append(0)
        sec_counts.append(hits)

# cumulative HI per second
hi = np.cumsum(np.array(sec_counts, dtype=np.int64))
print(f"Built HI with {len(hi)} seconds; last value={int(hi[-1])} events")

# save CSV (sec, hi)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["second","HI"])
    for i, v in enumerate(hi):
        w.writerow([i, int(v)])
print(f"Wrote {OUT_CSV}")

# %%
# Monotonicity & Trendability
def monotonicity(x):
    x = np.asarray(x, float)
    if len(x) < 2: return 0.0
    d = np.sign(np.diff(x))
    return abs(d.sum()) / (len(x)-1)

def trendability(x):
    x = np.asarray(x, float)
    if len(x) < 2: return 0.0
    t = np.arange(len(x), dtype=float)
    xm, tm = x.mean(), t.mean()
    cov = np.mean((x-xm)*(t-tm))
    stdx = x.std()+1e-12; stdt = t.std()+1e-12
    return abs(cov/(stdx*stdt))

m = monotonicity(hi)
tr = trendability(hi)
print(f"Monotonicity≈{m:.3f}, Trendability≈{tr:.3f}  (closer to 1 is better)")

fig, ax = plt.subplots(1,1, figsize=(10,3))
ax.plot(hi)
ax.set_title(f"HI (cumulative AE hits) — {FILE} S{SENSOR}  Mon≈{m:.2f}, Trend≈{tr:.2f}")
ax.set_xlabel("second"); ax.set_ylabel("cumulative hits")
plt.tight_layout(); plt.show()
