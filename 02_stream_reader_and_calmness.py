# %% [markdown]
# # 02 — Streaming Reader & Calmness Finder
#
# **Goal:** Stream *cell-power* efficiently and score per-second **calmness**
# (robust IQR) across AE sensors (S4–S7).

# %%
from pathlib import Path
import os, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = r"D:\Pipeline RUL Data"
FILE = "B.wfs"                 # start with B
N_CHANNELS = 8
DTYPE = np.dtype("<i2")
HEADER_BYTES = 2
FS = 1_000_000
CELL = 500                     # 0.5 ms
SENSORS = [4,5,6,7]            # sensors of interest
SLICE_MINUTES = 10             # scan first N minutes fast; increase later

# %%
def iter_cell_power(path, seconds_per_chunk=10.0):
    """Yield (start_cell_idx, cell_power_block) with block shape [n_channels, n_cells]."""
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
            # reshape → [cells, ch, CELL]
            cells = (arr.size // (N_CHANNELS*CELL))
            if cells == 0: break
            arr = arr[:cells*N_CHANNELS*CELL]
            arr = arr.reshape(cells, N_CHANNELS, CELL)
            cp = (arr.astype(np.float32)**2).mean(axis=2)  # [cells, ch]
            yield start_cell, cp.T.copy()
            start_cell += cells

# %%
fpath = os.path.join(DATA_DIR, FILE)
assert os.path.exists(fpath), f"Missing file: {fpath}"

cells_per_sec = FS // CELL
need_cells = int(SLICE_MINUTES*60 * cells_per_sec)

buckets = {s: defaultdict(list) for s in SENSORS}
seen = 0
for start, block in iter_cell_power(fpath, seconds_per_chunk=10.0):
    n = block.shape[1]
    if seen >= need_cells: break
    if seen + n > need_cells:
        n = need_cells - seen
        block = block[:, :n]
    secs = (start + np.arange(n)) // cells_per_sec
    uniq = np.unique(secs)
    for s in SENSORS:
        x = block[s]
        for u in uniq:
            m = (secs == u)
            buckets[s][int(u)].append(x[m])
    seen += n

seconds = sorted(set().union(*[set(buckets[s].keys()) for s in SENSORS]))
scores = []
for sec in seconds:
    spreads = []
    ok = True
    for s in SENSORS:
        if sec not in buckets[s]:
            ok=False; break
        arr = np.concatenate(buckets[s][sec])
        if arr.size < 100: ok=False; break
        q1, q3 = np.percentile(arr, [25, 75])
        spreads.append(float(q3-q1))
    if ok:
        scores.append((max(spreads), sec))
scores.sort(key=lambda t: t[0])

print(f"Scored {len(scores)} seconds from 0..{seconds[-1]} (first {SLICE_MINUTES} min).")
print('Calmest 10 seconds:', [sec for _,sec in scores[:10]])

# %%
# Visualize one calm vs one busy second on S4
def fetch_second(sec:int, sensor:int=4):
    seg = np.concatenate(buckets[sensor][sec])
    t = np.arange(seg.size)/cells_per_sec
    return t, seg

if scores:
    calm_sec = scores[0][1]
    busy_sec = scores[-1][1]
    t0, x0 = fetch_second(calm_sec, 4)
    t1, x1 = fetch_second(busy_sec, 4)
    fig, ax = plt.subplots(1,2, figsize=(12,3), sharey=True)
    ax[0].plot(t0, x0); ax[0].set_title(f"S4 calm sec {calm_sec}")
    ax[1].plot(t1, x1); ax[1].set_title(f"S4 busy sec {busy_sec}")
    for a in ax: a.set_xlabel("time (s)")
    ax[0].set_ylabel("cell power (counts²)")
    plt.tight_layout(); plt.show()
