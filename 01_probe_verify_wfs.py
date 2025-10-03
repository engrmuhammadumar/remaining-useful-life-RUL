# %% [markdown]
# # 01 — Probe & Verify WFS Layout
#
# **Goal:** Confirm the binary format of your `*.wfs` files and sanity-check timescales
# without loading the whole file.
#
# **Assumptions (edit if needed):**
# - Path: `D:\Pipeline RUL Data`
# - Files: `B.wfs, C.wfs, D.wfs, E.wfs`
# - 8 channels, interleaved, little-endian int16 (`<i2`), 2-byte header, 1 MHz
# - Cells of 500 samples (0.5 ms)

# %%
from pathlib import Path
import os, math, struct
import numpy as np
import matplotlib.pyplot as plt

# --- User-configurable ---
DATA_DIR = r"D:\Pipeline RUL Data"
FILES = ["B.wfs","C.wfs","D.wfs","E.wfs"]

N_CHANNELS = 8
DTYPE = np.dtype("<i2")   # little-endian int16
HEADER_BYTES = 2
FS = 1_000_000            # Hz
CELL = 500                # samples per cell (0.5 ms)
READ_S = 0.01             # 10 ms from each channel to plot

assert CELL > 0 and FS % CELL == 0, "FS must be divisible by CELL"
CELLS_PER_SEC = FS // CELL

# %%
def file_info(path: str):
    p = Path(path)
    size = p.stat().st_size
    return {"name": p.name, "size_bytes": size}

def derive_layout(size_bytes: int, header_bytes: int, dtype: np.dtype, n_channels: int):
    payload = size_bytes - header_bytes
    if payload <= 0 or payload % dtype.itemsize != 0:
        return None
    total_values = payload // dtype.itemsize
    if total_values % n_channels != 0:
        return None
    samples_per_ch = total_values // n_channels
    duration_s = samples_per_ch / FS
    return samples_per_ch, duration_s, total_values

def read_short_excerpt(path, start_s=0.0, read_s=0.01, ch=0):
    """Read 'read_s' seconds for one channel from interleaved <i2 WFS."""
    samp0 = int(start_s * FS)
    n_samp = int(read_s * FS)
    frame_bytes = N_CHANNELS * DTYPE.itemsize
    byte0 = HEADER_BYTES + samp0 * frame_bytes + ch * DTYPE.itemsize
    stride = frame_bytes
    data = np.empty(n_samp, dtype=DTYPE)
    with open(path, "rb") as f:
        f.seek(byte0)
        for i in range(n_samp):
            b = f.read(DTYPE.itemsize)
            if not b:
                data = data[:i]
                break
            data[i] = struct.unpack("<h", b)[0]
            f.seek(stride - DTYPE.itemsize, os.SEEK_CUR)
    return data.astype(np.float32)

def show_excerpt_grid(path, read_s=READ_S):
    cols = 4
    rows = math.ceil(N_CHANNELS/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6), sharex=True)
    axes = axes.ravel()
    for ch in range(N_CHANNELS):
        x = read_short_excerpt(path, start_s=0.0, read_s=read_s, ch=ch)
        t = np.arange(len(x))/FS
        ax = axes[ch]
        ax.plot(t*1e3, x)  # ms
        ax.set_title(f"Ch {ch}")
        if ch % cols == 0: ax.set_ylabel("counts")
        if ch//cols == rows-1: ax.set_xlabel("time (ms)")
    for k in range(N_CHANNELS, len(axes)):
        axes[k].axis("off")
    fig.suptitle(Path(path).name + f"  (first {read_s*1e3:.1f} ms per channel)")
    plt.tight_layout()
    plt.show()

# %%
for fname in FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] Missing: {fpath}")
        continue
    info = file_info(fpath)
    layout = derive_layout(info["size_bytes"], HEADER_BYTES, DTYPE, N_CHANNELS)
    print(f"\nFile: {fpath}  ({info['size_bytes']/1024/1024/1024:.2f} GB)")
    if not layout:
        print("  [!] Size does not align with assumed layout; check constants above.")
        continue
    samples_per_ch, duration_s, total_values = layout
    print(f"  samples/ch = {samples_per_ch:,}")
    print(f"  duration   = {duration_s:.2f} s  (~{duration_s/3600:.2f} h)")
    print(f"  dtype      = {DTYPE.str}  channels={N_CHANNELS}  header={HEADER_BYTES} B  interleaved")
    print(f"  cell size  = {CELL} samples  →  cells/sec = {CELLS_PER_SEC}")
    # quick plot
    try:
        show_excerpt_grid(fpath, read_s=READ_S)
    except Exception as e:
        print("  Plot failed:", e)
