# prepare_data.py
import os, yaml
import numpy as np
import pandas as pd
from loaders import read_wfs  # from the version I sent earlier with layout + header support
from hi import build_hi_for_channel_cellwise

def iter_chunks_interleaved(mm, ch_idx: int, chunk_samples: int, max_samples: int | None):
    N = mm.shape[0] if max_samples is None else min(mm.shape[0], max_samples)
    start = 0
    while start < N:
        end = min(N, start + chunk_samples)
        yield start, np.asarray(mm[start:end, ch_idx])
        start = end

def iter_chunks_contiguous(mm, ch_idx: int, chunk_samples: int, max_samples: int | None):
    N = mm.shape[1] if max_samples is None else min(mm.shape[1], max_samples)
    start = 0
    while start < N:
        end = min(N, start + chunk_samples)
        yield start, np.asarray(mm[ch_idx, start:end])
        start = end

def main():
    with open("config.yml","r") as f:
        cfg = yaml.safe_load(f)

    data_dir   = cfg["data_dir"]
    files      = cfg["file_names"]
    fs         = int(cfg["sampling_rate_hz"])
    n_channels = int(cfg["n_channels"])
    dtype      = cfg["dtype"]
    le         = bool(cfg["little_endian"])
    header     = int(cfg["header_bytes"])
    layout     = cfg.get("layout","interleaved")
    chunk_s    = int(cfg["chunk_seconds"])
    cfar_cfg   = cfg["cfar"]
    out_dir    = cfg["hi_output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # NEW: speed controls (you can set in config.yml; defaults below)
    channels_to_process = cfg.get("channels_to_process", [4,5,6,7])  # S5,S6,S7,S8 by default
    max_seconds_per_pipe = cfg.get("max_seconds_per_pipe", 120)      # quick test; set to null for full run

    # CFAR parameters
    pfa = float(cfar_cfg["pfa"])
    cell_size = int(cfar_cfg["cell_size"])
    n_guard_cells = int(cfar_cfg["n_guard_cells"])
    n_train_cells = int(cfar_cfg["n_train_cells"])

    chunk_samples = fs * chunk_s
    max_samples = None if max_seconds_per_pipe in (None, "null") else int(fs * max_seconds_per_pipe)

    for fname in files:
        path = os.path.join(data_dir, fname)
        print(f"Opening {path} (layout={layout})")
        mm, samples_per_ch = read_wfs(path, n_channels, dtype, layout, le, header)

        per_ch = []
        for ch in channels_to_process:
            print(f"  Channel {ch}: building HI (chunk={chunk_s}s, cell={cell_size} samples)")
            if layout == "interleaved":
                stream = iter_chunks_interleaved(mm, ch, chunk_samples, max_samples)
            else:
                stream = iter_chunks_contiguous(mm, ch, chunk_samples, max_samples)

            df = build_hi_for_channel_cellwise(
                stream_iter=stream,
                fs_hz=fs,
                cell_size=cell_size,
                pfa=pfa,
                n_guard_cells=n_guard_cells,
                n_train_cells=n_train_cells,
            )
            df["sensor"] = ch
            per_ch.append(df)

        big = pd.concat(per_ch, ignore_index=True)
        big["pipe"] = os.path.splitext(os.path.basename(fname))[0]
        out_csv = os.path.join(out_dir, f"{big['pipe'].iloc[0]}_HI.csv")
        big.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

        if max_seconds_per_pipe not in (None, "null"):
            print(f"(NOTE) Stopped at {max_seconds_per_pipe}s for speed. "
                  f"Set max_seconds_per_pipe: null to process full run.")

if __name__ == "__main__":
    main()
