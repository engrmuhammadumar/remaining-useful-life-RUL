
import argparse, os, sys, math, struct, io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

def guess_layout(file_size, fs, n_channels, header_candidates=(0, 512, 1024, 4096, 8192, 65536),
                 dtypes=("int16","int32","float32")):
    """
    Return list of plausible (dtype, header_bytes, samples_per_channel) candidates sorted by plausibility.
    """
    out = []
    for dt in dtypes:
        itemsize = np.dtype(dt).itemsize
        for hb in header_candidates:
            if file_size <= hb: 
                continue
            payload = file_size - hb
            # samples per channel (integer check)
            denom = itemsize * n_channels
            if payload % denom != 0:
                continue
            n_samples = payload // denom
            # require at least 1 second of data to be plausible
            if n_samples < fs:
                continue
            # penalize large headers except when exact
            score = 0
            # Prefer int16 when total size ~100 GB at 1 MHz, 8 ch (common for AE)
            if dt == "int16":
                score += 2
            # Prefer small header
            score += -math.log2(hb + 1)
            out.append((dt, hb, int(n_samples), score))
    # sort by score desc, then by samples
    out.sort(key=lambda x: (-x[3], -x[2]))
    return out

def band_energy_ratio(x, fs, lo=80_000, hi=400_000, lowband_hi=5_000):
    """
    Compute ratio of band-limited energy (lo..hi) to very-low-frequency energy (0..lowband_hi).
    Uses Welch PSD for robustness on a snippet.
    """
    # Welch
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 8192))
    # avoid zero
    eps = 1e-20
    band = np.logical_and(f >= lo, f <= hi)
    low  = f <= lowband_hi
    e_band = np.trapz(Pxx[band], f[band]) + eps
    e_low  = np.trapz(Pxx[low],  f[low])  + eps
    return float(e_band / e_low)

def summarize_chunk(w, fs):
    """Return per-channel quick stats for a chunk array w [samples, channels]"""
    # basic stats
    rms = np.sqrt(np.mean(w*w, axis=0))
    kurt = (np.mean((w - np.mean(w, axis=0))**4, axis=0) / (np.var(w, axis=0)**2 + 1e-12))
    # adaptive threshold for burst counts
    mu = np.mean(np.abs(w), axis=0); sd = np.std(np.abs(w), axis=0)
    thr = mu + 5*sd
    counts = (np.abs(w) > thr).sum(axis=0)
    # band energy ratios to classify channels
    ratios = [band_energy_ratio(w[:,ch], fs) for ch in range(w.shape[1])]
    # label: AE-like if band ratio is high
    label = ["AE-like" if r > 0.3 else "low-freq/pressure-like" for r in ratios]
    return rms, kurt, counts, ratios, label

def main():
    ap = argparse.ArgumentParser(description="Inspect a large .wfs waveform file safely and infer layout.")
    ap.add_argument("--path", required=True, help="Path to .wfs (or other raw) file")
    ap.add_argument("--fs", type=int, default=1_000_000, help="Sampling rate (default 1 MHz)")
    ap.add_argument("--n_channels", type=int, default=8, help="Number of interleaved channels (default 8)")
    ap.add_argument("--dtype", type=str, default="", help="Force dtype: int16|int32|float32|float64")
    ap.add_argument("--header_bytes", type=int, default=-1, help="Force header byte size (default: auto guess)")
    ap.add_argument("--seconds", type=float, default=5.0, help="Seconds to analyze from the start (default 5 s)")
    ap.add_argument("--outdir", type=str, default="wfs_inspect_out", help="Directory to save summary files/plots")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"File not found: {p}", file=sys.stderr)
        sys.exit(1)

    file_size = p.stat().st_size
    print(f"[INFO] File: {p.name} | Size: {file_size/1e9:.3f} GB")

    # Guess layout if not forced
    if args.dtype and args.header_bytes >= 0:
        candidates = [(args.dtype, args.header_bytes, None, 0)]
    else:
        candidates = guess_layout(file_size, args.fs, args.n_channels)
        if args.dtype:
            candidates = [c for c in candidates if c[0] == args.dtype]
        if args.header_bytes >= 0:
            candidates = [c for c in candidates if c[1] == args.header_bytes]
        if not candidates:
            print("[ERROR] Could not find a plausible (dtype, header_bytes) combo. Try forcing --dtype/--header_bytes.", file=sys.stderr)
            sys.exit(2)

    print("[INFO] Candidate layouts (dtype, header_bytes, samples_per_channel):")
    for (dt, hb, n_samples, score) in candidates[:5]:
        dur = n_samples/args.fs if n_samples else float('nan')
        print(f"   - {dt}, header={hb} bytes, samples/ch={n_samples:,} (~{dur:.2f} s)")

    # Use the top candidate
    dtype, header_bytes, n_samples, _ = candidates[0]
    print(f"[INFO] Using layout → dtype={dtype}, header_bytes={header_bytes}, samples_per_channel={n_samples:,} (~{n_samples/args.fs:.2f} s)")

    # Memory map just the needed portion for a short analysis window
    itemsize = np.dtype(dtype).itemsize
    # ensure we don't exceed file bounds
    samples_to_read = int(min(n_samples, args.seconds * args.fs))
    count = samples_to_read * args.n_channels
    # memmap and reshape
    raw = np.memmap(p, dtype=dtype, mode="r", offset=header_bytes, shape=(n_samples*args.n_channels,))
    arr = np.array(raw[:count], dtype=np.float32).reshape(samples_to_read, args.n_channels)
    del raw  # close mapping asap

    # Quick per-channel stats
    rms, kurt, counts, ratios, label = summarize_chunk(arr, args.fs)
    df = pd.DataFrame({
        "channel": np.arange(1, args.n_channels+1),
        "rms": rms,
        "kurtosis": kurt,
        "burst_counts(k=5σ)": counts,
        "AE_band_ratio(100-400k / 0-5k)": ratios,
        "auto_label": label
    })

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{p.stem}_summary_first{int(args.seconds)}s.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved summary → {csv_path}")

    # Optional: save a tiny snippet for visual inspection (first 0.02 s)
    try:
        import matplotlib.pyplot as plt
        snippet_samples = min(int(0.02*args.fs), arr.shape[0])
        t = np.arange(snippet_samples)/args.fs
        for ch in range(args.n_channels):
            plt.figure()
            plt.plot(t, arr[:snippet_samples, ch])
            plt.xlabel("Time [s]"); plt.ylabel(f"Ch {ch+1} amplitude (raw units)")
            plt.title(f"{p.stem} - Channel {ch+1} (first 0.02 s)")
            fig_path = outdir / f"{p.stem}_ch{ch+1}_snippet.png"
            plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
        print(f"[OK] Saved tiny waveform PNGs in {outdir}")
    except Exception as e:
        print(f"[WARN] Could not create plots: {e}")

    # Heuristic sensor count summary
    n_ae = sum(1 for lbl in label if "AE-like" in lbl)
    n_low = args.n_channels - n_ae
    print(f"[SUMMARY] Heuristic channel types → AE-like: {n_ae}, low-freq/pressure-like: {n_low}")
    print(df)
    print("\nIf AE-like ≈ 4 and low-freq ≈ 2, that matches your schematic (4 AE + 2 pressure).")
    print("If AE-like ≈ 8, then all 8 are AE sensors; if AE-like ≈ 6, then 6 AE + 2 others, etc.")

if __name__ == "__main__":
    main()
