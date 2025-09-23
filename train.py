# =========================
# config.py
# =========================
"""
Config for MonoSeq-RUL on PHM2010.
Edit BASE to your dataset root. Everything else can stay as-is to start.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import torch

# 1) BASIC PATHS & SPLITS
BASE = r"E:\\Collaboration Work\\With Farooq\\phm dataset\\PHM Challange 2010 Milling"
TRAIN_CUTTERS = ["c1", "c4", "c6"]  # labeled
TEST_CUTTERS  = ["c2", "c3", "c5"]  # unlabeled

ROOT_OUT = Path("artifacts_monoseq").resolve()
FEAT_DIR = ROOT_OUT / "features"
MODEL_DIR = ROOT_OUT / "models"
PLOT_DIR  = ROOT_OUT / "plots"
CSV_DIR   = ROOT_OUT / "csv"

# 2) SIGNAL / WINDOW SETTINGS
FS = 50_000
WIN = 4096
HOP = 2048
MAX_WINDOWS = 96

# 3) TRAINING SETTINGS
SEED = 42
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 35
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
VAL_RATIO = 0.15

# Backbone
HIDDEN = 256
DROPOUT = 0.2
NUM_LAYERS = 2

# Loss mixing
LAMBDA_SMOOTH_DELTA = 0.1
LAMBDA_PHASE = 0.0  # set 0.2 after we add phase labels

# Conformal (reserved)
CONF_ALPHA = 0.1

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    base: str = BASE
    train_cutters: list[str] = None
    test_cutters: list[str] = None
    fs: int = FS
    win: int = WIN
    hop: int = HOP
    max_windows: int = MAX_WINDOWS

    seed: int = SEED
    batch_size: int = BATCH_SIZE
    lr: float = LR
    epochs: int = EPOCHS
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    val_ratio: float = VAL_RATIO

    hidden: int = HIDDEN
    dropout: float = DROPOUT
    num_layers: int = NUM_LAYERS

    lambda_smooth_delta: float = LAMBDA_SMOOTH_DELTA
    lambda_phase: float = LAMBDA_PHASE

    conf_alpha: float = CONF_ALPHA

    root_out: Path = ROOT_OUT
    feat_dir: Path = FEAT_DIR
    model_dir: Path = MODEL_DIR
    plot_dir: Path = PLOT_DIR
    csv_dir: Path = CSV_DIR

    device: torch.device = DEVICE

    def __post_init__(self):
        if self.train_cutters is None:
            self.train_cutters = TRAIN_CUTTERS
        if self.test_cutters is None:
            self.test_cutters = TEST_CUTTERS

def make_dirs(cfg: Config):
    for p in [cfg.root_out, cfg.feat_dir, cfg.model_dir, cfg.plot_dir, cfg.csv_dir]:
        p.mkdir(parents=True, exist_ok=True)

def show(cfg: Config):
    print("\n=== MonoSeq-RUL Config ===")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
    print("Device:", cfg.device)
    print("==========================\n")

cfg = Config()
make_dirs(cfg)

# =========================
# data.py
# =========================
"""
Data utilities for PHM2010 MonoSeq-RUL.
- Reads wear CSVs for training cutters
- Discovers cut CSV files for each cutter
- Builds index lists (train/val/test)
"""
from __future__ import annotations
import os, re, glob
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from config import cfg

def read_wear_table(cutter_dir: str) -> Tuple[pd.DataFrame, float]:
    cands = [p for p in glob.glob(os.path.join(cutter_dir, "*.csv")) if "wear" in os.path.basename(p).lower()]
    if not cands:
        raise FileNotFoundError(f"No wear csv in {cutter_dir}")
    wear_file = cands[0]

    raw0 = pd.read_csv(wear_file, sep=None, engine="python", nrows=5)
    try:
        v = pd.to_numeric(raw0.iloc[0, 0], errors="coerce")
        use_header = bool(pd.isna(v))
    except Exception:
        use_header = True

    raw = (
        pd.read_csv(wear_file, sep=None, engine="python") if use_header
        else pd.read_csv(wear_file, sep=None, engine="python", header=None)
    )
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    def first_present(names):
        for n in names:
            if n in raw.columns:
                return n
        return None

    cut_col = first_present(["cut", "cut_number", "cut no", "cut_no", "c", "index", "id", "0"])
    f1_col  = first_present(["flute_1", "flute1", "f1", "flute 1", "1"])
    f2_col  = first_present(["flute_2", "flute2", "f2", "flute 2", "2"])
    f3_col  = first_present(["flute_3", "flute3", "f3", "flute 3", "3"])

    if cut_col is None or f1_col is None or f2_col is None or f3_col is None:
        tmp = raw.copy().dropna(axis=1, how="all")
        assert tmp.shape[1] >= 4, "Wear file must have >=4 usable columns"
        tmp.columns = [f"col_{i}" for i in range(tmp.shape[1])]
        cut_col, f1_col, f2_col, f3_col = "col_0", "col_1", "col_2", "col_3"
        raw = tmp

    cut_series = raw[cut_col].astype(str).str.extract(r"(\d+)", expand=False)
    cut_series = pd.to_numeric(cut_series, errors="coerce")

    f1 = pd.to_numeric(raw[f1_col], errors="coerce")
    f2 = pd.to_numeric(raw[f2_col], errors="coerce")
    f3 = pd.to_numeric(raw[f3_col], errors="coerce")

    df = pd.DataFrame({
        "cut_number": cut_series,
        "flute_1": f1, "flute_2": f2, "flute_3": f3
    }).dropna()
    df["cut_number"] = df["cut_number"].round().astype(int)

    df["wear_max"] = df[["flute_1", "flute_2", "flute_3"]].max(axis=1)
    EOL = float(df["wear_max"].max())

    eps = 1e-9
    df["f1_norm"] = df["flute_1"] / (EOL + eps)
    df["f2_norm"] = df["flute_2"] / (EOL + eps)
    df["f3_norm"] = df["flute_3"] / (EOL + eps)
    df["wear_norm"] = df["wear_max"] / (EOL + eps)
    df["rul_norm"]  = 1.0 - df["wear_norm"]
    df["RUL"] = EOL - df["wear_max"]

    return df.sort_values("cut_number").reset_index(drop=True), EOL


def discover_cut_files(cutter_dir: str, cutter_id: int) -> Dict[int, str]:
    all_csvs = glob.glob(os.path.join(cutter_dir, "**", "*.csv"), recursive=True)
    all_csvs = [p for p in all_csvs if "wear" not in os.path.basename(p).lower()]
    cuts = {}
    for p in all_csvs:
        name = os.path.basename(p).lower()
        m = re.search(rf"c[_-]?{cutter_id}[_-]?(\d+)\.csv$", name) or re.search(r"(\d+)\.csv$", name)
        if m:
            cuts[int(m.group(1))] = p
    return dict(sorted(cuts.items()))


def build_index_for_cutters(cutters: List[str], labeled: bool = True):
    index = []
    eol_map = {}
    for cname in cutters:
        cutter_dir = os.path.join(cfg.base, cname)
        cutter_id = int(re.findall(r"\d+", cname)[0])
        cut_files = discover_cut_files(cutter_dir, cutter_id)

        if labeled:
            wear_df, EOL = read_wear_table(cutter_dir)
            eol_map[cname] = EOL
            present = sorted(set(wear_df["cut_number"].astype(int)).intersection(cut_files.keys()))
            for cutn in present:
                row = wear_df.loc[wear_df["cut_number"] == cutn].iloc[0]
                y_norm = np.array([row["f1_norm"], row["f2_norm"], row["f3_norm"], row["wear_norm"], row["rul_norm"]], dtype=np.float32)
                y_raw  = np.array([row["flute_1"], row["flute_2"], row["flute_3"], row["wear_max"], row["RUL"]], dtype=np.float32)
                index.append({
                    "cutter": cname, "eol": EOL,
                    "cut_number": int(cutn),
                    "path": cut_files[int(cutn)],
                    "y_norm": y_norm, "y_raw": y_raw
                })
        else:
            present = sorted(cut_files.keys())
            for cutn in present:
                index.append({
                    "cutter": cname, "eol": None,
                    "cut_number": int(cutn),
                    "path": cut_files[int(cutn)],
                    "y_norm": None, "y_raw": None
                })
    return index, eol_map

# =========================
# features.py
# =========================
"""
Feature extraction for PHM2010 MonoSeq-RUL.
- Reads a single cut CSV (7 channels)
- Splits into windows (WIN, HOP)
- Computes per-window features (time + freq)
- Aggregates to fixed-length feature sequence (MAX_WINDOWS)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from config import cfg

_PER_CH_FEATURES = 9  # mean,std,min,max,skew,kurt,energy,psd_mean,psd_max


def _safe_read_csv(path: str) -> np.ndarray:
    try:
        df = pd.read_csv(path, header=None, engine="c", low_memory=False)
    except Exception:
        df = pd.read_csv(path, header=None, engine="python", low_memory=False)
    df = df.dropna(axis=1, how="all")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=0, how="any")
    return df.values.astype(np.float32)


def _feat_window(win: np.ndarray, fs: int = cfg.fs) -> np.ndarray:
    feats = []
    for i in range(win.shape[1]):
        x = win[:, i]
        feats += [x.mean(), x.std(), x.min(), x.max(), skew(x), kurtosis(x)]
        feats.append(float(np.sum(x**2) / max(len(x), 1)))
        f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
        feats.append(float(Pxx.mean()))
        feats.append(float(Pxx.max()))
    return np.array(feats, dtype=np.float32)


def extract_feature_sequence(path: str) -> np.ndarray:
    arr = _safe_read_csv(path)
    N, C = arr.shape
    feats = []
    if N >= cfg.win:
        for s in range(0, N - cfg.win + 1, cfg.hop):
            feats.append(_feat_window(arr[s:s+cfg.win]))
            if len(feats) >= cfg.max_windows:
                break
    else:
        # single window on the whole signal if shorter than WIN
        feats.append(_feat_window(arr))

    feats = np.stack(feats, axis=0)  # [T, 9*7=63]
    T = feats.shape[0]
    if T < cfg.max_windows:
        pad = np.zeros((cfg.max_windows - T, feats.shape[1]), dtype=np.float32)
        feats = np.vstack([feats, pad])
    else:
        feats = feats[:cfg.max_windows]
    return feats.astype(np.float32)

# =========================
# dataset.py
# =========================
"""
Dataset for MonoSeq-RUL.
- Wraps cut indices into sequences of window-features
- Provides normalized wear targets per time-step (replicated across windows)
- Batches to [B, T, F] for the model
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from config import cfg
from data import build_index_for_cutters
from features import extract_feature_sequence

class PHMSeqDataset(Dataset):
    def __init__(self, index, labeled: bool = True):
        self.index = index
        self.labeled = labeled

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        it = self.index[i]
        X = extract_feature_sequence(it["path"])  # [T,F]
        T, F = X.shape
        if self.labeled and it["y_norm"] is not None:
            wear_norm = np.float32(it["y_norm"][3])  # wear_norm scalar
            y_seq = np.full((T,), wear_norm, dtype=np.float32)
            eol = np.float32(it["eol"]) if it["eol"] is not None else np.float32(np.nan)
        else:
            y_seq = np.full((T,), np.nan, dtype=np.float32)
            eol = np.float32(np.nan)
        return (
            torch.tensor(X, dtype=torch.float32),  # [T,F]
            torch.tensor(y_seq, dtype=torch.float32),
            torch.tensor(eol, dtype=torch.float32),
            np.int32(it["cut_number"]),
            str(it["cutter"]),
        )

def collate_fn(batch):
    X, y, eol, cutn, cutter = zip(*batch)
    X = torch.stack(X)           # [B,T,F]
    y = torch.stack(y)           # [B,T]
    eol = torch.stack(eol)       # [B]
    return X, y, eol, np.array(cutn, dtype=int), np.array(cutter)

# =========================
# model.py
# =========================
"""
MonoSeq model: encoder + monotone increment head (cum-sum) + variance head.
Returns wear (normalized) over windows, its log-variance, phase logits (per-step), and increments.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = cfg.hidden):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden*2, hidden)

    def forward(self, x):  # x: [B,T,F]
        h, _ = self.rnn(x)
        h = F.relu(self.proj(h))  # [B,T,H]
        return h

class MonoSeqModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = cfg.hidden, n_phases: int = 3):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden)
        self.inc_raw = nn.Linear(hidden, 1)       # increments before softplus
        self.wear_logvar = nn.Linear(hidden, 1)   # log variance for wear
        self.phase_head = nn.Linear(hidden, n_phases)

    def forward(self, x):
        h = self.encoder(x)                   # [B,T,H]
        inc = F.softplus(self.inc_raw(h)).squeeze(-1)      # [B,T]  (nonnegative)
        wear = torch.cumsum(inc, dim=1)                     # [B,T]  (monotone)
        wear_logvar = self.wear_logvar(h).squeeze(-1)       # [B,T]
        phase_logits = self.phase_head(h)                   # [B,T,3]
        return wear, wear_logvar, phase_logits, inc

# =========================
# losses.py
# =========================
import torch
import torch.nn as nn

def nll_gaussian(y_pred, y_logvar, y_true, mask=None):
    if mask is None:
        mask = torch.ones_like(y_true)
    var = torch.exp(y_logvar)
    loss = 0.5 * (torch.log(var) + (y_true - y_pred) ** 2 / (var + 1e-9))
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-9)


def monotonic_smoothness_loss(increments, mask=None, lambda_smooth=0.1):
    if mask is None:
        mask = torch.ones_like(increments)
    neg_penalty = torch.relu(-increments) * mask
    diff = increments[:, 1:] - increments[:, :-1]
    smooth_penalty = diff.abs() * mask[:, 1:]
    loss = neg_penalty.sum() + lambda_smooth * smooth_penalty.sum()
    return loss / (mask.sum() + 1e-9)

# (phase loss reserved; set lambda_phase=0 for now)

# =========================
# train.py
# =========================
"""
End-to-end training + validation plots for MonoSeq-RUL (normalized wear).
Outputs:
- best model weights to models/best_monoseq.pt
- validation CSV + plots per cutter (wear & RUL in raw units)
- test predictions (approx, using median EOL of train) CSV + plots
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from config import cfg
from data import build_index_for_cutters
from features import extract_feature_sequence
from dataset import PHMSeqDataset, collate_fn
from model import MonoSeqModel
from losses import nll_gaussian, monotonic_smoothness_loss


def cutterwise_split(index, val_ratio):
    # last val_ratio of cuts per cutter -> validation
    train_idx, val_idx = [], []
    by_c = {}
    for i, it in enumerate(index):
        by_c.setdefault(it["cutter"], []).append(i)
    for c, idxs in by_c.items():
        # keep order by cut_number
        idxs = sorted(idxs, key=lambda k: index[k]["cut_number"])
        k = max(1, int(round(len(idxs)*val_ratio)))
        train_idx += idxs[:-k]
        val_idx   += idxs[-k:]
    return train_idx, val_idx


def plot_truth_pred(x, y_true, y_pred, title, ylab, path):
    plt.figure(figsize=(10,4))
    plt.plot(x, y_true, label="True", linewidth=2.2)
    plt.plot(x, y_pred, label="Pred", linestyle="--", linewidth=2.2)
    plt.title(title); plt.xlabel("Cut #"); plt.ylabel(ylab); plt.grid(True, ls='--', alpha=0.5)
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def main():
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    # --- Build indices ---
    train_index, train_eols = build_index_for_cutters(cfg.train_cutters, labeled=True)
    test_index,  _         = build_index_for_cutters(cfg.test_cutters,  labeled=False)
    print(f"Train cuts: {len(train_index)} | Test cuts: {len(test_index)}")

    # --- Split train/val by cutter tail ---
    tr_ids, va_ids = cutterwise_split(train_index, cfg.val_ratio)
    tr_index = [train_index[i] for i in tr_ids]
    va_index = [train_index[i] for i in va_ids]

    # --- Datasets ---
    train_ds = PHMSeqDataset(tr_index, labeled=True)
    val_ds   = PHMSeqDataset(va_index, labeled=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # --- Determine input dim ---
    Fdim = extract_feature_sequence(tr_index[0]["path"]).shape[1]

    # --- Model ---
    model = MonoSeqModel(input_dim=Fdim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    best = 1e18
    for ep in range(1, cfg.epochs+1):
        # Train
        model.train(); tr_loss = 0.0; n_tr = 0
        for X, y, eol, cutn, cutter in train_loader:
            X = X.to(cfg.device); y = y.to(cfg.device)
            opt.zero_grad()
            wear, wear_logvar, phase_logits, inc = model(X)
            mask = torch.isfinite(y).float()
            loss_w = nll_gaussian(wear, wear_logvar, y, mask)
            loss_m = monotonic_smoothness_loss(inc, mask, lambda_smooth=cfg.lambda_smooth_delta)
            loss = loss_w + loss_m
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_loss += float(loss.item()) * X.size(0); n_tr += X.size(0)
        tr_loss /= max(1, n_tr)

        # Val
        model.eval(); va_loss = 0.0; n_va = 0
        with torch.no_grad():
            for X, y, eol, cutn, cutter in val_loader:
                X = X.to(cfg.device); y = y.to(cfg.device)
                wear, wear_logvar, phase_logits, inc = model(X)
                mask = torch.isfinite(y).float()
                loss_w = nll_gaussian(wear, wear_logvar, y, mask)
                loss_m = monotonic_smoothness_loss(inc, mask, lambda_smooth=cfg.lambda_smooth_delta)
                loss = loss_w + loss_m
                va_loss += float(loss.item()) * X.size(0); n_va += X.size(0)
        va_loss /= max(1, n_va)
        sched.step(va_loss)

        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), (cfg.model_dir/"best_monoseq.pt").as_posix())
            print("  ↳ saved best")

    # --- Validation metrics/plots (raw units) ---
    model.eval()
    Yw=[]; Pw=[]; Yr=[]; Pr=[]; cc=[]; ct=[]
    with torch.no_grad():
        for X, y, eol, cutn, cutter in val_loader:
            X = X.to(cfg.device); y = y.to(cfg.device); eol = eol.numpy()
            wear, wear_logvar, _, _ = model(X)
            # use last time step prediction per cut
            wear_n = wear[:, -1].cpu().numpy()  # normalized
            wear_raw_pred = wear_n * eol
            rul_raw_pred  = (1.0 - wear_n) * eol
            # ground truth (constant along T)
            y_n = y[:, -1].cpu().numpy()
            wear_raw_true = y_n * eol
            rul_raw_true  = (1.0 - y_n) * eol
            Yw.append(wear_raw_true); Pw.append(wear_raw_pred)
            Yr.append(rul_raw_true);  Pr.append(rul_raw_pred)
            cc.append(cutn); ct.append(cutter)
    Yw = np.concatenate(Yw); Pw = np.concatenate(Pw)
    Yr = np.concatenate(Yr); Pr = np.concatenate(Pr)

    def rmse(a,b):
        return float(np.sqrt(np.mean((a-b)**2)))
    print(f"Val Wear RMSE {rmse(Yw,Pw):.2f} | R² {np.corrcoef(Yw,Pw)[0,1]**2:.3f}")
    print(f"Val  RUL RMSE {rmse(Yr,Pr):.2f} | R² {np.corrcoef(Yr,Pr)[0,1]**2:.3f}")

    # per-cutter plots on validation
    # rebuild a simple loader to access per-item info
    val_items = [va_index[i] for i in range(len(va_index))]
    by_c = {}
    for it in val_items:
        by_c.setdefault(it["cutter"], []).append(it)

    for cname, items in by_c.items():
        xs=[]; yt_w=[]; yp_w=[]; yt_r=[]; yp_r=[]
        for it in sorted(items, key=lambda t: t["cut_number"]):
            X = extract_feature_sequence(it["path"])  # [T,F]
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(cfg.device)
            with torch.no_grad():
                wear, wear_logvar, _, _ = model(X_t)
                wn = wear[:, -1].cpu().numpy()[0]
            eol = it["eol"]
            y_n = float(it["y_norm"][3])
            xs.append(it["cut_number"]) 
            yp_w.append(wn*eol); yp_r.append((1.0-wn)*eol)
            yt_w.append(y_n*eol); yt_r.append((1.0-y_n)*eol)
        plot_truth_pred(xs, yt_w, yp_w, f"{cname} — Validation Wear", "Wear (0.001 mm)", (cfg.plot_dir/f"val_{cname}_wear.png").as_posix())
        plot_truth_pred(xs, yt_r, yp_r, f"{cname} — Validation RUL",  "RUL (wear units)", (cfg.plot_dir/f"val_{cname}_rul.png").as_posix())

    # --- Test predictions (approx: use median EOL) ---
    EOL_REF = float(np.median(list(train_eols.values()))) if len(train_eols)>0 else 1.0
    by_ct = {}
    for it in test_index:
        by_ct.setdefault(it["cutter"], []).append(it)
    import csv
    with open((cfg.csv_dir/"test_predictions.csv").as_posix(), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["cutter","cut_number","wear_pred","rul_pred"])
        for cname, items in by_ct.items():
            xs=[]; yp_w=[]; yp_r=[]
            for it in sorted(items, key=lambda t: t["cut_number"]):
                X = extract_feature_sequence(it["path"])  # [T,F]
                X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    wear, _, _, _ = model(X_t)
                    wn = wear[:, -1].cpu().numpy()[0]
                xs.append(it["cut_number"]) 
                yp_w.append(wn*EOL_REF); yp_r.append((1.0-wn)*EOL_REF)
                w.writerow([cname, it["cut_number"], yp_w[-1], yp_r[-1]])
            plot_truth_pred(xs, yp_w, yp_w, f"{cname} — Test Wear (pred)", "Wear (0.001 mm)", (cfg.plot_dir/f"test_{cname}_wear.png").as_posix())
            plot_truth_pred(xs, yp_r, yp_r, f"{cname} — Test RUL (pred)",  "RUL (wear units)", (cfg.plot_dir/f"test_{cname}_rul.png").as_posix())

if __name__ == "__main__":
    main()
