# ==== Concrete AE Run-to-Failure Dataset Quick Scan ====
# Run this in a local Python environment (e.g., Jupyter, VSCode, or plain Python).
# Requires: pandas, numpy, scikit-learn, matplotlib

import os
import re
import json
import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------- CONFIG ---------------------
DATA_PATH = r"F:\concrete data\test 3\per_file_features_800.csv"  # your path
SAVE_DIR = Path(DATA_PATH).parent
PROFILE_MD = SAVE_DIR / "data_profile.md"

# If your file is delimited by semicolons or tabs, set sep accordingly
READ_CSV_KW = dict(sep=",", low_memory=False)

# For memory safety with very wide data
MAX_COLS_FOR_CORR = 50   # compute correlations only on the top-N most variable features
MAX_PREVIEW_COLS = 20    # limit preview width

# Potential name hints (edit if your headers are different)
ID_HINTS = ["id", "specimen", "sample", "file", "filename"]
TIME_HINTS = ["time", "ts", "timestamp", "cycle", "step", "index"]
TARGET_HINTS = ["rul", "label", "target", "damage", "failure", "life"]

# --------------------- LOAD ---------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find file at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, **READ_CSV_KW)

print("\n=== BASIC SHAPE ===")
print(df.shape)  # (rows, cols)

print("\n=== COLUMNS (first 50) ===")
print(list(df.columns[:50]))

print("\n=== DTYPE SUMMARY ===")
print(df.dtypes.value_counts())

# --------------------- QUICK STATS ---------------------
print("\n=== HEAD ===")
with pd.option_context("display.max_columns", MAX_PREVIEW_COLS):
    print(df.head(3))

print("\n=== TAIL ===")
with pd.option_context("display.max_columns", MAX_PREVIEW_COLS):
    print(df.tail(3))

print("\n=== NULL COUNTS (top 30) ===")
nulls = df.isna().sum().sort_values(ascending=False)
print(nulls.head(30))

# --------------------- HEURISTICS: ID/TIME/TARGET ---------------------
cols_lower = {c: c.lower() for c in df.columns}

def pick_first_by_hints(hints):
    for c in df.columns:
        cl = c.lower()
        if any(h in cl for h in hints):
            return c
    return None

id_col = pick_first_by_hints(ID_HINTS)
time_col = pick_first_by_hints(TIME_HINTS)
target_col = pick_first_by_hints(TARGET_HINTS)

print("\n=== HEURISTIC COLUMN GUESSES ===")
print(f"id_col     : {id_col}")
print(f"time_col   : {time_col}")
print(f"target_col : {target_col}  (check if this is RUL/damage/failure)")

# --------------------- DUPLICATES & CONSTANTS ---------------------
dup_rows = df.duplicated().sum()
print(f"\n=== DUPLICATE ROWS ===\n{dup_rows}")

nunique = df.nunique(dropna=True)
constant_cols = nunique[nunique <= 1].index.tolist()
print(f"\n=== CONSTANT/NEAR-CONSTANT FEATURES ===\nCount: {len(constant_cols)}")
if len(constant_cols) > 0:
    print(constant_cols[:30], "..." if len(constant_cols) > 30 else "")

# --------------------- FEATURE GROUPS (e.g., AE_ prefixed) ---------------------
prefix_counts = {}
for c in df.columns:
    m = re.match(r"([A-Za-z]+)[_\-].*", c)
    if m:
        prefix = m.group(1)
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

if prefix_counts:
    print("\n=== FEATURE GROUP PREFIX COUNTS (top 20) ===")
    for k, v in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{k}: {v}")

# --------------------- MISSINGNESS MAP (lightweight) ---------------------
missing_ratio = df.isna().mean().sort_values(ascending=False)
print("\n=== MISSING RATIO (top 30) ===")
print(missing_ratio.head(30))

# --------------------- TARGET INSPECTION (if present) ---------------------
def safe_describe(series):
    try:
        return series.describe()
    except Exception as e:
        return f"unable to describe: {e}"

target_summary = None
if target_col and target_col in df.columns:
    print(f"\n=== TARGET SUMMARY: {target_col} ===")
    print(safe_describe(df[target_col]))
    if pd.api.types.is_numeric_dtype(df[target_col]):
        # Plot target distribution
        plt.figure(figsize=(7,5))
        series = df[target_col].dropna()
        # simple histogram
        plt.hist(series, bins=50)
        plt.title(f"Distribution of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    else:
        # categorical breakdown
        print("\nValue counts (top 20):")
        print(df[target_col].value_counts().head(20))

# --------------------- TIME/CYCLE INSPECTION (if present) ---------------------
if time_col and time_col in df.columns:
    # Try parse datetimes if it smells like a timestamp
    if "time" in time_col.lower() or "timestamp" in time_col.lower():
        try:
            ts = pd.to_datetime(df[time_col], errors="coerce")
            print(f"\nParsed {time_col} to datetime; NaT count = {ts.isna().sum()}")
        except Exception as e:
            print(f"\nCould not parse {time_col} as datetime: {e}")

    # Basic monotonic check per id (if id present)
    if id_col and id_col in df.columns and pd.api.types.is_numeric_dtype(df.index):
        # Group monotonicity check
        try:
            grp = df.groupby(id_col)[time_col].apply(lambda s: s.is_monotonic_increasing or s.is_monotonic_decreasing)
            print(f"\n=== MONOTONIC TIME PER {id_col} (first 10) ===")
            print(grp.head(10))
        except Exception as e:
            print(f"\nMonotonic check failed: {e}")

# --------------------- NUMERIC FEATURE SUMMARY ---------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n=== NUMERIC FEATURES COUNT === {len(num_cols)}")
if len(num_cols) > 0:
    # Basic stats for a subset (avoid printing 800 cols)
    print("\n=== NUMERIC SUMMARY (first 20 features) ===")
    print(df[num_cols[:20]].describe().T)

# --------------------- CORRELATION SNAPSHOT ---------------------
corr_cols = num_cols
if len(num_cols) > MAX_COLS_FOR_CORR:
    # choose top-N by variance
    variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
    corr_cols = list(variances.head(MAX_COLS_FOR_CORR).index)
    print(f"\nUsing top-{MAX_COLS_FOR_CORR} most variable features for corr snapshot.")

if len(corr_cols) >= 2:
    corr = df[corr_cols].corr(numeric_only=True)
    # Very lightweight: report the strongest pairs
    tril_idx = np.tril_indices_from(corr, k=-1)
    pairs = []
    for i, j in zip(tril_idx[0], tril_idx[1]):
        pairs.append((corr.index[i], corr.columns[j], corr.values[i, j]))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:20]
    print("\n=== TOP 20 CORRELATED PAIRS (abs value) ===")
    for a, b, r in pairs_sorted:
        print(f"{a} ~ {b}: r={r:.3f}")

# --------------------- PCA SNAPSHOT ---------------------
# PCA can show structure even with many features
if len(num_cols) >= 2:
    try:
        # drop rows with NA for PCA
        df_num = df[num_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        # standardize
        X = StandardScaler().fit_transform(df_num.values)
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(X)

        plt.figure(figsize=(6,5))
        plt.scatter(Z[:,0], Z[:,1], s=6, alpha=0.6)
        plt.title("PCA (2D) of numeric features")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

        print("\nPCA explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)
    except Exception as e:
        print(f"\nPCA failed: {e}")

# --------------------- RUL/FAILURE CONSISTENCY CHECK (light) ---------------------
# If target looks like RUL (numeric, nonnegative), show min/max and a few quantiles
if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
    tgt = df[target_col].dropna()
    if (tgt >= 0).mean() > 0.95:  # mostly nonnegative
        print(f"\n=== {target_col} (RUL-like) Quick Stats ===")
        qs = tgt.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
        print(qs)

# --------------------- PER-ID TRAJECTORY PREVIEW ---------------------
# If we have id/time & a numeric target, plot a couple trajectories
def plot_example_trajectories(df, id_col, time_col, target_col, max_ids=3):
    try:
        ids = df[id_col].dropna().unique()[:max_ids]
        for uid in ids:
            sub = df[df[id_col] == uid].copy()
            if time_col in sub.columns:
                # sort by time (numeric or datetime)
                try:
                    sub["_t"] = pd.to_datetime(sub[time_col], errors="coerce")
                    if sub["_t"].isna().all():
                        sub["_t"] = sub[time_col]
                except:
                    sub["_t"] = sub[time_col]
                sub = sub.sort_values(by="_t")
            if target_col in sub.columns and pd.api.types.is_numeric_dtype(sub[target_col]):
                plt.figure(figsize=(7,4))
                plt.plot(sub["_t"] if "_t" in sub else np.arange(len(sub)), sub[target_col], marker=".")
                plt.title(f"{target_col} trajectory for {id_col}={uid}")
                plt.xlabel(time_col if time_col else "index")
                plt.ylabel(target_col)
                plt.tight_layout()
                plt.show()
    except Exception as e:
        print(f"Trajectory preview failed: {e}")

if id_col and target_col:
    plot_example_trajectories(df, id_col, time_col if time_col in df.columns else None, target_col, max_ids=3)

# --------------------- WRITE PROFILE SUMMARY ---------------------
def md_line(s=""):
    return s + "\n"

def md_bullet_list(items):
    return "".join([f"- {x}\n" for x in items])

summary = []
summary.append(md_line("# Concrete AE Dataset: Quick Profile"))
summary.append(md_line(f"**File:** `{DATA_PATH}`"))
summary.append(md_line(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns"))
summary.append(md_line("\n## Heuristic Columns"))
summary.append(md_line(f"- ID: `{id_col}`"))
summary.append(md_line(f"- TIME: `{time_col}`"))
summary.append(md_line(f"- TARGET: `{target_col}`"))
summary.append(md_line("\n## Types"))
dtype_counts = df.dtypes.value_counts()
summary.append(md_line(md_bullet_list([f"`{k}`: {v}" for k, v in dtype_counts.items()])))

summary.append(md_line("\n## Missingness (Top 20)"))
summary.append(md_line(md_bullet_list([f"`{k}`: {v} NAs" for k, v in nulls.head(20).items()])))

summary.append(md_line("\n## Constant/Low-Var Features"))
summary.append(md_line(f"- Count: {len(constant_cols)}"))
summary.append(md_line(md_bullet_list(constant_cols[:50])))

summary.append(md_line("\n## Numeric Feature Count"))
summary.append(md_line(f"- {len(num_cols)}"))

if prefix_counts:
    summary.append(md_line("\n## Feature Group Prefix Counts (Top 20)"))
    for k, v in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        summary.append(md_line(f"- {k}: {v}"))

with open(PROFILE_MD, "w", encoding="utf-8") as f:
    f.write("".join(summary))

print(f"\nWrote summary: {PROFILE_MD}")
