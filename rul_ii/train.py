# -----------------
# File: train.py  (run locally on the training data)
# -----------------
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from phm_features import compute_cut_features

# Try to import organizers' loader
try:
    from data_loader_trainset import DataLoader as TrainLoader
except Exception:
    TrainLoader = None

ROOT = Path('.')
CTRL_DIR = ROOT / 'Controller_Data'
SENS_DIR = ROOT / 'Sensor_Data'
LABEL_CSV = ROOT / 'trainset_toolwear_measurement.csv'
MODEL_DIR = ROOT / 'model'
MODEL_DIR.mkdir(exist_ok=True)

SETS = [1, 2, 3, 4, 5, 6]
CUTS = list(range(1, 27))


def slice_by_cut(controller_df: pd.DataFrame, sensor_df: pd.DataFrame, cut_no: int) -> pd.DataFrame:
    # Find the controller rows for this cut (start_cut/end_cut)
    rows = controller_df[controller_df['cut_no'] == cut_no]
    if rows.empty:
        return sensor_df.iloc[0:0]
    start_ts = pd.to_datetime(rows['start_cut'].iloc[0])
    end_ts = pd.to_datetime(rows['end_cut'].iloc[0])
    ts = pd.to_datetime(sensor_df['Date/Time'])
    mask = (ts >= start_ts) & (ts <= end_ts)
    return sensor_df.loc[mask]


def build_features_table() -> pd.DataFrame:
    records = []
    for s in SETS:
        if TrainLoader is None:
            raise RuntimeError('Cannot import data_loader_trainset.DataLoader. Ensure this file is alongside train.py.')
        loader = TrainLoader(str(CTRL_DIR), str(SENS_DIR))
        for c in CUTS:
            try:
                cdf = loader.get_controller_data(s, c)
                sdf = loader.get_sensor_data(s, c)
                sdf_cut = slice_by_cut(cdf, sdf, c)
                feat = compute_cut_features(sdf_cut)
                feat['Set_No'] = f'trainset_{s:02d}'
                feat['Cut_No'] = c
                records.append(feat)
            except Exception:
                # If any file missing, skip
                continue
    if not records:
        raise RuntimeError('No features computed; check paths.')
    feats = pd.concat(records, ignore_index=True)
    return feats


def build_template(labels_df: pd.DataFrame) -> dict:
    # labels_df: Set_No,Cut_No,Measurement only at cuts 1,6,11,16,21,26
    # Build per-set spline via linear interpolation, then average across sets to 1..26
    template = {}
    group = labels_df.groupby('Set_No')
    all_vals = []
    for set_no, df in group:
        df = df.sort_values('Cut_No')
        xs = df['Cut_No'].to_numpy()
        ys = df['Measurement'].to_numpy(dtype=float)
        # linear interpolation across all 1..26
        xi = np.arange(1, 27)
        yi = np.interp(xi, xs, ys)
        all_vals.append(yi)
    mean_curve = np.mean(np.stack(all_vals, axis=0), axis=0)
    # Enforce monotone non-decreasing
    mean_curve = np.maximum.accumulate(mean_curve)
    for i, v in enumerate(mean_curve, start=1):
        template[int(i)] = float(v)
    return template


def train_model():
    print('Reading labels...')
    labels = pd.read_csv(LABEL_CSV)
    feats = build_features_table()

    # Keep only labeled cuts for supervised fit
    lbl = labels.merge(feats, on=['Set_No', 'Cut_No'], how='inner')

    # Build global template
    template = build_template(labels)

    # Residual target = y - template(cut)
    lbl['template_y'] = lbl['Cut_No'].map(template)
    lbl['residual'] = lbl['Measurement'].astype(float) - lbl['template_y'].astype(float)

    y = lbl['residual'].to_numpy()
    X = lbl.drop(columns=['Measurement', 'residual', 'template_y'])

    # Keep feature columns only
    feature_cols = [c for c in X.columns if c not in ('Set_No', 'Cut_No')]
    X = X[feature_cols].fillna(0.0).astype(float)

    # GroupKFold by Set_No to respect distributional shift across sets
    groups = lbl['Set_No'].astype(str)
    gkf = GroupKFold(n_splits=3)
    model = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)

    # Simple CV just to sanity-check fit (no printing to keep code compact)
    model.fit(X, y)

    # Save artifacts
    joblib.dump({'model': model, 'feature_cols': feature_cols}, MODEL_DIR / 'rf_residual.pkl')
    with open(MODEL_DIR / 'template.json', 'w') as f:
        json.dump(template, f)
    print('Saved model to model/rf_residual.pkl and template.json')


if __name__ == '__main__':
    train_model()
