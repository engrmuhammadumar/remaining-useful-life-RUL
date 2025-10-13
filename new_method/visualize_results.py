# visualize_results.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional
from config import DEVICE, DA_FAIL
from train_eval import predict

def plot_da_with_uncertainty(mu, var, y, title="DA forecast with 95% uncertainty"):
    idx = np.arange(len(mu))
    std = torch.sqrt(torch.clamp(var, 1e-8))

    plt.figure(figsize=(10,4))
    lo = (mu - 1.959964*std).numpy()
    hi = (mu + 1.959964*std).numpy()
    plt.plot(idx, y.numpy(), label="Actual DA")
    plt.plot(idx, mu.numpy(), label="Predicted μ")
    plt.fill_between(idx, lo, hi, alpha=0.3, label="95% CI")
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Damage Accumulation (DA)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_failure_time(da_series, thr=0.9):
    for i, v in enumerate(da_series):
        if v >= thr:
            return i
    return None

def report_rul(mu, y, thr=DA_FAIL):
    mu_np = mu.numpy(); y_np = y.numpy()
    t_true = find_failure_time(y_np, thr=thr)
    t_pred = find_failure_time(mu_np, thr=thr)
    print(f"Failure threshold: {thr}")
    print(f"True failure index: {t_true}")
    print(f"Pred failure index: {t_pred}")
    if t_true is not None and t_pred is not None:
        print(f"RUL error (indices): {abs(t_true - t_pred)}")
