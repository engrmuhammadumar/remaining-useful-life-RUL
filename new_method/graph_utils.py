# graph_utils.py
import numpy as np
import torch
from typing import List
from config import SENSOR_COORDS, RBF_SIGMA, A_THRESH

def build_adj(sensors: List[str]) -> torch.Tensor:
    """RBF adjacency based on sensor coordinates, row-normalized."""
    pts = []
    for s in sensors:
        if s in SENSOR_COORDS:
            pts.append(SENSOR_COORDS[s])
        else:
            # fallback: place on a line using numeric part if available
            try:
                idx = int(s)
            except:
                idx = len(pts)
            pts.append((0.2*idx, 0.0))
    P = np.array(pts, dtype=np.float32)  # [N,2]
    dists = np.linalg.norm(P[None, :, :] - P[:, None, :], axis=-1)  # [N,N]

    A = np.exp(- (dists**2) / (2*RBF_SIGMA**2))
    np.fill_diagonal(A, 1.0)
    A[A < A_THRESH] = 0.0
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return torch.tensor(A, dtype=torch.float32)
