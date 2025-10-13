# train_eval.py
import json
import torch
from typing import Dict
from config import DEVICE, LAMBDA
from losses_metrics import evidential_loss, rmse, mape, picp

def train_one_epoch(model, dl, A, opt) -> float:
    model.train(); total = 0.0
    for x, yN in dl:
        y = yN.mean(dim=1)        # average DA across sensors → global target
        x = x.to(DEVICE); y = y.to(DEVICE)
        mu, v, alpha, beta = model(x, A)
        loss = evidential_loss(y, mu, v, alpha, beta, LAMBDA)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def predict(model, dl, A):
    model.eval()
    pred_mu, pred_var, y_true = [], [], []
    for x, yN in dl:
        y = yN.mean(dim=1)
        x = x.to(DEVICE)
        mu, v, alpha, beta = model(x, A)
        var = beta / (v * (alpha - 1 + 1e-6))  # approximate predictive variance
        pred_mu.append(mu.cpu()); pred_var.append(var.cpu()); y_true.append(y.cpu())
    return torch.cat(pred_mu), torch.cat(pred_var), torch.cat(y_true)

def evaluate(model, lte, A) -> Dict:
    mu, var, y = predict(model, lte, A)
    return {
        "rmse": rmse(mu, y),
        "mape": mape(mu, y),
        "picp95": picp(y, mu, var, q=0.95),
        "mu": mu,
        "var": var,
        "y": y,
    }

def save_best(model, metrics, path="stge_rul_best.pt"):
    torch.save({"model": model.state_dict()}, path)
    best = {k: float(v) for k,v in metrics.items() if isinstance(v, (int, float))}
    with open("stge_rul_metrics.json", "w") as f:
        json.dump(best, f, indent=2)
