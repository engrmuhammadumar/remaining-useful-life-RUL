# losses_metrics.py
import math
import torch

def nig_nll(y, mu, v, alpha, beta):
    two_beta_v = 2*beta*(1+v)
    nll = 0.5*torch.log(math.pi/v) \
          - alpha*torch.log(two_beta_v) \
          + (alpha+0.5)*torch.log(v*(y-mu)**2 + two_beta_v) \
          + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)
    return nll

def evidence_regularizer(y, mu, v, alpha, beta):
    err = torch.abs(y - mu)
    return err * (2*v + alpha)

def evidential_loss(y, mu, v, alpha, beta, lam=1.0):
    return nig_nll(y, mu, v, alpha, beta).mean() + lam * evidence_regularizer(y, mu, v, alpha, beta).mean()

def rmse(a,b):
    return float(torch.sqrt(torch.mean((a-b)**2)).item())

def mape(a,b):
    return float((torch.mean(torch.abs((a-b) / (a.abs()+1e-6))).item()))

def picp(y, mu, var, q=0.95):
    std = torch.sqrt(torch.clamp(var, 1e-8))
    z = torch.tensor(1.959964, device=mu.device) if q==0.95 else torch.tensor(1.644854, device=mu.device)
    lo = mu - z*std; hi = mu + z*std
    inside = ((y >= lo) & (y <= hi)).float().mean().item()
    return inside
