"""Experiment: evaluate LowRankHeteroHead as an output head (head-only finetune).

Trains the hetero head (keeping backbone fixed) on demo synthetic data and
evaluates NLL, coverage, and latency. Saves results to `results/hetero_head_results.json`.
"""
import os
import sys
import time
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.demo_backbone import DemoBackbone
from src.core.hetero_heads import LowRankHeteroHead


def load_demo_data():
    fn = os.path.join(REPO_ROOT, "data", "processed", "demo_synth.npz")
    if not os.path.exists(fn):
        raise FileNotFoundError("Run scripts/demo_prepare.py first")
    d = np.load(fn)
    X = d["X"].astype(np.float32)
    y = (d["X"].sum(axis=1) > 0).astype(np.float32)
    return X, y


def nll_gaussian(mean, var, target):
    eps = 1e-6
    var = var + eps
    return 0.5 * (np.log(var) + ((target - mean) ** 2) / var)


def main(rank: int = 4, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
    X, y = load_demo_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load backbone and freeze
    backbone = DemoBackbone(input_dim=X.shape[1], hidden=32, out_dim=8)
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    # create hetero head mapping backbone output -> scalar mean,var
    hetero = LowRankHeteroHead(in_dim=backbone.out_dim, out_dim=1, rank=rank)
    hetero.to(device)

    # simple dataset: run backbone to get features
    with torch.no_grad():
        feats = backbone(torch.from_numpy(X).to(device)).cpu().numpy()

    ds = TensorDataset(torch.from_numpy(feats.astype(np.float32)), torch.from_numpy(y.astype(np.float32)).unsqueeze(1))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optim = torch.optim.Adam(hetero.parameters(), lr=lr)

    hetero.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            mean, var = hetero(xb)
            loss = 0.5 * (torch.log(var + 1e-6) + (yb - mean) ** 2 / (var + 1e-6))
            loss = loss.mean()
            loss.backward()
            optim.step()

    # eval on full dataset
    hetero.eval()
    feats_t = torch.from_numpy(feats).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        mean_t, var_t = hetero(feats_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    latency = (t1 - t0)

    mean_np = mean_t.cpu().numpy().squeeze()
    var_np = var_t.cpu().numpy().squeeze()

    nll = float(nll_gaussian(mean_np, var_np, y).mean())
    sigma = np.sqrt(var_np)
    coverage = float(((np.abs(mean_np - y) <= sigma)).mean())

    out = {
        "rank": int(rank),
        "nll": nll,
        "coverage_1sigma": coverage,
        "eval_latency_s": float(latency),
    }

    out_fn = os.path.join(REPO_ROOT, "results", "hetero_head_results.json")
    with open(out_fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved hetero head results -> {out_fn}")


if __name__ == "__main__":
    main()
