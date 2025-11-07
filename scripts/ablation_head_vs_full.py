"""Ablation: head-only vs full finetune on the demo synthetic dataset.

Runs two short experiments:
 - head-only: freeze backbone, train TinyDeepEnsemble head params only
 - full: allow backbone to update as well

Measures:
 - Negative log-likelihood (Gaussian: mean,var)
 - 1-sigma coverage (fraction of targets within sqrt(var))
 - inference latency (avg per-batch)

Saves results to `results/ablation_head_vs_full.json`.
"""
import os
import sys
import time
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ensure repo root available for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.demo_backbone import DemoBackbone
from src.core.ensembles import TinyDeepEnsemble


def load_demo_data():
    fn = os.path.join(REPO_ROOT, "data", "processed", "demo_synth.npz")
    if not os.path.exists(fn):
        raise FileNotFoundError("Run scripts/demo_prepare.py first")
    d = np.load(fn)
    X = d["X"].astype(np.float32)
    y = (d["X"].sum(axis=1) > 0).astype(np.float32)
    return X, y


def nll_gaussian(mean, var, target):
    # mean,var,target: numpy arrays
    eps = 1e-6
    var = var + eps
    return 0.5 * (np.log(var) + ((target - mean) ** 2) / var)


def run_experiment(train_backbone: bool, epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
    X, y = load_demo_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DemoBackbone(input_dim=X.shape[1], hidden=32, out_dim=1)
    model.to(device)
    tde = TinyDeepEnsemble(model, num_members=4)

    # init by running a batch (after model is on device)
    with torch.no_grad():
        tde(torch.from_numpy(X[:8]).to(device))

    tde.to(device)

    # choose params
    if not train_backbone:
        for p in tde.base_model.parameters():
            p.requires_grad = False

    optim = torch.optim.Adam([p for p in tde.parameters() if p.requires_grad], lr=lr)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # train loop (short)
    tde.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            mean, var = tde(xb)
            # mean: (batch,1), var: (batch,1)
            # use gaussian NLL
            loss = 0.5 * (torch.log(var + 1e-6) + (yb - mean) ** 2 / (var + 1e-6))
            loss = loss.mean()
            loss.backward()
            optim.step()

    # eval
    tde.eval()
    Xt = torch.from_numpy(X).to(device)
    # measure inference latency
    iters = 10
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            m, v = tde(Xt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    avg_latency = (t1 - t0) / iters

    mean_np = m.cpu().numpy().squeeze()
    var_np = v.cpu().numpy().squeeze()
    target = y

    nll = float(nll_gaussian(mean_np, var_np, target).mean())
    # coverage: fraction within 1 sigma
    sigma = np.sqrt(var_np)
    coverage = float(((np.abs(mean_np - target) <= sigma)).mean())

    return {"train_backbone": bool(train_backbone),
            "nll": float(nll),
            "coverage_1sigma": float(coverage),
            "avg_inference_latency_s": float(avg_latency)}


def main():
    results = {}
    for mode in [False, True]:
        print(f"Running experiment train_backbone={mode}")
        res = run_experiment(train_backbone=mode, epochs=5)
        results[f"train_backbone_{int(mode)}"] = res

    out_fn = os.path.join(REPO_ROOT, "results", "ablation_head_vs_full.json")
    with open(out_fn, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablation results -> {out_fn}")


if __name__ == "__main__":
    main()
