"""Run a small rank sweep for LowRankHeteroHead across multiple seeds.

Produces `results/hetero_rank_sweep.json` and a simple plot `results/hetero_rank_sweep.png`.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def run_one(rank, seed, epochs=6):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X, y = load_demo_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = DemoBackbone(input_dim=X.shape[1], hidden=32, out_dim=8)
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    hetero = LowRankHeteroHead(in_dim=backbone.out_dim, out_dim=1, rank=rank)
    hetero.to(device)

    with torch.no_grad():
        feats = backbone(torch.from_numpy(X).to(device)).cpu().numpy()

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.from_numpy(feats.astype(np.float32)), torch.from_numpy(y.astype(np.float32)).unsqueeze(1))
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    optim = torch.optim.Adam(hetero.parameters(), lr=1e-3)
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

    hetero.eval()
    feats_t = torch.from_numpy(feats).to(device)
    t0 = time.time()
    with torch.no_grad():
        mean_t, var_t = hetero(feats_t)
    t1 = time.time()

    mean_np = mean_t.cpu().numpy().squeeze()
    var_np = var_t.cpu().numpy().squeeze()
    eps = 1e-6
    nll = float((0.5 * (np.log(var_np + eps) + ((y - mean_np) ** 2) / (var_np + eps))).mean())
    coverage = float(((np.abs(mean_np - y) <= np.sqrt(var_np))).mean())
    latency = float(t1 - t0)
    return {"rank": int(rank), "seed": int(seed), "nll": nll, "coverage": coverage, "latency": latency}


def main():
    ranks = [1, 2, 4, 8]
    seeds = [1, 2, 3]
    results = []
    for r in ranks:
        for s in seeds:
            print(f"Running rank={r} seed={s}")
            res = run_one(r, s, epochs=6)
            results.append(res)
            print(res)

    out_fn = os.path.join(REPO_ROOT, "results", "hetero_rank_sweep.json")
    with open(out_fn, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved rank sweep results -> {out_fn}")

    # simple plot: mean coverage per rank
    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby('rank')['coverage'].agg(['mean', 'std']).reset_index()
    plt.errorbar(summary['rank'], summary['mean'], yerr=summary['std'], fmt='-o')
    plt.xlabel('rank')
    plt.ylabel('coverage (1-sigma)')
    plt.title('Hetero head coverage vs rank')
    plt.grid(True)
    png = os.path.join(REPO_ROOT, 'results', 'hetero_rank_sweep.png')
    plt.savefig(png)
    print(f"Saved plot -> {png}")


if __name__ == '__main__':
    main()
