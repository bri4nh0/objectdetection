"""Compare TinyDeepEnsemble, MC-dropout and a simple full ensemble on the demo dataset.

Outputs `results/compare_baselines.json` and `results/compare_baselines.png`.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.demo_backbone import DemoBackbone
from src.core.ensembles import TinyDeepEnsemble
try:
    from src.core.uq_baselines import MCDropoutWrapper
except Exception:
    # fallback if module under different path
    from core.uq_baselines import MCDropoutWrapper  # type: ignore


def load_demo_data():
    fn = os.path.join(REPO_ROOT, "data", "processed", "demo_synth.npz")
    if not os.path.exists(fn):
        raise FileNotFoundError("Run scripts/demo_prepare.py first")
    d = np.load(fn)
    X = d["X"].astype(np.float32)
    y = (d["X"].sum(axis=1) > 0).astype(np.float32)
    return X, y


def eval_tde(model, X, device):
    model.eval()
    with torch.no_grad():
        m, v = model(torch.from_numpy(X).to(device))
    return m.cpu().numpy().squeeze(), v.cpu().numpy().squeeze()


def eval_mc(model, X, device, samples=16):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            p = model(torch.from_numpy(X).to(device))[0]
            preds.append(p.cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0).squeeze()
    var = preds.var(axis=0).squeeze()
    return mean, var


def eval_full_ensemble(num_members, X, device):
    # simple full ensemble: independent tiny backbones
    members = [DemoBackbone(input_dim=X.shape[1], hidden=32, out_dim=1).to(device) for _ in range(num_members)]
    # init by running one batch
    Xt = torch.from_numpy(X[:8]).to(device)
    with torch.no_grad():
        for m in members:
            _ = m(Xt)

    preds = []
    with torch.no_grad():
        for m in members:
            p = m(torch.from_numpy(X).to(device)).cpu().numpy()
            preds.append(p)
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0).squeeze()
    var = preds.var(axis=0).squeeze()
    return mean, var


def nll(mean, var, target):
    eps = 1e-6
    return 0.5 * (np.log(var + eps) + ((target - mean) ** 2) / (var + eps))


def main():
    X, y = load_demo_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TinyDE
    backbone = DemoBackbone(input_dim=X.shape[1], hidden=32, out_dim=1).to(device)
    tde = TinyDeepEnsemble(backbone, num_members=4).to(device)
    with torch.no_grad():
        tde(torch.from_numpy(X[:8]).to(device))
    mean_tde, var_tde = eval_tde(tde, X, device)

    # MC-dropout: wrap backbone with dropout in eval
    mc_model = MCDropoutWrapper(backbone, T=16).to(device)
    mean_mc, var_mc = eval_mc(mc_model, X, device, samples=16)

    # Full ensemble
    mean_full, var_full = eval_full_ensemble(4, X, device)

    res = {}
    for name, (m, v) in [('TinyDE', (mean_tde, var_tde)), ('MC', (mean_mc, var_mc)), ('Full', (mean_full, var_full))]:
        nlls = nll(m, v, y)
        res[name] = {
            'nll_mean': float(nlls.mean()),
            'coverage_1sigma': float(((np.abs(m - y) <= np.sqrt(v)).mean())),
        }

    out_fn = os.path.join(REPO_ROOT, 'results', 'compare_baselines.json')
    with open(out_fn, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved compare baselines -> {out_fn}")

    # plot
    labels = list(res.keys())
    nll_vals = [res[k]['nll_mean'] for k in labels]
    cov_vals = [res[k]['coverage_1sigma'] for k in labels]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar(labels, nll_vals, alpha=0.6)
    ax2.plot(labels, cov_vals, '-o', color='C1')
    ax1.set_ylabel('NLL')
    ax2.set_ylabel('Coverage (1-sigma)')
    ax1.set_title('Baseline comparison')
    plt.savefig(os.path.join(REPO_ROOT, 'results', 'compare_baselines.png'))
    print('Saved plot -> results/compare_baselines.png')


if __name__ == '__main__':
    main()
