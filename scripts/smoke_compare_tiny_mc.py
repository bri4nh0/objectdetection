import os
import sys
import time
import uuid
import numpy as np
import torch
from torch import nn

# ensure repo src importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.core.ensembles import TinyDeepEnsemble
from src.core.multimodal import FusionMLP
from src.core.uq_baselines import MCDropoutWrapper
from src.utils.metadata import save_metadata


def ece_score(probs, labels, n_bins=10):
    # simple ECE for binary/regression-normalized scores in [0,1]
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / probs.size) * abs(acc - conf)
    return ece


def bench(model, inp, runs=200, device='cpu'):
    model.to(device)
    inp = inp.to(device)
    # warmup
    for _ in range(10):
        _ = model(inp)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = None
    for _ in range(runs):
        out = model(inp)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / runs, out


def run_smoke():
    run_id = 'smoke_' + uuid.uuid4().hex[:8]
    results = {}
    # load FusionMLP
    model = FusionMLP(input_size=3)
    model_path = 'models/fusion/fusion_mlp_balanced.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print('Loaded FusionMLP weights')
        except Exception as e:
            print('Failed loading weights, continuing with fresh model:', e)

    # synthetic test set: 100 random samples, targets are binary via sigmoid
    X = torch.randn(100, 3)
    with torch.no_grad():
        raw = model(X).squeeze().numpy()
    probs = 1 / (1 + np.exp(-raw))
    # synthetic binary labels with noise
    labels = (probs + 0.1 * np.random.randn(*probs.shape) > 0.5).astype(float)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TinyEnsemble
    td = TinyDeepEnsemble(model, num_members=4)
    td_time, td_out = bench(td, X, runs=200, device=device)
    td_mean, td_var = td_out
    td_probs = 1 / (1 + np.exp(-td_mean.detach().cpu().numpy().flatten()))
    td_ece = ece_score(td_probs, labels)

    # MC-Dropout (T=30)
    mc = MCDropoutWrapper(model, T=30)
    mc_time, mc_out = bench(mc, X, runs=50, device=device)
    mc_mean, mc_var = mc_out
    mc_probs = 1 / (1 + np.exp(-mc_mean.detach().cpu().numpy().flatten()))
    mc_ece = ece_score(mc_probs, labels)

    results['td_time'] = td_time
    results['td_ece'] = float(td_ece)
    results['mc_time'] = mc_time
    results['mc_ece'] = float(mc_ece)
    results['device'] = device

    os.makedirs('results', exist_ok=True)
    out_file = os.path.join('results', f'{run_id}_smoke_results.json')
    import json
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    # save metadata
    save_metadata(run_id, repo_root, config={'test_samples': 100, 'td_members': 4, 'mc_T': 30}, models={'fusion_mlp': model_path if os.path.exists(model_path) else ''}, out_path='results')

    print('SMOKE RESULTS:', results)
    print('Metadata saved for run:', run_id)


if __name__ == '__main__':
    run_smoke()
