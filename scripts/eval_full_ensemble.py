"""Evaluate a saved full-ensemble checkpoint on demo inputs.

Usage:
  python scripts/eval_full_ensemble.py --ckpt results/full_ensemble_4x_ep5.pth --out results/full_ensemble_eval.json

This script loads the saved checkpoint containing member state_dicts, runs each
member on the demo synthetic inputs, computes predictive mean and variance per-sample,
and writes `results/full_ensemble_outputs.npz` with mean/var and calls the
evaluation utilities to write a standardized JSON.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

# ensure repo root in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.trd_uq_eval import summarize_predictions, write_results_file


def load_demo_data():
    fn = os.path.join(REPO_ROOT, 'data', 'processed', 'demo_synth.npz')
    if not os.path.exists(fn):
        raise FileNotFoundError('Run scripts/demo_prepare.py first')
    d = np.load(fn)
    return d['X'], d['y']


def build_member(input_dim):
    # small MLP matching train_full_ensemble.SmallMLP
    from torch import nn

    class SmallMLP(nn.Module):
        def __init__(self, input_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return SmallMLP(input_dim)


def main(ckpt_path: str, out_json: str = 'results/full_ensemble_eval.json', device: str = 'cpu'):
    X, y = load_demo_data()
    X_t = torch.from_numpy(X).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    members_sd = ckpt.get('members') if isinstance(ckpt, dict) and 'members' in ckpt else None
    if members_sd is None:
        raise ValueError('Checkpoint must contain a "members" list of state_dicts')

    input_dim = X.shape[1]
    preds = []
    for i, sd in enumerate(members_sd):
        m = build_member(input_dim).to(device)
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            p = m(X_t).cpu().numpy()
        preds.append(p)

    preds = np.stack(preds, axis=0)  # (members, n)
    mean = preds.mean(axis=0)
    var = preds.var(axis=0, ddof=0)

    # save outputs
    out_npz = os.path.join(os.path.dirname(out_json), 'full_ensemble_outputs.npz')
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    np.savez(out_npz, mean=mean, var=var)
    print('Saved ensemble outputs ->', out_npz)

    metrics = summarize_predictions(mean, var, y[:mean.shape[0]])
    run_meta = {'ckpt': os.path.basename(ckpt_path), 'n_members': int(preds.shape[0])}
    write_results_file(out_json, run_meta, metrics)
    print('Wrote evaluation ->', out_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default='results/full_ensemble_eval.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    main(args.ckpt, args.out, device=args.device)
