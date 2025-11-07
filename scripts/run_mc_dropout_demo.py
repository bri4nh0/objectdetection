"""Run MC-Dropout on the demo data and save mean/var outputs.

Saves outputs to the `results` directory as an .npz file.
"""
import os
import sys
import argparse
import json
import numpy as np
import torch

# ensure repo root in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.demo_backbone import DemoBackbone
from src.core.uq_baselines import MCDropoutWrapper


def main(T=50, device='cpu', out=None):
    data_fn = os.path.join(REPO_ROOT, 'data', 'processed', 'demo_synth.npz')
    if not os.path.exists(data_fn):
        raise FileNotFoundError('Run scripts/demo_prepare.py first')
    d = np.load(data_fn)
    X = d['X']

    results_dir = os.path.join(REPO_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(device)
    model = DemoBackbone(input_dim=X.shape[1], hidden=32, out_dim=8)
    # try to load backbone checkpoint if exists
    backbone_ckpt = os.path.join(results_dir, 'demo_backbone.pth')
    if os.path.exists(backbone_ckpt):
        model.load_state_dict(torch.load(backbone_ckpt, map_location=device))
    model.to(device)

    mc = MCDropoutWrapper(model, T=int(T))
    mc.to(device)
    mc.eval()

    Xt = torch.from_numpy(X[:16]).to(device)
    with torch.no_grad():
        mean, var = mc(Xt)

    out_fn = out or os.path.join(results_dir, 'mc_dropout_outputs.npz')
    np.savez(out_fn, mean=mean.cpu().numpy(), var=var.cpu().numpy())
    print('Saved MC-dropout outputs ->', out_fn)

    meta = {'T': int(T), 'device': str(device), 'rows': int(Xt.shape[0])}
    meta_fn = out_fn.replace('.npz', '_meta.json')
    with open(meta_fn, 'w') as f:
        json.dump(meta, f, indent=2)
    print('Wrote metadata ->', meta_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    main(T=args.T, device=args.device, out=args.out)
