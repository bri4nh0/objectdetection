"""Lightweight full-ensemble training harness for smoke experiments.

Trains an ensemble of independent small MLPs on the demo synthetic data.
This is intentionally small so it can run in CI for smoke (CPU-friendly).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ensure repo root in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


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


def train_one(model, loader, optim, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = ((pred - yb) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.size(0)
    return total / len(loader.dataset)


def main(epochs=5, num_members=4, batch_size=32, device='cpu'):
    # load demo data or create synthetic
    data_fn = os.path.join(REPO_ROOT, 'data', 'processed', 'demo_synth.npz')
    if os.path.exists(data_fn):
        d = np.load(data_fn)
        X = d['X'].astype(np.float32)
        y = d['y'].astype(np.float32)
    else:
        rng = np.random.RandomState(0)
        X = rng.randn(256, 16).astype(np.float32)
        y = (X.sum(axis=1) + 0.1 * rng.randn(256)).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device(device)

    input_dim = X.shape[1]
    members = [SmallMLP(input_dim) for _ in range(num_members)]
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in members]

    for m in members:
        m.to(device)

    history = {i: [] for i in range(num_members)}
    for ep in range(epochs):
        for i, m in enumerate(members):
            loss = train_one(m, loader, opts[i], device)
            history[i].append(loss)

    # save ensemble checkpoints as a list of state_dicts
    out_dir = os.path.join(REPO_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {'members': [m.state_dict() for m in members]}
    ckpt_fn = os.path.join(out_dir, f'full_ensemble_{num_members}x_ep{epochs}.pth')
    torch.save(ckpt, ckpt_fn)
    meta = {'num_members': num_members, 'epochs': epochs, 'input_dim': int(input_dim)}
    meta_fn = os.path.join(out_dir, f'full_ensemble_{num_members}x_ep{epochs}_meta.json')
    with open(meta_fn, 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved full-ensemble checkpoint ->', ckpt_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--members', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(epochs=args.epochs, num_members=args.members, batch_size=args.batch_size, device=args.device)
