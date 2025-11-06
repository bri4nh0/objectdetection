#!/usr/bin/env python3
"""Small training stub for TinyDeepEnsemble per-member head fine-tuning.

Creates a synthetic dataset, builds a small base model, wraps it with
TinyDeepEnsemble (if available) and trains only the per-member head params
for a few epochs. Saves a checkpoint to `results/`.

This is intended as a lightweight, reproducible smoke/training harness to
verify the TinyDeepEnsemble training flow on machines like an RTX 3050 Ti.
"""
import argparse
import os
import sys
from pathlib import Path
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.core.ensembles import TinyDeepEnsemble
except Exception:
    # fallback if running from repo root without package marker
    try:
        from core.ensembles import TinyDeepEnsemble
    except Exception:
        TinyDeepEnsemble = None

try:
    from src.utils.metadata import save_metadata
except Exception:
    try:
        from utils.metadata import save_metadata
    except Exception:
        save_metadata = None


def make_synthetic_data(n_samples=512, input_dim=32, out_dim=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_samples, input_dim)).astype(np.float32)
    # regression targets with small noise
    W = rng.normal(size=(input_dim, out_dim)).astype(np.float32)
    y = X.dot(W) + 0.1 * rng.normal(size=(n_samples, out_dim)).astype(np.float32)
    return X, y


def build_base_model(input_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


def train_head_only(model, loader, val_loader, device, epochs, lr, num_members):
    model.to(device)
    # trainable params: if TinyDeepEnsemble present, its head params; otherwise all params
    if TinyDeepEnsemble is not None and isinstance(model, TinyDeepEnsemble):
        params = [p for name, p in model.named_parameters() if 'scale' in name or 'bias' in name]
    else:
        params = list(model.parameters())

    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            # model may return (mean, var) tuple
            if isinstance(out, tuple) or isinstance(out, list):
                mean = out[0]
            else:
                mean = out
            loss = loss_fn(mean, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        dt = time.time() - t0
        train_loss = running / n

        # validation
        model.eval()
        with torch.no_grad():
            running_v = 0.0
            n_v = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                mean = out[0] if isinstance(out, (tuple, list)) else out
                running_v += float(loss_fn(mean, yb).item()) * xb.size(0)
                n_v += xb.size(0)
            val_loss = running_v / n_v

        # report
        print(f"epoch={ep+1}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={dt:.3f}s")

    return model


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--num_members', type=int, default=4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args(argv)

    out_dir = REPO_ROOT / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # data
    X, y = make_synthetic_data(n_samples=512, input_dim=32, out_dim=3, seed=args.seed)
    X_val, y_val = make_synthetic_data(n_samples=256, input_dim=32, out_dim=3, seed=args.seed + 1)

    train_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    base = build_base_model(input_dim=32, out_dim=3)

    if args.num_members > 1 and TinyDeepEnsemble is not None:
        model = TinyDeepEnsemble(base_model=base, num_members=args.num_members)
        print('Using TinyDeepEnsemble wrapper')
    else:
        model = base
        if args.num_members > 1:
            print('TinyDeepEnsemble not available; training base model only')

    print(f"device={args.device} torch.cuda.is_available()={torch.cuda.is_available()}")

    if args.dry_run:
        print('Dry run requested; exiting after setup')
        return 0

    device = torch.device(args.device)
    trained = train_head_only(model, train_loader, val_loader, device, args.epochs, args.lr, args.num_members)

    # evaluate quick stat
    trained.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_val).to(device)
        out = trained(xb)
        mean = out[0] if isinstance(out, (tuple, list)) else out
        mean = mean.cpu().numpy()
        var = None
        if isinstance(out, (tuple, list)) and len(out) > 1:
            var = out[1].cpu().numpy()

    # save checkpoint and metadata
    ckpt_path = out_dir / 'train_tiny_ensemble_stub.pth'
    try:
        torch.save({'state_dict': trained.state_dict(), 'num_members': args.num_members}, ckpt_path)
        print(f'Saved checkpoint to {ckpt_path}')
    except Exception as e:
        print('Failed to save checkpoint:', e)

    meta = {
        'script': str(Path(__file__).relative_to(REPO_ROOT)),
        'num_members': args.num_members,
        'epochs': args.epochs,
        'device': args.device,
        'mean_val_mean': float(mean.mean()),
        'mean_val_std': float(mean.std()),
        'var_present': var is not None,
    }

    meta_path = out_dir / 'train_tiny_ensemble_stub_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Wrote metadata to {meta_path}')

    # try to save richer metadata via helper if available
    if save_metadata is not None:
        try:
            # save_metadata expects a mapping of model name -> path
            save_metadata(run_id='train_tiny_ensemble_stub', repo_root=str(REPO_ROOT), config=meta, models={'train_tiny_ensemble_stub': str(ckpt_path)}, out_path=str(out_dir))
            print('save_metadata succeeded')
        except Exception as e:
            print('save_metadata failed:', e)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
