"""Smoke script comparing discounted fusion vs naive concatenation on synthetic data.

Produces small metric prints; intended for CI/demo smoke.
"""
import os
import sys
import numpy as np
import torch

# ensure repo root in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.discounted_fusion import DiscountedFusion


def make_synthetic(batch=64):
    rng = np.random.RandomState(0)
    # two modalities: A dim=8, B dim=4
    A = rng.randn(batch, 8).astype(np.float32)
    B = rng.randn(batch, 4).astype(np.float32)
    # target is small function of concatenated features
    X = np.concatenate([A, B], axis=1)
    y = (X.sum(axis=1) + 0.1 * rng.randn(batch)).astype(np.float32)
    return A, B, y


def main():
    A, B, y = make_synthetic()
    X = np.concatenate([A, B], axis=1)
    feats = torch.from_numpy(X)

    model = DiscountedFusion([8, 4], hidden=16)
    model.eval()
    with torch.no_grad():
        fused = model(feats)

    # naive baseline is identity concat
    baseline = feats

    # simple comparison: l2 distance between fused and baseline
    d = ((fused - baseline) ** 2).mean().item()
    print(f"Mean-squared-change-from-baseline: {d:.6f}")


if __name__ == '__main__':
    main()
