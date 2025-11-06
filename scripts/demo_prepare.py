"""Prepare self-contained demo artifacts: synthetic dataset and small checkpoints.

Produces:
 - data/processed/demo_synth.npy  (synthetic inputs)
 - results/demo_backbone.pth      (backbone state_dict)
 - results/demo_tde.pth           (TinyDeepEnsemble state_dict with initialized heads)

This script is intentionally simple and reproducible on CPU.
"""
import os
import sys
import json
import numpy as np
import torch

# ensure repo root in sys.path so `src` imports work when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.demo_backbone import DemoBackbone
try:
    from src.core.ensembles import TinyDeepEnsemble
except Exception:
    # fallback import path if package root differs
    from core.ensembles import TinyDeepEnsemble  # type: ignore


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_synthetic(seed: int = 1234, n: int = 128, dim: int = 16):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim).astype(np.float32)
    # simple binary labels by summing dims (not used heavily here)
    y = (X.sum(axis=1) > 0).astype(np.int64)
    return X, y


def main():
    X, y = make_synthetic()
    fn = os.path.join(DATA_DIR, "demo_synth.npz")
    np.savez(fn, X=X, y=y)
    print(f"Saved synthetic data -> {fn}")

    # build tiny backbone and save checkpoint
    model = DemoBackbone(input_dim=X.shape[1], hidden=32, out_dim=8)
    backbone_ckpt = os.path.join(RESULTS_DIR, "demo_backbone.pth")
    torch.save(model.state_dict(), backbone_ckpt)
    print(f"Saved backbone checkpoint -> {backbone_ckpt}")

    # create TinyDeepEnsemble wrapper and initialize head params by running a dummy forward
    tde = TinyDeepEnsemble(model, num_members=4)
    # run one batch to lazy-init parameters
    import torch as _t
    _x = _t.from_numpy(X[:8])
    with torch.no_grad():
        _ = tde(_x)

    tde_ckpt = os.path.join(RESULTS_DIR, "demo_tde.pth")
    torch.save(tde.state_dict(), tde_ckpt)
    print(f"Saved TinyDeepEnsemble checkpoint -> {tde_ckpt}")

    # write a minimal provenance file
    meta = {
        "synthetic_rows": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "backbone_ckpt": os.path.basename(backbone_ckpt),
        "tde_ckpt": os.path.basename(tde_ckpt),
    }
    meta_fn = os.path.join(RESULTS_DIR, "demo_prepare_metadata.json")
    with open(meta_fn, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata -> {meta_fn}")


if __name__ == "__main__":
    main()
