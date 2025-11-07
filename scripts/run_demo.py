"""Run the self-contained demo: load demo artifacts, run TinyDeepEnsemble, save outputs.

Produces:
 - results/demo_outputs.npz   (model outputs and simple metrics)
 - results/demo_run_metadata.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

# ensure repo root in sys.path so `src` imports work when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from src.core.demo_backbone import DemoBackbone
    from src.core.ensembles import TinyDeepEnsemble
except Exception:
    from core.demo_backbone import DemoBackbone  # type: ignore
    from core.ensembles import TinyDeepEnsemble  # type: ignore


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main(batch_size: int = 16, device_arg: str = 'auto'):
    fn = os.path.join(DATA_DIR, "demo_synth.npz")
    if not os.path.exists(fn):
        raise FileNotFoundError("Run scripts/demo_prepare.py first to create demo data")

    data = np.load(fn)
    X = data["X"]

    # resolve device
    if device_arg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device_arg.lower() == 'cuda' and not torch.cuda.is_available():
            print('Requested cuda but not available; falling back to cpu')
            device = torch.device('cpu')
        else:
            device = torch.device(device_arg)

    # load backbone and TDE
    backbone_ckpt = os.path.join(RESULTS_DIR, "demo_backbone.pth")
    tde_ckpt = os.path.join(RESULTS_DIR, "demo_tde.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DemoBackbone(input_dim=X.shape[1], hidden=32, out_dim=8)
    model.load_state_dict(torch.load(backbone_ckpt, map_location=device))
    model.to(device)

    tde = TinyDeepEnsemble(model, num_members=4)
    # load state dict into wrapper if present
    if os.path.exists(tde_ckpt):
        try:
            sd = torch.load(tde_ckpt, map_location=device)
            tde.load_state_dict(sd)
        except Exception:
            # some state dicts may contain base model state vs wrapper; ignore if incompatible
            pass
    tde.to(device)

    # run inference on first N rows
    Xt = torch.from_numpy(X[:batch_size]).to(device)
    with torch.no_grad():
        mean, var = tde(Xt)

    out_fn = os.path.join(RESULTS_DIR, "demo_outputs.npz")
    np.savez(out_fn, mean=mean.cpu().numpy(), var=var.cpu().numpy())
    print(f"Saved demo outputs -> {out_fn}")

    meta = {
        "device": str(device),
        "rows_used": int(Xt.shape[0]),
        "mean_shape": list(mean.shape),
        "var_shape": list(var.shape),
    }
    meta_fn = os.path.join(RESULTS_DIR, "demo_run_metadata.json")
    with open(meta_fn, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote run metadata -> {meta_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto', help="Device to run on: 'auto'|'cpu'|'cuda'")
    args = parser.parse_args()
    main(batch_size=args.batch_size, device_arg=args.device)
