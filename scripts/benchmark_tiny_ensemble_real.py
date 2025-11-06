import os
import sys
import time
import torch
from torch import nn

# ensure src importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.core.multimodal import FusionMLP
from src.core.ensembles import TinyDeepEnsemble


def bench_model(model, inp, runs=200, device='cpu'):
    model.to(device)
    inp = inp.to(device)
    # warmup
    for _ in range(10):
        _ = model(inp)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        out = model(inp)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    elapsed = (t1 - t0) / runs
    return elapsed, out


def run_real_bench(device):
    print(f'Running real benchmark on {device}')
    model = FusionMLP(input_size=3)
    # try load weights if available
    try:
        model.load_state_dict(torch.load('models/fusion/fusion_mlp_balanced.pth', map_location='cpu'))
        print('Loaded fusion weights')
    except Exception:
        print('No fusion weights found or failed to load; using fresh model')

    inp = torch.randn(1, 3)
    base_time, base_out = bench_model(model, inp, runs=200, device=device)

    td = TinyDeepEnsemble(model, num_members=4)
    td_time, td_out = bench_model(td, inp, runs=200, device=device)

    mean, var = td_out
    log = f"Device: {device}\nBase avg: {base_time:.6f}\nTD avg: {td_time:.6f}\nMEAN: {mean.detach().cpu().numpy()}\nVAR: {var.detach().cpu().numpy()}\n"
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'benchmark_tiny_ensemble_real.log'), 'w') as f:
        f.write(log)
    print(log)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_real_bench('cpu')
    if device == 'cuda':
        run_real_bench('cuda')
