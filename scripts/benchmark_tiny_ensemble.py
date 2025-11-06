import os
import sys
import time
import torch
from torch import nn

# Ensure src is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from core.ensembles import TinyDeepEnsemble
except Exception:
    try:
        from src.core.ensembles import TinyDeepEnsemble
    except Exception as e:
        print('ERROR: Could not import TinyDeepEnsemble:', e)
        raise


def bench_model(model, inp, runs=200, device='cpu'):
    model.to(device)
    inp = inp.to(device)
    # warmup
    for _ in range(10):
        _ = model(inp)
    torch.cuda.synchronize() if device == 'cuda' and torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(runs):
        out = model(inp)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    elapsed = (t1 - t0) / runs
    return elapsed, out


def run_bench(device):
    print(f'Running benchmark on device={device}')
    # Simple base model
    base = nn.Linear(3, 1)
    inp = torch.randn(1, 3)
    # baseline
    base_time, base_out = bench_model(base, inp, runs=500, device=device)

    # tiny ensemble
    td = TinyDeepEnsemble(base, num_members=4)
    td_time, td_out = bench_model(td, inp, runs=500, device=device)

    print(f'Base avg time (s): {base_time:.6f}, output: {base_out.detach().cpu().numpy()}')
    mean, var = td_out
    print(f'TinyEnsemble avg time (s): {td_time:.6f}, mean: {mean.detach().cpu().numpy()}, var: {var.detach().cpu().numpy()}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_bench('cpu')
    if device == 'cuda':
        run_bench('cuda')
