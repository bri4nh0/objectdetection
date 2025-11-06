import os
import sys
import traceback

# Ensure repo root and `src` are importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

log_path = os.path.join('results', 'verify_tiny_ensemble.log')
os.makedirs('results', exist_ok=True)

try:
    import torch
    from torch import nn
    # Prefer explicit package import, but fall back to sibling package if needed
    try:
        from src.core.ensembles import TinyDeepEnsemble
    except Exception:
        try:
            from core.ensembles import TinyDeepEnsemble
        except Exception:
            TinyDeepEnsemble = None

    class DummyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 1)
        def forward(self, x):
            return self.linear(x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = DummyBase().to(device)
    if TinyDeepEnsemble is None:
        raise ImportError('TinyDeepEnsemble could not be imported from src.core.ensembles or core.ensembles')
    td = TinyDeepEnsemble(base, num_members=4).to(device)
    td.eval()

    inp = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32).to(device)
    mean, var = td(inp)

    out = f"SUCCESS\nDevice: {device}\nMEAN: {mean.detach().cpu().numpy()}\nVAR: {var.detach().cpu().numpy()}\n"
except Exception:
    out = "ERROR\n" + traceback.format_exc()

with open(log_path, 'w') as f:
    f.write(out)

print(out)
