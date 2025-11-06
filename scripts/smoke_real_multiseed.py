import os
import sys
import time
import uuid
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.core.ensembles import TinyDeepEnsemble
from src.core.multimodal import FusionMLP
from src.core.uq_baselines import MCDropoutWrapper
from src.utils.metadata import save_metadata


def load_real_data():
    # Try a few known locations for features/labels
    candidates = [
        os.path.join('models', 'training', 'X_fighting.npy'),
        os.path.join('data', 'processed', 'pose_npy_output'),
        os.path.join('data', 'labels', 'behavior_escalation_labels.npy')
    ]
    # If X_fighting.npy exists, use it and corresponding y
    x_path = candidates[0]
    if os.path.exists(x_path):
        X = np.load(x_path)
        y_path = os.path.join('models', 'training', 'y_fighting.npy')
        y = np.load(y_path) if os.path.exists(y_path) else None
        return X, y

    # Otherwise try arbitrary .npy in pose_npy_output
    dir_path = candidates[1]
    if os.path.isdir(dir_path):
        npys = glob.glob(os.path.join(dir_path, '*.npy'))
        if npys:
            X = np.load(npys[0])
            # If labels file exists, load it
            y_path = candidates[2]
            y = np.load(y_path) if os.path.exists(y_path) else None
            return X, y

    return None, None


def ece_score(probs, labels, n_bins=15):
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / probs.size) * abs(acc - conf)
    return ece


def reliability_diagram(probs, labels, out_path, n_bins=15):
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    accs = []
    confs = []
    counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            accs.append(0.0)
            confs.append(0.0)
            counts.append(0)
            continue
        accs.append(labels[mask].mean())
        confs.append(probs[mask].mean())
        counts.append(mask.sum())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.plot(confs, accs, marker='o', linewidth=1)
    plt.fill_between(confs, accs, confs, color='gray', alpha=0.2)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability diagram')
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


def bench(model, X, runs=200, device='cpu'):
    model.to(device)
    inp = torch.tensor(X[:1]).float().to(device)
    # warmup
    for _ in range(10):
        _ = model(inp)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = None
    for _ in range(runs):
        out = model(inp)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / runs, out


def run_real_multiseed(seeds=(0, 1, 2, 3, 4)):
    run_id = 'real_' + uuid.uuid4().hex[:8]
    X, y = load_real_data()
    if X is None:
        print('No real dataset found; aborting. Place features in models/training/X_fighting.npy or data/processed/pose_npy_output/*.npy')
        return

    # if labels not found, synthesize from model predictions (falls back)
    if y is None:
        print('No labels found for dataset; will synthesize noisy labels from base model predictions (not ideal).')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    td_metrics = []
    mc_metrics = []

    model_path = os.path.join('models', 'fusion', 'fusion_mlp_balanced.pth')
    model_base = None
    if os.path.exists(model_path):
        # Try to load checkpoint into a model created with expected input size (common is 3)
        try:
            candidate = FusionMLP(input_size=3)
            candidate.load_state_dict(torch.load(model_path, map_location='cpu'))
            model_base = candidate
            print('Loaded fusion weights into input_size=3 model')
        except Exception:
            # fallback: use model sized for the dataset and skip loading weights
            print('Checkpoint incompatible with input_size=3; will use fresh model sized to data and skip loading weights')

    if model_base is None:
        model_base = FusionMLP(input_size=X.shape[1])

    # Ensure X matches model input dimensionality; if checkpoint expects fewer dims, slice features
    in_dim = model_base.fc[0].in_features if hasattr(model_base, 'fc') else X.shape[1]
    if X.shape[1] != in_dim:
        print(f'Warning: dataset feature dim {X.shape[1]} != model in_dim {in_dim}; slicing features to match')
        X = X[:, :in_dim]

    for s in seeds:
        np.random.seed(s)
        torch.manual_seed(s)

        # TinyEnsemble
        td = TinyDeepEnsemble(model_base, num_members=4)
        td_time, td_out = bench(td, X, runs=200, device=device)
        td_mean, td_var = td_out
        td_probs = 1 / (1 + np.exp(-td_mean.detach().cpu().numpy().flatten()))

        # MC
        mc = MCDropoutWrapper(model_base, T=30)
        mc_time, mc_out = bench(mc, X, runs=50, device=device)
        mc_mean, mc_var = mc_out
        mc_probs = 1 / (1 + np.exp(-mc_mean.detach().cpu().numpy().flatten()))

        # labels
        if y is not None:
            labels = y.flatten()[:td_probs.size]
        else:
            # synth labels from base model with noise
            labels = (td_probs + 0.1 * np.random.randn(*td_probs.shape) > 0.5).astype(float)

        td_ece = ece_score(td_probs, labels)
        mc_ece = ece_score(mc_probs, labels)

        td_metrics.append({'seed': s, 'time': td_time, 'ece': float(td_ece)})
        mc_metrics.append({'seed': s, 'time': mc_time, 'ece': float(mc_ece)})

    # compute mean and 95% CI across seeds
    def mean_ci(values):
        arr = np.array(values)
        mean = arr.mean()
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return mean, lo, hi

    td_times = [m['time'] for m in td_metrics]
    td_eces = [m['ece'] for m in td_metrics]
    mc_times = [m['time'] for m in mc_metrics]
    mc_eces = [m['ece'] for m in mc_metrics]

    td_time_mean, td_time_lo, td_time_hi = mean_ci(td_times)
    td_ece_mean, td_ece_lo, td_ece_hi = mean_ci(td_eces)
    mc_time_mean, mc_time_lo, mc_time_hi = mean_ci(mc_times)
    mc_ece_mean, mc_ece_lo, mc_ece_hi = mean_ci(mc_eces)

    results = {
        'td_time': {'mean': td_time_mean, 'ci': [td_time_lo, td_time_hi]},
        'td_ece': {'mean': td_ece_mean, 'ci': [td_ece_lo, td_ece_hi]},
        'mc_time': {'mean': mc_time_mean, 'ci': [mc_time_lo, mc_time_hi]},
        'mc_ece': {'mean': mc_ece_mean, 'ci': [mc_ece_lo, mc_ece_hi]},
        'device': device,
        'n_seeds': len(seeds)
    }

    os.makedirs('results', exist_ok=True)
    out_json = os.path.join('results', f'{run_id}_real_multiseed_results.json')
    import json
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # reliability diagram for TD using last seed's probs
    rel_png = os.path.join('results', f'{run_id}_td_reliability.png')
    reliability_diagram(td_probs, labels, rel_png)

    # save metadata
    save_metadata(run_id, repo_root, config={'td_members': 4, 'mc_T': 30, 'seeds': list(seeds)}, models={'fusion_mlp': model_path if os.path.exists(model_path) else ''}, out_path='results')

    print('MULTISEED RESULTS saved to', out_json)


if __name__ == '__main__':
    run_real_multiseed()
