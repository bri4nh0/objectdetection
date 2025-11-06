import os
import sys
import time
import uuid
import json
import numpy as np
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.core.ensembles import TinyDeepEnsemble
from src.core.multimodal import FusionMLP
from src.core.uq_baselines import MCDropoutWrapper
from src.utils.metadata import save_metadata


def ece_score(probs, labels, n_bins=10):
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc = []
    bin_conf = []
    bin_frac = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_frac.append(0)
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        frac = mask.sum() / probs.size
        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_frac.append(frac)
        ece += frac * abs(acc - conf)
    return ece, np.array(bin_conf), np.array(bin_acc), np.array(bin_frac)


def reliability_plot(confidences, accuracies, filename, title='Reliability'):
    bins = np.arange(len(confidences)) + 0.5
    plt.figure(figsize=(6, 6))
    # remove nan entries
    mask = ~np.isnan(accuracies)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.plot(confidences[mask], accuracies[mask], marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    vals = np.array(values)
    n = vals.size
    boots = []
    rng = np.random.RandomState(0)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots.append(vals[idx].mean())
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(np.mean(vals)), float(lo), float(hi)


def run_multiseed(seeds=5, td_members=4, mc_T=30, bootstrap_samples=1000):
    run_id = 'multiseed_' + uuid.uuid4().hex[:8]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare model
    base_model = FusionMLP(input_size=3)
    model_path = 'models/fusion/fusion_mlp_balanced.pth'
    if os.path.exists(model_path):
        try:
            base_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print('Loaded fusion weights')
        except Exception as e:
            print('Could not load fusion weights:', e)

    # synthetic dataset (or replace with real held-out if available)
    X = torch.randn(1000, 3)
    with torch.no_grad():
        raw = base_model(X).squeeze().numpy()
    probs_template = 1 / (1 + np.exp(-raw))
    # generate labels once from template with added noise
    labels = (probs_template + 0.05 * np.random.randn(*probs_template.shape) > 0.5).astype(float)

    td_times = []
    mc_times = []
    td_probs_all = []
    mc_probs_all = []

    for s in range(seeds):
        seed = 1000 + s
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # TinyEnsemble
        td = TinyDeepEnsemble(base_model, num_members=td_members)
        td_time, td_out = bench = bench_model(td, X, runs=50, device=device)
        td_mean, td_var = td_out
        td_probs = 1 / (1 + np.exp(-td_mean.detach().cpu().numpy().flatten()))

        # MC-dropout
        mc = MCDropoutWrapper(base_model, T=mc_T)
        mc_time, mc_out = bench_model(mc, X, runs=10, device=device)
        mc_mean, mc_var = mc_out
        mc_probs = 1 / (1 + np.exp(-mc_mean.detach().cpu().numpy().flatten()))

        td_times.append(td_time)
        mc_times.append(mc_time)
        td_probs_all.append(td_probs)
        mc_probs_all.append(mc_probs)

        print(f'seed={seed} td_time={td_time:.6f} mc_time={mc_time:.6f}')

    # aggregate
    td_probs_concat = np.concatenate(td_probs_all)
    mc_probs_concat = np.concatenate(mc_probs_all)
    labels_concat = np.tile(labels, seeds)

    td_ece, td_conf, td_acc, td_frac = ece_score(td_probs_concat, labels_concat, n_bins=15)
    mc_ece, mc_conf, mc_acc, mc_frac = ece_score(mc_probs_concat, labels_concat, n_bins=15)

    # reliability plots
    os.makedirs('results', exist_ok=True)
    reliability_td = os.path.join('results', f'{run_id}_reliability_td.png')
    reliability_mc = os.path.join('results', f'{run_id}_reliability_mc.png')
    reliability_plot(td_conf, td_acc, reliability_td, title='TinyDeepEnsemble Reliability')
    reliability_plot(mc_conf, mc_acc, reliability_mc, title='MC-Dropout Reliability')

    # bootstrap CIs across seeds for ECE and time
    td_ece_mean, td_ece_lo, td_ece_hi = bootstrap_ci([ece_score(p, labels, n_bins=15)[0] for p in td_probs_all], n_boot=bootstrap_samples)
    mc_ece_mean, mc_ece_lo, mc_ece_hi = bootstrap_ci([ece_score(p, labels, n_bins=15)[0] for p in mc_probs_all], n_boot=bootstrap_samples)
    td_time_mean, td_time_lo, td_time_hi = bootstrap_ci(td_times, n_boot=bootstrap_samples)
    mc_time_mean, mc_time_lo, mc_time_hi = bootstrap_ci(mc_times, n_boot=bootstrap_samples)

    results = {
        'run_id': run_id,
        'device': device,
        'seeds': seeds,
        'td_members': td_members,
        'mc_T': mc_T,
        'td_ece_mean': td_ece_mean,
        'td_ece_ci': [td_ece_lo, td_ece_hi],
        'mc_ece_mean': mc_ece_mean,
        'mc_ece_ci': [mc_ece_lo, mc_ece_hi],
        'td_time_mean': td_time_mean,
        'td_time_ci': [td_time_lo, td_time_hi],
        'mc_time_mean': mc_time_mean,
        'mc_time_ci': [mc_time_lo, mc_time_hi],
        'reliability_td': reliability_td,
        'reliability_mc': reliability_mc
    }

    out_file = os.path.join('results', f'{run_id}_multiseed_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    # save metadata
    save_metadata(run_id, repo_root, config={'seeds': seeds, 'td_members': td_members, 'mc_T': mc_T}, models={'fusion_mlp': model_path if os.path.exists(model_path) else ''}, out_path='results')

    print('Multiseed results saved to', out_file)
    print(json.dumps(results, indent=2))


def bench_model(model, inp, runs=50, device='cpu'):
    model.to(device)
    inp = inp.to(device)
    # warmup
    for _ in range(5):
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


if __name__ == '__main__':
    run_multiseed(seeds=5, td_members=4, mc_T=30, bootstrap_samples=1000)
