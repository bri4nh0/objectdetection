"""Compare TinyDE, Full Ensemble, and MC-Dropout across multiple seeds.

This script runs the demo pipeline for each seed and each method, collects
per-seed metrics, and computes paired statistics (mean/std and optional
Wilcoxon or paired t-test if scipy is available).
"""
import os
import sys
import json
import subprocess
from statistics import mean, stdev

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run(cmd):
    print('RUN:', cmd)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}')


def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def main(seeds=[123, 456, 789], device='cpu'):
    out_dir = os.path.join(REPO_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)

    methods = ['tinyde', 'full_ensemble', 'mc_dropout']
    per_method = {m: {} for m in methods}

    for s in seeds:
        print(f'=== seed {s} ===')
        run(f'python scripts/demo_prepare.py --seed {s}')

        # TinyDE
        run(f'python scripts/run_demo.py --device {device}')
        tiny_out = os.path.join('results', f'tinyde_eval_seed{s}.json')
        run(f'python scripts/run_eval.py --demo --out {tiny_out}')
        per_method['tinyde'][s] = load_json(tiny_out)['metrics']

        # Full ensemble: train small full ensemble and eval
        run(f'python scripts/train_full_ensemble.py --epochs 3 --members 4 --device {device}')
        ckpt = os.path.join('results', 'full_ensemble_4x_ep3.pth')
        fe_out = os.path.join('results', f'full_ensemble_eval_seed{s}.json')
        run(f'python scripts/eval_full_ensemble.py --ckpt {ckpt} --out {fe_out} --device {device}')
        per_method['full_ensemble'][s] = load_json(fe_out)['metrics']

        # MC-dropout
        mc_npz = os.path.join('results', f'mc_dropout_outputs_seed{s}.npz')
        mc_eval = os.path.join('results', f'mc_dropout_eval_seed{s}.json')
        run(f'python scripts/run_mc_dropout_demo.py --T 50 --device {device} --out {mc_npz}')
        run(f'python scripts/run_eval.py --input {mc_npz} --labels data/processed/demo_synth.npz --out {mc_eval}')
        per_method['mc_dropout'][s] = load_json(mc_eval)['metrics']

    # aggregate
    summary = {}
    metrics_keys = list(next(iter(per_method['tinyde'].values())).keys())
    for m in methods:
        summary[m] = {}
        for k in metrics_keys:
            vals = [per_method[m][s][k] for s in seeds]
            summary[m][k] = {'mean': mean(vals), 'std': stdev(vals) if len(vals) > 1 else 0.0, 'per_seed': vals}

    # paired tests if scipy exists
    stats = {}
    try:
        from scipy.stats import wilcoxon, ttest_rel
        stats['tiny_vs_full'] = {}
        stats['tiny_vs_mc'] = {}
        for k in metrics_keys:
            a = [per_method['tinyde'][s][k] for s in seeds]
            b = [per_method['full_ensemble'][s][k] for s in seeds]
            stats['tiny_vs_full'][k] = {
                'wilcoxon': _safe_test(wilcoxon, a, b),
                'ttest_rel': _safe_test(ttest_rel, a, b),
            }
            c = [per_method['mc_dropout'][s][k] for s in seeds]
            stats['tiny_vs_mc'][k] = {
                'wilcoxon': _safe_test(wilcoxon, a, c),
                'ttest_rel': _safe_test(ttest_rel, a, c),
            }
    except Exception:
        stats['note'] = 'scipy not available; skipped paired tests'

    out = {'per_method': per_method, 'summary': summary, 'stats': stats}
    out_fn = os.path.join(out_dir, 'method_comparison_multi_seed.json')
    with open(out_fn, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote comparison ->', out_fn)


def _safe_test(fn, a, b):
    try:
        res = fn(a, b)
        return {'statistic': float(getattr(res, 'statistic', res[0]) if res is not None else None), 'pvalue': float(getattr(res, 'pvalue', res[1]) if res is not None else None)}
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    main(seeds=[123, 456, 789], device='cpu')
