"""Simple evaluation CLI that computes standardized metrics from demo outputs.

Usage (demo-ready):
  python scripts/run_eval.py --demo

This will load `data/processed/demo_synth.npz` and `results/demo_outputs.npz`,
compute metrics and write `results/demo_eval.json`.
"""
import os
import sys
import argparse
import numpy as np

# ensure repo root in sys.path so imports like `src.*` work when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.trd_uq_eval import summarize_predictions, write_results_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run evaluation on bundled demo outputs')
    parser.add_argument('--input', default=None, help='Path to a single .npz file containing mean/var (keys: mean,var)')
    parser.add_argument('--preds', default=None, help='Path to predictions array (.npy or .npz with key "mean")')
    parser.add_argument('--vars', default=None, help='Path to variance array (.npy or .npz with key "var")')
    parser.add_argument('--labels', default=None, help='Path to labels/targets array (.npy or .npz with key "y")')
    parser.add_argument('--out', default='results/demo_eval.json', help='Output JSON path')
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if args.demo:
        data_fn = os.path.join(repo_root, 'data', 'processed', 'demo_synth.npz')
        out_fn = os.path.join(repo_root, args.out)
        demo_out = os.path.join(repo_root, 'results', 'demo_outputs.npz')
        if not os.path.exists(data_fn) or not os.path.exists(demo_out):
            raise FileNotFoundError('Run scripts/demo_prepare.py and scripts/run_demo.py first')

        d = np.load(data_fn)
        X = d['X']
        y = d['y']
        out = np.load(demo_out)
        mean = out['mean']
        var = out['var']

        # demo outputs may be multi-dimensional (batch, out_dim). Reduce to a
        # scalar prediction per-sample by averaging across output dims for demo.
        if mean.ndim > 1:
            mean_reduced = mean.mean(axis=1)
        else:
            mean_reduced = mean
        if var.ndim > 1:
            var_reduced = var.mean(axis=1)
        else:
            var_reduced = var

        metrics = summarize_predictions(mean_reduced, var_reduced, y[:mean_reduced.shape[0]])
        run_meta = {
            'mode': 'demo',
            'mean_shape': list(mean.shape),
        }
        write_results_file(out_fn, run_meta, metrics)
        print('Wrote evaluation ->', out_fn)
    else:
        # General evaluation mode: load arrays from provided paths
        out_fn = os.path.join(repo_root, args.out)
        def load_array(p, key_hint=None):
            if p is None:
                return None
            if not os.path.exists(p):
                raise FileNotFoundError(f'Path not found: {p}')
            if p.endswith('.npy'):
                return np.load(p)
            if p.endswith('.npz'):
                d = np.load(p)
                # try common keys
                if key_hint and key_hint in d:
                    return d[key_hint]
                for k in ('mean', 'preds', 'pred', 'y_pred'):
                    if k in d:
                        return d[k]
                for k in ('var', 'vars', 'variance', 'unc'):
                    if k in d and key_hint is None:
                        return d[k]
                # fallback to first array in archive
                keys = list(d.keys())
                if len(keys) > 0:
                    return d[keys[0]]
                raise ValueError(f'No arrays found in {p}')
            # try CSV
            if p.endswith('.csv'):
                return np.loadtxt(p, delimiter=',')
            raise ValueError(f'Unsupported file extension for {p}')

        if args.input:
            d = np.load(os.path.join(repo_root, args.input))
            mean = d['mean'] if 'mean' in d else d[list(d.keys())[0]]
            var = d['var'] if 'var' in d else None
            if var is None:
                # try second array
                keys = list(d.keys())
                if len(keys) >= 2:
                    var = d[keys[1]]
            if var is None:
                raise ValueError('Input .npz must contain mean and var arrays')
            if args.labels:
                y = load_array(os.path.join(repo_root, args.labels), key_hint='y')
            else:
                # try labels in same archive
                y = d['y'] if 'y' in d else None
                if y is None:
                    raise ValueError('Labels not provided; use --labels or include "y" in the input .npz')
        else:
            mean = load_array(os.path.join(repo_root, args.preds) if args.preds else None, key_hint='mean')
            var = load_array(os.path.join(repo_root, args.vars) if args.vars else None, key_hint='var')
            y = load_array(os.path.join(repo_root, args.labels) if args.labels else None, key_hint='y')

        mean = np.asarray(mean)
        var = np.asarray(var)
        y = np.asarray(y)

        # If predictions are multi-d (batch, out_dim) but labels are scalar per-sample,
        # reduce predictions to a scalar per-sample by averaging across output dim.
        if mean.ndim > 1 and y.ndim == 1:
            mean = mean.mean(axis=1)
        if var.ndim > 1 and y.ndim == 1:
            var = var.mean(axis=1)

        metrics = summarize_predictions(mean, var, y[:mean.shape[0]])
        run_meta = {'mode': 'general', 'mean_shape': list(mean.shape)}
        write_results_file(out_fn, run_meta, metrics)
        print('Wrote evaluation ->', out_fn)


if __name__ == '__main__':
    main()
