"""Run the demo prepare -> demo run -> eval pipeline for multiple random seeds.

Writes per-seed eval JSONs and a summary JSON `results/multi_seed_demo_summary.json`.
"""
import os
import sys
import json
import subprocess
from statistics import mean, stdev

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_cmd(cmd):
    print("Running:", cmd)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def load_metrics(path):
    with open(path, 'r') as f:
        return json.load(f)


def main(seeds=[0, 1, 2], device='cpu'):
    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)
    per_seed = {}
    for s in seeds:
        print(f"== seed {s} ==")
        run_cmd(f"python scripts/demo_prepare.py --seed {s}")
        run_cmd(f"python scripts/run_demo.py --device {device}")
        out_json = f"results/demo_eval_seed{s}.json"
        run_cmd(f"python scripts/run_eval.py --demo --out {out_json}")
        metrics = load_metrics(out_json)
        per_seed[s] = metrics

    # aggregate simple stats for core metrics
    keys = ['nll_mean', 'rmse', 'coverage_1sigma']
    summary = {}
    for k in keys:
        vals = [per_seed[s]['metrics'][k] for s in per_seed]
        summary[k] = {'mean': mean(vals), 'std': stdev(vals) if len(vals) > 1 else 0.0}

    out_summary = os.path.join(REPO_ROOT, 'results', 'multi_seed_demo_summary.json')
    with open(out_summary, 'w') as f:
        json.dump({'per_seed': per_seed, 'summary': summary}, f, indent=2)
    print('Wrote summary ->', out_summary)


if __name__ == '__main__':
    # default seeds for a smoke run
    main(seeds=[123, 456, 789], device='cpu')
