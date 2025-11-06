import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.training.train_trd_uq import generate_trd_uq_training_data
from src.core.trd_uq_system import TRDUQSystem


def expected_calibration_error(probs, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
            continue
        acc = np.mean(labels[mask])
        conf = np.mean(probs[mask])
        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(np.sum(mask))
        ece += (np.sum(mask) / len(probs)) * abs(acc - conf)
    return ece, bins, bin_accs, bin_confs, bin_counts


def plot_reliability(bins, bin_accs, bin_confs, outpath):
    centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.plot(centers, bin_accs, marker='o', label='Accuracy')
    plt.bar(centers, np.array(bin_confs) - np.array(bin_accs), bottom=bin_accs, width=0.08, alpha=0.3, label='Conf-Acc gap')
    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical frequency')
    plt.title('Reliability diagram')
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath)
    plt.close()


def main():
    os.makedirs('results', exist_ok=True)

    print("🔎 Generating evaluation data...")
    sequences, labels = generate_trd_uq_training_data()
    X = sequences[:, -1, :]
    y = labels[:, 0]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize TRD-UQ system
    tq = TRDUQSystem()

    mc_samples = 50
    preds = []
    total_vars = []
    epistemic_uncs = []
    aleatoric_uncs = []

    print(f"🔁 Running MC inference with {mc_samples} samples on {len(X_val)} validation samples...")
    for i in range(len(X_val)):
        obj, beh, prox = map(float, X_val[i])
        mean_risk, total_unc, epi, alea = tq.monte_carlo_uncertainty(
            torch.tensor([[obj, beh, prox]], dtype=torch.float32), num_samples=mc_samples
        )
        preds.append(mean_risk)
        epistemic_uncs.append(epi)
        aleatoric_uncs.append(alea)
        # approximate predictive variance
        total_vars.append(epi ** 2 + alea ** 2 + 1e-6)

    preds = np.array(preds)
    y_val = np.array(y_val)
    total_vars = np.array(total_vars)
    epistemic_uncs = np.array(epistemic_uncs)
    aleatoric_uncs = np.array(aleatoric_uncs)

    # Metrics
    rmse = np.sqrt(np.mean((preds - y_val) ** 2))
    nll = 0.5 * (np.log(2 * np.pi * total_vars) + ((y_val - preds) ** 2) / total_vars)
    mean_nll = np.mean(nll)

    # Brier/ECE for binary event: critical (>=3.0)
    y_bin = (y_val >= 3.0).astype(float)
    probs = np.clip(preds / 4.0, 0.0, 1.0)
    brier = np.mean((probs - y_bin) ** 2)
    ece, bins, bin_accs, bin_confs, bin_counts = expected_calibration_error(probs, y_bin, n_bins=10)

    # Save per-sample CSV and summary
    out_samples = np.stack([y_val, preds, epistemic_uncs, aleatoric_uncs, total_vars], axis=1)
    np.savetxt('results/trd_uq_eval_samples.csv', out_samples, delimiter=',', header='y_true,pred_mean,epistemic_std,aleatoric_std,total_var', comments='')

    summary = {
        'rmse': float(rmse),
        'mean_nll': float(mean_nll),
        'brier': float(brier),
        'ece': float(ece),
        'mc_samples': mc_samples,
        'n_val': int(len(X_val))
    }

    with open('results/trd_uq_eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✅ Evaluation summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Reliability plot
    plot_reliability(bins, bin_accs, bin_confs, 'results/trd_uq_reliability.png')
    print("✅ Reliability diagram saved to results/trd_uq_reliability.png")


if __name__ == '__main__':
    main()
