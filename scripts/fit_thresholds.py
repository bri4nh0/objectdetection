import argparse
import csv
import numpy as np


def read_csv(path):
    scores, labels = [], []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = float(row['score'])
            y = int(row['label'])
            scores.append(s)
            labels.append(1 if y > 0 else 0)
    return np.array(scores, dtype=float), np.array(labels, dtype=int)


def fit_threshold(scores, labels, target_recall=0.9):
    # Grid search thresholds on 0..4 to meet or exceed target recall, maximizing precision
    ts = np.linspace(0.0, 4.0, 401)
    best = None
    for t in ts:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if recall + 1e-9 >= target_recall:
            if best is None or precision > best['precision']:
                best = {'t': float(t), 'recall': float(recall), 'precision': float(precision), 'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
    return best


def main():
    parser = argparse.ArgumentParser(description='Fit risk threshold for critical alarms to hit target recall')
    parser.add_argument('--val-csv', required=True, help='CSV with columns: score,label')
    parser.add_argument('--target-recall', type=float, default=0.9)
    args = parser.parse_args()

    scores, labels = read_csv(args.val_csv)
    best = fit_threshold(scores, labels, target_recall=args.target_recall)
    if not best:
        print('No threshold meets the target recall. Try lowering --target-recall.')
        return
    print('Best threshold for target recall:')
    print(f"  threshold: {best['t']:.3f}")
    print(f"  recall: {best['recall']:.3f}")
    print(f"  precision: {best['precision']:.3f}")
    print(f"  counts: TP={best['tp']} FP={best['fp']} TN={best['tn']} FN={best['fn']}")
    print('\nAdd to configs/system_config.yaml under decision:')
    print(f"  risk_threshold_critical: {best['t']:.3f}")


if __name__ == '__main__':
    main()


