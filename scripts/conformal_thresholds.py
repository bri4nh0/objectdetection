import argparse
import csv
import numpy as np


def read_csv(path):
    scores, labels = [], []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['score']))
            labels.append(int(row['label']))
    return np.array(scores, float), np.array(labels, int)


def conformal_threshold(scores_cal, labels_cal, delta=0.1):
    # For positive-class detection, use quantile of positive scores to achieve (1-delta) coverage
    pos_scores = scores_cal[labels_cal == 1]
    if len(pos_scores) == 0:
        raise ValueError('No positive samples in calibration set')
    q = np.quantile(pos_scores, delta)  # lower quantile; threshold at this ensures >= (1-delta) recall under exchangeability
    return float(q)


def evaluate(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return dict(threshold=float(threshold), recall=float(recall), precision=float(precision), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def main():
    parser = argparse.ArgumentParser(description='Conformal threshold for CRITICAL detection with coverage guarantees')
    parser.add_argument('--cal-csv', required=True, help='Calibration CSV (score,label)')
    parser.add_argument('--test-csv', required=True, help='Test CSV (score,label)')
    parser.add_argument('--delta', type=float, default=0.1, help='Allowed error rate (1-coverage)')
    args = parser.parse_args()

    scores_cal, labels_cal = read_csv(args.cal_csv)
    scores_test, labels_test = read_csv(args.test_csv)

    thr = conformal_threshold(scores_cal, labels_cal, delta=args.delta)
    print(f'Conformal threshold (delta={args.delta:.2f}): {thr:.3f}')

    stats = evaluate(scores_test, labels_test, thr)
    print('Test stats:')
    for k, v in stats.items():
        print(f'  {k}: {v}')
    print('\nAdd to configs/system_config.yaml under decision:')
    print(f'  risk_threshold_critical: {thr:.3f}')


if __name__ == '__main__':
    main()


