import argparse
import csv
from typing import List
from utils.calibration import PlattScaler, expected_calibration_error, brier_score


def read_csv(path: str) -> (List[float], List[int]):
    scores, labels = [], []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expect columns: score (0-4 risk), label (0/1 for critical)
            s = float(row['score'])
            y = int(row['label'])
            scores.append(max(0.0, min(4.0, s)) / 4.0)
            labels.append(1 if y > 0 else 0)
    return scores, labels


def main():
    parser = argparse.ArgumentParser(description='Fit Platt scaling for fusion/TRD-UQ outputs')
    parser.add_argument('--val-csv', required=True, help='CSV with columns: score,label')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.1)
    args = parser.parse_args()

    scores, labels = read_csv(args.val_csv)
    scaler = PlattScaler(lr=args.lr, epochs=args.epochs)
    scaler.fit(scores, labels)
    probs_raw = scores
    probs_cal = scaler.predict_proba(scores)

    ece_raw = expected_calibration_error(probs_raw, labels)
    ece_cal = expected_calibration_error(probs_cal, labels)
    brier_raw = brier_score(probs_raw, labels)
    brier_cal = brier_score(probs_cal, labels)

    print('Platt parameters:')
    print(f"  a: {scaler.a:.6f}")
    print(f"  b: {scaler.b:.6f}")
    print('Metrics (lower is better):')
    print(f"  ECE raw: {ece_raw:.4f} -> calibrated: {ece_cal:.4f}")
    print(f"  Brier raw: {brier_raw:.4f} -> calibrated: {brier_cal:.4f}")
    print('\nPaste into configs/system_config.yaml under calibration:')
    print(f"  a: {scaler.a:.6f}")
    print(f"  b: {scaler.b:.6f}")


if __name__ == '__main__':
    main()


