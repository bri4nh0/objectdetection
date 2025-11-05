import math
from typing import List, Tuple


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class PlattScaler:
    """Sigmoid calibration: p = sigmoid(a * s + b).

    Train with simple SGD on binary labels to minimize logistic loss.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, lr: float = 0.1, epochs: int = 200):
        self.a = float(a)
        self.b = float(b)
        self.lr = float(lr)
        self.epochs = int(epochs)

    def fit(self, scores: List[float], labels: List[int]) -> None:
        n = max(1, len(scores))
        a, b = self.a, self.b
        for _ in range(self.epochs):
            grad_a = 0.0
            grad_b = 0.0
            for s, y in zip(scores, labels):
                p = sigmoid(a * float(s) + b)
                # gradient of BCE wrt logits (a*s + b) is (p - y)
                diff = (p - float(y))
                grad_a += diff * float(s)
                grad_b += diff
            grad_a /= n
            grad_b /= n
            a -= self.lr * grad_a
            b -= self.lr * grad_b
        self.a, self.b = a, b

    def predict_proba(self, scores: List[float]) -> List[float]:
        return [sigmoid(self.a * float(s) + self.b) for s in scores]


def expected_calibration_error(probs: List[float], labels: List[int], bins: int = 10) -> float:
    assert len(probs) == len(labels)
    if len(probs) == 0:
        return 0.0
    bin_size = 1.0 / bins
    ece = 0.0
    for i in range(bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size
        indices = [j for j, p in enumerate(probs) if (p >= lo and (p < hi or (i == bins - 1 and p <= hi)))]
        if not indices:
            continue
        avg_conf = sum(probs[j] for j in indices) / len(indices)
        avg_acc = sum(1.0 if labels[j] == (1 if probs[j] >= 0.5 else 0) else 0.0 for j in indices) / len(indices)
        ece += (len(indices) / len(probs)) * abs(avg_conf - avg_acc)
    return ece


def brier_score(probs: List[float], labels: List[int]) -> float:
    if len(probs) == 0:
        return 0.0
    return sum((float(y) - float(p)) ** 2 for p, y in zip(probs, labels)) / len(probs)


