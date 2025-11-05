import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def heteroscedastic_gaussian_nll(mean, log_var, target):
    # mean, log_var, target: (N,1)
    var = torch.exp(log_var).clamp_min(1e-8)
    return 0.5 * (torch.log(var) + (target - mean) ** 2 / var)


def load_synthetic(n_per_level=500, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for level in range(4):
        for _ in range(n_per_level):
            if level == 0:
                obj = rng.integers(0, 2)
                beh = 0
                prox = rng.uniform(0.0, 0.3)
            elif level == 1:
                obj = rng.integers(1, 3)
                beh = 0
                prox = rng.uniform(0.2, 0.5)
            elif level == 2:
                obj = rng.integers(2, 4)
                beh = 1
                prox = rng.uniform(0.3, 0.6)
            else:
                obj = rng.integers(2, 4)
                beh = 1
                prox = rng.uniform(0.5, 1.0)
            X.append([obj / 3.0, beh, prox])
            y.append(level)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y


class HeteroFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.2)
        self.mean = nn.Linear(32, 1)
        self.log_var = nn.Linear(32, 1)

    def forward(self, x, mc_dropout=True):
        h = self.act1(self.fc1(x))
        h = self.do1(h) if mc_dropout else h
        h = self.act2(self.fc2(h))
        h = self.do2(h) if mc_dropout else h
        mean = torch.sigmoid(self.mean(h)) * 4.0
        log_var = self.log_var(h)
        return mean, log_var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    args = parser.parse_args()

    if args.synthetic:
        X, y = load_synthetic()
    else:
        # Placeholder: load from exported CSVs
        raise SystemExit('Provide training data from logs or use --synthetic')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)

    model = HeteroFusion()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        mu, log_var = model(X_train, mc_dropout=True)
        loss = heteroscedastic_gaussian_nll(mu, log_var, y_train).mean()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            mu_val, log_var_val = model(X_val, mc_dropout=False)
            val_loss = heteroscedastic_gaussian_nll(mu_val, log_var_val, y_val).mean()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{args.epochs} | Train NLL: {loss.item():.4f} | Val NLL: {val_loss.item():.4f}')

    torch.save(model.state_dict(), 'models/fusion/trd_uq_fusion_hetero.pth')
    print('Saved: models/fusion/trd_uq_fusion_hetero.pth')


if __name__ == '__main__':
    main()


