import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def heteroscedastic_gaussian_nll(mean, log_var, target):
    var = torch.exp(log_var).clamp_min(1e-8)
    return 0.5 * (torch.log(var) + (target - mean) ** 2 / var)


def read_csv(path):
    xs, ys = [], []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expect columns: object_risk, behavior_risk, proximity_risk, label_level
            try:
                obj = float(row['object_risk'])
                beh = float(row['behavior_risk'])
                prox = float(row['proximity_risk'])
                lvl = float(row['label_level'])
            except KeyError:
                # Fallback: if file only has fused score and label, skip
                raise SystemExit('CSV must include object_risk, behavior_risk, proximity_risk, label_level')
            xs.append([obj / 3.0, beh, prox])
            ys.append(lvl)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32).reshape(-1, 1)


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
    parser = argparse.ArgumentParser(description='Train heteroscedastic fusion from CSV logs')
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', default='models/fusion/trd_uq_fusion_hetero.pth')
    args = parser.parse_args()

    X, y = read_csv(args.train_csv)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=args.val_split, random_state=42)
    X_tr = torch.tensor(X_tr)
    y_tr = torch.tensor(y_tr)
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)

    model = HeteroFusion()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        mu, log_var = model(X_tr, mc_dropout=True)
        loss = heteroscedastic_gaussian_nll(mu, log_var, y_tr).mean()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            mu_val, log_var_val = model(X_val, mc_dropout=False)
            val_loss = heteroscedastic_gaussian_nll(mu_val, log_var_val, y_val).mean()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{args.epochs} | Train NLL: {loss.item():.4f} | Val NLL: {val_loss.item():.4f}')

    torch.save(model.state_dict(), args.out)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()


