import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ====== Generate Realistic & Balanced Dataset ======
X = []
y = []

def jitter(val, scale=0.1):
    return np.clip(val + np.random.normal(0, scale), 0.0, 1.0)

for level in range(4):
    for _ in range(500):  # 500 samples per level
        if level == 0:
            obj = np.random.randint(0, 2)
            beh = 0
            prox = np.random.uniform(0.0, 0.3)
        elif level == 1:
            obj = np.random.randint(1, 3)
            beh = 0
            prox = np.random.uniform(0.2, 0.5)
        elif level == 2:
            obj = np.random.randint(2, 4)
            beh = 1
            prox = np.random.uniform(0.3, 0.6)
        elif level == 3:
            obj = np.random.randint(2, 4)
            beh = 1
            prox = np.random.uniform(0.5, 1.0)
        
        # Add jitter for realism
        X.append([jitter(obj / 3.0), jitter(beh), jitter(prox)])
        y.append(level)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

# ====== Train/Test Split ======
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# ====== FusionMLP Model (Expanded) ======
class FusionMLP(nn.Module):
    def __init__(self):
        super(FusionMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

model = FusionMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====== Convert to Torch Tensors ======
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val)

# ====== Training Loop ======
EPOCHS = 200
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t)
        val_losses.append(val_loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# ====== Save Model and Plot ======
torch.save(model.state_dict(), "fusion_mlp_balanced.pth")
print("✅ Model saved as 'fusion_mlp_balanced.pth'")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("FusionMLP Training and Validation Loss (500 samples/level + jitter)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fusion_mlp_loss_curve.png")
plt.show()
