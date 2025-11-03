import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ====== Load Your Model Definition (must match your training code) ======
class FusionMLP(nn.Module):
    def __init__(self):
        super(FusionMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single scalar output
        )

    def forward(self, x):
        return self.fc(x)

# ====== Load Model and Weights ======
model = FusionMLP()
model.load_state_dict(torch.load("fusion_mlp_scalar.pth", map_location=torch.device("cpu")))
model.eval()

# ====== Simulate a Range of Inputs (Object × Behavior × Proximity) ======
frame_ids = []
fusion_scores = []
risk_weights = torch.tensor([0, 1, 2, 3], dtype=torch.float32)

for i in range(100):
    # Simulate inputs (you can modify this pattern)
    obj_risk = np.clip(np.sin(i / 15) * 3, 0, 3)           # cyclic object risk
    beh_risk = 1 if i % 20 > 10 else 0                     # intermittent behavior risk
    prox_risk = np.clip(np.random.rand() * 0.5, 0, 1)      # small proximity risk

    input_tensor = torch.tensor([[obj_risk, beh_risk, prox_risk]], dtype=torch.float32)
    with torch.no_grad():
        fusion_score = model(input_tensor).item()


    frame_ids.append(i)
    fusion_scores.append(fusion_score)

# ====== Plot Escalation Scores ======
plt.figure(figsize=(12, 5))
plt.plot(frame_ids, fusion_scores, label="FusionMLP Escalation Score", color="blue")
plt.axhline(1.5, color="gray", linestyle=":", label="Hazard Threshold (1.5)")
plt.axhline(2.0, color="gray", linestyle=":")
plt.axhline(2.5, color="gray", linestyle=":", label="Critical Threshold (2.5)")
plt.xlabel("Synthetic Frame")
plt.ylabel("Escalation Score")
plt.title("FusionMLP Escalation Score Simulation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
