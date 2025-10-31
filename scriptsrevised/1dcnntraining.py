# ========== Attention-1D-CNN Training Script (with Metrics Logging) ==========

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ========== CONFIG ==========
POSE_DATA_DIR = "pose_npy_output"  # Folder with pose .npy files
RESULTS_DIR = "results"
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
SEQ_LEN = 30
NUM_KEYPOINTS = 17
NUM_CLASSES = 2
MODEL_SAVE_PATH = "attention_1dcnn_behavior.pth"

# ========== DATASET ==========
class PoseSequenceDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = []
        for fname in os.listdir(folder_path):
            if fname.endswith(".npy"):
                label = 0 if "Neutral" in fname else 1  # Adjust based on your naming
                self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)         # (30, 34) already!
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ========== MODEL ==========
class Attention1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Attention1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.attention = nn.Linear(128, 1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        attn_scores = torch.softmax(self.attention(x), dim=1)
        x = (x * attn_scores).sum(dim=1)

        out = self.fc(x)
        return out

# ========== TRAINING FUNCTION ==========
def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = PoseSequenceDataset(POSE_DATA_DIR)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Attention1DCNN(input_dim=NUM_KEYPOINTS*2, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    train_losses, train_accuracies, val_accuracies = [], [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, MODEL_SAVE_PATH))
            print(f"\u2705 Saved new best model with Val Acc: {best_val_acc:.4f}")

    # Plot Loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))

    # Plot Accuracy
    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral", "Fighting"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))

    # Final Accuracies
    with open(os.path.join(RESULTS_DIR, "final_accuracy.txt"), "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Final Training Accuracy: {train_accuracies[-1]:.4f}\n")

    print("Training finished! Metrics and figures saved.")

# ========== RUN ==========
if __name__ == "__main__":
    train()
