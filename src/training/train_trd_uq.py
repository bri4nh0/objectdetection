import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_trd_uq_training_data():
    """Generate training data with temporal patterns for TRD-UQ"""
    sequences = []
    labels = []
    
    # Generate different risk patterns
    patterns = ['escalating', 'deescalating', 'volatile', 'stable']
    
    for pattern in patterns:
        for _ in range(250):  # 250 sequences per pattern
            seq_length = 30
            sequence = []
            
            if pattern == 'escalating':
                # Gradually increasing risk
                base_risk = np.linspace(0.5, 3.5, seq_length)
                noise = np.random.normal(0, 0.2, seq_length)
            elif pattern == 'deescalating':
                # Gradually decreasing risk
                base_risk = np.linspace(3.0, 0.5, seq_length)
                noise = np.random.normal(0, 0.2, seq_length)
            elif pattern == 'volatile':
                # High variance risk
                base_risk = np.random.normal(2.0, 1.0, seq_length)
                noise = np.random.normal(0, 0.3, seq_length)
            else:  # stable
                # Stable risk with small variations
                base_level = np.random.uniform(1.0, 2.0)
                base_risk = np.ones(seq_length) * base_level
                noise = np.random.normal(0, 0.1, seq_length)
            
            # Apply noise and clip
            risk_sequence = np.clip(base_risk + noise, 0, 4.0)
            
            # Create multi-dimensional features for each time step
            for i in range(seq_length):
                obj_risk = risk_sequence[i] / 4.0  # Normalize
                beh_risk = 1 if risk_sequence[i] > 2.0 else 0  # High risk → fighting behavior
                prox_risk = min(risk_sequence[i] / 3.0, 1.0)  # Proportional to risk
                
                sequence.append([obj_risk, beh_risk, prox_risk])
            
            sequences.append(sequence)
            # Label: final risk level and pattern type
            labels.append([risk_sequence[-1], patterns.index(pattern)])
    
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

def train_trd_uq_model():
    print("🚀 Training TRD-UQ Fusion Model with Temporal Patterns...")
    
    # Generate training data
    sequences, labels = generate_trd_uq_training_data()
    
    # Use only the last frame for initial fusion training
    X = sequences[:, -1, :]  # Last frame of each sequence
    y = labels[:, 0]  # Final risk level
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val).unsqueeze(1)
    
    # Initialize model
    from src.core.trd_uq_system import BayesianFusionLayer
    model = BayesianFusionLayer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(150):
        # Training
        optimizer.zero_grad()
        predictions, uncertainties = model(X_train_t, mc_dropout=False)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        with torch.no_grad():
            val_pred, _ = model(X_val_t, mc_dropout=False)
            val_loss = criterion(val_pred, y_val_t)
            val_losses.append(val_loss.item())
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch}/150 | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), "models/fusion/trd_uq_fusion.pth")
    print("✅ TRD-UQ model saved as models/fusion/trd_uq_fusion.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('TRD-UQ Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/trd_uq_training_curve.png')
    plt.show()
    
    print("🎉 TRD-UQ training completed!")

if __name__ == "__main__":
    train_trd_uq_model()