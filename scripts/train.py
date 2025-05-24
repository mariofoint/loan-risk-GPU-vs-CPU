import torch
import torch.nn as nn
import pandas as pd
import os
from pathlib import Path
import pandas as pd

# Dynamically resolve paths relative to this script
base_dir = Path(__file__).resolve().parent.parent
X_train_path = base_dir / "data" / "processed" / "X_train.csv"
y_train_path = base_dir / "data" / "processed" / "y_train.csv"

X_train = pd.read_csv(X_train_path).astype('float32').values
y_train = pd.read_csv(y_train_path).astype('float32').values.reshape(-1, 1)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)  # shape: [n_samples, n_features]
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # shape: [n_samples, 1]

# Use GPU if Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Define the Model
# This is a 3-layer neural network: input → hidden → hidden → output
# It uses ReLU for non-linearity and sigmoid at the end to produce probability [0, 1]
class LoanRiskModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# initialize model and move to device
model = LoanRiskModel(input_dim=X_train.shape[1]).to(device)

# The Loss Function and Optimizer
# Binary Cross-Entropy Loss since it is a binary classification problem
loss_fn = nn.BCELoss()  # measures error between predicted probs and true labels
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # adaptive learning rate

# Train the Model 
epochs = 100  # Number of full passes over the dataset

for epoch in range(epochs):
    model.train()  # set model to training mode
    # Ensure data is on the correct device (GPU or CPU)

    # move inputs and targets to device
    inputs = X_train.to(device)
    targets = y_train.to(device)

    # Forward pass: predict y from X
    outputs = model(inputs)

    # Compute loss between predicted and true values
    loss = loss_fn(outputs, targets)

    # Backpropagation: calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Gradient descent step
    optimizer.step()

    # Print progress
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

# Save the Model
# This stores learned weights (not the structure), which can be reloaded or exported.
os.makedirs("../models", exist_ok=True)
print(f"Saving model to: {Path('../models/loan_model.pt').resolve()}")
models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "loan_model.pt"
torch.save(model.state_dict(), model_path)

print(f"✅ Model saved to: {model_path}")
torch.save(model.state_dict(), "../models/loan_model.pt")
print("Model saved to /models/loan_model.pt")
