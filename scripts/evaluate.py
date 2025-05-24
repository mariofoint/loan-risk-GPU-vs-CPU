import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np

# Load Test Data
X_test = pd.read_csv("../data/processed/X_test.csv").astype('float32').values
y_test = pd.read_csv("../data/processed/y_test.csv").values.reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Load Model Definition
class LoanRiskModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

#Load Trained Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LoanRiskModel(input_dim=X_test.shape[1]).to(device)
model.load_state_dict(torch.load("../models/loan_model.pt"))
model.eval()

# Predict on Test Data
with torch.no_grad():
    outputs = model(X_test.to(device))
    predictions = outputs.cpu().numpy()

# (Sigmoid → binary class)
y_pred_labels = (predictions >= 0.5).astype(int)

# metrics
acc = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)
auc = roc_auc_score(y_test, predictions)
cm = confusion_matrix(y_test, y_pred_labels)

print("✅ Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"AUC:       {auc:.4f}")
print("Confusion Matrix:")
print(cm)
