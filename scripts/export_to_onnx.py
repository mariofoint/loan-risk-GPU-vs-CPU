from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd

# === Setup Paths ===
base_path = Path(__file__).resolve().parent
x_train_path = base_path.parent / "data" / "processed" / "X_train.csv"
model_path = base_path.parent / "models" / "loan_model.pt"
onnx_path = base_path.parent / "models" / "loan_model.onnx"

# === Load Sample Input ===
X_sample = pd.read_csv(x_train_path).astype('float32')
dummy_input = torch.randn(1, X_sample.shape[1], dtype=torch.float32)


# === Model Definition ===
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

# === Load Model Weights ===
model = LoanRiskModel(input_dim=X_sample.shape[1])
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# === Export to ONNX ===
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # Include weights
    input_names=["input"],
    output_names=["output"],
    opset_version=12,      # Use newer opset for best compatibility
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ Model exported to {onnx_path} with dynamic batch size support")
