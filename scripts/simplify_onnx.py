from onnxsim import simplify
import onnx
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
onnx_path = base_path / "models" / "loan_model.onnx"
output_path = base_path / "models" / "loan_model_optimized.onnx"

# Load the original model
model = onnx.load(str(onnx_path))

# Simplify it (constant folding, shape inference, etc.)
model_simplified, check = simplify(model)

assert check, "ONNX simplification failed!"

# Save the simplified model
onnx.save(model_simplified, output_path)

print(f"✅ Optimized ONNX model saved to: {output_path}")
