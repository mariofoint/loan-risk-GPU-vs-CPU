import onnx
import onnxoptimizer

model = onnx.load("models/loan_model.onnx")
passes = onnxoptimizer.get_available_passes()
optimized = onnxoptimizer.optimize(model, passes)
onnx.save(optimized, "models/loan_model_optimized.onnx")
print("✅ Optimized ONNX model saved.")
