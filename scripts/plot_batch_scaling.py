import onnxruntime as ort
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# === Config ===
BATCH_SIZES = [1, 10, 100, 1000, 5000, 10000]
REPEATS = 3

# === Paths ===
base_path = Path(__file__).resolve().parent
model_path = base_path.parent / "models" / "loan_model_optimized.onnx"
X_test_path = base_path.parent / "data" / "processed" / "X_test.csv"

# === Load data
X_full = pd.read_csv(X_test_path).astype('float32').values
input_dim = X_full.shape[1]

# === Benchmarking function
def benchmark_batch(session, input_name, batch_size):
    times = []
    for _ in range(REPEATS):
        x_batch = X_full[:batch_size].astype('float32')
        start = time.perf_counter()
        session.run(None, {input_name: x_batch})
        end = time.perf_counter()
        avg = (end - start) / batch_size
        times.append(avg)
    return np.mean(times) * 1000  # ms/sample

# === Prepare sessions
def get_session(provider):
    return ort.InferenceSession(str(model_path), providers=[provider])

input_name = ort.InferenceSession(str(model_path)).get_inputs()[0].name
providers = ort.get_available_providers()

session_cpu = get_session("CPUExecutionProvider")
session_gpu = get_session("CUDAExecutionProvider") if "CUDAExecutionProvider" in providers else None
session_trt = get_session("TensorrtExecutionProvider") if "TensorrtExecutionProvider" in providers else None

# === Collect results
results = {"Batch Size": [], "CPU": [], "GPU": [], "TensorRT": []}
print(f"🧪 Dataset has {len(X_full)} samples\n")

for size in BATCH_SIZES:
    if size > len(X_full):
        print(f"⚠️ Skipping batch size {size} (dataset too small)")
        continue

    print(f"🔁 Benchmarking batch size: {size}")
    results["Batch Size"].append(size)

    cpu = benchmark_batch(session_cpu, input_name, size)
    results["CPU"].append(cpu)
    print(f"  🧠 CPU: {cpu:.5f} ms/sample")

    if session_gpu:
        gpu = benchmark_batch(session_gpu, input_name, size)
        results["GPU"].append(gpu)
        print(f"  ⚡ GPU: {gpu:.5f} ms/sample")
    else:
        results["GPU"].append(np.nan)

    if session_trt:
        trt = benchmark_batch(session_trt, input_name, size)
        results["TensorRT"].append(trt)
        print(f"  🔥 TensorRT: {trt:.5f} ms/sample")
    else:
        results["TensorRT"].append(np.nan)

# === Plot results
plt.figure(figsize=(10, 6))
plt.plot(results["Batch Size"], results["CPU"], marker='o', label="CPU")
plt.plot(results["Batch Size"], results["GPU"], marker='o', label="GPU")
plt.plot(results["Batch Size"], results["TensorRT"], marker='o', label="TensorRT")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Batch Size (log scale)")
plt.ylabel("Latency per Sample (ms, log scale)")
plt.title("Inference Latency vs Batch Size (Log-Log)")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()

plot_path = base_path.parent / "models" / "batch_scaling_plot.png"
plt.savefig(plot_path)
plt.show()
print(f"✅ Plot saved to {plot_path}")
