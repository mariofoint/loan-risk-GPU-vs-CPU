import onnxruntime as ort
import pandas as pd
import numpy as np
import time
from pathlib import Path

# === Config ===
BATCH_SIZE = 128
NUM_BATCHES = 100000  # run 100 batches to measure sustained performance
SAMPLES = 128      # input shape must match exported ONNX batch size

# === Paths ===
base_path = Path(__file__).resolve().parent
model_path = base_path.parent / "models" / "loan_model_optimized.onnx"
X_test_path = base_path.parent / "data" / "processed" / "X_test.csv"

# === Load and prepare test data ===
X_test = pd.read_csv(X_test_path).astype('float32').values
X_benchmark = X_test[:SAMPLES].astype('float32')

# === Benchmarking Function: Repeat the batch N times ===
def benchmark_repeated_batch(session, input_name, batch_input, num_batches=NUM_BATCHES):
    times = []
    for _ in range(5):  # repeat 5 times for averaging
        start = time.perf_counter()
        for _ in range(num_batches):
            session.run(None, {input_name: batch_input})
        end = time.perf_counter()
        total_time = end - start
        avg_per_sample = total_time / (num_batches * BATCH_SIZE)
        times.append(avg_per_sample)
    return np.mean(times) * 1000  # ms/sample

# === Benchmark ===
print(f"🔁 Benchmarking 100x batch runs of size {BATCH_SIZE}...\n")

providers = ort.get_available_providers()
input_data = X_benchmark  # shape [128, input_dim]

# CPU
session_cpu = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
input_name = session_cpu.get_inputs()[0].name
cpu_latency = benchmark_repeated_batch(session_cpu, input_name, input_data)
print(f"🧠 CPU Latency (Batch x{NUM_BATCHES}): {cpu_latency:.5f} ms/sample")

# GPU
if "CUDAExecutionProvider" in providers:
    session_gpu = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    input_name = session_gpu.get_inputs()[0].name
    gpu_latency = benchmark_repeated_batch(session_gpu, input_name, input_data)
    print(f"⚡ GPU Latency (Batch x{NUM_BATCHES}): {gpu_latency:.5f} ms/sample")
    print(f"🚀 GPU Speedup: {cpu_latency / gpu_latency:.2f}×")
else:
    gpu_latency = None
    print("⚠️ CUDAExecutionProvider not available.")

# TensorRT
if "TensorrtExecutionProvider" in providers:
    session_trt = ort.InferenceSession(str(model_path), providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
    input_name = session_trt.get_inputs()[0].name
    trt_latency = benchmark_repeated_batch(session_trt, input_name, input_data)
    print(f"🔥 TensorRT Latency (Batch x{NUM_BATCHES}): {trt_latency:.5f} ms/sample")
    print(f"🚀 TensorRT Speedup: {cpu_latency / trt_latency:.2f}×")
else:
    print("⚠️ TensorRT provider not available.")
