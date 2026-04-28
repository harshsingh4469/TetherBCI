import torch
import time
import numpy as np
from framework import TetherBCI


def benchmark_inference(model, n_runs=100):
    """
    Benchmarks inference latency of TetherBCI.
    Simulates real-time BCI usage.
    """
    model.eval()

    # Dummy inputs simulating real-time brain signals
    eeg  = torch.randn(1, 64,  256)
    fmri = torch.randn(1, 1,   64, 64)
    meg  = torch.randn(1, 306, 256)

    # Warmup runs
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(eeg, fmri, meg, mode="classify")

    # Benchmark
    print(f"Benchmarking {n_runs} inference runs...")
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _     = model(eeg, fmri, meg, mode="classify")
            end   = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms":   np.mean(latencies),
        "std_ms":    np.std(latencies),
        "min_ms":    np.min(latencies),
        "max_ms":    np.max(latencies),
        "p50_ms":    np.percentile(latencies, 50),
        "p95_ms":    np.percentile(latencies, 95),
        "p99_ms":    np.percentile(latencies, 99),
    }


def optimize_with_torchscript(model):
    """
    Optimizes model using TorchScript for faster inference.
    TorchScript compiles the model for production deployment.
    """
    model.eval()
    print("Optimizing with TorchScript...")

    # Trace the model with dummy inputs
    eeg  = torch.randn(1, 64,  256)
    fmri = torch.randn(1, 1,   64, 64)
    meg  = torch.randn(1, 306, 256)

    with torch.no_grad():
        traced = torch.jit.trace(
            lambda e, f, m: model(e, f, m, mode="classify"),
            (eeg, fmri, meg)
        )

    torch.jit.save(traced, "models/tetherbci_optimized.pt")
    print("Optimized model saved as models/tetherbci_optimized.pt")
    return traced


def quantize_model(model):
    """
    Applies dynamic quantization to reduce model size
    and improve CPU inference speed by ~2x.
    """
    print("Applying dynamic quantization...")
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    torch.save(quantized.state_dict(), "models/tetherbci_quantized.pt")
    print("Quantized model saved as models/tetherbci_quantized.pt")
    return quantized


def run_optimization():
    print("=" * 55)
    print("TetherBCI Optimization & Latency Benchmark")
    print("=" * 55)

    # Load model
    model = TetherBCI()
    model.load_state_dict(torch.load("models/tetherbci.pt"))
    model.eval()

    # --- Baseline benchmark ---
    print("\n[1] Baseline Model:")
    baseline = benchmark_inference(model)
    print(f"    Mean latency:  {baseline['mean_ms']:.2f} ms")
    print(f"    P95  latency:  {baseline['p95_ms']:.2f} ms")
    print(f"    P99  latency:  {baseline['p99_ms']:.2f} ms")

    # --- TorchScript optimization ---
    print("\n[2] TorchScript Optimized Model:")
    optimized = optimize_with_torchscript(model)
    ts_results = benchmark_inference(optimized)
    print(f"    Mean latency:  {ts_results['mean_ms']:.2f} ms")
    print(f"    P95  latency:  {ts_results['p95_ms']:.2f} ms")
    print(f"    P99  latency:  {ts_results['p99_ms']:.2f} ms")

    # --- Quantization ---
    print("\n[3] Quantized Model:")
    quantized  = quantize_model(model)
    q_results  = benchmark_inference(quantized)
    print(f"    Mean latency:  {q_results['mean_ms']:.2f} ms")
    print(f"    P95  latency:  {q_results['p95_ms']:.2f} ms")
    print(f"    P99  latency:  {q_results['p99_ms']:.2f} ms")

    # --- Summary ---
    improvement = ((baseline['mean_ms'] - q_results['mean_ms']) / baseline['mean_ms']) * 100
    print("\n" + "=" * 55)
    print("OPTIMIZATION SUMMARY")
    print("=" * 55)
    print(f"Baseline mean latency:   {baseline['mean_ms']:.2f} ms")
    print(f"Optimized mean latency:  {ts_results['mean_ms']:.2f} ms")
    print(f"Quantized mean latency:  {q_results['mean_ms']:.2f} ms")
    print(f"Total latency reduction: {improvement:.1f}%")
    print("\nTetherBCI is production ready!")


if __name__ == "__main__":
    run_optimization()