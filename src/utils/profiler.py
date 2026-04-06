"""GPU profiling utilities for latency, throughput, and memory measurement."""

import time
import torch


class InferenceProfiler:
    """Profiles inference runs and collects efficiency metrics."""

    def __init__(self, num_warmup=3, num_runs=10):
        self.num_warmup = num_warmup
        self.num_runs = num_runs

    def profile(self, inference_fn, **kwargs):
        """Run inference_fn multiple times and collect metrics.

        Args:
            inference_fn: Callable that performs a single inference step.

        Returns:
            dict with latency_ms, throughput, peak_memory_mb.
        """
        device = torch.device("cuda")

        # Warmup
        for _ in range(self.num_warmup):
            inference_fn(**kwargs)
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        latencies = []

        for _ in range(self.num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            inference_fn(**kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        avg_latency = sum(latencies) / len(latencies)

        return {
            "latency_ms": avg_latency,
            "latency_std_ms": (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5,
            "throughput_samples_per_sec": 1000.0 / avg_latency,
            "peak_gpu_memory_mb": peak_memory_mb,
        }

    @staticmethod
    def find_max_batch_size(inference_fn, min_bs=1, max_bs=64, **kwargs):
        """Binary search for the maximum batch size before OOM.

        Args:
            inference_fn: Callable(batch_size=N, **kwargs) that runs inference.

        Returns:
            Maximum feasible batch size.
        """
        best = min_bs
        lo, hi = min_bs, max_bs
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                torch.cuda.empty_cache()
                inference_fn(batch_size=mid, **kwargs)
                torch.cuda.synchronize()
                best = mid
                lo = mid + 1
            except torch.cuda.OutOfMemoryError:
                hi = mid - 1
                torch.cuda.empty_cache()
        return best
