# Phase 1 Benchmark Summary

## Winning Configuration
- max_num_batched_tokens: `512`
- concurrent_requests: `4`
- gpu_memory_utilization: `0.9900`
- kv_cache_dtype: `auto`
- dtype: `float16`
- quantization: `gptq`

## Best Measured Sweep Result
- avg_ttft_ms: `245.3743`
- avg_itl_ms: `23.5052`
- throughput_tokens_per_sec: `30.3538`
- peak_vram_mb: `8089.2070`

## Quantization Comparison
### gptq_4bit_vllm
- rouge_l: `0.1492`
- throughput_tokens_per_sec: `34.3976`
- peak_vram_mb: `8089.2070`
- average_latency_ms: `3114.3809`
- notes: Primary benchmarked vLLM deployment.

### cpu_reference_transformers
- rouge_l: `0.0000`
- throughput_tokens_per_sec: `0.0000`
- peak_vram_mb: `0.0000`
- average_latency_ms: `0.0000`
- notes: Skipped on this machine because VLLM__ENABLE_CPU_REFERENCE is disabled. This avoids OOM risk on low-RAM WSL setups.

## Recommendation
Lock in the GPTQ 4-bit vLLM deployment for subsequent phases because it successfully fits the RTX 4060 Laptop and delivered 30.3538 tokens/sec with 245.3743 ms TTFT at max_num_batched_tokens=512, concurrent_requests=4, and gpu_memory_utilization=0.9900. The CPU reference baseline was skipped on this machine due to host RAM limits, so the production GPTQ path should remain the locked configuration for subsequent phases. Notes from the skipped CPU baseline: Skipped on this machine because VLLM__ENABLE_CPU_REFERENCE is disabled. This avoids OOM risk on low-RAM WSL setups.
