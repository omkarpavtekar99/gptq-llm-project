"""Prompt library for the Phase 1 benchmark."""

from __future__ import annotations

from mizan.serving.models import PromptSample

PHASE1_PROMPTS: list[PromptSample] = [
    PromptSample(prompt_id="p01", prompt="Explain what a GPU KV cache does in one paragraph.", expected_output="A KV cache stores attention keys and values from prior tokens so the model can reuse them during generation, reducing repeated computation and improving latency."),
    PromptSample(prompt_id="p02", prompt="Summarize why batching improves LLM serving efficiency.", expected_output="Batching improves efficiency by sharing GPU work across multiple requests, increasing throughput and utilization while often reducing per-token overhead."),
    PromptSample(prompt_id="p03", prompt="What is time to first token in LLM serving?", expected_output="Time to first token is the latency between sending a request and receiving the first generated token from the model."),
    PromptSample(prompt_id="p04", prompt="Define inter-token latency in one sentence.", expected_output="Inter-token latency is the average delay between consecutive generated tokens after the first token arrives."),
    PromptSample(prompt_id="p05", prompt="Give two tradeoffs of quantizing an LLM.", expected_output="Quantization reduces memory use and can improve speed, but it may slightly reduce model quality or numerical stability."),
    PromptSample(prompt_id="p06", prompt="Why might GPU memory utilization be capped below 1.0?", expected_output="Capping GPU memory utilization leaves headroom for runtime overhead, KV cache growth, and prevents out-of-memory crashes during concurrent requests."),
    PromptSample(prompt_id="p07", prompt="What is GPTQ in the context of LLMs?", expected_output="GPTQ is a post-training quantization method that compresses model weights, often to 4-bit precision, while preserving much of the original model quality."),
    PromptSample(prompt_id="p08", prompt="Explain why concurrency affects throughput and latency.", expected_output="Higher concurrency can raise throughput by keeping the GPU busy, but it may also increase queueing and latency for individual requests."),
    PromptSample(prompt_id="p09", prompt="Describe a good health check for an OpenAI-compatible server.", expected_output="A good health check confirms the server is reachable and can answer a lightweight completion request or list models successfully."),
    PromptSample(prompt_id="p10", prompt="What does max_num_batched_tokens control?", expected_output="max_num_batched_tokens limits the total tokens the scheduler can batch together across active requests at one time."),
    PromptSample(prompt_id="p11", prompt="Why measure both latency and throughput in benchmarking?", expected_output="Latency captures responsiveness while throughput measures overall capacity, so both are needed to pick the best serving configuration."),
    PromptSample(prompt_id="p12", prompt="Write one sentence on why WSL2 GPU support matters for local ML infra.", expected_output="WSL2 GPU support lets Linux-based ML tooling access Windows-hosted NVIDIA hardware for local development and benchmarking."),
    PromptSample(prompt_id="p13", prompt="What is the purpose of MLflow in an experimentation loop?", expected_output="MLflow records parameters, metrics, and artifacts so experiments can be compared, reproduced, and tracked over time."),
    PromptSample(prompt_id="p14", prompt="Define throughput in tokens per second.", expected_output="Throughput in tokens per second is the number of output tokens generated per second across one request or a workload."),
    PromptSample(prompt_id="p15", prompt="Why should benchmark reports include VRAM measurements?", expected_output="VRAM measurements show whether a configuration fits the hardware budget and help explain throughput or stability differences."),
    PromptSample(prompt_id="p16", prompt="What is a golden prompt set?", expected_output="A golden prompt set is a fixed collection of representative prompts and reference answers used to compare model behavior across runs."),
    PromptSample(prompt_id="p17", prompt="Explain why CPU-offloaded inference is useful as a reference baseline.", expected_output="CPU-offloaded inference is useful as a slower quality reference when GPU-optimized runs need to be compared against a less resource-constrained execution path."),
    PromptSample(prompt_id="p18", prompt="What does an OpenAI-compatible API provide?", expected_output="An OpenAI-compatible API exposes familiar completion and chat endpoints so existing clients and tooling can work with alternative model servers."),
    PromptSample(prompt_id="p19", prompt="Why is a warmup phase important before benchmarking?", expected_output="Warmup removes one-time initialization overhead so the measured benchmark better reflects steady-state serving behavior."),
    PromptSample(prompt_id="p20", prompt="Give one sentence on why benchmark summaries should round results consistently.", expected_output="Consistent rounding makes reports easier to compare and avoids misleading precision in benchmark results."),
]


def get_phase1_prompts(limit: int = 20) -> list[PromptSample]:
    """Return the fixed prompt sample used for Phase 1."""

    return PHASE1_PROMPTS[:limit]
