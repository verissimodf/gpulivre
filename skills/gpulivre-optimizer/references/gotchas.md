# Hardware and server gotchas

## All backends

- A second GPU workload invalidates the comparison and can cause OOM.
- Run on stable power and record clocks before and after each trial.
- Long-context proxy timeouts are not decode measurements. Test the engine directly.
- A faster response with empty or looping content is a failed trial.
- GPULivre's current preemption status is accounting, not proof that physical VRAM was released.

## NVIDIA CUDA

- Tensor Core unlock state is meaningful only on NVIDIA and may remain unknown.
- Ampere has no native FP8 units; choose quantization from measured kernels, not format labels.
- Include NVML process memory in the footprint.

## AMD ROCm

- PyTorch may expose ROCm through the CUDA-compatible API while the backend is ROCm.
- Graph support and stability vary by engine and version; validate output and memory after every change.

## Intel XPU/Arc

- Unified-memory allocations may be invisible to cgroup RSS. Include DRM driver residency.
- Graph capture and large prefill shapes can consume substantial transient memory.

## Apple MPS

- Use actual unified-memory and wired-limit information; never assume a fixed memory size.
- Deep speculation can lose on bandwidth-limited chips.

## OpenAI-compatible engines

- vLLM and SGLang may emit several tokens per SSE chunk under speculative decoding; final `usage` is authoritative.
- llama.cpp and ExLlama-family servers require the same exact needle and memory gates as any other engine.
- Ollama's OpenAI endpoint may keep models resident after a request; include that residency in later measurements.
- Missing `usage` requires a real tokenizer supplied by the caller. There is no chunk-count fallback.
