# Optimization levers

Change one lever per experiment. These are priors, not promises; applicability depends on the declared engine and hardware.

1. **Persistent caches:** retain JIT, autotune, graph and compilation caches to reduce startup.
2. **Decode graphs:** enable CUDA/XPU/HIP graphs only when the engine supports the model and shape.
3. **Attention implementation:** test supported flash/fused attention backends without assuming one is universal.
4. **KV layout and dtype:** measure both speed and needle accuracy. Quantized KV can be slower on some models.
5. **Batch and microbatch:** tune prefill chunking against concurrent decode stalls.
6. **Speculative decoding:** sweep draft depth for each context/concurrency; measure accepted tokens and step cost separately.
7. **Draft-head vocabulary:** prune only when target verification remains exact and the engine supports it.
8. **GPU sampling:** avoid host synchronization when supported.
9. **Concurrency/KV sizing:** size request and KV pools deliberately, then prove them with a full-KV soak.
10. **Tensor parallelism and GPU layers:** use only for multi-GPU or partial-offload configurations, with memory and interconnect measurements.

Process flags require a managed `launch`/`stop` lifecycle. For an already-running unknown endpoint, restrict candidates to documented OpenAI request fields in `extra_body`; never send engine-specific flags blindly.

Prioritize `ceiling_gap × likelihood / cost`. Stop after three consecutive rejected levers and report the remaining bottleneck.
