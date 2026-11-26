# Physical ceilings

Use measured device-copy bandwidth and sustained large-GEMM FLOPs. A practical prefill ceiling uses 50% of measured peak by default.

## Decode

Decode is normally memory-bandwidth bound:

```text
verify_bytes = dense + lm_head + effective_routed_experts + kv_bytes_per_token × context
step_bytes   = verify_bytes + draft_steps × (draft_layer + draft_head)
decode_tok_s = bandwidth_bytes_s × accepted_tokens × concurrency / step_bytes
```

For MoE concurrency, approximate the routed-expert union by:

```text
union = 1 - (1 - top_k / n_experts) ** ((drafts + 1) × concurrency)
```

Checkpoint tensor sizes are preferred over `parameters × bits`. The latter is only an estimate and must be labelled.

## Prefill

```text
flops_per_token ≈ 2 × active_linear_params
                + 4 × full_attention_layers × heads × head_dim × (context / 2)
prefill_tok_s = 0.5 × measured_peak_flops / flops_per_token
```

Select the FP16, BF16 or INT8 peak that matches the measured workload. Do not claim an INT8 ceiling for an FP16 workload.

## KV capacity

```text
kv_available = vram_cap - resident_weights - runtime_reserve
max_tokens   = floor(kv_available / kv_bytes_per_token)
max_streams  = floor(max_tokens / required_context)
```

Respect the effective cap imposed by `GPUAllocator`, including headroom and existing reservations. A theoretical KV fit is not permission to oversubscribe the allocator.

## Progress

For higher-is-better metrics, `achieved / ceiling × 100`. For TTFT and startup, compare against their estimated floors and explain scheduling/tokenization overhead separately. Reject a target above the applicable physical ceiling before starting a hill-climb.

Adapted from the MIT-licensed `inference-optimisation-skill` reference methodology.
