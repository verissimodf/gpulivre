---
name: gpulivre-optimizer
description: Optimize any OpenAI-compatible local LLM server on CUDA, ROCm, XPU, MPS or unified memory using GPULivre ceilings, interleaved benchmarks, VRAM watchdogs, quality gates and an append-only experiment ledger. Use for hill-climb, tok/s, TTFT, prefill, KV cache, speculative decoding or inference tuning requests.
---

# GPULivre Optimizer

Improve inference serving with measurements, not guesses.

## Required sequence

1. Read prior failures with `gpulivre_get_ledger_history`.
2. Measure hardware bandwidth and FLOPs; never infer them from a GPU marketing name.
3. Compute the physical ceiling with `gpulivre_compute_ceiling`.
4. Reserve the experiment's VRAM through `GPUAllocator` and serialize all GPU work.
5. Measure a pinned baseline with real `usage` token counts or a real tokenizer.
6. Change exactly one lever.
7. Run `ctl → cand → ctl`, at least two samples per leg, and compare medians.
8. Accept only a candidate that beats both controls and regresses nothing by more than 3%.
9. Require exact long-context needles and a concurrent full-KV soak before production promotion.
10. Append every success and failure to the ledger and PLAN report.

## Hard rules

- Never count SSE chunks as tokens.
- Never publish cache-hit prefill as cold prefill.
- Never run candidates in parallel on the same accelerator.
- Never kill a server by a process-name pattern. Abort, drain, then stop only a managed server.
- Stop a trial when the memory watchdog trips. Unified-memory accounting must include driver residency.
- Reject empty, looping or failed generations even when their apparent speed is high.
- Keep P1 interactive work protected. P1 optimizes TTFT, P2 decode with TTFT guards, and P3 aggregate throughput.
- Missing physical measurements produce unknown ceilings, not fabricated values.

## References

- `references/ceilings.md`: decode, prefill and KV arithmetic.
- `references/methodology.md`: sampling, memory and measurement traps.
- `references/levers.md`: ordered optimization levers.
- `references/gotchas.md`: hardware and engine caveats.

## MCP workflow

Use `gpulivre_run_interleaved_bench` for one candidate. Use `gpulivre_hill_climb` for a bounded session of one-lever candidates. Inspect the returned decision, flags and gates before declaring a winner.

Report baseline, candidate, ceiling percentage, sample count, memory peak and every limitation. Distinguish mocked/unit-tested backends from hardware actually measured.

## Attribution

Methodology adapted from [sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill), Copyright (c) 2026 Sybil Solutions contributors, MIT License. This integration is provided by GPULivre under the repository's MIT license.
