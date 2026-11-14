# GPU Unlock Benchmark — unlocked-cuda-coexist

- **Timestamp:** 2026-11-13T22:12:06+00:00
- **Host:** app (7.0.0-43-generic)
- **GPU:** NVIDIA GeForce RTX 3060 (12288 MiB, driver 595.99.02)
- **gpu_unlock active:** True (ss0=0x88888888)
- **Commit:** 6ab57b6

| Workload | Status | Metrics |
| :--- | :--- | :--- |
| whisper | ok | 14.132s audio in 0.3297s (42.86x realtime) |

> `micro` measures BF16 with FP32 accumulation through cuBLAS — the exact
> tensor-core rate that gpu_unlock changes. Whisper and Stable Diffusion
> numbers reflect whatever backend their service is configured to use; if
> they run on CPU, the boot-time unlock does not change them.
