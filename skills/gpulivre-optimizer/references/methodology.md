# Measurement methodology

## Tokens and time

- Build prompts before starting the clock and make cold prompts unique to defeat prefix caching.
- TTFT ends at the first content token; decode uses `(completion_tokens - 1) / decode_window`.
- Cold prefill uses server-reported prompt tokens divided by TTFT.
- Concurrent throughput is total completion tokens divided by the wall interval from first start to last finish.
- Keep a fixed prompt panel; temperature and content affect speculative acceptance.

## Noise

- Expect run-to-run variation of roughly ±5–10%.
- Use `ctl → cand → ctl`, at least two samples per leg, and medians.
- A candidate must beat both neighbouring controls. Discard trials with power, clock, memory or request flags.
- Do not compare numbers obtained on different hardware, engines or days as if they were controlled.

## Memory

Sample the complete footprint: process anonymous memory plus accelerator/driver residency. On Linux unified memory, deduplicate DRM clients before summing `drm-resident-*`. The watchdog aborts the trial at the cap and records its peak. It never force-kills an unknown server.

## Quality

Use three random retrieval codes at different context depths and require every code exactly. Then run concurrent full-context requests to saturate the intended KV pool while staying under the allocator cap. Needle and soak are production gates, not optional speed metrics.

Adapted from the MIT-licensed `inference-optimisation-skill` methodology.
