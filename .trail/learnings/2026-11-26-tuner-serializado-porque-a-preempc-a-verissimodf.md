---
title: Tuner serializado porque a preempção do allocator é apenas contábil
author: Verissimodf
date: 2026-11-26
branch: feature/universal-inference-tuner
id: l-68723743
scope: repo
status: active
src: agent
touches:
  - CHANGELOG.md
  - README.md
  - docs/ARQUITETURA.md
  - gpulivre/core/monitor.py
  - gpulivre/mcp/server.py
  - tasks/lessons.md
---

## Decided
- O hill-climb executa experimentos serializados e cooperativos, sem depender da preempção do GPUAllocator para liberar VRAM fisicamente. request_slot marca slots como "preempted" sem descarregar o workload (allocator.py), então confiar nele entre trials produziria tetos e benchmarks falsos.
## Watch out
- Corrigir preempção física, oversubscription (tracked_allocated nunca entra no cálculo de disponibilidade) e corridas em _active_slots continua pendente; qualquer feature que dependa de liberação física real esbarra nisso.
## In the code:
- gpulivre/core/hill_climber.py (execução serializada ctl-cand-ctl)
- gpulivre/core/bench_harness.py (watchdog aborta amostra; stop só com lifecycle gerenciado)
- gpulivre/core/allocator.py:L145-187 (preempção contábil)
