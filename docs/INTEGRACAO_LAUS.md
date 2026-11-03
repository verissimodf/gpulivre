# Integração: Laus & GPULivre

Este documento especifica a orquestração do ecossistema autônomo `Laus` (Pilar 2 — MiroFish Swarm & Pilar 1 — Jarvis Orchestrator) com o **GPULivre**.

---

## 1. Concorrência de Enxames de IA (MiroFish Swarm)
O enxame MiroFish executa simulações com até 20 personas paralelas para prever consenso de mercado e validar ordens financeiras. Cada simulação pode requerer entre 1 e 4 GB de VRAM.

No padrão **GPULivre**, essas simulações são classificadas como **`P3_BATCH` (Prioridade 3)**:
- Podem ser preemptadas graciosamente por tarefas humanas de voz (`P1_CRITICAL`).
- Se a VRAM local estiver 100% saturada, são desviadas automaticamente para instâncias remotas de GPU sem interromper a esteira de análise.

---

## 2. Uso do `SwarmSimulationAdapter`

```python
from gpulivre.adapters.swarm_adapter import SwarmSimulationAdapter
from gpulivre.core.allocator import GPUAllocator
from gpulivre.core.hybrid_fallback import hybrid_fallback_engine

adapter = SwarmSimulationAdapter(
    allocator=GPUAllocator(),
    fallback_engine=hybrid_fallback_engine
)

def rodar_consenso_mirofish(dados_mercado):
    return swarm_orchestrator.run_simulation(dados_mercado)

# Execução protegida com fallback automático
resultado = adapter.execute_simulation(
    task_name="mirofish_arbitrage_consensus",
    simulation_callable=rodar_consenso_mirofish,
    payload={"steps": 20, "agents_count": 20, "pair": "BTC/BRL"},
    dados_mercado=orderbook,
    required_vram_mb=2048.0
)
```

---

## 3. Comportamento sob Preempção e Fallback

1. **Memória Local Disponível:** A simulação roda acelerada na GPU local.
2. **Chegada de Áudio / Voz Concorrente:** O slot P3 é marcado como `preempted`. O enxame pausa temporariamente e retoma assim que o áudio for liberado.
3. **Memória 100% Saturada:** O adaptador aciona imediatamente o `HybridFallbackEngine`, que empacota o job e despacha para o **Google Colab** (GPU T4/A100) via `colab-drive-modularizer`.
4. **Modo Bypass Legado:** Se o usuário desabilitou o GPULivre, a simulação roda localmente sem orquestração.
