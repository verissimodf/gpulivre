# Changelog — GPULivre

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

## [0.2.0] - 2026-11-02

### Adicionado
- **Mecanismo de Toggle pelo Usuário (`UserConfigManager`):** Permite ao usuário do `AssistentePessoal` habilitar ou desabilitar o GPULivre e o Fallback Híbrido em tempo de execução com persistência em `~/.antigravity/gpulivre_user_config.json`.
- **Motor de Fallback Híbrido (`HybridFallbackEngine`):** Desvio automático de simulações MiroFish para o Google Colab (GPUs T4/A100) e chamadas de linguagem para a Gemini Cloud API quando a VRAM local atinge 100% de saturação.
- **Adaptadores de Integração Concomitante:**
  - `VoicePipelineAdapter`: Proteção de prioridade P1_CRITICAL com preempção e liberação automática de VRAM para o `AssistentePessoal`.
  - `SwarmSimulationAdapter`: Gerenciamento de simulações P3_BATCH do `Laus` com suporte a preempção graciosa e fallback para Colab.
- **Ferramentas MCP Atualizadas:** `gpulivre_get_user_config` e `gpulivre_set_user_toggle`.
- **Suíte de Testes E2E Concomitante:** [`test_hybrid_e2e_validation.py`](tests/test_hybrid_e2e_validation.py) validando bypass de toggle, preempção concorrente e saturação de VRAM.

## [0.1.0] - 2026-11-02

### Adicionado
- Estrutura inicial do projeto com `GPUMonitor`, `GPUAllocator` e `GPUCleaner`.
- Monitoramento híbrido via NVML, PyTorch CUDA, Apple Metal MPS e CPU fallback.
- Sistema de preempção de prioridade (P1 > P2 > P3).
- Expurgo forçado de tensores com `gc.collect()` e `torch.cuda.empty_cache()`.
- Servidor MCP nativo e schemas JSON.
- Suíte inicial de testes unitários em `tests/test_allocator.py`.
- Governança sob o padrão Laytech em `tasks/lessons.md`.
