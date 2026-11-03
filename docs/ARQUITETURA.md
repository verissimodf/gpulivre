# Arquitetura Técnica — GPULivre

O **GPULivre** é arquitetado como uma biblioteca e microsserviço assíncrono em Python (>=3.10) sem dependências externas pesadas no núcleo.

---

## 1. Módulos Internos

### 1.1 `gpulivre.core.monitor` (`GPUMonitor`)
- **Objetivo:** Fornecer telemetria precisa de VRAM e temperatura de forma agnóstica de plataforma.
- **Hierarquia de Provedores:**
  1. **NVIDIA NVML (`pynvml`):** Leitura de baixo nível diretamente do driver. Informa VRAM usada, livre, total, temperatura e lista de PIDs associados.
  2. **PyTorch CUDA (`torch.cuda`):** Fallback primário em ambientes com PyTorch instalado. Mede `memory_allocated` e `memory_reserved`.
  3. **Apple Silicon Metal (`mps`):** Reconhecimento de memória unificada em ambientes macOS.
  4. **CPU Fallback:** Em ambientes sem aceleração física, retorna `backend="cpu"` com 0 MB de VRAM sem disparar exceções.

### 1.2 `gpulivre.core.allocator` (`GPUAllocator`)
- **Objetivo:** Gerenciar concessão e preempção de slots de VRAM.
- **Regras de Negócio:**
  - Mantém uma margem de segurança configurável (`safety_headroom_mb = 512.0 MB`) para evitar que o sistema operacional congele ou o driver trave por esgotamento absoluto de VRAM.
  - Se a memória disponível for insuficiente para atender uma solicitação **P1_CRITICAL** ou **P2_HIGH**, o alocador busca slots ativos de menor prioridade (**P3_BATCH**), marcando-os como `"preempted"` até que o espaço necessário seja liberado.

### 1.3 `gpulivre.core.cleaner` (`GPUCleaner`)
- **Objetivo:** Eliminar fragmentação e tensores zumbis.
- **Ações:**
  - Executa `gc.collect()` para desreferenciar tensores Python.
  - Executa `torch.cuda.empty_cache()` para devolver blocos não utilizados ao driver.
  - Executa `torch.cuda.ipc_collect()` para limpar buffers de IPC compartilhado entre processos.

### 1.4 `gpulivre.core.hybrid_fallback` (`HybridFallbackEngine`)
- **Objetivo:** Garantir continuidade ininterrupta para agentes autônomos quando a VRAM local estiver 100% ocupada.
- **Roteamento Inteligente:**
  - **MiroFish Swarms & Simulações:** Empacotados em payloads JSON e transmitidos para o **Google Colab** (via `colab-drive-modularizer`), executando em GPUs Tesla T4 / A100 gratuitas.
  - **Prompts & Embeddings:** Roteados para o **Google AI Studio** (`gemini-2.5-flash`), dispensando recursos locais de GPU.

### 1.5 `gpulivre.adapters.user_toggle` (`UserConfigManager`)
- **Objetivo:** Dar autonomia e controle ao usuário do `AssistentePessoal`.
- **Configuração:**
  - Persistida em `~/.antigravity/gpulivre_user_config.json`.
  - Override via variáveis de ambiente `GPULIVRE_ENABLED` e `GPULIVRE_FALLBACK_ENABLED`.
  - Modo *bypass legado* quando desativado.
