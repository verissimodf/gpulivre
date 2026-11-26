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

---

## 2. Universal Inference Auto-Tuner

### 2.1 `gpulivre.core.ceiling`
- Consolida identidade do `GPUMonitor` com medições explícitas de bandwidth e FLOPs.
- Calcula limites de decode, prefill e capacidade de KV para CUDA, ROCm, XPU, MPS e memória unificada.
- Não deriva especificações físicas do nome comercial da GPU: dados ausentes ficam desconhecidos e geram avisos.
- Considera contexto, concorrência, quantização e custo/aceitação de speculative decoding.

### 2.2 `gpulivre.core.bench_harness`
- Cliente stdlib para endpoints OpenAI-compatible, sem acoplamento a llama.cpp, Ollama, vLLM, SGLang ou ExLlama.
- SSE é usado para TTFT e conteúdo; contagem de tokens vem exclusivamente de `usage` ou de tokenizador real injetado.
- Mede startup, TTFT, decode c1/cN, cold prefill e needle retrieval.
- O watchdog inclui memória do processo/driver e aborta a amostra ao ultrapassar o cap. Servidores desconhecidos nunca são encerrados por padrão de processo.

### 2.3 `gpulivre.core.hill_climber`
- Executa `controle → candidato → controle`, com no mínimo seis amostras agregadas e medianas.
- Altera um lever por candidato e rejeita regressões superiores a 3%.
- Exige needle retrieval e full-KV soak antes da promoção para produção.
- Persiste todas as decisões em `~/.antigravity/gpulivre_ledger.jsonl` e no relatório `gpulivre_PLAN.md`.

### 2.4 Integração MCP
- `gpulivre_compute_ceiling`: calcula os tetos físicos.
- `gpulivre_run_interleaved_bench`: executa uma rodada A/B controlada.
- `gpulivre_hill_climb`: executa uma sessão serializada de candidatos.
- `gpulivre_get_ledger_history`: recupera tentativas anteriores e evita repetição de falhas.

A metodologia operacional também está disponível em `skills/gpulivre-optimizer/`.
