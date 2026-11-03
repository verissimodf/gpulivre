# ⚡ GPULivre — Autonomous GPU & VRAM Resource Orchestrator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: >=3.10](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Laytech Protocol](https://img.shields.io/badge/Protocol-Laytech-purple.svg)](../../skills/laytech/SKILL.md)
[![MCP Enabled](https://img.shields.io/badge/MCP-Server-orange.svg)](gpulivre/mcp/server.py)

O **GPULivre** é a camada de infraestrutura e orquestração de aceleração de hardware (GPU/VRAM) projetada especificamente para o ecossistema autônomo local (`AssistentePessoal` e `Laus`). 

Ele resolve o conflito de concorrência de hardware quando múltiplos agentes de IA (simulações de enxames MiroFish, transcrição por voz Whisper, síntese de voz TTS, RAG vetorial e modelos quantitativos) competem pela mesma GPU/VRAM.

---

## 🏛️ Topologia do Sistema

```mermaid
flowchart TD
    User["👤 Usuário (AssistentePessoal / Chat)"] -->|Configura Toggle| Toggle["UserConfigManager<br/>(Ativo / Inativo / Auto)"]
    
    subgraph Ativado ["GPULivre Ativado (Modo Orquestrado)"]
        Voice["🎙️ Voz / TTS / Whisper (AssistentePessoal)"] -->|Prioridade P1 (Crítica)| Allocator["GPULivre Allocator"]
        RAG["📚 RAG / Embeddings / Visão"] -->|Prioridade P2 (Alta)| Allocator
        Swarm["🐟 MiroFish Swarm / Quant (Laus)"] -->|Prioridade P3 (Batch)| Allocator
        
        Allocator -->|Memória Disponível| LocalGPU["🖥️ GPU Local (CUDA / MPS)"]
        Allocator -->|Concorrência P1 vs P3| Preempt["⚡ Preempção Graciosa:<br/>Pausa P3 e libera VRAM para P1"]
        Allocator -->|VRAM 100% Saturada| Fallback["🌐 HybridFallbackEngine"]
        
        Fallback -->|Enxames / Batch| Colab["🚀 Google Colab (Tesla T4 / A100)<br/>(colab-drive-modularizer)"]
        Fallback -->|LLM / Embeddings| Gemini["✨ Gemini Cloud API<br/>(AI Studio / Flash 2.5)"]
    end
    
    subgraph Desativado ["Modo Bypass Legado"]
        DirectRun["Execução Direta no Host<br/>(Sem reserva de VRAM / Sem Fallback)"]
    end
    
    Toggle -->|gpulivre_enabled: true| Allocator
    Toggle -->|gpulivre_enabled: false| DirectRun
```

---

## 📂 Estrutura de Arquivos e Módulos do Repositório

```
gpulivre/
├── gpulivre/
│   ├── __init__.py                    # Exports públicos e versionamento
│   ├── core/
│   │   ├── monitor.py                 # Telemetria agnóstica (NVML, PyTorch CUDA, MPS, CPU fallback)
│   │   ├── allocator.py               # Alocador de VRAM com preempção prioritária (P1 > P2 > P3)
│   │   ├── cleaner.py                 # Expurgo forçado de tensores residuais e coleta de lixo
│   │   └── hybrid_fallback.py         # Motor de transbordo (Google Colab GPU & Gemini Cloud API)
│   ├── adapters/
│   │   ├── user_toggle.py             # Gerenciador de preferências do usuário e modo bypass
│   │   ├── voice_adapter.py           # Adaptador P1_CRITICAL para pipeline de voz (AssistentePessoal)
│   │   └── swarm_adapter.py           # Adaptador P3_BATCH para enxames MiroFish e quant (Laus)
│   └── mcp/
│       └── server.py                  # Servidor MCP expondo 7 ferramentas para agentes autônomos
├── docs/
│   ├── ARQUITETURA.md                 # Especificação técnica e subsistemas
│   ├── INTEGRACAO_ASSISTENTE_PESSOAL.md # Manual de integração de voz
│   ├── INTEGRACAO_LAUS.md             # Manual de integração com simulações MiroFish
│   └── FALLBACK_HIBRIDO.md            # Manual do motor de transbordo para Colab/Gemini
├── tests/
│   ├── test_allocator.py              # Testes unitários do núcleo
│   └── test_hybrid_e2e_validation.py  # Testes de integração E2E com toggle e concorrência
├── tasks/
│   └── lessons.md                     # Registro de autocorreção contínua (Padrão Laytech)
├── pyproject.toml                     # Metadados de build e dependências
├── CHANGELOG.md                       # Histórico de releases
└── README.md                          # Apresentação do projeto
```

---

## 🚀 Funcionalidades Principais

1. **Monitoramento Híbrido em Tempo Real (`gpulivre.core.monitor`):**
   - Telemetria de VRAM total, usada e livre via NVIDIA NVML (`pynvml`), `torch.cuda`, Apple Metal (`mps`) e fallback seguro para CPU/RAM.
   - Detecção de temperatura, utilização de computação e processos ativos por PID.

2. **Alocador de VRAM com Fila de Preempção (`gpulivre.core.allocator`):**
   - Escalonamento com 3 níveis de prioridade:
     - **P1 - CRITICAL / REALTIME:** Voz Full-Duplex, VAD, Barge-In, respostas interativas de chat.
     - **P2 - HIGH:** Embeddings RAG, geração de imagens, inferências analíticas pontuais.
     - **P3 - BATCH / SWARM:** Simulações MiroFish (enxames multi-agente), backtesting quantitativo, fine-tuning.
   - Preempção graciosa de tarefas P3 quando uma solicitação P1 chega sem VRAM disponível suficiente.

3. **Limpador Ativo de Memória (`gpulivre.core.cleaner`):**
   - Coleta de lixo forçada (`gc.collect`).
   - Esvaziamento do pool de tensores do PyTorch (`torch.cuda.empty_cache()` e `torch.cuda.ipc_collect()`).
   - Eliminação de processos órfãos e mitigação de vazamentos de memória.

4. **Motor de Fallback Híbrido (`gpulivre.core.hybrid_fallback`):**
   - Quando a VRAM física local atinge 100% de ocupação:
     - Simulações pesadas (MiroFish Swarm) são empacotadas e roteadas para **Google Colab** (GPU T4/A100) via padrão `colab-drive-modularizer`.
     - Inferências de texto e visão são desviadas para o **Gemini Cloud API** (Gemini 2.5 Flash via `ai_studio_client`).

5. **Mecanismo de Toggle pelo Usuário (`gpulivre.adapters.user_toggle`):**
   - Permite ao usuário do `AssistentePessoal` ligar ou desligar a orquestração e o fallback a qualquer momento via arquivo de configuração (`~/.antigravity/gpulivre_user_config.json`), variáveis de ambiente ou ferramentas MCP.

6. **Servidor MCP Nativo (`gpulivre.mcp.server`):**
   - Fornece ferramentas para agentes LLM (Antigravity, Claude Code, Cursor) consultarem a saúde da GPU, reservarem slots, liberarem memória e alternarem preferências em tempo real.

---

## 🛠️ Ferramentas MCP (`gpulivre-mcp`)

| Ferramenta | Descrição |
| :--- | :--- |
| `gpulivre_get_status` | Retorna telemetria da GPU (VRAM total, usada, livre em MB, temperatura, backend ativo). |
| `gpulivre_request_slot` | Reserva um slot de memória com prioridade (P1 Voz, P2 RAG, P3 Swarm). |
| `gpulivre_release_slot` | Libera um slot de alocação de GPU e expurga o cache. |
| `gpulivre_purge_cache` | Dispara `gc.collect()`, `torch.cuda.empty_cache()` e `torch.cuda.ipc_collect()`. |
| `gpulivre_list_slots` | Lista todos os slots ativos em execução no momento. |
| `gpulivre_get_user_config` | Consulta as preferências do usuário no AssistentePessoal. |
| `gpulivre_set_user_toggle` | Habilita ou desabilita o GPULivre e o Fallback Híbrido em tempo de execução. |

---

## 📦 Instalação e Execução

```bash
# Clone e instalação no ambiente local (fora do Google Drive):
git clone https://github.com/verissimodf/gpulivre.git
cd gpulivre
pip install -e .

# Com aceleração CUDA total:
pip install -e ".[cuda]"
```

Executando a suíte de testes de validação:
```bash
python -m unittest tests/test_hybrid_e2e_validation.py
```

Iniciando o servidor MCP:
```bash
python -m gpulivre.mcp.server
```

---

## 📖 Documentação Detalhada

- [Arquitetura e Fluxos Técnicos](docs/ARQUITETURA.md)
- [Integração com AssistentePessoal (Voz)](docs/INTEGRACAO_ASSISTENTE_PESSOAL.md)
- [Integração com Laus (MiroFish Swarm)](docs/INTEGRACAO_LAUS.md)
- [Motor de Fallback Híbrido (Colab & Gemini)](docs/FALLBACK_HIBRIDO.md)
- [Lições Aprendidas e Autocorreção](tasks/lessons.md)
- [Changelog](CHANGELOG.md)
