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
│   │   ├── hybrid_fallback.py         # Motor de transbordo (Google Colab GPU & Gemini Cloud API)
│   │   ├── ceiling.py                 # Tetos de decode, prefill e KV em qualquer backend
│   │   ├── bench_harness.py           # Benchmark OpenAI-compatible e watchdog de memória
│   │   └── hill_climber.py            # Otimização intercalada, gates e ledger institucional
│   ├── adapters/
│   │   ├── user_toggle.py             # Gerenciador de preferências do usuário e modo bypass
│   │   ├── voice_adapter.py           # Adaptador P1_CRITICAL para pipeline de voz (AssistentePessoal)
│   │   └── swarm_adapter.py           # Adaptador P3_BATCH para enxames MiroFish e quant (Laus)
│   └── mcp/
│       └── server.py                  # Servidor MCP expondo 11 ferramentas para agentes autônomos
├── docs/
│   ├── ARQUITETURA.md                 # Especificação técnica e subsistemas
│   ├── INTEGRACAO_ASSISTENTE_PESSOAL.md # Manual de integração de voz
│   ├── INTEGRACAO_LAUS.md             # Manual de integração com simulações MiroFish
│   └── FALLBACK_HIBRIDO.md            # Manual do motor de transbordo para Colab/Gemini
├── tests/
│   ├── test_allocator.py              # Testes unitários do núcleo
│   ├── test_hybrid_e2e_validation.py  # Testes de integração E2E com toggle e concorrência
│   └── test_hill_climber.py           # Tetos, harness, watchdog, hill-climb e MCP
├── skills/gpulivre-optimizer/         # Skill nativa de otimização de inferência
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

7. **Universal Inference Auto-Tuner:**
   - Calcula tetos físicos sem assumir GPU, quantização ou engine específicas.
   - Mede endpoints OpenAI-compatible usando `usage` real, nunca chunks de stream.
   - Compara `controle → candidato → controle` com medianas e watchdog de VRAM.
   - Só promove candidatos sem regressão >3% e com needle/full-KV soak aprovados.

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
| `gpulivre_compute_ceiling` | Calcula tetos de decode/prefill e capacidade máxima de KV para o hardware e modelo informados. |
| `gpulivre_run_interleaved_bench` | Compara controle e candidato em `ctl → cand → ctl`, com pelo menos seis amostras e cap de memória. |
| `gpulivre_hill_climb` | Executa uma sessão serializada de otimização por prioridade P1/P2/P3 e gates de qualidade. |
| `gpulivre_get_ledger_history` | Consulta sucessos e falhas anteriores por hardware, modelo, lever e decisão. |

### Primeiro ciclo de hill-climb

O exemplo abaixo usa um servidor local já iniciado em `http://127.0.0.1:8000`. Como o endpoint externo não é reiniciado pelo GPULivre, o candidato altera somente campos documentados de `extra_body`. Os gates são executados pelo harness e a memória é limitada pelo sampler configurado no ambiente Python.

```python
from gpulivre.mcp.server import dispatch_tool

result = dispatch_tool("gpulivre_hill_climb", {
    "baseline": {
        "name": "base",
        "base_url": "http://127.0.0.1:8000",
        "model": "meu-modelo",
        "concurrency": 4,
        "prefill_lengths": [4096, 32768],
        "long_context_tokens": 32768,
        "memory_cap_mb": 14000
    },
    "levers": [{
        "name": "candidate_request_option",
        "likelihood": 0.5,
        "config": {
            "name": "candidate",
            "base_url": "http://127.0.0.1:8000",
            "model": "meu-modelo",
            "concurrency": 4,
            "prefill_lengths": [4096, 32768],
            "long_context_tokens": 32768,
            "memory_cap_mb": 14000,
            "extra_body": {"cache_prompt": True}
        }
    }],
    "priority": 2,
    "max_experiments": 1,
    "hardware": "minha-gpu-ou-memoria-unificada"
})
print(result)
```

`cache_prompt` é apenas um exemplo para servidores que documentam esse campo; remova-o ou substitua-o por uma opção válida na sua engine. Flags de `launch` exigem lifecycle gerenciado e não devem ser enviadas a um endpoint desconhecido. Consulte `skills/gpulivre-optimizer/SKILL.md` antes de otimizar produção.

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

Executando toda a suíte de testes de validação:
```bash
python -m unittest discover tests
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
- [Skill GPULivre Optimizer](skills/gpulivre-optimizer/SKILL.md)
- [Lições Aprendidas e Autocorreção](tasks/lessons.md)
- [Changelog](CHANGELOG.md)

## 🌿 Branches (Git Flow)
- `main` = produção (cada commit é uma versão, tagueada `vX.Y.Z`) · `develop` = integração
- Features: `feature/<nome>` a partir de `develop`, PR para `develop`
- Releases: `release/X.Y.Z` a partir de `develop`; merge em `main` + tag + volta para `develop`
- Hotfix: `hotfix/X.Y.Z` a partir de `main`; merge em `main` + tag + `develop`
- Versionamento SemVer; convenções em `.gitflow`
