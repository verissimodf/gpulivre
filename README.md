# 🔓 gpulivre: Liberdade Total para Tensor Cores em GPUs NVIDIA GeForce

> **"Your GeForce runs AI at half speed. This frees it."**

O **gpulivre** é um projeto open-source voltado à comunidade de IA local e engenheiros de machine learning que rodam modelos de linguagem (LLMs) em servidores Linux com placas domésticas NVIDIA GeForce (especialmente arquitetura **Ampere GA106/GA107**, como a **RTX 3050 6GB/8GB**). 

Ele remove o estrangulamento artificial imposto de fábrica nos *Tensor Cores* (`FP16`, `BF16`, `TF32`), dobrando a taxa de despacho em operações matriciais e viabilizando inferência de alta performance com **Ollama** e **llama.cpp**.

---

## 📊 O Salto de Performance Real (MSI RTX 3050 6GB - `10de:2584`)

| Métrica de Carga | Stock (Padrão NVIDIA) | gpulivre (Desbloqueado + FA) | Ganho Relativo |
| :--- | :--- | :--- | :--- |
| **Seletor de Velocidade (`SS0`)** | `0x30501002` (Capado em 50%) | **`0x88888888` (`SS1=0x8`)** | **2× TFLOPS FP16/BF16** |
| **Prefill / Leitura de Prompt (1.026 tokens)** | `158.22 tokens/s` | **`1.043.83 tokens/s`** | **+559%** |
| **Latência Inicial (TTFT - 61 tokens)** | `385.53 ms` | **`39.02 ms`** | **~10× Menor Latência** |
| **Geração de Texto (`eval rate` - Q4_0)** | `15.59 tokens/s` | **`38.47 a 43.84 tokens/s`** | **+146% a +181%** |

---

## 🧠 Arquitetura Técnica: O Desbloqueio do Silício GA107 (`0x177`)

Placas baseadas no chip **GA107** (como a RTX 3050 6GB) retornam a assinatura `0xb77000a1` no registrador `PMC_BOOT_0`, correspondendo ao chip ID binário **`0x177`**. O **gpulivre** introduz a macro estrutural `GA107(id) { id, 0x177, &ga10x }` na árvore de receitas do driver de boot, permitindo que o payload execute via SEC2/Falcon no `initramfs` antes que o driver proprietário da NVIDIA seja carregado.

### Compatibilidade Homologada
- **RTX 3050 6GB Desktop:** `[10de:2584]` (Chip GA107)
- **RTX 3050 8GB Desktop:** `[10de:2507]` / `[10de:2582]` (Chips GA106 / GA107)

---

## 🚀 Guia Prático de Implementação

### 1. Clonar e Instalar Automaticamente
```bash
git clone https://github.com/verissimodf/gpulivre.git
cd gpulivre
sudo bash scripts/install-gpulivre.sh
```

> **Sem etapa extra de patch.** O motor de desbloqueio já está embutido neste
> repositório em `upstream/gpu-unlock/` (subtree de [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) v2.0.0)
> com as receitas **GA107** (`0x2584` RTX 3050 6GB e `0x2582` RTX 3050 8GB)
> commitadas em `kernel/gpu_unlock/recipes.c`. O payload equivalente, portável
> para aplicar sobre upstream fresco, está em
> [`patches/0001-support-ga107.patch`](patches/0001-support-ga107.patch) e é
> verificado no CI a cada push.

### 2. Sincronismo de Boot (Docker + NVIDIA)
Para evitar que o daemon do Docker suba antes da inicialização completa do driver gráfico (nvidia-drm / /dev/nvidia0), o script configura automaticamente a barreira no systemd:

[Unit]
After=systemd-udev-settle.service

[Service]
ExecStartPre=/bin/sh -c 'until /usr/bin/nvidia-smi >/dev/null 2>&1; do sleep 1; done'

### 3. Executando o Ollama Otimizado (docker-compose.yml)

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-llm
    restart: unless-stopped
    environment:
      - OLLAMA_FLASH_ATTENTION=1
      - OLLAMA_KV_CACHE_TYPE=f16
      - OLLAMA_VULKAN=0
    ports:
      - "11434:11434"
    volumes:
      - ./volumes/ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

🤝 Como Contribuir
Contribuições, relatos de compatibilidade com outros modelos de placas e melhorias nos scripts de automação são extremamente bem-vindos! Abra uma Issue ou envie um Pull Request.

## 📁 Estrutura do Repositório

```
gpulivre/
├── scripts/install-gpulivre.sh   # instalador único (deps + build + GRUB + Docker)
├── docker/docker-compose.yml     # Ollama otimizado (FlashAttention, KV cache)
├── patches/0001-support-ga107.patch  # delta GA107 portável sobre upstream v2.0.0
├── upstream/gpu-unlock/          # motor vendido (subtree Zagitta/gpu-unlock v2.0.0)
│   ├── kernel/gpu_unlock/        # módulo DKMS das receitas (GA106/GA107/GA107 0x2584, 0x2582)
│   ├── firmware/                 # extratores de firmware NVIDIA (fetch/extract/notice)
│   ├── debian/                   # empacotamento .deb original
│   ├── packaging/                # Arch, RPM, Unraid e conf de initramfs (mkinitcpio/dracut)
│   ├── tests/                    # testes de módulo e empacotamento
│   ├── licenses/ + docs/         # licenças GPL-2.0/NVIDIA-MIT e doc técnica
│   └── AGENTS.md                 # diretrizes do upstream
└── docs/                         # engenharia silício GA107 + visualizações GitHub Pages
```

## 🏆 Inspiração, Créditos e Outros Modelos Suportados

O **gpulivre** nasce como uma extensão focada na usabilidade, automação de infraestrutura (Docker/Ollama) e suporte aos chips de entrada (GA107), construída sobre a base brilhante de engenharia reversa do projeto upstream [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock).

Enquanto o **gpulivre** foca na democratização da arquitetura de entrada **Ampere GA107/GA106** (RTX 3050, 3060), o método base de interceptação do `initramfs` descoberto pela comunidade também suporta o desbloqueio de Tensor Cores e limitação de vGPU para as seguintes arquiteturas:
* **Turing (TU102):** RTX 2080 Ti, Titan RTX.
* **Ampere High-End (GA102/GA104):** RTX 3070, 3080, 3090 e Titan RTX.
* **Data Center (GA100):** A100.

Se você possui placas superiores à RTX 3060, o repositório original do Zagitta oferece as receitas estruturais (`sec2`/`booter`) para liberar a potência total dessas GPUs.

## 📈 Gráficos Interativos de Performance
Explore os dados de benchmark e a estrutura do projeto renderizados em tempo real pelo GitHub Pages:
* 🚀 **[Gráfico de Ganhos de Performance e Latência](https://verissimodf.github.io/gpulivre/docs/visualizations/ai-performance-gains.html)**
* 🗂️ **[Status de Sincronização do Repositório](https://verissimodf.github.io/gpulivre/docs/visualizations/repo-sync-status.html)**
