# Benchmark do gpu_unlock

`scripts/benchmark-gpu-unlock.py` mede cargas reais do pipeline em dois boots:
**stock** (sem unlock) e **unlocked** (`gpu_unlock.targets=all`). O resultado vai
para `benchmarks/results/` em JSON e Markdown (diretório ignorado pelo git).

## O que mede

| Workload | Carga | Como |
| :--- | :--- | :--- |
| `micro` | GEMM BF16 8192³ com acumulação FP32 | cuBLAS via `scripts/benchmark_bf16.cu` |
| `whisper` | transcrição de áudio real | `whisper-server` (`/inference`) |
| `sd` | texto → imagem | `sd-server` (`/sdapi/v1/txt2img`) |
| `llamacpp` | prefill + decode | `llama-bench` na imagem oficial CUDA |
| `ollama` | prefill + decode | `ollama-llm` (`/api/generate`) |

Todos usam o modelo/checkpoint já configurado no ambiente:

- Whisper: `ggml-small.bin` no container `assistente-pessoal-stt`.
- Stable Diffusion: `sd-turbo.safetensors` no container `assistente-pessoal-studio-sd`.
- llama.cpp e Ollama: `qwen2.5:7b` (blob GGUF do volume do Ollama).

## Uso

```bash
# boot desbloqueado
./scripts/benchmark-gpu-unlock.py --label unlocked

# subconjunto, útil para smoke tests
./scripts/benchmark-gpu-unlock.py --only micro,ollama --label smoke

# comparação (stock primeiro)
./scripts/benchmark-gpu-unlock.py --compare \
  benchmarks/results/<stock>.json benchmarks/results/<unlocked>.json
```

O script é stdlib-only. A imagem `ghcr.io/ggml-org/llama.cpp:full-cuda` é baixada
sob demanda com `--pull`; o binário `llama-bench` é localizado em `/app`.

## Baseline stock (boot sem unlock)

O unlock é aplicado no boot pelo cmdline do kernel. Neste host o parâmetro está
em `/etc/default/grub.d/99-gpu-unlock.cfg`:

```
GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT gpu_unlock.targets=all"
```

**Opção A — um boot só, sem alterar a configuração:** no menu do GRUB edite a
entrada (tecla `e`), apague `gpu_unlock.targets=all`, boote com `Ctrl+X` e rode a
suíte com `--label stock`. O próximo boot normal volta ao estado desbloqueado.

**Opção B — persistente até restaurar:**

```bash
sudo mv /etc/default/grub.d/99-gpu-unlock.cfg{,.disabled}
sudo update-grub
sudo reboot

# após o boot stock:
./scripts/benchmark-gpu-unlock.py --label stock

# restaurar o unlock:
sudo mv /etc/default/grub.d/99-gpu-unlock.cfg{.disabled,}
sudo update-grub
sudo reboot
```

Antes da suíte completa, confirme que o boot é mesmo stock:

```bash
./scripts/benchmark-gpu-unlock.py --only micro --label stock
# deve mostrar: gpu_unlock active: False
```

## Observações do ambiente medido

- **Whisper** roda com `--no-gpu` (CPU). O unlock não altera esse número.
- **Stable Diffusion** usa o backend CPU: o `sd-server` não encontra devices
  Vulkan/CUDA. Não há SDXL instalado; o checkpoint medido é o `sd-turbo`.
- **Ollama**: no primeiro boot do container o cold start do backend CUDA pode
  estourar o watchdog de discovery (30 s). Reinicie o container com o cache
  quente e confirme `size_vram > 0` em `/api/ps` antes de medir.
- **llama.cpp**: a medição usa CUDA (`-ngl 99`) e o mesmo blob `qwen2.5:7b` do
  Ollama, o que torna os dois números comparáveis.
