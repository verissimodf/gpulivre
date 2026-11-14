# Benchmark do gpu_unlock

`scripts/benchmark-gpu-unlock.py` mede cargas reais do pipeline em dois boots:
**stock** (sem unlock) e **unlocked** (`gpu_unlock.targets=all`). O resultado vai
para `benchmarks/results/` em JSON e Markdown (diretório ignorado pelo git).

Os artefatos da captura de 2026-11-13 estão congelados em
[`benchmarks/evidence/2026-11-13-cuda/`](evidence/2026-11-13-cuda/README.md),
com ambiente, resultados brutos e testes automatizados.

## O que mede

| Workload | Carga | Como |
| :--- | :--- | :--- |
| `micro` | GEMM BF16 8192³ com acumulação FP32 | cuBLAS via `scripts/benchmark_bf16.cu` |
| `whisper` | transcrição de áudio real | `whisper-server` (`/inference`) |
| `sd` | texto → imagem | `sd-server` (`/sdapi/v1/txt2img`) |
| `llamacpp` | prefill + decode | `llama-bench` na imagem oficial CUDA |
| `ollama` | prefill + decode | `ollama-llm` (`/api/generate`) |

Todos usam o modelo/checkpoint já configurado no ambiente:

- Whisper: `ggml-small.bin` no container `assistente-pessoal-stt` (CUDA).
- Stable Diffusion: `sd_xl_turbo_1.0_fp16.q8_0.gguf` (SDXL-Turbo) no container
  `assistente-pessoal-studio-sd`; o `sd-turbo.safetensors` continua no volume
  como fallback.
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

- **Whisper** roda em CUDA (`whisper.cpp` compilado com `GGML_CUDA`; o
  `--no-gpu` foi removido do entrypoint). Mede ~42,9x tempo real.
- **Stable Diffusion** usa o backend CUDA do `sd-server`, com SDXL-Turbo em
  GGUF q8_0. A geração padrão é 1024x1024/4 steps (~6,05 s/imagem, ~9,9/min)
  com `--vae-tiling` e `--eager-load`. `STUDIO_MAX_VRAM=6` reserva espaço para
  Ollama e Whisper coexistirem na mesma GPU.
- **Ollama** usa a imagem `ollama-llm:warmup`, que pré-aquece os backends
  `cuda_v12`/`cuda_v13` antes do `serve`. O discovery cai para ~1,6 s e o
  `qwen2.5:7b` sobe 100% GPU; confirme `size_vram > 0` em `/api/ps`.
- **llama.cpp**: a medição usa CUDA (`-ngl 99`) e o mesmo blob `qwen2.5:7b` do
  Ollama, o que torna os dois números comparáveis.

## Último resultado (unlocked, CUDA nativo)

Medido em 2026-11-13 com a suíte completa:

| Workload | Resultado |
| :--- | :--- |
| `micro` | 49,5 TFLOPS BF16 |
| `whisper` | 14,132 s de áudio em 0,3294 s (42,9x tempo real) |
| `sd` | SDXL-Turbo 1024 px/4 steps: 6,05 s/imagem (9,92/min) |
| `ollama` | prefill 2341,9 tok/s, decode 65,5 tok/s (1007 tokens de prompt) |

Comando:

```bash
# o studio-sd não publica porta no host; exponha temporariamente para o benchmark
docker run -d --rm --name sd-proxy \
  --network assistente-pessoal_studio_internal \
  -p 127.0.0.1:17870:7870 alpine/socat \
  tcp-listen:7870,fork,reuseaddr tcp:studio-sd:7870

./scripts/benchmark-gpu-unlock.py --only micro,whisper,sd,ollama \
  --sd-steps 4 --sd-size 1024 --sd-url http://127.0.0.1:17870 \
  --label unlocked-cuda-all
```
