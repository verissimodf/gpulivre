# Evidência de teste — CUDA nativo, SDXL e coexistência na RTX 3060

**Data da captura:** 2026-11-13 (UTC), host `app`
**Branch:** `feature/benchmark-gpu-unlock`
**Commits de referência:** `6ab57b6` (rodadas de 21:56–22:15 UTC) e `1942b1c` (rodada de 20:42 UTC)

Este diretório congela os artefatos brutos da suíte
[`scripts/benchmark-gpu-unlock.py`](../../../scripts/benchmark-gpu-unlock.py) e dos
testes automatizados do GPULivre. O objetivo é permitir auditoria e reprodução sem
depender do estado volátil do host.

## 1. Ambiente medido

| Item | Valor |
| :--- | :--- |
| Host / kernel | `app` / `7.0.0-43-generic` |
| GPU | NVIDIA GeForce RTX 3060, 12.288 MiB |
| Driver | 595.99.02 |
| SM clock máx. / power limit | 2100 MHz / 170 W |
| `gpu_unlock` | ativo, `ss0=0x88888888` |
| Kernel cmdline | `... gpu_unlock.targets=all ...` |
| stable-diffusion.cpp | commit `fc350be`, CUDA 12.6.3, arch 86 |
| whisper.cpp | commit `21ba6d2`, CUDA 12.6.3, arch 86 |
| Ollama | imagem `ollama-llm:warmup` (pré-aquece `cuda_v12`/`cuda_v13`) |
| Modelo SDXL | `sd_xl_turbo_1.0_fp16.q8_0.gguf` |
| SHA-256 do SDXL ativo | `f3387ead491b0c14c0f17693eb140b924efe1b70bcd4e89195e8be7705728c67` |

A captura completa está em [`environment.txt`](environment.txt), incluindo
`nvidia-smi`, processos na GPU, containers, versões e hash do modelo.

## 2. Artefatos brutos

Os arquivos abaixo são cópias exatas de `benchmarks/results/` no momento da captura.

| Arquivo | O que registra |
| :--- | :--- |
| [`results/2026-11-13T20-42-27-00-00-unlocked.json`](results/2026-11-13T20-42-27-00-00-unlocked.json) | Linha de base pré-CUDA: Whisper em CPU, SD `sd-turbo` 256 px, llama.cpp e Ollama |
| [`results/2026-11-13T21-56-35-00-00-unlocked-cuda.json`](results/2026-11-13T21-56-35-00-00-unlocked-cuda.json) | Whisper recompilado com CUDA |
| [`results/2026-11-13T22-12-06-00-00-unlocked-cuda-coexist.json`](results/2026-11-13T22-12-06-00-00-unlocked-cuda-coexist.json) | Whisper em CUDA com SDXL e Ollama carregados na mesma GPU |
| [`results/2026-11-13T22-12-19-00-00-unlocked-cuda-sdxl.json`](results/2026-11-13T22-12-19-00-00-unlocked-cuda-sdxl.json) | SDXL-Turbo 1024×1024 / 4 steps |
| [`results/2026-11-13T22-15-49-00-00-unlocked-cuda-all.json`](results/2026-11-13T22-15-49-00-00-unlocked-cuda-all.json) | **Rodada canônica**: micro, Whisper, SDXL e Ollama na mesma execução |
| [`environment.txt`](environment.txt) | Host, kernel, GPU, VRAM, containers, versões e hash do modelo |
| [`tests.txt`](tests.txt) | Saída de `unittest` e `pytest` (14 testes) |
| [`ci.txt`](ci.txt) | Checks de CI executados localmente (patch GA107 e receitas) |

Cada JSON contém `meta` (timestamp, commit, kernel, GPU, status do unlock e cauda do
log do kernel) e `results` (métricas e amostras). Os `.md` correspondentes são o
resumo renderizado pela própria suíte.

## 3. Resultados

### Rodada canônica — `2026-11-13T22-15-49Z` (`unlocked-cuda-all`)

| Workload | Métrica | Resultado |
| :--- | :--- | ---: |
| `micro` | GEMM BF16 (FP32 acc) | **49,5 TFLOPS** |
| `whisper` | 14,132 s de áudio | **0,3294 s** = **42,9× tempo real** |
| `sd` | SDXL-Turbo 1024 px / 4 steps | **6,0499 s/imagem** = **9,92 imagens/min** |
| `ollama` | `qwen2.5:7b`, 1007 tokens de prompt | **2341,86 tok/s** prefill, **65,52 tok/s** decode |

Amostras: Whisper `0,329 / 0,322 / 0,339 s`; SD `6,050 / 6,045 / 6,067 s`; 3 repetições
por workload (mediana). O `micro` usa 30 iterações.

### Progressão das rodadas

| Rodada | Workload | Resultado | Observação |
| :--- | :--- | ---: | :--- |
| 20:42 UTC | `micro` | 51,0 TFLOPS | Variação normal frente aos 49,5 TFLOPS |
| 20:42 UTC | `whisper` | 5,578 s (2,53×) | Ainda em CPU |
| 20:42 UTC | `sd` | 30,26 s (1,98/min) | `sd-turbo`, 256 px, 2 steps, CPU |
| 20:42 UTC | `llamacpp` | 2465,71 tok/s prefill; 69,04 tok/s decode | 512 tokens de prompt |
| 20:42 UTC | `ollama` | 2360,47 tok/s prefill; 66,04 tok/s decode | 1006 tokens de prompt |
| 21:56 UTC | `whisper` | 0,3303 s (42,79×) | Primeira medição com CUDA |
| 22:12 UTC | `whisper` | 0,3297 s (42,86×) | SDXL + Ollama carregados |
| 22:12 UTC | `sd` | 6,0688 s (9,89/min) | SDXL 1024 px / 4 steps |

> A rodada de 20:42 serve como linha de base operacional, mas **não** é um baseline
> stock do `gpu_unlock`: nela o Whisper e o SD ainda rodavam em CPU. Para comparar o
> efeito do unlock, use duas rodadas com o mesmo backend e `gpu_unlock` ligado/desligado.

## 4. Coexistência de VRAM

Snapshot de [`environment.txt`](environment.txt), com os três serviços ativos:

| Processo | VRAM |
| :--- | ---: |
| `whisper-server` | 840 MiB |
| `/opt/sd/sd-server` | 4.170 MiB |
| `/usr/lib/ollama/llama-server` | 4.662 MiB |
| **Total usado** | **9.816 MiB de 12.288 MiB** |
| **Livre** | **~2.472 MiB** |

Durante a rodada de 22:15 UTC o total observado foi **9.830 MiB**, com Ollama em
4.676 MiB. A diferença é ruído de alocação. O `qwen2.5:7b` reportou **100% GPU** em
`ollama ps`, e o SDXL manteve ~6 s por imagem com os três modelos residentes.

## 5. Testes automatizados e checks de CI

Arquivos: [`tests.txt`](tests.txt) e [`ci.txt`](ci.txt).

```bash
python -m unittest discover -s tests -v
python -m pytest tests -q
```

| Suíte | Resultado |
| :--- | :--- |
| `unittest` | **14 testes, 0 falhas** (`OK`) |
| `pytest` | **14 passed** (1 warning de depreciação do `pynvml`) |
| Checks de CI (locais) | sintaxe do instalador, arquivos base, patch GA107 e receitas `0x2582`/`0x2584`: **OK** |

Cobertura funcional:

- `tests/test_allocator.py` (7 testes): fallback de monitor para CPU, purge de cache,
  alocação feliz, rejeição por VRAM insuficiente, preempção P1 sobre P3, limpeza de
  slots e despacho das ferramentas MCP.
- `tests/test_hybrid_e2e_validation.py` (7 testes): toggle do usuário (modo legado),
  voz P1 com VRAM suficiente, concorrência P1×P3 com preempção, fallback para Colab
  em saturação, fallback para Gemini, rejeição limpa com fallback desligado e toggle
  via MCP.

O `pyproject.toml` foi ajustado com `[tool.setuptools.packages.find]` para que
`pip install -e ".[test]"` funcione no layout flat do repositório. O comando usado na
captura foi:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests -q
```

## 6. Como reproduzir o benchmark

O `studio-sd` não publica porta no host; para o benchmark é preciso um proxy
temporário na rede interna:

```bash
docker run -d --rm --name sd-proxy \
  --network assistente-pessoal_studio_internal \
  -p 127.0.0.1:17870:7870 alpine/socat \
  tcp-listen:7870,fork,reuseaddr tcp:studio-sd:7870

./scripts/benchmark-gpu-unlock.py --only micro,whisper,sd,ollama \
  --sd-steps 4 --sd-size 1024 --sd-reps 3 \
  --sd-url http://127.0.0.1:17870 \
  --ollama-model qwen2.5:7b \
  --label unlocked-cuda-all

docker rm -f sd-proxy
```

Para comparar dois estados (stock × unlocked):

```bash
./scripts/benchmark-gpu-unlock.py --compare \
  benchmarks/results/<stock>.json benchmarks/results/<unlocked>.json
```

## 7. Limitações e critérios de validade

- **Sem baseline stock nesta captura.** Todas as rodadas têm `gpu_unlock.active=true`.
  O delta do unlock exige um boot sem `gpu_unlock.targets=all`.
- **SDXL aquecido.** Os tempos excluem o carregamento inicial do GGUF (~76 s a frio);
  `--eager-load` move essa leitura para o start do container, coberto pelo
  `start_period: 240s`.
- **Ollama em GPU.** Os números só valem com `ollama ps` mostrando `100% GPU`; o
  backend pode cair para CPU se a descoberta CUDA falhar no start a frio.
- **Variação do `micro`.** 49,5–51,0 TFLOPS entre execuções; a métrica é sensível a
  clock/power/temperatura.
- **Um host, uma GPU.** Não há normalização por temperatura, potência ou versão de
  driver além do que está registrado em `meta`.
- **`llamacpp` não participou da rodada canônica.** O resultado de 20:42 é mantido
  como referência separada.

## 8. Validação após reboot

Sequência usada para confirmar que o unlock e os backends CUDA sobrevivem a um boot
completo. Executar na janela do host:

```bash
# 0) proxy temporário para o studio-sd (não publica porta no host)
cd /opt/laytech-ai/gpulivre
docker run -d --rm --name sd-proxy \
  --network assistente-pessoal_studio_internal \
  -p 127.0.0.1:17870:7870 alpine/socat \
  tcp-listen:7870,fork,reuseaddr tcp:studio-sd:7870

# 1) antes do reboot: nova rodada com label próprio
./scripts/benchmark-gpu-unlock.py --only micro,whisper,sd,ollama \
  --sd-steps 4 --sd-size 1024 --sd-url http://127.0.0.1:17870 \
  --label pre-reboot

# 2) reboot
sudo reboot
```

Após o boot:

```bash
# 3) confirmar unlock e GPU
grep -o 'gpu_unlock.targets=all' /proc/cmdline
journalctl -k -b | grep 'gpu_unlock: unlocked' | tail -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 4) confirmar serviços (aguarde o healthcheck)
docker ps --format '{{.Names}} | {{.Image}} | {{.Status}}' | grep -E 'assistente-pessoal|ollama-llm'
curl -s --max-time 300 http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:7b","prompt":"ok","stream":false,"keep_alive":"10m","options":{"num_predict":1}}' >/dev/null
docker exec ollama-llm ollama ps   # deve mostrar 100% GPU

# 5) nova rodada e comparação
cd /opt/laytech-ai/gpulivre
./scripts/benchmark-gpu-unlock.py --only micro,whisper,sd,ollama \
  --sd-steps 4 --sd-size 1024 --sd-url http://127.0.0.1:17870 \
  --label post-reboot
./scripts/benchmark-gpu-unlock.py --compare \
  benchmarks/results/<pre-reboot>.json benchmarks/results/<post-reboot>.json

# 6) limpeza e desligamento total
docker rm -f sd-proxy
sudo poweroff
```

> O `--compare` emite aviso quando os dois arquivos têm o mesmo estado de unlock;
> isso é esperado no teste de persistência (pre × post reboot), não no baseline stock.
