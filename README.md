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
git clone [https://github.com/verissimodf/gpulivre.git](https://github.com/verissimodf/gpulivre.git)
cd gpulivre
sudo bash scripts/install-gpulivre.sh
cat << 'EOF' > patches/ga107-rtx3050.patch
diff --git a/kernel/gpu_unlock/recipes.c b/kernel/gpu_unlock/recipes.c
index 8c0d111..4f98220 100644
--- a/kernel/gpu_unlock/recipes.c
+++ b/kernel/gpu_unlock/recipes.c
@@ -383,6 +383,7 @@ static const struct gu_family ga10x = {
#define GA103(id) { id, 0x173, &ga10x }
#define GA104(id) { id, 0x174, &ga10x }
#define GA106(id) { id, 0x176, &ga10x }
+#define GA107(id) { id, 0x177, &ga10x }

static const struct gu_card cards[] = {
/* TU102: RTX 2080 Ti, Titan RTX */
@@ -431,6 +432,7 @@ static const struct gu_card cards[] = {
GA104(0x24e0),

GA106(0x2501), GA106(0x2503), GA106(0x2504), GA106(0x2505),
GA106(0x2507), GA106(0x2508), GA106(0x2509), GA106(0x252f),

GA106(0x2507), GA107(0x2584), GA107(0x2582), GA106(0x2508), GA106(0x2509), GA106(0x252f),
GA106(0x2544),
};

## 🏆 Inspiração, Créditos e Outros Modelos Suportados

O **gpulivre** nasce como uma extensão focada na usabilidade, automação de infraestrutura (Docker/Ollama) e suporte aos chips de entrada (GA107), construída sobre a base brilhante de engenharia reversa do projeto upstream [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock).

Enquanto o **gpulivre** foca na democratização da arquitetura de entrada **Ampere GA107/GA106** (RTX 3050, 3060), o método base de interceptação do `initramfs` descoberto pela comunidade também suporta o desbloqueio de Tensor Cores e limitação de vGPU para as seguintes arquiteturas:
* **Turing (TU102):** RTX 2080 Ti, Titan RTX.
* **Ampere High-End (GA102/GA104):** RTX 3070, 3080, 3090 e Titan RTX.
* **Data Center (GA100):** A100.

Se você possui placas superiores à RTX 3060, o repositório original do Zagitta oferece as receitas estruturais (`sec2`/`booter`) para liberar a potência total dessas GPUs.
