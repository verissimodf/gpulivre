# 🚀 gpulivre — Desbloqueio de Silício (GA107/GA106) e Otimização Extrema de IA Local em GPUs GeForce RTX

O **gpulivre** é um conjunto de patches de kernel, automação de boot e tuning de containers para destravar 100% da capacidade dos **Tensor Cores (FP16 / BF16 / TF32)** em placas domésticas NVIDIA GeForce (com foco especial na arquitetura **Ampere GA107 / GA106**, como a **RTX 3050 6GB/8GB** e variantes laptop/desktop) rodando em **Ubuntu Server (24.04 / 26.04 LTS)** com **Ollama / llama.cpp**.

---

## 📊 Resultados Reais Comprovados (Homologado em MSI RTX 3050 6GB — `10de:2584`)

| Métrica | Configuração Padrão (Stock) | Com `gpulivre` (`SS0=0x88888888` + CUDA + FA) | Ganho Obtido |
| :--- | :--- | :--- | :--- |
| **Registrador de Tensor Cores (`SS0`)** | `0x30501002` (Capado em 50%) | **`0x88888888` (`SS1=0x8` Full Rate)** | **2x Taxa de Despacho FP16/BF16** |
| **Prefill / Leitura de Prompt (1.026 tokens)** | `158.22 tokens/s` | **`1.043,83 tokens/s`** | **+559% de Throughput** |
| **Latência Inicial (TTFT em 61 tokens)** | `385.53 ms` | **`39.02 ms`** | **~10x Menor Latência** |
| **Geração de Texto (`eval rate` Q4_0)** | `15.59 tokens/s` | **`38.47 a 43.84 tokens/s`** | **+146% a +181% de Velocidade** |

---

## 🧠 A Descoberta Técnica: O Patch do Silício GA107 (`0x177`)

Por padrão, o projeto upstream `Zagitta/gpu-unlock` inclui suporte apenas para o chip **GA106 (`0x176`)** usado nas primeiras RTX 3050 de 8GB (`0x2507`), falhando com erro `-19` (`-ENODEV`) nas placas equipadas com o die **GA107** (cujo registrador `PMC_BOOT_0` retorna `0xb77000a1`, onde `(chip >> 20) & 0x1ff == 0x177`).

O **gpulivre** introduz a macro `#define GA107(id) { id, 0x177, &ga10x }` na tabela de receitas do módulo de kernel, permitindo que toda a linha de chips **GA107** execute o payload `ga102-v67` no SEC2/Falcon durante o `initramfs` e desbloqueie os seletores `SS0=0x88888888` e `SS1=0x8` antes da carga do driver proprietário da NVIDIA.

### Placas RTX Semelhantes Beneficiadas pelo Método `GA107` (`0x177`)
Qualquer placa baseada no silício **GA107** pode ser adicionada à macro `GA107(0xXXXX)` no arquivo `recipes.c`:
* **RTX 3050 6GB Desktop:** `0x2584`
* **RTX 3050 8GB (Variante GA107 Desktop):** `0x2582`
* **RTX 3050 Mobile / Laptop:** `0x25a2`, `0x25a5`, `0x25e2`, `0x25e5`
* **RTX 3050 Ti Mobile / Laptop:** `0x25a0`, `0x25e0`

*(Para descobrir o ID exato da sua placa no Linux, execute `lspci -nn | grep -i nvidia` e observe o código `[10de:XXXX]`).*

---

## 🛠️ Passo a Passo de Implementação

### 1. Pré-requisitos de BIOS e Sistema Operacional
1. **Desative o Secure Boot** na BIOS/UEFI da placa-mãe (ou cadastre a chave MOK local via `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der`).
2. Instale as dependências de compilação do kernel:
   ```bash
   sudo apt update
   sudo apt install -y git build-essential debhelper dkms linux-headers-$(uname -r) curl ca-certificates xz-utils python3

3. Clonar Upstream, Aplicar o Patch GA107 e Instalar via DKMS
Bash
git clone [https://github.com/Zagitta/gpu-unlock.git](https://github.com/Zagitta/gpu-unlock.git) /opt/gpu-unlock
cd /opt/gpu-unlock

# 1. Injeta a macro GA107 (chip 0x177) no recipes.c
sed -i 's/#define GA106(id) { id, 0x176, \&ga10x }/#define GA106(id) { id, 0x176, \&ga10x }\n#define GA107(id) { id, 0x177, \&ga10x }/' kernel/gpu_unlock/recipes.c

# 2. Adiciona os PCI IDs das RTX 3050 GA107 (0x2584 6GB e 0x2582 8GB)
sed -i 's/GA106(0x2507),/GA106(0x2507), GA107(0x2584), GA107(0x2582),/' kernel/gpu_unlock/recipes.c

# 3. Compila os pacotes Debian (.deb) e instala forçando reconstrução do DKMS
make deb
sudo dpkg -i build/deb/gpu-unlock-dkms_*_all.deb build/deb/gpu-unlock-firmware_*_all.deb
3. Configurar o Bootloader (Drop-in GRUB) e Initramfs
Em versões modernas do Ubuntu (como o 26.04 LTS), utilize um arquivo drop-in em /etc/default/grub.d/ para garantir que o parâmetro gpu_unlock.targets=all seja preservado:

Bash
echo 'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT gpu_unlock.targets=all"' | sudo tee /etc/default/grub.d/99-gpu-unlock.cfg
sudo update-grub
sudo update-initramfs -u -k all
4. Sincronismo de Boot do Docker com o Driver NVIDIA
Em servidores rápidos com SSD NVMe, o daemon do Docker pode iniciar os containers (~10s de boot) antes que o módulo nvidia-drm finalize a inicialização da GPU (~35s de boot), fazendo o Ollama cair silenciosamente para 100% CPU. Crie a trava de sincronismo no systemd:

Bash
sudo mkdir -p /etc/systemd/system/docker.service.d
cat << 'EOF' | sudo tee /etc/systemd/system/docker.service.d/wait-for-nvidia.conf
[Unit]
After=systemd-udev-settle.service

[Service]
ExecStartPre=/bin/sh -c 'until /usr/bin/nvidia-smi >/dev/null 2>&1; do sleep 1; done'
EOF
sudo systemctl daemon-reload
5. Configuração Otimizada do Container Ollama (docker-compose.yml)
Ative o FlashAttention (OLLAMA_FLASH_ATTENTION=1), mantenha o cache KV em FP16 (OLLAMA_KV_CACHE_TYPE=f16) e desative explicitamente o backend Vulkan (OLLAMA_VULKAN=0) para usar o CUDA v13 puro:

YAML
services:
  ollama-llm:
    image: ollama/ollama:latest
    container_name: ollama-llm
    restart: unless-stopped
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - OLLAMA_FLASH_ATTENTION=1
      - OLLAMA_KV_CACHE_TYPE=f16
      - OLLAMA_VULKAN=0
    ports:
      - "127.0.0.1:11434:11434"
    volumes:
      - ./volumes/ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
🔍 Auditoria e Verificação Pós-Reboot
Após reiniciar o servidor (sudo reboot), execute:

Bash
sudo dmesg | grep -i gpu_unlock
Saída esperada confirmando o desbloqueio físico dos seletores ss0=0x88888888 e ss1=0x8:

Plaintext
pci 0000:01:00.0: gpu_unlock: stage=gfw plm=0xffffff8f ss0=0x30501002 ss1=0x0
pci 0000:01:00.0: gpu_unlock: stage=host ss0=0x88888888 ss1=0x8 plm_feat=0xffffffff
pci 0000:01:00.0: gpu_unlock: stage=flr post_wpr2=0x1ffffe00:0x0 ss0=0x88888888 ss1=0x8
pci 0000:01:00.0: gpu_unlock: unlocked

---

### Arquivo 2: Gerando o Arquivo `.patch` Oficial no Seu Servidor

Para deixar o seu projeto `gpulivre` profissional, você pode extrair o `git diff` exato da modificação que fizemos em `/opt/laytech-ai/gpu-unlock` e salvá-lo em `/opt/laytech-ai/gpulivre/patches/ga107-rtx3050.patch` rodando:

```bash
cd /opt/laytech-ai/gpu-unlock
git diff kernel/gpu_unlock/recipes.c > /opt/laytech-ai/gpulivre/patches/ga107-rtx3050.patch
cat /opt/laytech-ai/gpulivre/patches/ga107-rtx3050.patch
