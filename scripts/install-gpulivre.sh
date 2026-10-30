#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "[+] gpulivre: Instalador Automatizado de Desbloqueio e Otimização"
echo ""

sudo apt update
sudo apt install -y git build-essential debhelper dkms linux-headers-$(uname -r) curl ca-certificates xz-utils python3

# Motor: árvore vendida neste repositório (subtree de Zagitta/gpu-unlock)
# já com as receitas GA107 (0x2584 / 0x2582) commitadas — sem clone remoto
# nem edição via sed. Payload portável equivalente em patches/0001-support-ga107.patch.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../upstream/gpu-unlock" && pwd)"
cd "$SRC_DIR"

make deb
sudo dpkg -i build/deb/gpu-unlock-dkms_*_all.deb build/deb/gpu-unlock-firmware_*_all.deb

echo 'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT gpu_unlock.targets=all"' | sudo tee /etc/default/grub.d/99-gpu-unlock.cfg
sudo update-grub
sudo update-initramfs -u -k all

sudo systemctl enable --now nvidia-persistenced
sudo mkdir -p /etc/systemd/system/docker.service.d
cat << 'EOT' | sudo tee /etc/systemd/system/docker.service.d/wait-for-nvidia.conf
[Unit]
After=systemd-udev-settle.service

[Service]
ExecStartPre=/bin/sh -c 'until /usr/bin/nvidia-smi >/dev/null 2>&1; do sleep 1; done'
EOT
sudo systemctl daemon-reload

echo ""
echo "[✔] Instalação concluída! Reinicie o servidor com 'sudo reboot'."
echo ""
