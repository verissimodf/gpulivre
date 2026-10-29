#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "[+] gpulivre: Instalador Automatizado de Desbloqueio e Otimização"
echo ""

sudo apt update
sudo apt install -y git build-essential debhelper dkms linux-headers-$(uname -r) curl ca-certificates xz-utils python3

SRC_DIR="/opt/gpu-unlock"
if [ ! -d "$SRC_DIR" ]; then
sudo git clone https://github.com/Zagitta/gpu-unlock.git "$SRC_DIR"
fi

cd "$SRC_DIR"
git pull

if ! grep -q "GA107" kernel/gpu_unlock/recipes.c; then
sed -i 's/#define GA106(id) { id, 0x176, &ga10x }/#define GA106(id) { id, 0x176, &ga10x }\n#define GA107(id) { id, 0x177, &ga10x }/' kernel/gpu_unlock/recipes.c
sed -i 's/GA106(0x2507),/GA106(0x2507), GA107(0x2584), GA107(0x2582),/' kernel/gpu_unlock/recipes.c
fi

make deb
sudo dpkg -i build/deb/gpu-unlock-dkms_all.deb build/deb/gpu-unlock-firmware_all.deb || sudo apt-get install -f -y

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
