# Integração upstream: Zagitta/gpu-unlock → gpulivre

Este documento descreve como o motor de desbloqueio está integrado ao gpulivre,
como atualizá-lo e como validar mudanças **sem comprometer o host em produção**.

## 1. Proveniência

- **Upstream:** [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) v2.0.0, commit `1fc6d62`.
- **Forma de integração:** `git subtree` com squash em `upstream/gpu-unlock/`.
- **Delta gpulivre:** receitas GA107 (macro `GA107(id) { id, 0x177, &ga10x }` e
  device IDs `0x2584` — RTX 3050 6GB — e `0x2582` — RTX 3050 8GB) commitadas
  diretamente em `upstream/gpu-unlock/kernel/gpu_unlock/recipes.c`.
- **Payload portável:** o mesmo delta, no formato `git apply`, está em
  `patches/0001-support-ga107.patch`. O CI aplica `--check` sobre um worktree do
  commit pristine `1fe5c8c` a cada push.

## 2. O que o gpulivre acrescentou sobre a base

1. Suporte homologado GA107/GA106 de entrada (RTX 3050 6GB/8GB).
2. Instalador único `scripts/install-gpulivre.sh` que constrói o motor **a partir
   da árvore vendida** (sem clone remoto e sem edição `sed`).
3. Config do GRUB por drop-in (`/etc/default/grub.d/99-gpu-unlock.cfg`).
4. Barreira Docker↔driver (`docker.service.d/wait-for-nvidia.conf`).
5. Perfil Ollama otimizado (`docker/docker-compose.yml`).
6. Documentação PT-BR e visualizações interativas (GitHub Pages).

## 3. Atualizando o upstream no futuro

```sh
# na raiz do gpulivre
git subtree pull --prefix upstream/gpu-unlock https://github.com/Zagitta/gpu-unlock main --squash
```

Se o upstream mudar a seção de receitas aproveitadas pelo patch:

```sh
# validação portabilidade
git worktree add --detach /tmp/upstream <commit-pristine>
git apply --check --directory=upstream/gpu-unlock patches/0001-support-ga107.patch
```

## 4. Validação segura (sem tocar no host em produção)

- **Teste de build (não invasivo):** `make deb` dentro de
  `upstream/gpu-unlock/` — gera `.deb` em `build/deb/` sem instalar nada no
  sistema (`*.deb`, `build/` são ignorados pelo `.gitignore` do upstream).
- **Verificação runtime (somente leitura):**
  ```sh
  journalctl -k -b | grep gpu_unlock   # prova de unlock aplicado no boot
  lspci -nnk -s 0000:01:00.0           # driver em bind na placa
  nvidia-smi                           # estado funcional da GPU
  ```
- **Nunca em produção:** `dpkg -i`, `dkms remove/install`, `update-initramfs`
  ou `update-grub` fora do fluxo do instalador em janela de manutenção.

## 5. Créditos e conformidade

O motor permanece GPL-2.0 (crédito Zagitta/gpu-unlock); atribuições NVIDIA em
`upstream/gpu-unlock/licenses/` e `upstream/gpu-unlock/firmware/NOTICE.md`
devem acompanhar qualquer distribuição de pacotes gerados.
