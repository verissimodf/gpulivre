
🔬 Documentação de Engenharia: Suporte à Arquitetura GA107
1. O Problema Silicioso
A NVIDIA implementa limites de hardware em placas de consumo através de seletores de velocidade internos (SS0 e SS1). Em GPUs RTX série 30 baseadas em GA106 e GA107, os Tensor Cores operam por padrão com metade da taxa de transferência em operações de precisão mista FP16/BF16.

2. O Mecanismo de Interceptação (initramfs)
O utilitário atua nos primeiros instantes do boot do Linux (fase init-top do initramfs):

Claim: O módulo de kernel intercepta o barramento PCI do dispositivo antes que o driver nvidia realize o bind.

PLM & Firmware Execution: O payload binário executa via motor SEC2/Falcon.

Register Overwrite: Os registradores de controle de clock dos Tensor Cores são reescritos de 0x30501002 para 0x88888888.

FLR (Function Level Reset): Um reset controlado é aplicado à placa, garantindo persistência quando o driver oficial assume o controle.
