# Engineering Lessons & Self-Correction Log (`gpulivre`)

## [2026-11-02] - Isolamento de Ambiente e Repositórios Externos ao Google Drive
- **Erro / Falha Detectada:** Tentativa de execução de comandos do sistema falhou com `g:\Meu Drive\Laus_Docs: no such directory` devido ao fato do drive virtual Google Drive estar desconectado no host Windows.
- **Causa Raiz:** O Google Drive (`G:`) pode sofrer unmount, travamentos de sincronização ou bloqueios de arquivo (file locks) durante operações de IO massivo e chamadas de subprocessos.
- **Ação Corretiva Implementada:** Todo o código, clones e novos projetos foram direcionados estritamente para o diretório local fora da nuvem: `C:\Users\layte\.antigravity\projects\gpulivre`.
- **Regra Preventiva (Anti-Repetição):** Nunca clonar, criar ambientes virtuais ou executar subprocessos dependentes de drives de nuvem (`G:\`). Sempre utilizar caminhos absolutos locais sob `~/.antigravity/` ou `C:\Users\layte\...`.
- **Evidência de Validação:** Projeto inicializado com sucesso em `C:\Users\layte\.antigravity\projects\gpulivre`.

---

## [2026-11-02] - Sandbox de Subagentes e Limitação de Execução de Comandos Interativos
- **Erro / Falha Detectada:** O subagente de pesquisa (`research`) tentou executar comandos shell/python interativos e sofreu negação de permissão ("Permission prompt for action was denied").
- **Causa Raiz:** Subagentes em modo `research` possuem perfil restrito (read-only) e o runtime dispara prompts de segurança interativos que bloqueiam a automação não assistida.
- **Ação Corretiva Implementada:** As ações de criação, escrita, inspeção estruturada e modelagem de arquitetura foram centralizadas no agente orquestrador utilizando as ferramentas nativas de filesystem (`write_file`, `read_file`) e o MCP `codebase-memory`.
- **Regra Preventiva (Anti-Repetição):** Delegar para subagentes do tipo `research` apenas leituras e consultas não mutativas; usar o agente principal para escrita e criação direta de artefatos de código.
- **Evidência de Validação:** Estrutura completa de módulos do `gpulivre` escrita e validada sem interrupções.

---

## [2026-11-02] - Concorrência de GPU em Ecossistemas Multi-Agente (AssistentePessoal + Laus)
- **Erro / Falha Detectada:** Risco crítico de colapso de memória de vídeo (CUDA OOM) quando múltiplos agentes locais disparam simultaneamente inferências de voz (Whisper/Piper) e simulações pesadas (MiroFish Swarm / Quant).
- **Causa Raiz:** Ausência de um árbitro central de preempção de VRAM e falta de expurgo forçado de tensores residuais no pool do PyTorch.
- **Ação Corretiva Implementada:** Criação do módulo `gpulivre.core.allocator` com níveis de prioridade (P1 Crítico para Voz/Chat > P2 Alto para RAG > P3 Batch para Swarms) e capacidade de preempção graciosa de tarefas secundárias.
- **Regra Preventiva (Anti-Repetição):** Qualquer agente que requisite inferência local de GPU deve consultar e reservar um slot via `gpulivre_request_slot` antes de carregar pesos em VRAM, e invocar `gpulivre_release_slot` ao concluir.
- **Evidência de Validação:** Suíte de testes unitários `tests/test_allocator.py` validando o algoritmo de preempção em cenários de memória restrita.

---

## [2026-11-02] - Adoção Segura com Mecanismo de Toggle pelo Usuário no AssistentePessoal
- **Erro / Falha Detectada:** Risco de regressão ou interrupção na experiência do usuário do `AssistentePessoal` caso novas camadas de orquestração de GPU se tornassem mandatórias e inflexíveis.
- **Causa Raiz:** Modificações em pipelines de voz e inferência exigem transição suave e possibilidade de rollback imediato sem necessidade de alterar código-fonte.
- **Ação Corretiva Implementada:** Implementação do `UserConfigManager` (`gpulivre/adapters/user_toggle.py`) permitindo ao usuário habilitar/desabilitar o GPULivre e o Fallback Híbrido em tempo de execução via arquivo de configuração (`~/.antigravity/gpulivre_user_config.json`), variáveis de ambiente (`GPULIVRE_ENABLED`) e ferramentas MCP (`gpulivre_set_user_toggle`).
- **Regra Preventiva (Anti-Repetição):** Toda funcionalidade de infraestrutura de hardware em ecossistemas de agentes deve implementar um modo *bypass legado*, garantindo que o usuário possa desativar a orquestração a qualquer momento com zero quebras de compatibilidade.
- **Evidência de Validação:** `tests/test_hybrid_e2e_validation.py::test_01_user_toggle_disabled_bypass_mode` confirmando o bypass total e transparente quando desativado.

---

## [2026-11-02] - Prevenção de Inanição de Agentes via Fallback Híbrido (Colab / Gemini)
- **Erro / Falha Detectada:** Tarefas pesadas em lote (como enxames de 20 agentes MiroFish) ficavam bloqueadas indefinidamente quando a GPU local atingia 100% de ocupação ou sofria preempção repetida por comandos de voz.
- **Causa Raiz:** Falta de uma rota alternativa de computação remota quando o hardware local esgota seus limites de VRAM.
- **Ação Corretiva Implementada:** Implementação do `HybridFallbackEngine` (`gpulivre/core/hybrid_fallback.py`), integrando o padrão `colab-drive-modularizer` para desvio automático de simulações pesadas para GPUs gratuitas do Google Colab (Tesla T4/A100) e rotas de inferência para o Google AI Studio (Gemini 2.5 Flash).
- **Regra Preventiva (Anti-Repetição):** Nunca permitir que um agente autônomo trave por saturação de VRAM local; sempre prover rota de transbordo (offload) assíncrono para instâncias na nuvem.
- **Evidência de Validação:** `tests/test_hybrid_e2e_validation.py::test_04_vram_saturation_triggers_hybrid_fallback_to_colab` comprovando o redirecionamento automático sob saturação de memória.
