# Design — Universal Hardware & Inference Auto-Tuner

- **Data:** 2026-11-26
- **Status:** aprovado para implementação
- **Branch:** `feature/universal-inference-tuner`
- **Base:** `develop` (`1942b1c`)
- **Worktree:** `/tmp/opencode/gpulivre-universal-inference-tuner`

## 1. Objetivo

Fundir a orquestração de VRAM e telemetria do GPULivre com a metodologia científica de medição e autoaperfeiçoamento do `inference-optimisation-skill`, sem acoplar o resultado a uma GPU, quantização ou engine específica.

O subsistema deve:

1. calcular tetos físicos de decode, prefill e KV;
2. medir servidores OpenAI-compatible sem contar chunks de stream como tokens;
3. executar experimentos intercalados `controle → candidato → controle`;
4. promover somente configurações que superem os controles e os portões de qualidade;
5. persistir todas as tentativas, inclusive falhas;
6. expor o ciclo completo por ferramentas MCP;
7. oferecer a metodologia como skill nativa do repositório.

## 2. Princípios e invariantes

1. **Agnosticismo:** CUDA, ROCm, XPU, MPS, memória unificada, CPU e servidores OpenAI-compatible são detectados por adapters independentes.
2. **Sem números inventados:** largura de banda, TFLOPs e estado do patch só entram no cálculo quando têm origem verificável. Ausência de medição resulta em campo desconhecido e erro explícito quando a fórmula depende dele.
3. **Contagem real:** tokens medidos vêm de `usage` do servidor ou de um tokenizador real fornecido pelo chamador. Chunks SSE nunca são usados como contador.
4. **Uma variável por vez:** um candidato altera apenas um lever.
5. **Controle estatístico:** comparação por medianas de controles vizinhos, com no mínimo seis amostras agregadas por rodada.
6. **Retrocompatibilidade:** as sete ferramentas MCP existentes e os contratos de `GPUMonitor` e `GPUAllocator` permanecem válidos.
7. **Isolamento:** experimentos serializados; nenhuma parada por padrão de processo ou `pkill -f`.
8. **Falha segura:** watchdog interrompe a coleta, sinaliza violação de memória e apenas encerra servidor quando existe comando de lifecycle definido pelo operador.

## 3. Arquitetura

```text
                         agentes externos
                                │
                        gpulivre/mcp/server.py
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
       core/ceiling.py   core/bench_harness.py   core/hill_climber.py
             │                  │                  │
       hardware profile    OpenAI-compatible     ledger.jsonl / PLAN.md
             │                  │
             └──────────┬───────┘
                        │
          GPUMonitor ───┴─── GPUAllocator / VRAM cap
```

Os módulos são aditivos. `ceiling.py` e `bench_harness.py` não importam o hill-climber; o hill-climber consome ambos. O servidor MCP conserva o mapa `TOOL_DISPATCHER` e adiciona quatro handlers.

## 4. Módulo `gpulivre/core/ceiling.py`

### 4.1 Contratos públicos

- `HardwareCapabilities`
  - backend, nome e identificador do dispositivo;
  - memória total, dedicada e unificada;
  - largura de banda efetiva em GB/s;
  - picos FP16, BF16 e INT8 em TFLOPs;
  - Tensor Cores desbloqueadas, com estado `enabled`, `disabled`, `unknown` ou `not_applicable`;
  - origem e timestamp de cada medição.
- `ModelSpec`
  - parâmetros ativos;
  - bytes de pesos estimados ou preferencialmente medidos;
  - bits por peso;
  - bytes de KV por token;
  - dimensões de atenção;
  - estado opcional de draft head e comprimento de draft.
- `DraftSpec`
  - bytes do draft layer e do draft head;
  - comprimento e taxa de aceitação observada.
- `CeilingReport`
  - teto de decode por contexto e concorrência;
  - teto de prefill por contexto;
  - bytes de KV por stream;
  - streams completas e tokens máximos dentro do cap;
  - avisos de medição incompleta.

### 4.2 Descoberta de hardware

A precedência é:

1. medições fornecidas explicitamente pelo operador;
2. medições persistidas no cache local do GPULivre;
3. probe específico do backend disponível;
4. estado desconhecido.

Backends possíveis:

- NVIDIA: NVML para identidade, memória e estado do patch; probe de bandwidth/FLOPs quando solicitado;
- ROCm: `torch.cuda` com backend ROCm ou `rocm-smi` quando disponível;
- Intel XPU: `torch.xpu` e informações do dispositivo;
- Apple MPS: `torch.backends.mps`, memória unificada e `iogpu.wired_limit_mb` quando legível;
- memória unificada Linux/DRM: `/proc/<pid>/fdinfo` e `drm-resident-*`;
- CPU: RAM do sistema, sem inventar aceleração de GPU.

O patch de Tensor Cores é específico da integração NVIDIA existente. Em outros backends o estado é `not_applicable`; em NVIDIA desconhecido permanece `unknown`.

### 4.3 Fórmulas

- **Decode:** `bandwidth_gbs * 1e9 / bytes_read_per_emitted_token`.
- **Bytes por token:** pesos dense, experts efetivamente roteados quando aplicável, LM head, KV do contexto e custo de draft.
- **Prefill:** `0.5 * peak_tflops * 1e12 / flops_per_token`, com atenção causal média em `L/2`.
- **KV máximo:** `(vram_cap_mb - pesos - runtime_reserve_mb) / kv_bytes_per_token`.
- **Streams completas:** `floor(tokens_max / contexto)`.

Os cálculos retornam `None` e um aviso quando uma grandeza obrigatória não foi medida. `compute_theoretical_ceiling` nunca divide por zero nem inventa defaults físicos.

## 5. Módulo `gpulivre/core/bench_harness.py`

### 5.1 Cliente OpenAI-compatible

Cliente stdlib com `urllib.request`, sem dependência de SDK:

- `GET /v1/models` para health check;
- `POST /v1/chat/completions` com `stream=true` e `stream_options.include_usage=true`;
- parsing SSE apenas para obter TTFT e conteúdo;
- contagem de tokens exclusivamente pelo bloco final `usage` ou por `token_counter` real injetado;
- ausência de `usage` e ausência de tokenizador resulta em amostra inválida, nunca em contagem por chunks.

### 5.2 Métricas

- startup do servidor, quando há comando `launch`;
- TTFT curto e longo;
- c1 decode, mediana de N amostras;
- cN aggregate e throughput por stream;
- cold prefill em 4K, 32K, 128K e 256K, escalonados à VRAM disponível;
- needle retrieval com três códigos únicos em profundidades distintas e correspondência exata;
- pegada de memória real e pico;
- flags `LOOP`, `EMPTY_OUTPUT`, `REQ_FAIL`, `THERMAL_THROTTLE`, `CLOCK_DROP`, `MEM_CAP`, `ON_BATTERY`.

### 5.3 Watchdog de memória

O watchdog recebe um sampler de bytes e um callback de interrupção:

1. amostra antes da requisição;
2. mantém amostragem durante o teste;
3. encerra imediatamente ao ultrapassar o cap;
4. cancela workers e fecha a resposta ativa;
5. chama `stop` apenas para servidor gerenciado, tentando drain antes do encerramento;
6. persiste `MEM_CAP` e a tentativa como falha.

Em servidor externo sem lifecycle, a chamada cliente é abortada e o operador recebe instrução para encerrar o processo pelo próprio gerenciador.

### 5.4 Thermal throttling

A detecção usa telemetria opcional do `GPUMonitor`: comparação de clock e temperatura antes/depois, com limiares configuráveis. Quando não há sensor, a ausência de clock não é tratada como throttling.

## 6. Módulo `gpulivre/core/hill_climber.py`

### 6.1 Rodada intercalada

A ordem é `ctl → cand → ctl`. Cada perna recebe `samples_per_leg`, mínimo de dois, totalizando no mínimo seis amostras. Cada amostra executa o mesmo conjunto de métricas. As medianas de cada perna alimentam a decisão.

Para evitar drift de uma única sequência, a API suporta repetição da sequência completa; a decisão padrão exige os dois controles vizinhos.

### 6.2 Critério de promoção

Para a métrica alvo:

- P1: menor TTFT;
- P2: maior decode c1, com TTFT como restrição;
- P3: maior throughput agregado cN, ou decode/prefill conforme configuração.

O candidato é aceito somente quando:

1. supera as medianas dos dois controles na métrica alvo;
2. não regride mais de 3% em nenhuma métrica comparável;
3. não apresenta flags inválidas;
4. passa needle retrieval;
5. passa full-KV concurrent soak dentro do cap.

Falha em qualquer portão rejeita a configuração, com o motivo exato no ledger.

### 6.3 Seleção de lever

Cada lever declara:

- nome e descrição;
- patch aplicável a uma engine/configuração;
- métricas afetadas;
- probabilidade estimada de ganho;
- aplicabilidade opcional.

Prioridade: `gap_para_o_teto × probabilidade_de_ganho`, com desempate por custo. Somente um lever é alterado por candidato. Candidatos genéricos podem usar `extra_body`; flags de processo exigem `launch` gerenciado e engine declarada.

### 6.4 Persistência

- Ledger: `~/.antigravity/gpulivre_ledger.jsonl`, append-only, uma linha JSON por tentativa.
- Plano: `~/.antigravity/gpulivre_PLAN.md`, relatório legível.
- Registro: timestamp UTC, fingerprint do hardware, modelo, quantização, engine, configuração, lever, métricas, medianas, flags, portões, decisão e motivo.
- Consulta histórica filtra por hardware, modelo e lever, permitindo pular tentativas já rejeitadas.

## 7. Ferramentas MCP

As ferramentas retornam JSON string, como as existentes.

1. `gpulivre_compute_ceiling`
   - entradas: especificação do modelo, contexto, concorrência e cap;
   - saída: `CeilingReport` serializado.
2. `gpulivre_run_interleaved_bench`
   - entradas: controle, candidato, repetições, cap e perfil;
   - saída: medianas, regressões, portões e decisão.
3. `gpulivre_hill_climb`
   - entradas: endpoint, modelo, baseline, levers, prioridade e limites;
   - saída: sessão resumida, vencedor, ledger e plano.
4. `gpulivre_get_ledger_history`
   - entradas: filtros e limite;
   - saída: histórico filtrado e tentativas rejeitadas que devem ser evitadas.

O toggle `gpulivre_enabled` continua controlando a orquestração. O tuner deve respeitar o bypass legado: quando desabilitado, apenas consulta e benchmark são permitidos; experimentos que exigem lifecycle retornam erro explícito.

## 8. Skill embutida

Diretório `skills/gpulivre-optimizer/`:

- `SKILL.md`: procedimento, gates, ordem de levers e reporting;
- `references/ceilings.md`: fórmulas e exemplos;
- `references/levers.md`: levers genéricos e específicos por engine;
- `references/gotchas.md`: CUDA, ROCm, XPU, MPS, memória unificada e engines OpenAI-compatible;
- `references/methodology.md`: contagem, ruído, watchdog e interleaving.

A skill declara atribuição explícita ao `inference-optimisation-skill`, licença MIT e URL da fonte.

## 9. Testes

`tests/test_hill_climber.py` usa `unittest` e mocks, sem rede real:

1. teto de GPU dedicada com Tensor Cores versus memória unificada;
2. rejeição por regressão de TTFT superior a 3%;
3. rejeição por falha de needle retrieval;
4. interrupção imediata do watchdog ao exceder o cap;
5. persistência de sucesso e falha no `ledger.jsonl`;
6. ordem serializada `ctl → cand → ctl` e mínimo de seis amostras;
7. contagem por `usage`, rejeição de ausência de usage e rejeição de contagem por chunks;
8. dispatcher MCP para as quatro ferramentas novas.

A suíte existente será preservada. A implementação será TDD, com os testes novos vermelhos antes do código de produção. Testes MCP usarão singletons e caminhos temporários injetados, sem alterar `~/.antigravity/gpulivre_user_config.json` nem o ledger real do usuário.

## 10. Documentação e lições

- `README.md`: novo subsistema, quatro ferramentas, uso e exemplo de hill-climb;
- `CHANGELOG.md`: seção `[Não lançado]`;
- `tasks/lessons.md`: decisões e armadilhas descobertas durante a implementação;
- `docs/ARQUITETURA.md`: fluxo e contratos do novo subsistema;
- `.trail/learnings/`: decisão durável quando o código não expressar claramente a justificativa.

## 11. Verificação

1. testes novos e existentes com `python3 -m unittest discover tests`;
2. smoke test do import e do dispatcher MCP;
3. `trail check` antes de revisão;
4. inspeção do diff contra `develop`;
5. atualização de `CHANGELOG.md`, `README.md`, `tasks/lessons.md` e `AGENTS.md` se a estrutura mudar.

O ambiente de teste será isolado em `.venv` dentro do worktree, sem alterar o Python global.

## 12. Riscos e rollback

- **Medição insuficiente:** os tetos ficam explicitamente incompletos; não há resultado silenciosamente incorreto.
- **Endpoint sem `usage`:** a amostra é rejeitada e o operador deve fornecer tokenizador real.
- **Candidato que exige reinício:** só é considerado com lifecycle gerenciado e seguro.
- **Preempção atual é contábil:** marcar um slot como `preempted` não descarrega o workload. O tuner será serializado e cooperativo, mas não prometerá liberação física pela preempção existente. Corrigir preempção, oversubscription de reservas, transporte MCP e o entry point de CLI fica fora deste escopo para evitar regressões não relacionadas.
- **Ledger corrompido:** cada linha é validada na leitura; linhas inválidas são ignoradas e reportadas.
- **Regressão em produção:** as quatro ferramentas são aditivas e o toggle legado permanece disponível.
- **Rollback:** remover os módulos novos, os handlers novos e o diretório da skill restaura o comportamento anterior sem migração de dados.
