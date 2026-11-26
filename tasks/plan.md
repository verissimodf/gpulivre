# Plano — Universal Inference Tuner

## Objetivo e resultado observável

Entregar o subsistema aprovado em `docs/superpowers/specs/2026-11-26-universal-inference-tuner-design.md`, com cálculo de tetos, benchmark OpenAI-compatible, hill-climb intercalado, quatro ferramentas MCP, skill embutida, documentação e testes verdes.

## Critérios de aceitação

1. Tetos de decode/prefill/KV calculados para hardware dedicado e unificado, sem valores físicos inventados.
2. Tokens derivados de `usage` ou tokenizador real injetado, nunca de chunks SSE.
3. Watchdog interrompe a amostra ao exceder o cap e registra `MEM_CAP`.
4. Rodada `ctl → cand → ctl` possui ao menos seis amostras agregadas e usa medianas.
5. Candidato é rejeitado por regressão >3%, flags ou falha de needle/soak.
6. Sucessos e falhas persistem em JSONL e podem ser consultados por filtros.
7. As quatro ferramentas MCP novas funcionam por `dispatch_tool`; as sete anteriores permanecem intactas.
8. `python3 -m unittest discover tests` passa no ambiente virtual isolado.
9. README, CHANGELOG, arquitetura, skill e `tasks/lessons.md` refletem o comportamento entregue.

## Premissas

- O allocator real é `GPUAllocator`; `VRAMAllocator` na linguagem do pedido refere-se a esse componente.
- Métricas físicas não disponíveis ficam desconhecidas e impedem somente os cálculos que dependem delas.
- Servidores sem `usage` exigem tokenizador real injetado; não há estimativa por chunks.
- O worktree baseado em `develop` não contém os benchmarks one-shot presentes apenas em outra branch; eles não serão copiados para esta feature.
- Não haverá benchmark de hardware real sem endpoint e modelo explícitos do usuário.

## Escopo incluído

- `gpulivre/core/ceiling.py`
- `gpulivre/core/bench_harness.py`
- `gpulivre/core/hill_climber.py`
- quatro handlers em `gpulivre/mcp/server.py`
- `tests/test_hill_climber.py`
- `skills/gpulivre-optimizer/`
- documentação e lições da feature

## Fora de escopo

- corrigir preempção física, oversubscription de slots ou corridas preexistentes no allocator;
- implementar transporte MCP stdio/rede ou o entry point de CLI ausente;
- alterar motor vendored de desbloqueio;
- executar cargas reais, instalar engines ou iniciar/modelar servidores sem solicitação explícita;
- commit, push ou PR do código da feature sem nova autorização.

## Passos de implementação

1. Preparar `/tmp/opencode/gpulivre-universal-inference-tuner-venv`, instalar dependências do projeto e medir a suíte baseline.
2. Escrever testes de ceiling, harness, watchdog, hill-climb, ledger e MCP; confirmar falha RED por ausência dos módulos.
3. Implementar os contratos e fórmulas de `ceiling.py`; rodar testes focados.
4. Implementar cliente SSE, métricas, watchdog e needle em `bench_harness.py`; rodar testes focados.
5. Implementar interleaving, regressões, gates, levers e ledger em `hill_climber.py`; rodar testes focados.
6. Integrar os quatro handlers MCP sem alterar os contratos existentes.
7. Criar a skill e atualizar README, CHANGELOG, arquitetura e lições.
8. Revisar o diff, executar toda a suíte, smoke tests e `trail check`.

## Verificação

- RED: `python3 -m unittest tests/test_hill_climber.py` antes dos módulos.
- GREEN focado após cada módulo.
- Gate final: `/tmp/opencode/gpulivre-universal-inference-tuner-venv/bin/python -m unittest discover tests`.
- Smoke: import dos módulos e chamadas MCP com mocks.
- Diff: `git diff --check develop...HEAD` e inspeção de working tree.
- Não declarar suporte de hardware real validado sem teste no respectivo backend.

## Riscos e rollback

- Dependências indisponíveis: instalar somente em `/tmp/opencode/gpulivre-universal-inference-tuner-venv`; se a rede falhar, registrar a lacuna e não modificar o host.
- APIs extensas demais: manter modelos pequenos e stdlib, sem frameworks novos.
- Efeitos colaterais em `~/.antigravity`: todos os testes usam diretórios temporários e mocks.
- Rollback: excluir os três módulos, quatro handlers, testes e skill; nenhum dado ou contrato existente é migrado.
