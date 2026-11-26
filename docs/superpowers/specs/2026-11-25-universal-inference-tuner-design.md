# GPULivre — otimizador universal de inferência

## Contrato de execução

Qualquer endpoint OpenAI-compatible pode participar da medição. A qualificação para produção exige um adaptador no host da inferência capaz de verificar a identidade do servidor, aplicar e confirmar a configuração, observar a residência de memória da carga e interrompê-la. O cliente pode estar em outra máquina. Cada resultado identifica os recursos disponíveis e a origem dos dados. O controle usa o GPUAllocator e os slots de TaskPriority existentes. A extensão do MCP é aditiva.

## Física e orçamento

ceiling.py combina GPUMonitor com sondas opcionais de CUDA, XPU, ROCm e MPS. Cada valor tem unidade, fonte e escopo por dispositivo. O estado do desbloqueio admite ativo, inativo, desconhecido e inaplicável. A presença de uma arquitetura ou de um arquivo de patch, por si só, é insuficiente para confirmar execução acelerada. Ausências admitem especificações explícitas e identificadas como tais.

Para decode c1 denso, a estimativa em tok/s divide a largura de banda efetiva em bytes/s pelos bytes lidos por token, incluindo pesos ativos, leitura do KV no contexto escolhido e custos do draft quando conhecidos. Para prefill, divide 50% do pico aplicável em operações/s pelas operações por token. A eficiência assumida e a precisão do kernel acompanham o resultado. O alvo acima de uma estimativa prática recebe aviso. A rejeição antecipada por impossibilidade exige um limite físico demonstrável sob as hipóteses do ensaio.

A capacidade KV em tokens usa a parte do orçamento do slot restante após pesos residentes, draft, buffers e margem, dividida pelos bytes KV por token. Em MoE, pesos ativos definem o custo de leitura e pesos residentes definem capacidade. Uma carga que excede o orçamento mesmo com contexto vazio é inviável. Contextos e concorrência compartilham o orçamento.

## Medição e interrupção

bench_harness.py usa HTTP e relógio monotônico da biblioteca padrão. TTFT termina no primeiro conteúdo gerado. O stream fornece eventos de tempo. A contagem de tokens vem de usage ou de tokenizer correspondente à revisão do modelo e ao template. Decode distingue o primeiro token dos seguintes. c1 usa a mediana das repetições e cN divide os tokens totais pela janela comum. Startup vai do lançamento controlado à prontidão verificada. Prefill frio requer evidência de contexto processado sem reaproveitamento de prefixo e registra a origem da duração. Uso de prompt dividido por TTFT pode ser informado como estimativa separada.

O watchdog observa o servidor durante carregamento, geração, silêncio do stream e gates concorrentes. Usa residência do processo ou carga identificada e verifica se o slot continua ativo. Amostras de memória global requerem atribuição verificável. Memória unificada exige contabilidade sem dupla contagem. A primeira amostra acima da reserva, preempção ou perda de observabilidade cancela a qualificação. O adaptador interrompe a carga, verifica o encerramento e registra o resultado da recuperação. O relatório expõe o intervalo de amostragem e identifica picos que podem escapar entre leituras. Falhas HTTP, resposta vazia, repetição degenerada ou evidência de throttling invalidam a amostra.

## Seleção e qualificação

hill_climber.py serializa o uso experimental do dispositivo e altera uma variável confirmada por vez. Seleciona levers suportados por lacuna relativa na métrica alvo multiplicada pela probabilidade estimada de ganho. P1 prioriza TTFT, P3 prioriza throughput e P2 explicita o objetivo equilibrado antes do ensaio. Cada bloco controle, candidato e controle tem ao menos duas amostras válidas sob carga equivalente. Compara medianas e direções por métrica. A promoção exige vitória estrita sobre os dois controles e regressão de no máximo 3% em cada métrica secundária frente a cada controle. Falta de evidência obrigatória impede promoção.

O candidato vencedor passa por recuperação exata de uma chave aleatória em contexto longo de comprimento verificado e por requisições simultâneas que ocupam o orçamento KV planejado. O soak verifica sobreposição, término das respostas e pico de memória. Os gates têm de ser executados com a configuração candidata reconfirmada após o último controle. Falhas restauram a configuração de controle e registram o estado efetivamente alcançado.

## Persistência e interfaces

Cada tentativa recebe ID e registro em ~/.antigravity/gpulivre_ledger.jsonl, incluindo revisões de hardware, driver, modelo e servidor, configurações, slot, amostras, gates, motivo e recuperação. Escritas concorrentes são serializadas. Uma tentativa interrompida permanece identificável. Histórico de rejeições evita repetir a mesma hipótese no mesmo ambiente. PLAN.md exibe antes, depois e percentual do teto aplicável.

O MCP acrescenta gpulivre_compute_ceiling, gpulivre_run_interleaved_bench, gpulivre_hill_climb e gpulivre_get_ledger_history. Relatórios distinguem medição, rejeição, interrupção e promoção. Os guias em skills/gpulivre-optimizer descrevem como interpretar limites, escolher uma variável e diagnosticar resultados inconclusivos.

## Critérios de entrega

Testes unittest com endpoint, controle e telemetria simulados cobrem memória dedicada e unificada, contagem independente dos chunks, ganho de decode com perda de TTFT superior a 3%, falha de needle ou soak, preempção durante stream silencioso, perda de telemetria, restauração e persistência de sucessos e falhas. Executar python -m unittest discover tests e informar o número de testes, falhas e skips. Documentar em README.md, CHANGELOG.md e tasks/lessons.md. O exemplo local deve identificar pré-requisitos, modelo, endpoint e como verificar o resultado no ledger. Resultados simulados e medições em hardware real recebem identificação explícita.
