# Integração: AssistentePessoal & GPULivre

Este documento especifica como o sistema `AssistentePessoal` (Jarvis) interage com o **GPULivre** para garantir síntese e reconhecimento de voz sem latência e sem riscos de Out of Memory (OOM).

---

## 1. O Problema de Concorrência no Jarvis
No `AssistentePessoal`, os modelos de voz (Piper TTS, Kokoro, XTTS v2 e Whisper VAD) necessitam de VRAM dedicada durante a fala humana. Quando outros processos concorrentes (como simulações de mercado ou indexações RAG) ocupam a GPU, ocorrem engasgos de áudio, atrasos inaceitáveis no Barge-In e falhas de CUDA OOM.

---

## 2. A Solução: `VoicePipelineAdapter` (Prioridade P1_CRITICAL)

Todo pipeline de voz no `AssistentePessoal` é encapsulado pelo `VoicePipelineAdapter`:

```python
from gpulivre.adapters.voice_adapter import VoicePipelineAdapter
from gpulivre.core.allocator import GPUAllocator

adapter = VoicePipelineAdapter(allocator=GPUAllocator())

def sintetizar_audio(texto: str):
    # Chamada ao Kokoro / Piper / XTTS
    return engine.synth(texto)

# Execução com proteção de slot P1
resultado = adapter.execute_voice_task(
    task_name="barge_in_response",
    voice_callable=sintetizar_audio,
    texto="Olá, mestre. A cotação do Bitcoin está em alta.",
    required_vram_mb=800.0
)
```

### O que acontece nos bastidores:
1. O GPULivre intercepta a requisição.
2. Se a VRAM estiver justa, localiza qualquer simulação ou tarefa em lote P3 ativa e dispara a **preempção graciosa**.
3. O modelo de áudio ganha prioridade imediata e sintetiza a fala humana com zero atraso.
4. Ao final do áudio, o slot é liberado no bloco `finally:` e o cache de VRAM é expurgado automaticamente.

---

## 3. Mecanismo de Habilitar / Desabilitar pelo Usuário

O usuário pode ativar ou desativar o GPULivre a qualquer momento sem reiniciar a aplicação:

### Via Chat / MCP Tool Call:
```json
{
  "tool": "gpulivre_set_user_toggle",
  "arguments": {
    "gpulivre_enabled": true,
    "hybrid_fallback_enabled": true
  }
}
```

### Via Arquivo de Configuração:
Editar `~/.antigravity/gpulivre_user_config.json`:
```json
{
  "gpulivre_enabled": false,
  "hybrid_fallback_enabled": false
}
```

### Efeito no Sistema:
- **`gpulivre_enabled: false` (Modo Bypass Legado):** O adaptador simplesmente chama a função de voz de forma direta, sem consultar slots ou preempções.
- **`gpulivre_enabled: true` (Modo Orquestrado):** Governança ativa com telemetria, preempção de P3 e expurgo de memória.
