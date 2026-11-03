"""
GPULivre Voice Pipeline Adapter for AssistentePessoal.
Intercepts TTS (Kokoro/Piper/XTTS) and Whisper VAD/Barge-In tasks.
Enforces P1_CRITICAL priority with preemption and user toggle support.
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional

from gpulivre.core.allocator import GPUAllocator, TaskPriority
from gpulivre.adapters.user_toggle import user_config_manager

logger = logging.getLogger("gpulivre.adapter.voice")


class VoicePipelineAdapter:
    """
    Wraps voice execution in AssistentePessoal with GPULivre VRAM governance.
    """

    def __init__(self, allocator: Optional[GPUAllocator] = None, config_manager=None):
        self.allocator = allocator
        self.config_manager = config_manager or user_config_manager

    def execute_voice_task(
        self,
        task_name: str,
        voice_callable: Callable[..., Any],
        *args,
        required_vram_mb: float = 800.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executes a voice synthesis or transcription task.
        If GPULivre is disabled by the user, bypasses slot allocation directly.
        If enabled, allocates with P1_CRITICAL priority, preempting P3 batch tasks.
        """
        # 1. User Toggle Bypass
        if not self.config_manager.is_gpulivre_enabled():
            logger.info(f"[Legacy Mode] GPULivre disabled by user. Running voice task '{task_name}' directly.")
            output = voice_callable(*args, **kwargs)
            return {
                "success": True,
                "mode": "legacy_direct",
                "output": output,
                "preempted_other_tasks": [],
            }

        # 2. GPULivre Orchestrated Execution
        slot_id = None
        preempted_slots = []
        if self.allocator:
            res = self.allocator.request_slot(
                agent_id="assistente_voice_engine",
                task_name=task_name,
                required_vram_mb=required_vram_mb,
                priority=TaskPriority.P1_CRITICAL,
            )
            if not res.success:
                logger.error(f"Failed to reserve VRAM for critical voice task: {res.reason}")
                # Voice must not fail; run in best-effort fallback
            else:
                slot_id = res.slot.slot_id if res.slot else None
                preempted_slots = res.preempted_slots
                if preempted_slots and self.config_manager.config.notify_on_preemption:
                    logger.info(f"P1 Voice preempted {len(preempted_slots)} background task(s) to guarantee zero audio lag.")

        try:
            output = voice_callable(*args, **kwargs)
            return {
                "success": True,
                "mode": "gpulivre_p1_orchestrated",
                "slot_id": slot_id,
                "preempted_other_tasks": preempted_slots,
                "output": output,
            }
        finally:
            # 3. Clean up VRAM immediately after audio generation
            if slot_id and self.allocator:
                self.allocator.release_slot(slot_id)
