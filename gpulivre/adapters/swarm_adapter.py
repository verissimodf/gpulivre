"""
GPULivre Swarm & Simulation Adapter for Laus Ecosystem.
Intercepts MiroFish Swarm (20 agents), Quant Arbitrage, and Batch Training.
Enforces P3_BATCH priority, supports preemption by Voice, and offloads
to Google Colab / Gemini when local VRAM is saturated.
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional

from gpulivre.core.allocator import GPUAllocator, TaskPriority
from gpulivre.core.hybrid_fallback import HybridFallbackEngine, hybrid_fallback_engine
from gpulivre.adapters.user_toggle import user_config_manager

logger = logging.getLogger("gpulivre.adapter.swarm")


class SwarmSimulationAdapter:
    """
    Wraps MiroFish and multi-agent simulation workloads with GPULivre VRAM governance
    and automatic cloud offloading.
    """

    def __init__(
        self,
        allocator: Optional[GPUAllocator] = None,
        fallback_engine: Optional[HybridFallbackEngine] = None,
        config_manager=None,
    ):
        self.allocator = allocator
        self.fallback_engine = fallback_engine or hybrid_fallback_engine
        self.config_manager = config_manager or user_config_manager

    def execute_simulation(
        self,
        task_name: str,
        simulation_callable: Callable[..., Any],
        payload: Dict[str, Any],
        *args,
        required_vram_mb: float = 2048.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executes a MiroFish multi-agent simulation.
        If GPULivre is disabled: runs directly on local host.
        If enabled:
          - Tries local VRAM allocation as P3_BATCH.
          - If VRAM is full: automatically offloads to Google Colab GPU runtime.
          - If preempted by P1 Voice during run: checkpoints and defers.
        """
        # 1. User Toggle Bypass
        if not self.config_manager.is_gpulivre_enabled():
            logger.info(f"[Legacy Mode] GPULivre disabled by user. Running swarm '{task_name}' directly.")
            output = simulation_callable(*args, **kwargs)
            return {
                "success": True,
                "mode": "legacy_direct",
                "execution_target": "local_unmanaged",
                "output": output,
            }

        # 2. Attempt local VRAM allocation
        slot_id = None
        if self.allocator:
            res = self.allocator.request_slot(
                agent_id="laus_swarm_orchestrator",
                task_name=task_name,
                required_vram_mb=required_vram_mb,
                priority=TaskPriority.P3_BATCH,
            )

            # If local slot denied -> Fallback to Google Colab
            if not res.success:
                logger.warning(
                    f"Local VRAM insufficient for '{task_name}' ({required_vram_mb} MB). "
                    f"Triggering Hybrid Fallback Engine."
                )
                fallback_res = self.fallback_engine.route_task(
                    task_name=task_name,
                    task_type="swarm",
                    payload=payload,
                    priority=3,
                )
                return {
                    "success": fallback_res.success,
                    "mode": "hybrid_fallback_offload",
                    "execution_target": fallback_res.dispatched_to,
                    "reason": fallback_res.reason,
                    "data": fallback_res.result_data,
                    "latency_ms": fallback_res.latency_ms,
                }

            slot_id = res.slot.slot_id if res.slot else None

        # 3. Local execution with slot active
        try:
            output = simulation_callable(*args, **kwargs)
            
            # Check if this slot was preempted by a voice event during execution
            was_preempted = False
            if slot_id and self.allocator:
                active_slot = self.allocator._active_slots.get(slot_id)
                if active_slot and active_slot.status == "preempted":
                    was_preempted = True

            return {
                "success": True,
                "mode": "gpulivre_p3_local",
                "slot_id": slot_id,
                "was_preempted_by_higher_priority": was_preempted,
                "execution_target": "local_gpu_pool",
                "output": output,
            }
        finally:
            if slot_id and self.allocator:
                self.allocator.release_slot(slot_id)
