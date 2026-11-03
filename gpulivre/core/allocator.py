"""
GPULivre Core: Prioritized VRAM allocator with preemption for multi-agent workloads.
"""

from __future__ import annotations
import uuid
import time
import logging
from enum import IntEnum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from gpulivre.core.monitor import GPUMonitor, GPUStatus
from gpulivre.core.cleaner import GPUCleaner

logger = logging.getLogger("gpulivre.allocator")


class TaskPriority(IntEnum):
    P1_CRITICAL = 1   # Interactive Voice, VAD, User Chat (Highest)
    P2_HIGH = 2       # RAG Embeddings, Vision, Immediate Tool calls
    P3_BATCH = 3      # MiroFish Swarms, Backtesting, Quant Simulations (Preemptible)


class AllocationSlot(BaseModel):
    slot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    task_name: str
    required_vram_mb: float
    priority: TaskPriority
    allocated_at: float = Field(default_factory=time.time)
    status: str = "active"  # active | preempted | released


class AllocationResult(BaseModel):
    success: bool
    slot: Optional[AllocationSlot] = None
    preempted_slots: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    current_free_vram_mb: float = 0.0


class GPUAllocator:
    """
    Manages VRAM allocations across competing agents.
    Guarantees that high-priority interactive tasks (Voice/Chat) preempt
    low-priority background batch tasks (Swarm/Quant) when VRAM is constrained.
    """

    def __init__(
        self,
        monitor: Optional[GPUMonitor] = None,
        cleaner: Optional[GPUCleaner] = None,
        safety_headroom_mb: float = 512.0,
    ):
        self.monitor = monitor or GPUMonitor()
        self.cleaner = cleaner or GPUCleaner(self.monitor)
        self.safety_headroom_mb = safety_headroom_mb
        self._active_slots: Dict[str, AllocationSlot] = {}

    def get_allocated_vram_mb(self) -> float:
        """Returns total VRAM accounted for in active software slots."""
        return sum(
            slot.required_vram_mb
            for slot in self._active_slots.values()
            if slot.status == "active"
        )

    def request_slot(
        self,
        agent_id: str,
        task_name: str,
        required_vram_mb: float,
        priority: TaskPriority = TaskPriority.P2_HIGH,
    ) -> AllocationResult:
        """
        Attempts to allocate a VRAM slot. If VRAM is insufficient and the requesting
        task has higher priority than existing batch tasks, preempts P3 tasks.
        """
        status = self.monitor.get_status()
        
        # If no GPU is available, allow allocation only if 0 MB requested, or return CPU fallback
        if not status.is_available:
            logger.info("No physical GPU detected. Granting CPU-fallback virtual slot.")
            slot = AllocationSlot(
                agent_id=agent_id,
                task_name=task_name,
                required_vram_mb=0.0,
                priority=priority,
                status="active",
            )
            self._active_slots[slot.slot_id] = slot
            return AllocationResult(
                success=True,
                slot=slot,
                reason="Allocated on CPU fallback (no hardware GPU).",
                current_free_vram_mb=0.0,
            )

        # Available memory taking into account safety headroom and existing software reservations
        tracked_allocated = self.get_allocated_vram_mb()
        effective_free_mb = max(0.0, status.free_vram_mb - self.safety_headroom_mb)

        # Case 1: Memory is immediately available
        if effective_free_mb >= required_vram_mb:
            slot = AllocationSlot(
                agent_id=agent_id,
                task_name=task_name,
                required_vram_mb=required_vram_mb,
                priority=priority,
                status="active",
            )
            self._active_slots[slot.slot_id] = slot
            logger.info(f"Slot {slot.slot_id} granted for {task_name} ({required_vram_mb} MB).")
            return AllocationResult(
                success=True,
                slot=slot,
                current_free_vram_mb=effective_free_mb,
            )

        # Case 2: Memory is tight. Attempt purge first
        logger.warning(
            f"Insufficient VRAM for {task_name}: need {required_vram_mb} MB, free {effective_free_mb:.1f} MB. Triggering purge."
        )
        self.cleaner.purge_cache()
        status = self.monitor.get_status()
        effective_free_mb = max(0.0, status.free_vram_mb - self.safety_headroom_mb)

        if effective_free_mb >= required_vram_mb:
            slot = AllocationSlot(
                agent_id=agent_id,
                task_name=task_name,
                required_vram_mb=required_vram_mb,
                priority=priority,
                status="active",
            )
            self._active_slots[slot.slot_id] = slot
            return AllocationResult(
                success=True,
                slot=slot,
                reason="Allocated after cache purge.",
                current_free_vram_mb=effective_free_mb,
            )

        # Case 3: Still tight. Check if preemption can reclaim enough space
        preempted_ids = []
        if priority in (TaskPriority.P1_CRITICAL, TaskPriority.P2_HIGH):
            # Sort active P3 batch slots (newest first)
            candidate_p3_slots = [
                s for s in self._active_slots.values()
                if s.status == "active" and s.priority > priority
            ]
            candidate_p3_slots.sort(key=lambda s: s.allocated_at, reverse=True)

            reclaimable_mb = sum(s.required_vram_mb for s in candidate_p3_slots)
            if effective_free_mb + reclaimable_mb >= required_vram_mb:
                for cand in candidate_p3_slots:
                    cand.status = "preempted"
                    preempted_ids.append(cand.slot_id)
                    effective_free_mb += cand.required_vram_mb
                    logger.warning(f"Preempted low-priority task {cand.task_name} (Slot: {cand.slot_id})")
                    if effective_free_mb >= required_vram_mb:
                        break

                # Now grant slot
                slot = AllocationSlot(
                    agent_id=agent_id,
                    task_name=task_name,
                    required_vram_mb=required_vram_mb,
                    priority=priority,
                    status="active",
                )
                self._active_slots[slot.slot_id] = slot
                return AllocationResult(
                    success=True,
                    slot=slot,
                    preempted_slots=preempted_ids,
                    reason=f"Allocated via preemption of {len(preempted_ids)} background task(s).",
                    current_free_vram_mb=effective_free_mb,
                )

        # Case 4: Cannot allocate even after preemption
        return AllocationResult(
            success=False,
            reason=f"Insufficient VRAM: required {required_vram_mb} MB exceeds available {effective_free_mb:.1f} MB (Headroom: {self.safety_headroom_mb} MB).",
            current_free_vram_mb=effective_free_mb,
        )

    def release_slot(self, slot_id: str) -> bool:
        """Releases an active allocation slot and triggers optional purge."""
        if slot_id in self._active_slots:
            slot = self._active_slots[slot_id]
            slot.status = "released"
            del self._active_slots[slot_id]
            logger.info(f"Released slot {slot_id} ({slot.task_name}).")
            self.cleaner.purge_cache()
            return True
        return False

    def list_active_slots(self) -> List[AllocationSlot]:
        """Lists all currently active allocation slots."""
        return [s for s in self._active_slots.values() if s.status == "active"]
