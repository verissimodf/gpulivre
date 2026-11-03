"""
GPULivre Core: Memory purge, cache eviction, and defragmentation routines.
"""

from __future__ import annotations
import gc
import logging
from typing import Dict, Any
from pydantic import BaseModel

logger = logging.getLogger("gpulivre.cleaner")


class PurgeReport(BaseModel):
    gc_objects_collected: int
    cuda_empty_cache_executed: bool
    cuda_ipc_collect_executed: bool
    reclaimed_vram_mb: float = 0.0
    details: Dict[str, Any] = {}


class GPUCleaner:
    """Safely cleans up orphaned VRAM allocations, stale CUDA tensors, and caches."""

    def __init__(self, monitor=None):
        self.monitor = monitor

    def purge_cache(self) -> PurgeReport:
        """Forces full garbage collection and purges CUDA tensor caches."""
        vram_before = 0.0
        if self.monitor:
            status_before = self.monitor.get_status()
            vram_before = status_before.free_vram_mb

        # 1. Python GC
        collected = gc.collect()

        # 2. PyTorch CUDA cache
        cuda_executed = False
        ipc_executed = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_executed = True
                try:
                    torch.cuda.ipc_collect()
                    ipc_executed = True
                except Exception:
                    pass
        except ImportError:
            pass

        # 3. Calculate reclaimed delta
        reclaimed = 0.0
        if self.monitor:
            status_after = self.monitor.get_status()
            reclaimed = max(0.0, status_after.free_vram_mb - vram_before)

        report = PurgeReport(
            gc_objects_collected=collected,
            cuda_empty_cache_executed=cuda_executed,
            cuda_ipc_collect_executed=ipc_executed,
            reclaimed_vram_mb=round(reclaimed, 2),
            details={
                "gc_unreachable_count": collected,
                "cuda_available": cuda_executed,
            },
        )
        logger.info(f"Purge complete: {collected} objects collected, reclaimed {reclaimed:.2f} MB VRAM.")
        return report
