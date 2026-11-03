"""
GPULivre: Autonomous GPU & VRAM Resource Orchestrator for Multi-Agent AI Ecosystems.
"""

from gpulivre.core.monitor import GPUMonitor, GPUStatus
from gpulivre.core.allocator import GPUAllocator, AllocationSlot, TaskPriority
from gpulivre.core.cleaner import GPUCleaner
from gpulivre.core.hybrid_fallback import (
    HybridFallbackEngine,
    FallbackExecutionResult,
    hybrid_fallback_engine,
)
from gpulivre.adapters.user_toggle import (
    UserConfigManager,
    GPULivreUserConfig,
    user_config_manager,
)
from gpulivre.adapters.voice_adapter import VoicePipelineAdapter
from gpulivre.adapters.swarm_adapter import SwarmSimulationAdapter

__version__ = "0.2.0"
__all__ = [
    "GPUMonitor",
    "GPUStatus",
    "GPUAllocator",
    "AllocationSlot",
    "TaskPriority",
    "GPUCleaner",
    "HybridFallbackEngine",
    "FallbackExecutionResult",
    "hybrid_fallback_engine",
    "UserConfigManager",
    "GPULivreUserConfig",
    "user_config_manager",
    "VoicePipelineAdapter",
    "SwarmSimulationAdapter",
]
