"""
GPULivre Core: Hardware-agnostic GPU and VRAM monitoring engine.
"""

from __future__ import annotations
import os
import sys
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("gpulivre.monitor")


class GPUProcessInfo(BaseModel):
    pid: int
    process_name: str
    vram_used_mb: float


class GPUStatus(BaseModel):
    device_id: int
    name: str
    backend: str = Field(..., description="cuda | mps | rocm | cpu")
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float
    utilization_percent: float = 0.0
    temperature_celsius: Optional[float] = None
    is_available: bool = True
    processes: List[GPUProcessInfo] = Field(default_factory=list)


class GPUMonitor:
    """Monitors hardware GPU state across CUDA, MPS, and CPU fallbacks."""

    def __init__(self, preferred_device_id: int = 0):
        self.device_id = preferred_device_id
        self._nvml_initialized = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.info("NVML initialized successfully.")
        except Exception as e:
            self._nvml_initialized = False
            logger.debug(f"NVML unavailable: {e}. Falling back to PyTorch/CPU detection.")

    def get_status(self) -> GPUStatus:
        """Retrieves comprehensive GPU status using the best available backend."""
        # 1. Try NVML (Direct driver access)
        if self._nvml_initialized:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="ignore")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                try:
                    temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    temp = None

                # Process list
                procs = []
                try:
                    running_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in running_procs:
                        procs.append(
                            GPUProcessInfo(
                                pid=p.pid,
                                process_name=f"pid_{p.pid}",
                                vram_used_mb=round(p.usedGpuMemory / (1024 * 1024), 2),
                            )
                        )
                except Exception:
                    pass

                total_mb = round(mem_info.total / (1024 * 1024), 2)
                used_mb = round(mem_info.used / (1024 * 1024), 2)
                free_mb = round(mem_info.free / (1024 * 1024), 2)

                return GPUStatus(
                    device_id=self.device_id,
                    name=name,
                    backend="cuda",
                    total_vram_mb=total_mb,
                    used_vram_mb=used_mb,
                    free_vram_mb=free_mb,
                    utilization_percent=float(util_rates.gpu),
                    temperature_celsius=temp,
                    is_available=True,
                    processes=procs,
                )
            except Exception as e:
                logger.warning(f"NVML status query failed: {e}. Trying secondary backend.")

        # 2. Try PyTorch CUDA / MPS
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.device_id)
                total_mb = round(props.total_memory / (1024 * 1024), 2)
                reserved_mb = round(torch.cuda.memory_reserved(self.device_id) / (1024 * 1024), 2)
                allocated_mb = round(torch.cuda.memory_allocated(self.device_id) / (1024 * 1024), 2)
                free_mb = max(0.0, total_mb - reserved_mb)

                return GPUStatus(
                    device_id=self.device_id,
                    name=props.name,
                    backend="cuda",
                    total_vram_mb=total_mb,
                    used_vram_mb=allocated_mb,
                    free_vram_mb=free_mb,
                    utilization_percent=round((allocated_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0,
                    is_available=True,
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Metal MPS
                return GPUStatus(
                    device_id=0,
                    name="Apple Silicon Metal (Unified Memory)",
                    backend="mps",
                    total_vram_mb=16384.0,  # Estimated unified memory
                    used_vram_mb=0.0,
                    free_vram_mb=16384.0,
                    utilization_percent=0.0,
                    is_available=True,
                )
        except ImportError:
            pass

        # 3. CPU / RAM Fallback
        return GPUStatus(
            device_id=-1,
            name="CPU Fallback (No Hardware GPU Detected)",
            backend="cpu",
            total_vram_mb=0.0,
            used_vram_mb=0.0,
            free_vram_mb=0.0,
            utilization_percent=0.0,
            temperature_celsius=None,
            is_available=False,
        )

    def close(self) -> None:
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except Exception:
                pass
