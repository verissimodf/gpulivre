"""Hardware-agnostic theoretical inference ceilings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

MIB = 1024 * 1024


class TensorCoreState(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class HardwareCapabilities(BaseModel):
    name: str
    backend: str
    device_id: int = 0
    total_memory_mb: float = Field(ge=0)
    unified_memory: bool = False
    bandwidth_gbs: Optional[float] = Field(default=None, gt=0)
    peak_tflops_fp16: Optional[float] = Field(default=None, gt=0)
    peak_tflops_bf16: Optional[float] = Field(default=None, gt=0)
    peak_tflops_int8: Optional[float] = Field(default=None, gt=0)
    tensor_cores_unlocked: TensorCoreState = TensorCoreState.UNKNOWN
    measurement_sources: dict[str, str] = Field(default_factory=dict)
    measurement_timestamps: dict[str, datetime] = Field(default_factory=dict)


class DraftSpec(BaseModel):
    draft_layer_bytes: int = Field(ge=0)
    draft_head_bytes: int = Field(ge=0)
    drafts: int = Field(default=1, ge=1)
    accepted_tokens: float = Field(gt=0)


class ModelSpec(BaseModel):
    active_params_b: float = Field(gt=0)
    bits_per_weight: float = Field(gt=0)
    dense_weight_bytes: Optional[int] = Field(default=None, ge=0)
    total_weight_bytes: Optional[int] = Field(default=None, ge=0)
    kv_bytes_per_token: int = Field(gt=0)
    attn_layers: int = Field(ge=0)
    attn_heads: int = Field(ge=0)
    head_dim: int = Field(ge=0)
    lm_head_bytes: int = Field(default=0, ge=0)
    routed_expert_bytes: int = Field(default=0, ge=0)
    extra_flops_per_token: float = Field(default=0.0, ge=0)
    draft: Optional[DraftSpec] = None


class KVCapacity(BaseModel):
    context: int
    kv_bytes_per_stream: int
    available_kv_bytes: int
    max_tokens: int
    max_full_streams: Optional[int]


class CeilingReport(BaseModel):
    hardware: HardwareCapabilities
    model: ModelSpec
    decode_tok_s: dict[str, Optional[float]]
    prefill_tok_s: dict[str, Optional[float]]
    kv: dict[str, KVCapacity]
    warnings: list[str] = Field(default_factory=list)


def discover_hardware(
    monitor: Any,
    measurements: Optional[Mapping[str, Optional[float]]] = None,
) -> HardwareCapabilities:
    """Build capabilities from monitor identity and explicit measurements only."""
    status = monitor.get_status()
    supplied = measurements or {}
    backend = status.backend.lower()
    now = datetime.now(timezone.utc)
    measurement_names = (
        "bandwidth_gbs",
        "peak_tflops_fp16",
        "peak_tflops_bf16",
        "peak_tflops_int8",
    )
    values = {name: supplied.get(name) for name in measurement_names}
    sources = {
        "name": "monitor",
        "backend": "monitor",
        "device_id": "monitor",
        "total_memory_mb": "monitor",
    }
    timestamps = {
        name: now for name in (*sources, *measurement_names) if name in sources or values[name] is not None
    }
    sources.update({name: "operator" for name in measurement_names if values[name] is not None})

    if backend == "cuda":
        tensor_state = getattr(status, "tensor_cores_unlocked", TensorCoreState.UNKNOWN)
        tensor_state = TensorCoreState(tensor_state)
    else:
        tensor_state = TensorCoreState.NOT_APPLICABLE

    return HardwareCapabilities(
        name=status.name,
        backend=backend,
        device_id=status.device_id,
        total_memory_mb=status.total_vram_mb,
        unified_memory=backend == "mps" or "unified" in status.name.lower(),
        tensor_cores_unlocked=tensor_state,
        measurement_sources=sources,
        measurement_timestamps=timestamps,
        **values,
    )


def compute_theoretical_ceiling(
    hardware: HardwareCapabilities,
    model: ModelSpec,
    vram_cap_mb: float,
    contexts: Sequence[int],
    concurrency: int = 1,
    precision: str = "bf16",
    runtime_reserve_mb: float = 0.0,
) -> CeilingReport:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if vram_cap_mb < 0 or runtime_reserve_mb < 0:
        raise ValueError("memory values must be non-negative")
    if any(context < 0 for context in contexts):
        raise ValueError("contexts must be non-negative")

    precision = precision.lower()
    peaks = {
        "fp16": hardware.peak_tflops_fp16,
        "bf16": hardware.peak_tflops_bf16,
        "int8": hardware.peak_tflops_int8,
    }
    if precision not in peaks:
        raise ValueError(f"unsupported precision: {precision}")

    warnings: list[str] = []
    if hardware.bandwidth_gbs is None:
        warnings.append("decode ceiling unavailable: bandwidth_gbs was not measured")
    peak_tflops = peaks[precision]
    if peak_tflops is None:
        warnings.append(f"prefill ceiling unavailable: peak_tflops_{precision} was not measured")

    estimated_weight_bytes = int(model.active_params_b * 1e9 * model.bits_per_weight / 8)
    dense_weight_bytes = (
        model.dense_weight_bytes
        if model.dense_weight_bytes is not None
        else estimated_weight_bytes
    )
    total_weight_bytes = (
        model.total_weight_bytes
        if model.total_weight_bytes is not None
        else dense_weight_bytes + model.lm_head_bytes + model.routed_expert_bytes
    )
    if model.dense_weight_bytes is None or model.total_weight_bytes is None:
        warnings.append("weight bytes estimated from active parameters and bits per weight")

    accepted_tokens = model.draft.accepted_tokens if model.draft else 1.0
    draft_bytes = (
        model.draft.drafts * (model.draft.draft_layer_bytes + model.draft.draft_head_bytes)
        if model.draft
        else 0
    )
    weight_bytes_per_step = (
        dense_weight_bytes + model.lm_head_bytes + model.routed_expert_bytes
    )

    decode: dict[str, Optional[float]] = {}
    prefill: dict[str, Optional[float]] = {}
    kv: dict[str, KVCapacity] = {}
    available_kv_bytes = int(
        vram_cap_mb * MIB - runtime_reserve_mb * MIB - total_weight_bytes
    )
    if available_kv_bytes < 0:
        warnings.append("model weights and runtime reserve exceed the VRAM cap")
        available_kv_bytes = 0
    max_tokens = available_kv_bytes // model.kv_bytes_per_token

    for context in contexts:
        key = str(context)
        kv_bytes_per_stream = context * model.kv_bytes_per_token
        bytes_per_step = (
            weight_bytes_per_step + concurrency * (kv_bytes_per_stream + draft_bytes)
        )
        if hardware.bandwidth_gbs is None:
            decode[key] = None
        elif bytes_per_step == 0:
            decode[key] = None
            warnings.append(f"decode ceiling unavailable for context {context}: zero bytes per step")
        else:
            decode[key] = (
                hardware.bandwidth_gbs
                * 1e9
                * accepted_tokens
                * concurrency
                / bytes_per_step
            )

        flops_per_token = 2 * model.active_params_b * 1e9
        flops_per_token += 2 * model.attn_layers * model.attn_heads * model.head_dim * context
        flops_per_token += model.extra_flops_per_token
        if peak_tflops is None:
            prefill[key] = None
        elif flops_per_token == 0:
            prefill[key] = None
            warnings.append(f"prefill ceiling unavailable for context {context}: zero FLOPs per token")
        else:
            prefill[key] = 0.5 * peak_tflops * 1e12 / flops_per_token

        kv[key] = KVCapacity(
            context=context,
            kv_bytes_per_stream=kv_bytes_per_stream,
            available_kv_bytes=available_kv_bytes,
            max_tokens=max_tokens,
            max_full_streams=max_tokens // context if context else None,
        )

    return CeilingReport(
        hardware=hardware,
        model=model,
        decode_tok_s=decode,
        prefill_tok_s=prefill,
        kv=kv,
        warnings=warnings,
    )
