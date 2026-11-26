"""
GPULivre MCP Server: Exposes GPU resource management and user toggle tools to LLM agents.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict

from gpulivre.core.monitor import GPUMonitor
from gpulivre.core.allocator import GPUAllocator, TaskPriority
from gpulivre.core.cleaner import GPUCleaner
from gpulivre.core.hybrid_fallback import hybrid_fallback_engine
from gpulivre.core.ceiling import (
    HardwareCapabilities,
    ModelSpec,
    compute_theoretical_ceiling,
    discover_hardware,
)
from gpulivre.core.bench_harness import BenchmarkConfig, BenchmarkRunner
from gpulivre.core.hill_climber import (
    ExperimentLedger,
    HillClimber,
    Lever,
    run_interleaved_bench,
    target_metric_for_priority,
)
from gpulivre.adapters.user_toggle import user_config_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpulivre.mcp")

# Initialize global singletons
monitor = GPUMonitor()
cleaner = GPUCleaner(monitor)
allocator = GPUAllocator(monitor, cleaner)


def handle_get_status(_args: Dict[str, Any]) -> str:
    status = monitor.get_status()
    return status.model_dump_json(indent=2)


def handle_purge_cache(_args: Dict[str, Any]) -> str:
    report = cleaner.purge_cache()
    return report.model_dump_json(indent=2)


def handle_request_slot(args: Dict[str, Any]) -> str:
    agent_id = str(args.get("agent_id", "unknown_agent"))
    task_name = str(args.get("task_name", "unnamed_task"))
    required_vram_mb = float(args.get("required_vram_mb", 1024.0))
    priority_val = int(args.get("priority", 2))
    
    try:
        priority = TaskPriority(priority_val)
    except ValueError:
        priority = TaskPriority.P2_HIGH

    res = allocator.request_slot(
        agent_id=agent_id,
        task_name=task_name,
        required_vram_mb=required_vram_mb,
        priority=priority,
    )
    return res.model_dump_json(indent=2)


def handle_release_slot(args: Dict[str, Any]) -> str:
    slot_id = str(args.get("slot_id", ""))
    success = allocator.release_slot(slot_id)
    return json.dumps({"success": success, "slot_id": slot_id})


def handle_list_slots(_args: Dict[str, Any]) -> str:
    slots = allocator.list_active_slots()
    return json.dumps([s.model_dump() for s in slots], indent=2)


def handle_get_user_config(_args: Dict[str, Any]) -> str:
    return user_config_manager.config.model_dump_json(indent=2)


def handle_set_user_toggle(args: Dict[str, Any]) -> str:
    if "gpulivre_enabled" in args:
        user_config_manager.set_gpulivre_enabled(bool(args["gpulivre_enabled"]))
    if "hybrid_fallback_enabled" in args:
        user_config_manager.set_fallback_enabled(bool(args["hybrid_fallback_enabled"]))
    if "preferred_fallback" in args:
        user_config_manager.config.preferred_fallback = str(args["preferred_fallback"])
        user_config_manager.save()
    return user_config_manager.config.model_dump_json(indent=2)


def _priority(value: Any) -> TaskPriority:
    try:
        return TaskPriority(int(value))
    except (TypeError, ValueError):
        return TaskPriority.P2_HIGH


def _ledger_from_args(args: Dict[str, Any]) -> ExperimentLedger:
    kwargs = {}
    if args.get("ledger_path"):
        kwargs["path"] = args["ledger_path"]
    if args.get("plan_path"):
        kwargs["plan_path"] = args["plan_path"]
    return ExperimentLedger(**kwargs)


def handle_compute_ceiling(args: Dict[str, Any]) -> str:
    hardware_data = args.get("hardware")
    hardware = (
        HardwareCapabilities(**hardware_data)
        if hardware_data
        else discover_hardware(monitor, measurements=args.get("measurements"))
    )
    model = ModelSpec(**args["model"])
    report = compute_theoretical_ceiling(
        hardware=hardware,
        model=model,
        vram_cap_mb=float(args["vram_cap_mb"]),
        contexts=[int(value) for value in args.get("contexts", [4096])],
        concurrency=int(args.get("concurrency", 1)),
        precision=str(args.get("precision", "bf16")),
    )
    return report.model_dump_json(indent=2)


def handle_run_interleaved_bench(args: Dict[str, Any]) -> str:
    control = BenchmarkConfig(**args["control"])
    candidate = BenchmarkConfig(**args["candidate"])
    if not user_config_manager.is_gpulivre_enabled() and (control.launch or candidate.launch):
        return json.dumps({"error": "Managed lifecycle is disabled while GPULivre bypass is active."})
    priority = _priority(args.get("priority", 2))
    experiment = run_interleaved_bench(
        runner=BenchmarkRunner(),
        control=control,
        candidate=candidate,
        target_metric=str(args.get("target_metric") or target_metric_for_priority(priority, control.concurrency)),
        samples_per_leg=int(args.get("samples_per_leg", 2)),
        ledger=_ledger_from_args(args),
        hardware=str(args.get("hardware", "unknown")),
        model=str(args.get("model", candidate.model)),
        lever=str(args.get("lever", "candidate")),
        require_quality_gates=bool(args.get("require_quality_gates", True)),
        ceiling_percent=args.get("ceiling_percent"),
    )
    return experiment.model_dump_json(indent=2)


def handle_hill_climb(args: Dict[str, Any]) -> str:
    baseline = BenchmarkConfig(**args["baseline"])
    levers = [Lever(**item) for item in args.get("levers", [])]
    if not user_config_manager.is_gpulivre_enabled() and (
        baseline.launch or any(lever.config.launch for lever in levers)
    ):
        return json.dumps({"error": "Managed lifecycle is disabled while GPULivre bypass is active."})
    climber = HillClimber(runner=BenchmarkRunner(), ledger=_ledger_from_args(args))
    session = climber.hill_climb(
        baseline=baseline,
        levers=levers,
        priority=_priority(args.get("priority", 2)),
        ceiling_gap=float(args.get("ceiling_gap", 1.0)),
        max_experiments=args.get("max_experiments"),
        hardware=str(args.get("hardware", "unknown")),
        require_quality_gates=bool(args.get("require_quality_gates", True)),
    )
    return session.model_dump_json(indent=2)


def handle_get_ledger_history(args: Dict[str, Any]) -> str:
    history = _ledger_from_args(args).history(
        hardware=args.get("hardware"),
        model=args.get("model"),
        lever=args.get("lever"),
        accepted=args.get("accepted"),
        limit=int(args.get("limit", 100)),
    )
    return json.dumps(history, indent=2, ensure_ascii=False)


# Dispatcher map for programmatic or MCP stdio execution
TOOL_DISPATCHER = {
    "gpulivre_get_status": handle_get_status,
    "gpulivre_purge_cache": handle_purge_cache,
    "gpulivre_request_slot": handle_request_slot,
    "gpulivre_release_slot": handle_release_slot,
    "gpulivre_list_slots": handle_list_slots,
    "gpulivre_get_user_config": handle_get_user_config,
    "gpulivre_set_user_toggle": handle_set_user_toggle,
    "gpulivre_compute_ceiling": handle_compute_ceiling,
    "gpulivre_run_interleaved_bench": handle_run_interleaved_bench,
    "gpulivre_hill_climb": handle_hill_climb,
    "gpulivre_get_ledger_history": handle_get_ledger_history,
}


def dispatch_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Dispatches tool execution by name."""
    handler = TOOL_DISPATCHER.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return handler(arguments)
    except Exception as e:
        logger.exception(f"Error handling tool {name}")
        return json.dumps({"error": str(e)})


def main():
    print("GPULivre MCP Service initialized. Ready for tool execution.")


if __name__ == "__main__":
    main()
