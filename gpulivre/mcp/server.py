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


# Dispatcher map for programmatic or MCP stdio execution
TOOL_DISPATCHER = {
    "gpulivre_get_status": handle_get_status,
    "gpulivre_purge_cache": handle_purge_cache,
    "gpulivre_request_slot": handle_request_slot,
    "gpulivre_release_slot": handle_release_slot,
    "gpulivre_list_slots": handle_list_slots,
    "gpulivre_get_user_config": handle_get_user_config,
    "gpulivre_set_user_toggle": handle_set_user_toggle,
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
