"""
Comprehensive Unit & Integration Tests for GPULivre.
Verifies allocator, preemption, monitor, cleaner, and MCP tool dispatching.
"""

import unittest
from unittest.mock import MagicMock

from gpulivre.core.monitor import GPUMonitor, GPUStatus
from gpulivre.core.allocator import GPUAllocator, TaskPriority
from gpulivre.core.cleaner import GPUCleaner
from gpulivre.mcp.server import dispatch_tool


class TestGPULivre(unittest.TestCase):

    def setUp(self):
        # Create a mock monitor with 8GB total, 4GB used, 4GB free (4096 MB free)
        self.mock_monitor = MagicMock(spec=GPUMonitor)
        self.mock_status = GPUStatus(
            device_id=0,
            name="Mock NVIDIA RTX 4090",
            backend="cuda",
            total_vram_mb=8192.0,
            used_vram_mb=4096.0,
            free_vram_mb=4096.0,
            utilization_percent=50.0,
            temperature_celsius=55.0,
            is_available=True,
        )
        self.mock_monitor.get_status.return_value = self.mock_status

        self.mock_cleaner = MagicMock(spec=GPUCleaner)
        # Headroom 512 MB -> Effective free is 4096 - 512 = 3584 MB
        self.allocator = GPUAllocator(
            monitor=self.mock_monitor,
            cleaner=self.mock_cleaner,
            safety_headroom_mb=512.0,
        )

    def test_01_monitor_cpu_fallback(self):
        """Verifies that monitor handles environments without hardware GPUs safely."""
        real_monitor = GPUMonitor()
        status = real_monitor.get_status()
        self.assertIsNotNone(status)
        self.assertIn(status.backend, ["cuda", "mps", "cpu"])
        real_monitor.close()

    def test_02_cleaner_purge(self):
        """Verifies cache purge routine produces a valid PurgeReport."""
        cleaner = GPUCleaner(self.mock_monitor)
        report = cleaner.purge_cache()
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report.gc_objects_collected, 0)

    def test_03_allocator_happy_path(self):
        """Verifies slot allocation when memory is sufficient."""
        # Request 1024 MB (Effective free is 3584 MB)
        res = self.allocator.request_slot(
            agent_id="agent_whisper",
            task_name="voice_transcription",
            required_vram_mb=1024.0,
            priority=TaskPriority.P1_CRITICAL,
        )
        self.assertTrue(res.success)
        self.assertIsNotNone(res.slot)
        self.assertEqual(res.slot.status, "active")
        self.assertEqual(len(self.allocator.list_active_slots()), 1)

    def test_04_allocator_insufficient_memory_rejection(self):
        """Verifies rejection when requested VRAM exceeds total capacity."""
        # Request 5000 MB (Exceeds effective free 3584 MB)
        res = self.allocator.request_slot(
            agent_id="agent_huge_model",
            task_name="70b_quant",
            required_vram_mb=5000.0,
            priority=TaskPriority.P3_BATCH,
        )
        self.assertFalse(res.success)
        self.assertIn("Insufficient VRAM", res.reason)

    def test_05_preemption_of_batch_tasks(self):
        """
        Verifies that high-priority interactive tasks (P1 Voice) preempt
        running low-priority background batch tasks (P3 MiroFish Swarm).
        """
        # 1. Simulate state: only 1500 MB free (Headroom 512 -> 988 MB effective free)
        low_mem_status = GPUStatus(
            device_id=0,
            name="Mock NVIDIA RTX 4090",
            backend="cuda",
            total_vram_mb=8192.0,
            used_vram_mb=6692.0,
            free_vram_mb=1500.0,
            is_available=True,
        )
        self.mock_monitor.get_status.return_value = low_mem_status

        # 2. Allocate P3 Swarm job requiring 800 MB (fits within 988 MB)
        res_p3 = self.allocator.request_slot(
            agent_id="agent_mirofish",
            task_name="swarm_simulation_run",
            required_vram_mb=800.0,
            priority=TaskPriority.P3_BATCH,
        )
        self.assertTrue(res_p3.success)
        p3_slot_id = res_p3.slot.slot_id

        # 3. Simulate memory updated: now only 700 MB free (Effective: 700 - 512 = 188 MB)
        crowded_status = GPUStatus(
            device_id=0,
            name="Mock NVIDIA RTX 4090",
            backend="cuda",
            total_vram_mb=8192.0,
            used_vram_mb=7492.0,
            free_vram_mb=700.0,
            is_available=True,
        )
        self.mock_monitor.get_status.return_value = crowded_status

        # 4. User starts speaking: P1 Critical Voice task arrives requiring 600 MB
        # Effective free is only 188 MB, but P3 has 800 MB -> Preemption should trigger!
        res_p1 = self.allocator.request_slot(
            agent_id="agent_jarvis_voice",
            task_name="full_duplex_barge_in",
            required_vram_mb=600.0,
            priority=TaskPriority.P1_CRITICAL,
        )

        self.assertTrue(res_p1.success)
        self.assertIn(p3_slot_id, res_p1.preempted_slots)
        self.assertEqual(res_p3.slot.status, "preempted")
        self.assertEqual(res_p1.slot.status, "active")

    def test_06_release_slot(self):
        """Verifies releasing allocated slots."""
        res = self.allocator.request_slot(
            agent_id="agent_test",
            task_name="quick_embed",
            required_vram_mb=256.0,
        )
        slot_id = res.slot.slot_id
        released = self.allocator.release_slot(slot_id)
        self.assertTrue(released)
        self.assertEqual(len(self.allocator.list_active_slots()), 0)

    def test_07_mcp_dispatcher_tools(self):
        """Verifies MCP tool dispatching end-to-end."""
        # 1. Get status
        status_raw = dispatch_tool("gpulivre_get_status", {})
        self.assertIn("device_id", status_raw)

        # 2. Request slot
        req_raw = dispatch_tool(
            "gpulivre_request_slot",
            {"agent_id": "mcp_agent", "task_name": "mcp_test", "required_vram_mb": 512.0, "priority": 1},
        )
        self.assertIn('"success": true', req_raw)

        # 3. List slots
        list_raw = dispatch_tool("gpulivre_list_slots", {})
        self.assertIn("mcp_test", list_raw)

        # 4. Purge cache
        purge_raw = dispatch_tool("gpulivre_purge_cache", {})
        self.assertIn("gc_objects_collected", purge_raw)


if __name__ == "__main__":
    unittest.main()
