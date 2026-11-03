"""
Comprehensive End-to-End Validation Tests for GPULivre Hybrid System.
Validates:
1. User Enable/Disable Toggle Mechanism (AssistentePessoal Control).
2. Priority Concurrency: P1 Voice preempts P3 MiroFish Swarms.
3. Automatic Cloud Offload (Google Colab & Gemini AI Studio) on VRAM Saturation.
4. MCP Toggle Tools.
"""

import unittest
from unittest.mock import MagicMock

from gpulivre.core.monitor import GPUMonitor, GPUStatus
from gpulivre.core.allocator import GPUAllocator, TaskPriority
from gpulivre.core.cleaner import GPUCleaner
from gpulivre.core.hybrid_fallback import HybridFallbackEngine
from gpulivre.adapters.user_toggle import UserConfigManager, GPULivreUserConfig
from gpulivre.adapters.voice_adapter import VoicePipelineAdapter
from gpulivre.adapters.swarm_adapter import SwarmSimulationAdapter
from gpulivre.mcp.server import dispatch_tool


class TestHybridE2EValidation(unittest.TestCase):

    def setUp(self):
        # 1. Custom mock user config manager (in-memory)
        self.mock_config = GPULivreUserConfig(
            gpulivre_enabled=True,
            hybrid_fallback_enabled=True,
            preferred_fallback="auto",
            safety_headroom_mb=512.0,
        )
        self.config_manager = MagicMock(spec=UserConfigManager)
        self.config_manager.config = self.mock_config
        self.config_manager.is_gpulivre_enabled.side_effect = lambda: self.mock_config.gpulivre_enabled
        self.config_manager.is_fallback_enabled.side_effect = lambda: self.mock_config.hybrid_fallback_enabled

        # 2. Mock GPU Monitor (Total: 4096 MB, Used: 2048 MB, Free: 2048 MB)
        self.mock_monitor = MagicMock(spec=GPUMonitor)
        self.mock_status = GPUStatus(
            device_id=0,
            name="Mock NVIDIA RTX 4070 (8GB)",
            backend="cuda",
            total_vram_mb=4096.0,
            used_vram_mb=2048.0,
            free_vram_mb=2048.0,
            utilization_percent=50.0,
            is_available=True,
        )
        self.mock_monitor.get_status.return_value = self.mock_status

        # 3. Cleaner and Allocator
        self.mock_cleaner = MagicMock(spec=GPUCleaner)
        self.allocator = GPUAllocator(
            monitor=self.mock_monitor,
            cleaner=self.mock_cleaner,
            safety_headroom_mb=512.0,
        )

        # 4. Fallback Engine and Adapters
        self.fallback_engine = HybridFallbackEngine(config_manager=self.config_manager)
        self.voice_adapter = VoicePipelineAdapter(
            allocator=self.allocator, config_manager=self.config_manager
        )
        self.swarm_adapter = SwarmSimulationAdapter(
            allocator=self.allocator,
            fallback_engine=self.fallback_engine,
            config_manager=self.config_manager,
        )

    def test_01_user_toggle_disabled_bypass_mode(self):
        """
        Validates that when the user disables GPULivre in AssistentePessoal,
        both voice and swarm tasks execute in direct legacy mode without slot reservations.
        """
        # User toggles GPULivre off
        self.mock_config.gpulivre_enabled = False

        # Execute voice task
        mock_tts = MagicMock(return_value="audio_buffer_wav")
        res_voice = self.voice_adapter.execute_voice_task("tts_synthesis", mock_tts, "Olá, Jarvis!")

        self.assertTrue(res_voice["success"])
        self.assertEqual(res_voice["mode"], "legacy_direct")
        self.assertEqual(res_voice["output"], "audio_buffer_wav")
        self.assertEqual(len(self.allocator.list_active_slots()), 0)

        # Execute swarm task
        mock_swarm = MagicMock(return_value={"consensus": "BUY", "confidence": 0.88})
        res_swarm = self.swarm_adapter.execute_simulation(
            "mirofish_consensus", mock_swarm, payload={"steps": 5}
        )

        self.assertTrue(res_swarm["success"])
        self.assertEqual(res_swarm["mode"], "legacy_direct")
        self.assertEqual(res_swarm["execution_target"], "local_unmanaged")
        self.assertEqual(len(self.allocator.list_active_slots()), 0)

    def test_02_voice_happy_path_with_toggle_enabled(self):
        """Validates normal voice synthesis when toggle is enabled and VRAM is sufficient."""
        self.mock_config.gpulivre_enabled = True

        mock_tts = MagicMock(return_value="audio_buffer_wav")
        res_voice = self.voice_adapter.execute_voice_task(
            "tts_piper", mock_tts, "Mensagem falada", required_vram_mb=600.0
        )

        self.assertTrue(res_voice["success"])
        self.assertEqual(res_voice["mode"], "gpulivre_p1_orchestrated")
        # Slot should be automatically released in finally block
        self.assertEqual(len(self.allocator.list_active_slots()), 0)
        self.mock_cleaner.purge_cache.assert_called()

    def test_03_concomitant_concurrency_p1_voice_preempts_p3_swarm(self):
        """
        Validates real-time concurrency:
        1. Swarm (P3) requests 1200 MB VRAM.
        2. Status updates to crowded (effective free VRAM drops to only 300 MB).
        3. User starts speaking: Voice (P1) requests 800 MB.
        4. GPULivre immediately preempts Swarm (P3) so Voice executes without latency.
        """
        self.mock_config.gpulivre_enabled = True

        # 1. Swarm starts and holds a slot
        res_p3 = self.allocator.request_slot(
            agent_id="laus_swarm_agent",
            task_name="mirofish_swarm_sim",
            required_vram_mb=1200.0,
            priority=TaskPriority.P3_BATCH,
        )
        self.assertTrue(res_p3.success)
        swarm_slot_id = res_p3.slot.slot_id

        # 2. Simulate tight VRAM on device (only 700 MB total free -> 188 MB effective with 512 MB headroom)
        tight_status = GPUStatus(
            device_id=0,
            name="Mock NVIDIA RTX 4070 (8GB)",
            backend="cuda",
            total_vram_mb=4096.0,
            used_vram_mb=3396.0,
            free_vram_mb=700.0,
            is_available=True,
        )
        self.mock_monitor.get_status.return_value = tight_status

        # 3. Voice arrives with P1_CRITICAL requesting 800 MB
        mock_whisper = MagicMock(return_value="transcription_text")
        res_voice = self.voice_adapter.execute_voice_task(
            "whisper_vad_barge_in",
            mock_whisper,
            b"audio_bytes",
            required_vram_mb=800.0,
        )

        # 4. Assert Voice succeeded and Swarm was preempted
        self.assertTrue(res_voice["success"])
        self.assertIn(swarm_slot_id, res_voice["preempted_other_tasks"])
        self.assertEqual(res_p3.slot.status, "preempted")

    def test_04_vram_saturation_triggers_hybrid_fallback_to_colab(self):
        """
        Validates that when local VRAM is 100% saturated (0 MB effective free),
        a heavy MiroFish Swarm simulation is automatically offloaded to Google Colab.
        """
        self.mock_config.gpulivre_enabled = True
        self.mock_config.hybrid_fallback_enabled = True

        # Simulate 100% full VRAM
        full_status = GPUStatus(
            device_id=0,
            name="Mock NVIDIA RTX 4070",
            backend="cuda",
            total_vram_mb=4096.0,
            used_vram_mb=4096.0,
            free_vram_mb=0.0,
            is_available=True,
        )
        self.mock_monitor.get_status.return_value = full_status

        mock_local_swarm = MagicMock()
        res = self.swarm_adapter.execute_simulation(
            task_name="mirofish_monte_carlo",
            simulation_callable=mock_local_swarm,
            payload={"steps": 50, "agents_count": 20},
            required_vram_mb=2048.0,
        )

        # Local callable should not be called because it was offloaded to Colab
        mock_local_swarm.assert_not_called()
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "hybrid_fallback_offload")
        self.assertEqual(res["execution_target"], "colab")
        self.assertIn("Colab Tesla T4", res["data"]["target_accelerator"])
        self.assertEqual(res["data"]["personas_simulated"], 20)

    def test_05_vram_saturation_triggers_hybrid_fallback_to_gemini(self):
        """
        Validates that when local VRAM is saturated, LLM inference tasks
        are automatically offloaded to Gemini Cloud API.
        """
        self.mock_config.gpulivre_enabled = True
        self.mock_config.hybrid_fallback_enabled = True

        res = self.fallback_engine.route_task(
            task_name="market_macro_sentiment",
            task_type="llm_inference",
            payload={"prompt": "Analise o spread BTC/BRL com risco macrocambial", "token_budget": 1024},
        )

        self.assertTrue(res.success)
        self.assertEqual(res.dispatched_to, "gemini")
        self.assertIn("gemini-2.5-flash", res.result_data["model"])
        self.assertIn("Gemini Cloud Response", res.result_data["text"])

    def test_06_fallback_disabled_by_user_rejects_cleanly(self):
        """
        Validates that if the user explicitly disables hybrid fallback,
        overloaded tasks are rejected with a clear explanation instead of making remote calls.
        """
        self.mock_config.gpulivre_enabled = True
        self.mock_config.hybrid_fallback_enabled = False  # Disabled by user

        res = self.fallback_engine.route_task(
            task_name="heavy_swarm",
            task_type="swarm",
            payload={"steps": 10},
        )

        self.assertFalse(res.success)
        self.assertEqual(res.dispatched_to, "rejected")
        self.assertIn("disabled by user", res.reason)

    def test_07_mcp_user_toggle_dispatch(self):
        """Validates MCP tools for getting and setting user toggle preferences."""
        # 1. Get config
        cfg_str = dispatch_tool("gpulivre_get_user_config", {})
        self.assertIn("gpulivre_enabled", cfg_str)

        # 2. Toggle off
        updated_str = dispatch_tool("gpulivre_set_user_toggle", {"gpulivre_enabled": False})
        self.assertIn('"gpulivre_enabled": false', updated_str)

        # 3. Toggle on
        updated_str2 = dispatch_tool(
            "gpulivre_set_user_toggle",
            {"gpulivre_enabled": True, "hybrid_fallback_enabled": True, "preferred_fallback": "colab"},
        )
        self.assertIn('"gpulivre_enabled": true', updated_str2)
        self.assertIn('"preferred_fallback": "colab"', updated_str2)


if __name__ == "__main__":
    unittest.main()
