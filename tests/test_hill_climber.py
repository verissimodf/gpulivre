"""Tests for hardware ceilings, OpenAI benchmarking and empirical hill-climbing."""

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from gpulivre.core.allocator import TaskPriority
from gpulivre.core.bench_harness import (
    BenchmarkConfig,
    BenchmarkResult,
    MemoryCapExceeded,
    MemoryWatchdog,
    MissingUsageError,
    OpenAICompatibleClient,
    build_needle_document,
    exact_needle_match,
    looping,
)
from gpulivre.core.ceiling import (
    DraftSpec,
    HardwareCapabilities,
    ModelSpec,
    TensorCoreState,
    compute_theoretical_ceiling,
    discover_hardware,
)
from gpulivre.core.hill_climber import (
    ExperimentLedger,
    HillClimber,
    Lever,
    run_interleaved_bench,
    target_metric_for_priority,
)
from gpulivre.core.monitor import GPUStatus
from gpulivre.mcp.server import TOOL_DISPATCHER, dispatch_tool


MIB = 1024 * 1024


def hardware(**overrides):
    values = {
        "name": "Mock Accelerator",
        "backend": "cuda",
        "total_memory_mb": 24_576.0,
        "unified_memory": False,
        "bandwidth_gbs": 100.0,
        "peak_tflops_fp16": 90.0,
        "peak_tflops_bf16": 100.0,
        "peak_tflops_int8": 200.0,
        "tensor_cores_unlocked": TensorCoreState.ENABLED,
    }
    values.update(overrides)
    return HardwareCapabilities(**values)


def model(**overrides):
    values = {
        "active_params_b": 1.0,
        "bits_per_weight": 4.0,
        "dense_weight_bytes": 500_000_000,
        "total_weight_bytes": 500_000_000,
        "kv_bytes_per_token": 1_000,
        "attn_layers": 2,
        "attn_heads": 4,
        "head_dim": 64,
    }
    values.update(overrides)
    return ModelSpec(**values)


def config(name="ctl"):
    return BenchmarkConfig(name=name, model="mock-model", base_url="http://mock")


def result(name, decode=50.0, ttft=100.0, needle=True, soak=True, flags=None):
    return BenchmarkResult(
        name=name,
        ok=not flags,
        metrics={"c1_decode_tok_s": decode, "ttft_short_ms": ttft},
        flags=flags or [],
        needle_passed=needle,
        soak_passed=soak,
    )


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


class TestCeilings(unittest.TestCase):
    def test_dedicated_gpu_decode_prefill_and_kv(self):
        report = compute_theoretical_ceiling(hardware(), model(), vram_cap_mb=1024, contexts=[0])

        self.assertAlmostEqual(report.decode_tok_s["0"], 200.0)
        self.assertAlmostEqual(report.prefill_tok_s["0"], 25_000.0)
        expected_tokens = int((1024 * MIB - 500_000_000) / 1000)
        self.assertEqual(report.kv["0"].max_tokens, expected_tokens)
        self.assertIsNone(report.kv["0"].max_full_streams)

    def test_unified_memory_int8_and_unknown_tensor_patch(self):
        unified = hardware(
            name="Unified Arc",
            backend="xpu",
            unified_memory=True,
            tensor_cores_unlocked=TensorCoreState.NOT_APPLICABLE,
        )
        report = compute_theoretical_ceiling(
            unified, model(), vram_cap_mb=1024, contexts=[0], precision="int8"
        )

        self.assertAlmostEqual(report.prefill_tok_s["0"], 50_000.0)
        self.assertEqual(report.hardware.backend, "xpu")
        self.assertTrue(report.hardware.unified_memory)
        self.assertEqual(report.hardware.tensor_cores_unlocked, TensorCoreState.NOT_APPLICABLE)

    def test_speculative_decode_counts_draft_and_acceptance(self):
        speculative = model(
            draft=DraftSpec(
                draft_layer_bytes=50_000_000,
                draft_head_bytes=50_000_000,
                drafts=1,
                accepted_tokens=2.0,
            )
        )
        report = compute_theoretical_ceiling(hardware(), speculative, 1024, contexts=[0])

        self.assertAlmostEqual(report.decode_tok_s["0"], 100e9 * 2 / 600_000_000)

    def test_bpw_is_explicitly_labelled_as_weight_estimate(self):
        report = compute_theoretical_ceiling(
            hardware(),
            model(dense_weight_bytes=None, total_weight_bytes=None),
            1024,
            contexts=[0],
        )

        self.assertAlmostEqual(report.decode_tok_s["0"], 200.0)
        self.assertTrue(any("estimated" in warning for warning in report.warnings))

    def test_missing_measurement_returns_warning_not_fabricated_number(self):
        report = compute_theoretical_ceiling(
            hardware(bandwidth_gbs=None, peak_tflops_bf16=None, peak_tflops_fp16=None),
            model(),
            1024,
            contexts=[0],
        )

        self.assertIsNone(report.decode_tok_s["0"])
        self.assertIsNone(report.prefill_tok_s["0"])
        self.assertGreaterEqual(len(report.warnings), 2)

    def test_hardware_discovery_uses_monitor_and_measurements(self):
        monitor = MagicMock()
        monitor.get_status.return_value = GPUStatus(
            device_id=0,
            name="Intel Arc Unified Memory",
            backend="xpu",
            total_vram_mb=65_536.0,
            used_vram_mb=0.0,
            free_vram_mb=65_536.0,
        )

        capabilities = discover_hardware(
            monitor, measurements={"bandwidth_gbs": 115.0, "peak_tflops_bf16": 36.0}
        )

        self.assertEqual(capabilities.backend, "xpu")
        self.assertTrue(capabilities.unified_memory)
        self.assertEqual(capabilities.bandwidth_gbs, 115.0)
        self.assertEqual(capabilities.tensor_cores_unlocked, TensorCoreState.NOT_APPLICABLE)


class TestOpenAIHarness(unittest.TestCase):
    def test_chat_uses_usage_and_never_stream_chunk_count(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3}}\n',
            b"data: [DONE]\n",
        ]
        clock = iter([0.0, 0.2, 0.7])
        client = OpenAICompatibleClient(
            config(), opener=lambda *_args, **_kwargs: FakeResponse(lines), clock=lambda: next(clock)
        )

        measured = client.chat("Say hello", max_tokens=3)

        self.assertEqual(measured.completion_tokens, 3)
        self.assertEqual(measured.prompt_tokens, 2)
        self.assertAlmostEqual(measured.ttft_s, 0.2)
        self.assertAlmostEqual(measured.decode_tok_s, 4.0)
        self.assertEqual(measured.text, "Hello world")

    def test_missing_usage_is_invalid_without_real_tokenizer(self):
        lines = [b'data: {"choices":[{"delta":{"content":"one two three"}}]}\n', b"data: [DONE]\n"]
        client = OpenAICompatibleClient(
            config(), opener=lambda *_args, **_kwargs: FakeResponse(lines)
        )

        with self.assertRaises(MissingUsageError):
            client.chat("one two", max_tokens=3)

    def test_real_tokenizer_is_allowed_when_usage_is_absent(self):
        lines = [b'data: {"choices":[{"delta":{"content":"one two three"}}]}\n', b"data: [DONE]\n"]
        client = OpenAICompatibleClient(
            config(),
            opener=lambda *_args, **_kwargs: FakeResponse(lines),
            token_counter=lambda text: len(text.split()),
        )

        measured = client.chat("one two", max_tokens=3)

        self.assertEqual(measured.prompt_tokens, 2)
        self.assertEqual(measured.completion_tokens, 3)

    def test_memory_watchdog_trips_immediately(self):
        callback = MagicMock()
        watchdog = MemoryWatchdog(cap_mb=1, sampler=lambda: 2 * MIB, on_trip=callback)

        with self.assertRaises(MemoryCapExceeded):
            watchdog.sample_once()

        self.assertTrue(watchdog.tripped)
        callback.assert_called_once_with()
        watchdog.close()

    def test_needle_exact_match_and_loop_detection(self):
        _prompt, codes = build_needle_document(128, seed=7)

        self.assertTrue(exact_needle_match(" ".join(codes), codes))
        self.assertFalse(exact_needle_match(" ".join(codes[:-1]), codes))
        self.assertTrue(looping("one two three four five six seven eight " * 60))


class TestHillClimber(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.tempdir.name) / "ledger.jsonl"
        self.plan_path = Path(self.tempdir.name) / "PLAN.md"
        self.ledger = ExperimentLedger(self.ledger_path, self.plan_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def run_experiment(self, samples, **kwargs):
        runner = MagicMock()
        runner.run.side_effect = samples
        kwargs.setdefault("samples_per_leg", 2)
        experiment = run_interleaved_bench(
            runner=runner,
            control=config("ctl"),
            candidate=config("cand"),
            target_metric="c1_decode_tok_s",
            ledger=self.ledger,
            hardware="mock-gpu",
            model="mock-model",
            lever="flash_attention",
            **kwargs,
        )
        return experiment, runner

    def test_interleaving_and_successful_persistence(self):
        samples = [
            result("ctl", 50), result("ctl", 52),
            result("cand", 60), result("cand", 62),
            result("ctl", 51), result("ctl", 53),
        ]

        experiment, runner = self.run_experiment(samples)

        self.assertTrue(experiment.accepted)
        self.assertEqual(len(experiment.samples), 6)
        self.assertEqual([call.args[0].name for call in runner.run.call_args_list],
                         ["ctl", "ctl", "cand", "cand", "ctl", "ctl"])
        rows = [json.loads(line) for line in self.ledger_path.read_text().splitlines()]
        self.assertTrue(rows[0]["accepted"])
        self.assertIn("Before", self.plan_path.read_text())
        self.assertIn("After", self.plan_path.read_text())

    def test_decode_win_with_ttft_regression_over_three_percent_is_rejected(self):
        samples = [
            result("ctl", 50, 100), result("ctl", 52, 100),
            result("cand", 60, 104), result("cand", 62, 104),
            result("ctl", 51, 100), result("ctl", 53, 100),
        ]

        experiment, _runner = self.run_experiment(samples)

        self.assertFalse(experiment.accepted)
        self.assertIn("regression", experiment.reason)
        self.assertFalse(self.ledger.history()[0]["accepted"])

    def test_needle_failure_is_rejected(self):
        samples = [
            result("ctl", 50), result("ctl", 52),
            result("cand", 60, needle=False), result("cand", 62, needle=False),
            result("ctl", 51), result("ctl", 53),
        ]

        experiment, _runner = self.run_experiment(samples)

        self.assertFalse(experiment.accepted)
        self.assertIn("needle", experiment.reason)

    def test_invalid_sample_and_soak_failure_are_rejected(self):
        invalid = result("cand", 100, flags=["MEM_CAP"])
        soak_failure = result("cand", 100, soak=False)
        for bad_sample, expected in ((invalid, "MEM_CAP"), (soak_failure, "soak")):
            samples = [result("ctl"), result("ctl"), bad_sample, bad_sample, result("ctl"), result("ctl")]
            experiment, _runner = self.run_experiment(samples)
            self.assertFalse(experiment.accepted)
            self.assertIn(expected, experiment.reason)

    def test_minimum_samples_is_enforced(self):
        with self.assertRaises(ValueError):
            self.run_experiment([], samples_per_leg=1)

    def test_history_filters_and_priority_metrics(self):
        self.run_experiment([
            result("ctl"), result("ctl"), result("cand", 60), result("cand", 60),
            result("ctl"), result("ctl"),
        ])

        self.assertEqual(len(self.ledger.history(hardware="mock-gpu", lever="flash_attention")), 1)
        self.assertEqual(len(self.ledger.history(hardware="other")), 0)
        self.assertEqual(target_metric_for_priority(TaskPriority.P1_CRITICAL), "ttft_short_ms")
        self.assertEqual(target_metric_for_priority(TaskPriority.P2_HIGH), "c1_decode_tok_s")
        self.assertEqual(target_metric_for_priority(TaskPriority.P3_BATCH, concurrency=4),
                         "c4_aggregate_tok_s")

    def test_hill_climb_changes_one_lever_and_updates_best(self):
        runner = MagicMock()
        runner.run.side_effect = [
            result("ctl"), result("ctl"), result("cand-a", 60), result("cand-a", 60),
            result("ctl"), result("ctl"),
        ]
        climber = HillClimber(runner=runner, ledger=self.ledger)
        levers = [
            Lever(name="a", config=config("cand-a"), likelihood=0.9),
            Lever(name="b", config=config("cand-b"), likelihood=0.1),
        ]

        session = climber.hill_climb(config("ctl"), levers, TaskPriority.P2_HIGH, max_experiments=1)

        self.assertEqual(session.best_config.name, "cand-a")
        self.assertEqual(len(session.experiments), 1)
        self.assertEqual(session.experiments[0].lever, "a")


class TestMonitorBackends(unittest.TestCase):
    def test_rocm_is_not_misreported_as_cuda(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.version.hip = "6.2"
        props = MagicMock()
        props.name = "Mock Radeon"
        props.total_memory = 8 * 1024**3
        fake_torch.cuda.get_device_properties.return_value = props
        fake_torch.cuda.memory_reserved.return_value = 1024**3
        fake_torch.cuda.memory_allocated.return_value = 512 * 1024**2

        with patch.dict(sys.modules, {"torch": fake_torch}):
            from gpulivre.core.monitor import GPUMonitor
            status = GPUMonitor().get_status()

        self.assertEqual(status.backend, "rocm")

    def test_xpu_is_discovered_without_cuda(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.version.hip = None
        fake_torch.xpu.is_available.return_value = True
        props = MagicMock()
        props.name = "Mock Arc"
        props.total_memory = 16 * 1024**3
        fake_torch.xpu.get_device_properties.return_value = props
        fake_torch.xpu.memory_reserved.return_value = 1024**3
        fake_torch.xpu.memory_allocated.return_value = 512 * 1024**2

        with patch.dict(sys.modules, {"torch": fake_torch}):
            from gpulivre.core.monitor import GPUMonitor
            status = GPUMonitor().get_status()

        self.assertEqual(status.backend, "xpu")
        self.assertEqual(status.total_vram_mb, 16_384.0)


class TestMCPIntegration(unittest.TestCase):
    def test_all_new_tools_are_registered(self):
        expected = {
            "gpulivre_compute_ceiling",
            "gpulivre_run_interleaved_bench",
            "gpulivre_hill_climb",
            "gpulivre_get_ledger_history",
        }
        self.assertTrue(expected.issubset(TOOL_DISPATCHER))

    def test_compute_ceiling_dispatch(self):
        raw = dispatch_tool("gpulivre_compute_ceiling", {
            "hardware": hardware().model_dump(mode="json"),
            "model": model().model_dump(),
            "vram_cap_mb": 1024,
            "contexts": [0],
        })
        payload = json.loads(raw)

        self.assertAlmostEqual(payload["decode_tok_s"]["0"], 200.0)

    def test_ledger_history_dispatch_uses_injected_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            path.write_text(json.dumps({"hardware": "gpu", "model": "m", "accepted": False}) + "\n")
            raw = dispatch_tool("gpulivre_get_ledger_history", {
                "ledger_path": str(path), "hardware": "gpu"
            })

        self.assertEqual(len(json.loads(raw)), 1)


if __name__ == "__main__":
    unittest.main()
