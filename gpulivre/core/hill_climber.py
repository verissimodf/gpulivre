"""Noise-controlled, serialized hill-climbing for OpenAI-compatible inference servers."""

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from gpulivre.core.allocator import TaskPriority
from gpulivre.core.bench_harness import BenchmarkConfig, BenchmarkResult

logger = logging.getLogger("gpulivre.hill_climber")

DEFAULT_LEDGER_PATH = Path.home() / ".antigravity" / "gpulivre_ledger.jsonl"
DEFAULT_PLAN_PATH = Path.home() / ".antigravity" / "gpulivre_PLAN.md"


class ExperimentSample(BaseModel):
    leg: str
    result: BenchmarkResult


class ExperimentResult(BaseModel):
    accepted: bool
    reason: str
    target_metric: str
    lever: str
    hardware: str
    model: str
    samples: List[ExperimentSample]
    control_before: Dict[str, float] = Field(default_factory=dict)
    candidate: Dict[str, float] = Field(default_factory=dict)
    control_after: Dict[str, float] = Field(default_factory=dict)
    regressions_percent: Dict[str, float] = Field(default_factory=dict)
    ceiling_percent: Optional[float] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Lever(BaseModel):
    name: str
    config: BenchmarkConfig
    likelihood: float = Field(default=0.5, ge=0.0, le=1.0)
    cost: float = Field(default=1.0, gt=0.0)
    affected_metrics: List[str] = Field(default_factory=list)

    def priority_score(self, ceiling_gap: float) -> float:
        return max(0.0, ceiling_gap) * self.likelihood / self.cost


class HillClimbSession(BaseModel):
    best_config: BenchmarkConfig
    experiments: List[ExperimentResult] = Field(default_factory=list)
    plateau: bool = False


class ExperimentLedger:
    """Append-only experiment history with a human-readable companion report."""

    def __init__(
        self,
        path: Path | str = DEFAULT_LEDGER_PATH,
        plan_path: Path | str = DEFAULT_PLAN_PATH,
    ):
        self.path = Path(path).expanduser()
        self.plan_path = Path(plan_path).expanduser()
        self._lock = threading.Lock()

    def append(self, experiment: ExperimentResult) -> None:
        record = experiment.model_dump(mode="json")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._append_plan(experiment)

    def _append_plan(self, experiment: ExperimentResult) -> None:
        before = experiment.control_before.get(experiment.target_metric)
        after = experiment.candidate.get(experiment.target_metric)
        verdict = "ACCEPTED" if experiment.accepted else "REJECTED"
        lines = [
            f"\n## {experiment.timestamp} — {experiment.lever}\n",
            f"- Hardware/model: `{experiment.hardware}` / `{experiment.model}`",
            f"- Decision: **{verdict}** — {experiment.reason}",
            f"- Before -> After ({experiment.target_metric}): `{before}` -> `{after}`",
        ]
        if experiment.ceiling_percent is not None:
            lines.append(f"- Theoretical ceiling reached: {experiment.ceiling_percent:.2f}%")
        if experiment.regressions_percent:
            regressions = ", ".join(
                f"{name}={value:.2f}%"
                for name, value in experiment.regressions_percent.items()
            )
            lines.append(f"- Regressions: {regressions}")
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        with self.plan_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def history(
        self,
        hardware: Optional[str] = None,
        model: Optional[str] = None,
        lever: Optional[str] = None,
        accepted: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if limit < 1 or not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Ignoring invalid ledger line %s", line_number)
                    continue
                if hardware is not None and row.get("hardware") != hardware:
                    continue
                if model is not None and row.get("model") != model:
                    continue
                if lever is not None and row.get("lever") != lever:
                    continue
                if accepted is not None and row.get("accepted") is not accepted:
                    continue
                rows.append(row)
        return rows[-limit:]


def target_metric_for_priority(priority: TaskPriority, concurrency: int = 1) -> str:
    if priority == TaskPriority.P1_CRITICAL:
        return "ttft_short_ms"
    if priority == TaskPriority.P3_BATCH and concurrency > 1:
        return f"c{concurrency}_aggregate_tok_s"
    return "c1_decode_tok_s"


def _lower_is_better(metric: str) -> bool:
    return metric.startswith(("ttft", "startup", "latency", "mem_peak"))


def _better(candidate: float, control: float, metric: str) -> bool:
    return candidate < control if _lower_is_better(metric) else candidate > control


def _regression(candidate: float, control: float, metric: str) -> float:
    if control == 0:
        if candidate == control:
            return 0.0
        return math.inf if not _better(candidate, control, metric) else 0.0
    if _lower_is_better(metric):
        return (candidate - control) / abs(control) * 100.0
    return (control - candidate) / abs(control) * 100.0


def _median_metrics(samples: Sequence[ExperimentSample]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for sample in samples:
        for metric, value in sample.result.metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.setdefault(metric, []).append(float(value))
    return {metric: statistics.median(items) for metric, items in values.items() if items}


def _failed_result(config: BenchmarkConfig, error: Exception) -> BenchmarkResult:
    return BenchmarkResult(
        name=config.name,
        ok=False,
        flags=["REQ_FAIL"],
        error=repr(error),
        needle_passed=False,
        soak_passed=False,
    )


def run_interleaved_bench(
    runner: Any,
    control: BenchmarkConfig,
    candidate: BenchmarkConfig,
    target_metric: str,
    samples_per_leg: int = 2,
    ledger: Optional[ExperimentLedger] = None,
    hardware: str = "unknown",
    model: Optional[str] = None,
    lever: str = "candidate",
    require_quality_gates: bool = True,
    ceiling_percent: Optional[float] = None,
) -> ExperimentResult:
    """Run ctl -> cand -> ctl and apply strict promotion and quality gates."""
    if samples_per_leg < 2:
        raise ValueError("samples_per_leg must be >= 2 (six samples total)")

    samples: List[ExperimentSample] = []
    for leg, config in (("control_before", control), ("candidate", candidate), ("control_after", control)):
        for _ in range(samples_per_leg):
            try:
                measured = runner.run(config)
            except Exception as error:  # a failed trial is institutional memory too
                measured = _failed_result(config, error)
            samples.append(ExperimentSample(leg=leg, result=measured))

    before = _median_metrics([s for s in samples if s.leg == "control_before"])
    cand = _median_metrics([s for s in samples if s.leg == "candidate"])
    after = _median_metrics([s for s in samples if s.leg == "control_after"])
    candidate_samples = [s.result for s in samples if s.leg == "candidate"]
    all_flags = sorted({flag for sample in samples for flag in sample.result.flags})

    accepted = False
    reason = ""
    regressions: Dict[str, float] = {}
    if all_flags:
        reason = f"invalid samples: {', '.join(all_flags)}"
    elif any(not sample.result.ok for sample in samples):
        reason = "one or more benchmark samples failed"
    elif target_metric not in before or target_metric not in cand or target_metric not in after:
        reason = f"missing target metric: {target_metric}"
    elif not (_better(cand[target_metric], before[target_metric], target_metric) and
              _better(cand[target_metric], after[target_metric], target_metric)):
        reason = f"candidate does not beat both controls on {target_metric}"
    else:
        for metric in cand.keys() & before.keys() & after.keys():
            if metric == target_metric:
                continue
            regression = max(
                _regression(cand[metric], before[metric], metric),
                _regression(cand[metric], after[metric], metric),
            )
            regressions[metric] = regression
            if regression > 3.0:
                reason = f"regression on {metric}: {regression:.2f}% > 3%"
                break
        else:
            if require_quality_gates and any(sample.needle_passed is not True for sample in candidate_samples):
                reason = "needle retrieval gate failed"
            elif require_quality_gates and any(sample.soak_passed is not True for sample in candidate_samples):
                reason = "full-KV soak gate failed"
            else:
                accepted = True
                reason = "candidate beat both controls without regressions and passed all gates"

    experiment = ExperimentResult(
        accepted=accepted,
        reason=reason,
        target_metric=target_metric,
        lever=lever,
        hardware=hardware,
        model=model or candidate.model,
        samples=samples,
        control_before=before,
        candidate=cand,
        control_after=after,
        regressions_percent=regressions,
        ceiling_percent=ceiling_percent,
    )
    if ledger is not None:
        ledger.append(experiment)
    return experiment


class HillClimber:
    """Serialize one-lever experiments and keep only fully qualified winners."""

    def __init__(self, runner: Any, ledger: Optional[ExperimentLedger] = None):
        self.runner = runner
        self.ledger = ledger

    def hill_climb(
        self,
        baseline: BenchmarkConfig,
        levers: Sequence[Lever],
        priority: TaskPriority = TaskPriority.P2_HIGH,
        ceiling_gap: float = 1.0,
        max_experiments: Optional[int] = None,
        hardware: str = "unknown",
        require_quality_gates: bool = True,
    ) -> HillClimbSession:
        ordered = sorted(levers, key=lambda lever: lever.priority_score(ceiling_gap), reverse=True)
        if max_experiments is not None:
            ordered = ordered[:max_experiments]
        best = baseline
        experiments: List[ExperimentResult] = []
        consecutive_rejections = 0
        target = target_metric_for_priority(priority, baseline.concurrency)

        for lever in ordered:
            if self.ledger is not None and self.ledger.history(
                hardware=hardware, model=best.model, lever=lever.name, accepted=False, limit=1
            ):
                consecutive_rejections += 1
                if consecutive_rejections >= 3:
                    break
                continue
            experiment = run_interleaved_bench(
                runner=self.runner,
                control=best,
                candidate=lever.config,
                target_metric=target,
                ledger=self.ledger,
                hardware=hardware,
                model=best.model,
                lever=lever.name,
                require_quality_gates=require_quality_gates,
            )
            experiments.append(experiment)
            if experiment.accepted:
                best = lever.config
                consecutive_rejections = 0
            else:
                consecutive_rejections += 1
                if consecutive_rejections >= 3:
                    break

        plateau = consecutive_rejections >= 3
        return HillClimbSession(best_config=best, experiments=experiments, plateau=plateau)
