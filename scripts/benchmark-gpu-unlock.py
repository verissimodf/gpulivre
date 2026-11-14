#!/usr/bin/env python3
"""Benchmark real GPU workloads before/after the gpu_unlock boot-time unlock.

The suite is intentionally dependency-free (Python standard library only) and
records enough environment metadata to compare a stock boot with an unlocked
boot:

    scripts/benchmark-gpu-unlock.py --label stock
    scripts/benchmark-gpu-unlock.py --label unlocked
    scripts/benchmark-gpu-unlock.py --compare \
        benchmarks/results/<stock>.json benchmarks/results/<unlocked>.json

Workloads
---------
micro     BF16 GEMM through cuBLAS (the exact tensor-core rate gpu_unlock changes)
whisper   whisper.cpp HTTP server (/inference), CPU or GPU build
sd        stable-diffusion.cpp sd-server (A1111-compatible /sdapi/v1/txt2img)
llamacpp  llama-bench (local binary or the official CUDA container)
ollama    Ollama /api/generate (prompt prefill + token decode)

Every workload can be skipped; a failure in one does not stop the others.
Results are written as JSON plus a Markdown summary under benchmarks/results/.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "results"


class SkipBenchmark(RuntimeError):
    """Raised when a workload is unavailable in the current environment."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], timeout: int = 120, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def docker_available() -> bool:
    return shutil.which("docker") is not None and run(["docker", "info"], timeout=10).returncode == 0


def container_ip(name: str) -> str | None:
    if not docker_available():
        return None
    try:
        proc = run(
            ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", name],
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    for ip in proc.stdout.split():
        if ip and ip != "0.0.0.0":
            return ip
    return None


def url_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 600) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_multipart(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    timeout: int = 600,
) -> Any:
    boundary = f"----gpulivre{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")
    for name, (filename, content, content_type) in files.items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        body += content + b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")

    req = Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def git_commit() -> str:
    try:
        proc = run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], timeout=5)
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def gpu_info() -> dict[str, Any]:
    try:
        proc = run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit",
                "--format=csv,noheader",
            ],
            timeout=10,
        )
        line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "name": parts[0],
            "memory_total": parts[1],
            "driver": parts[2],
            "max_sm_clock": parts[3],
            "power_limit": parts[4],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def gpu_unlock_status() -> dict[str, Any]:
    lines: list[str] = []
    for cmd in (["journalctl", "-k", "-b", "--no-pager"], ["dmesg"]):
        try:
            proc = run(cmd, timeout=10)
        except Exception:
            continue
        if proc.returncode == 0:
            lines = [line for line in proc.stdout.splitlines() if "gpu_unlock" in line]
            if lines:
                break

    active = any("gpu_unlock: unlocked" in line for line in lines)
    ss0 = None
    for line in lines:
        match = re.search(r"ss0=(0x[0-9a-fA-F]+)", line)
        if match:
            ss0 = match.group(1)
    return {
        "active": active,
        "ss0": ss0,
        "log_tail": lines[-4:],
    }


def median_of(values: list[float]) -> float:
    return round(statistics.median(values), 4)


def find_audio() -> Path | None:
    for directory in (
        Path("/opt/laytech-ai/AssistentePessoal/data/voices"),
        REPO_ROOT / "benchmarks" / "assets",
    ):
        if directory.is_dir():
            for pattern in ("*.wav", "*.mp3", "*.ogg", "*.flac", "*.m4a"):
                files = sorted(directory.glob(pattern))
                if files:
                    return files[0]
    return None


def audio_duration(path: Path) -> float | None:
    if shutil.which("ffprobe"):
        try:
            proc = run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(path),
                ],
                timeout=30,
            )
            if proc.returncode == 0:
                return float(proc.stdout.strip())
        except Exception:
            pass
    if path.suffix.lower() == ".wav":
        try:
            import wave

            with wave.open(str(path), "rb") as wav:
                return wav.getnframes() / float(wav.getframerate())
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# workloads
# ---------------------------------------------------------------------------


def bench_micro(args: argparse.Namespace) -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise SkipBenchmark("nvcc not found")
    source = Path(__file__).resolve().parent / "benchmark_bf16.cu"
    if not source.is_file():
        raise SkipBenchmark(f"missing {source}")

    with tempfile.TemporaryDirectory(prefix="gpulivre-bench-") as tmp:
        binary = Path(tmp) / "benchmark_bf16"
        build = run([nvcc, "-O3", "-arch=native", "-o", str(binary), str(source), "-lcublas"], timeout=300)
        if build.returncode != 0:
            # -arch=native is not available on older toolchains.
            build = run([nvcc, "-O3", "-arch=sm_86", "-o", str(binary), str(source), "-lcublas"], timeout=300)
        if build.returncode != 0:
            raise RuntimeError(f"nvcc failed: {build.stderr.strip()[:500]}")
        proc = run([str(binary)], timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"benchmark failed: {proc.stderr.strip()[:500]}")
        match = re.search(r"([0-9.]+)\s+TFLOPS", proc.stdout)
        if not match:
            raise RuntimeError(f"unexpected output: {proc.stdout.strip()[:300]}")
        return {
            "tflops": float(match.group(1)),
            "n": 8192,
            "iters": 30,
            "raw": proc.stdout.strip(),
        }


def bench_whisper(args: argparse.Namespace) -> dict[str, Any]:
    url = args.whisper_url.rstrip("/")
    if not url_ok(url, timeout=3):
        raise SkipBenchmark(f"whisper server unreachable at {url}")

    source = Path(args.whisper_audio) if args.whisper_audio else find_audio()
    if source is None or not source.is_file():
        raise SkipBenchmark("no audio sample found (use --whisper-audio)")

    duration = audio_duration(source)
    if duration is None:
        raise SkipBenchmark(f"could not read audio duration for {source}")

    send_path = source
    tmpdir: tempfile.TemporaryDirectory | None = None
    if source.suffix.lower() != ".wav" and shutil.which("ffmpeg"):
        tmpdir = tempfile.TemporaryDirectory(prefix="gpulivre-audio-")
        wav = Path(tmpdir.name) / "sample.wav"
        proc = run(
            ["ffmpeg", "-y", "-i", str(source), "-ar", "16000", "-ac", "1", str(wav)],
            timeout=120,
        )
        if proc.returncode == 0:
            send_path = wav

    content_type = mimetypes.guess_type(send_path.name)[0] or "application/octet-stream"
    payload = send_path.read_bytes()

    def transcribe() -> float:
        start = time.perf_counter()
        http_multipart(
            f"{url}/inference",
            {"response_format": "json", "language": "auto", "temperature": "0"},
            {"file": (send_path.name, payload, content_type)},
            timeout=args.timeout,
        )
        return time.perf_counter() - start

    try:
        transcribe()  # warm-up, keeps the model resident
        samples = [transcribe() for _ in range(args.whisper_reps)]
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    elapsed = median_of(samples)
    return {
        "backend": "http-server",
        "audio_file": str(source),
        "audio_seconds": round(duration, 3),
        "elapsed_seconds": elapsed,
        "x_realtime": round(duration / elapsed, 2) if elapsed else None,
        "reps": args.whisper_reps,
        "samples": [round(s, 3) for s in samples],
    }


def bench_sd(args: argparse.Namespace) -> dict[str, Any]:
    url = args.sd_url.rstrip("/")
    if not url_ok(url, timeout=3):
        raise SkipBenchmark(f"sd-server unreachable at {url}")

    try:
        options = http_json(f"{url}/sdapi/v1/options", timeout=10)
        model = options.get("sd_model_checkpoint", "unknown")
    except Exception:
        model = "unknown"

    payload = {
        "prompt": args.sd_prompt,
        "negative_prompt": "blurry, low quality, watermark",
        "steps": args.sd_steps,
        "width": args.sd_size,
        "height": args.sd_size,
        "cfg_scale": 1.0,
        "seed": args.sd_seed,
        "batch_size": 1,
    }

    def generate() -> tuple[float, int]:
        start = time.perf_counter()
        response = http_json(f"{url}/sdapi/v1/txt2img", payload, timeout=args.timeout)
        elapsed = time.perf_counter() - start
        images = response.get("images") or []
        if not images:
            raise RuntimeError("sd-server returned no images")
        return elapsed, len(images)

    generate()  # warm-up
    samples: list[float] = []
    for _ in range(args.sd_reps):
        elapsed, _ = generate()
        samples.append(elapsed)

    elapsed = median_of(samples)
    return {
        "backend": "http-server",
        "model": model,
        "steps": args.sd_steps,
        "size": args.sd_size,
        "seed": args.sd_seed,
        "elapsed_seconds": elapsed,
        "images_per_minute": round(60.0 / elapsed, 2) if elapsed else None,
        "reps": args.sd_reps,
        "samples": [round(s, 3) for s in samples],
    }


def resolve_ollama_blob(container: str, model: str) -> Path:
    proc = run(["docker", "exec", container, "ollama", "show", model, "--modelfile"], timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ollama show failed: {proc.stderr.strip()[:300]}")

    blob = None
    for line in proc.stdout.splitlines():
        if line.startswith("FROM ") and "/models/blobs/" in line:
            blob = line.split("/models/blobs/", 1)[1].strip()
            break
    if not blob:
        raise RuntimeError(f"no blob found in Modelfile for {model}")

    mount = run(
        [
            "docker",
            "inspect",
            "-f",
            '{{range .Mounts}}{{if eq .Destination "/root/.ollama"}}{{.Source}}{{end}}{{end}}',
            container,
        ],
        timeout=10,
    )
    source = mount.stdout.strip()
    if mount.returncode != 0 or not source:
        raise RuntimeError(f"could not resolve Ollama data mount for {container}")

    path = Path(source) / "models" / "blobs" / blob
    if not path.is_file():
        raise RuntimeError(f"blob not found on host: {path}")
    return path


def find_llamacpp_entrypoint(image: str) -> str:
    candidates = [
        "/app/llama-bench",
        "/llama-bench",
        "/usr/local/bin/llama-bench",
        "/usr/bin/llama-bench",
    ]
    check = "for p in " + " ".join(candidates) + '; do [ -x "$p" ] && { echo "$p"; exit 0; }; done'
    try:
        proc = run(
            ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-c", check],
            timeout=60,
        )
        if proc.returncode == 0:
            found = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            if found:
                return found
    except Exception:
        pass
    return candidates[0]


def bench_llamacpp(args: argparse.Namespace) -> dict[str, Any]:
    image = args.llamacpp_image
    model: Path | None = Path(args.llamacpp_model) if args.llamacpp_model else None

    if model is None:
        if not docker_available():
            raise SkipBenchmark("llama-bench not on PATH and Docker unavailable (use --llamacpp-model)")
        model = resolve_ollama_blob(args.ollama_container, args.llamacpp_ollama_model)

    if not model.is_file():
        raise SkipBenchmark(f"GGUF model not found: {model}")

    local_bin = args.llamacpp_bin or shutil.which("llama-bench")
    if local_bin and not Path(local_bin).is_file():
        raise SkipBenchmark(f"llama-bench binary not found: {local_bin}")
    if local_bin:
        cmd = [
            local_bin,
            "-m",
            str(model),
            "-p",
            str(args.llamacpp_prompt),
            "-n",
            str(args.llamacpp_gen),
            "-r",
            str(args.llamacpp_reps),
            "-ngl",
            "99",
            "-o",
            "json",
        ]
    else:
        if not docker_available():
            raise SkipBenchmark("llama-bench not found and Docker unavailable")
        if run(["docker", "image", "inspect", image], timeout=20).returncode != 0:
            if not args.pull:
                raise SkipBenchmark(f"container image {image} not present (use --pull)")
            pull = run(["docker", "pull", image], timeout=1800)
            if pull.returncode != 0:
                raise RuntimeError(f"docker pull failed: {pull.stderr.strip()[:500]}")
        entrypoint = args.llamacpp_entrypoint or find_llamacpp_entrypoint(image)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{model}:/models/model.gguf:ro",
            "--entrypoint",
            entrypoint,
            image,
            "-m",
            "/models/model.gguf",
            "-p",
            str(args.llamacpp_prompt),
            "-n",
            str(args.llamacpp_gen),
            "-r",
            str(args.llamacpp_reps),
            "-ngl",
            "99",
            "-o",
            "json",
        ]

    proc = run(cmd, timeout=args.timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed: {(proc.stderr or proc.stdout).strip()[:800]}")

    text = proc.stdout
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise RuntimeError(f"llama-bench returned no JSON: {text.strip()[:500]}")
    rows = json.loads(text[start : end + 1])

    pp = tg = None
    for row in rows:
        if row.get("n_prompt") == args.llamacpp_prompt and row.get("n_gen") == 0:
            pp = row.get("avg_ts")
        if row.get("n_prompt") == 0 and row.get("n_gen") == args.llamacpp_gen:
            tg = row.get("avg_ts")

    if pp is None and tg is None:
        raise RuntimeError(f"could not map llama-bench rows: {rows}")

    return {
        "model": model.name,
        "model_path": str(model),
        "backend": "local" if local_bin else f"docker:{image}",
        "prompt_tokens": args.llamacpp_prompt,
        "gen_tokens": args.llamacpp_gen,
        "prefill_tok_s": round(pp, 2) if pp is not None else None,
        "decode_tok_s": round(tg, 2) if tg is not None else None,
        "reps": args.llamacpp_reps,
    }


def build_long_prompt(tokens: int) -> str:
    paragraph = (
        "The system monitors GPU telemetry, reserves VRAM for high-priority voice "
        "requests, pauses batch simulations when memory is exhausted, and falls back "
        "to remote compute when local capacity is insufficient. "
    )
    words_target = max(8, int(tokens * 0.75))
    words = paragraph.split()
    repeated = (words * ((words_target // len(words)) + 1))[:words_target]
    return "Context: " + " ".join(repeated) + "\n\nQuestion: summarize the policy above."


def bench_ollama(args: argparse.Namespace) -> dict[str, Any]:
    url = args.ollama_url.rstrip("/")
    if not url_ok(url, timeout=3):
        raise SkipBenchmark(f"Ollama unreachable at {url}")

    prompt = build_long_prompt(args.ollama_prompt_tokens)

    def generate(text: str, predict: int) -> dict[str, Any]:
        payload = {
            "model": args.ollama_model,
            "prompt": text,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_predict": predict,
                "temperature": 0,
                "seed": 42,
                "num_ctx": 4096,
            },
        }
        return http_json(f"{url}/api/generate", payload, timeout=args.timeout)

    # Warm-up: loads the model into VRAM and initializes the runner.
    generate("warm-up", 1)

    samples: list[dict[str, float]] = []
    for index in range(args.ollama_reps):
        # A unique prefix defeats prompt/KV cache reuse and forces a full prefill.
        unique = f"[measurement {index} {uuid.uuid4().hex[:8]}]\n\n"
        response = generate(unique + prompt, args.ollama_predict)
        prompt_count = response.get("prompt_eval_count") or 0
        prompt_ns = response.get("prompt_eval_duration") or 0
        eval_count = response.get("eval_count") or 0
        eval_ns = response.get("eval_duration") or 0
        if not prompt_count or not prompt_ns or not eval_count or not eval_ns:
            continue
        samples.append(
            {
                "prefill_tok_s": prompt_count / (prompt_ns / 1e9),
                "decode_tok_s": eval_count / (eval_ns / 1e9),
                "prompt_tokens": float(prompt_count),
                "eval_tokens": float(eval_count),
            }
        )

    if not samples:
        raise RuntimeError("Ollama returned no eval timings")

    return {
        "backend": "http-server",
        "model": args.ollama_model,
        "prompt_tokens": int(median_of([s["prompt_tokens"] for s in samples])),
        "gen_tokens": int(median_of([s["eval_tokens"] for s in samples])),
        "prefill_tok_s": median_of([s["prefill_tok_s"] for s in samples]),
        "decode_tok_s": median_of([s["decode_tok_s"] for s in samples]),
        "reps": len(samples),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


BENCHMARKS = {
    "micro": bench_micro,
    "whisper": bench_whisper,
    "sd": bench_sd,
    "llamacpp": bench_llamacpp,
    "ollama": bench_ollama,
}


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    selected = [name.strip() for name in args.only.split(",") if name.strip()] if args.only else list(BENCHMARKS)
    unknown = [name for name in selected if name not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"unknown benchmark(s): {', '.join(unknown)}")

    meta = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "kernel": run(["uname", "-r"], timeout=5).stdout.strip(),
        "gpu": gpu_info(),
        "gpu_unlock": gpu_unlock_status(),
        "host": os.uname().nodename,
    }

    results: dict[str, Any] = {}
    for index, name in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {name} ...", flush=True)
        try:
            result = BENCHMARKS[name](args)
            result["status"] = "ok"
            print(f"    ok: {json.dumps({k: v for k, v in result.items() if k != 'raw'})[:300]}", flush=True)
        except SkipBenchmark as exc:
            result = {"status": "skipped", "reason": str(exc)}
            print(f"    skipped: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one failed workload must not stop the suite
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"    error: {result['error']}", flush=True)
        results[name] = result

    return {"meta": meta, "results": results}


def write_outputs(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9A-Za-z_.-]+", "-", report["meta"]["timestamp"])
    base = out_dir / f"{stamp}-{report['meta']['label']}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    gpu = meta.get("gpu", {})
    unlock = meta.get("gpu_unlock", {})
    lines = [
        f"# GPU Unlock Benchmark — {meta['label']}",
        "",
        f"- **Timestamp:** {meta['timestamp']}",
        f"- **Host:** {meta.get('host', 'unknown')} ({meta.get('kernel', 'unknown')})",
        f"- **GPU:** {gpu.get('name', 'unknown')} ({gpu.get('memory_total', '?')}, driver {gpu.get('driver', '?')})",
        f"- **gpu_unlock active:** {unlock.get('active')} (ss0={unlock.get('ss0')})",
        f"- **Commit:** {meta.get('git_commit', 'unknown')}",
        "",
        "| Workload | Status | Metrics |",
        "| :--- | :--- | :--- |",
    ]
    for name, result in report["results"].items():
        status = result.get("status", "unknown")
        if status != "ok":
            detail = result.get("reason") or result.get("error") or ""
            lines.append(f"| {name} | {status} | {detail} |")
            continue
        metrics = []
        if name == "micro":
            metrics.append(f"{result['tflops']} TFLOPS BF16")
        elif name == "whisper":
            metrics.append(
                f"{result['audio_seconds']}s audio in {result['elapsed_seconds']}s "
                f"({result['x_realtime']}x realtime)"
            )
        elif name == "sd":
            metrics.append(
                f"{result['model']} {result['size']}px/{result['steps']} steps: "
                f"{result['elapsed_seconds']}s per image ({result['images_per_minute']}/min)"
            )
        elif name == "llamacpp":
            metrics.append(
                f"prefill {result['prefill_tok_s']} tok/s, decode {result['decode_tok_s']} tok/s"
            )
        elif name == "ollama":
            metrics.append(
                f"prefill {result['prefill_tok_s']} tok/s, decode {result['decode_tok_s']} tok/s "
                f"({result['prompt_tokens']} prompt tokens)"
            )
        lines.append(f"| {name} | ok | {'; '.join(metrics)} |")
    lines += [
        "",
        "> `micro` measures BF16 with FP32 accumulation through cuBLAS — the exact",
        "> tensor-core rate that gpu_unlock changes. Whisper and Stable Diffusion",
        "> numbers reflect whatever backend their service is configured to use; if",
        "> they run on CPU, the boot-time unlock does not change them.",
        "",
    ]
    return "\n".join(lines)


def metric_rows(report: dict[str, Any]) -> dict[tuple[str, str], tuple[float, str, bool]]:
    """Return {(workload, metric): (value, unit, higher_is_better)}."""
    rows: dict[tuple[str, str], tuple[float, str, bool]] = {}
    results = report.get("results", {})

    for workload, metric, unit, higher in (
        ("micro", "tflops", "TFLOPS", True),
        ("whisper", "x_realtime", "x realtime", True),
        ("whisper", "elapsed_seconds", "s", False),
        ("sd", "elapsed_seconds", "s/image", False),
        ("sd", "images_per_minute", "img/min", True),
        ("llamacpp", "prefill_tok_s", "tok/s", True),
        ("llamacpp", "decode_tok_s", "tok/s", True),
        ("ollama", "prefill_tok_s", "tok/s", True),
        ("ollama", "decode_tok_s", "tok/s", True),
    ):
        result = results.get(workload, {})
        if result.get("status") == "ok" and result.get(metric) is not None:
            rows[(workload, metric)] = (float(result[metric]), unit, higher)
    return rows


def compare_reports(stock_path: Path, unlocked_path: Path) -> str:
    stock = json.loads(stock_path.read_text(encoding="utf-8"))
    unlocked = json.loads(unlocked_path.read_text(encoding="utf-8"))

    stock_rows = metric_rows(stock)
    unlocked_rows = metric_rows(unlocked)
    keys = sorted(set(stock_rows) | set(unlocked_rows), key=lambda item: (item[0], item[1]))

    lines = [
        "# GPU Unlock — stock vs unlocked",
        "",
        f"- **Stock:** {stock['meta']['timestamp']} (unlock active: {stock['meta']['gpu_unlock']['active']})",
        f"- **Unlocked:** {unlocked['meta']['timestamp']} (unlock active: {unlocked['meta']['gpu_unlock']['active']})",
        "",
        "| Workload | Metric | Stock | Unlocked | Delta | Gain |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]

    for workload, metric in keys:
        s = stock_rows.get((workload, metric))
        u = unlocked_rows.get((workload, metric))
        if s and u:
            s_value, unit, higher = s
            u_value = u[0]
            delta = u_value - s_value
            pct = (delta / s_value * 100.0) if s_value else 0.0
            gain = pct if higher else -pct
            lines.append(
                f"| {workload} | {metric} | {s_value:g} {unit} | {u_value:g} {unit} | "
                f"{delta:+g} | {gain:+.1f}% |"
            )
        elif s:
            lines.append(f"| {workload} | {metric} | {s[0]:g} {s[1]} | — | — | — |")
        else:
            lines.append(f"| {workload} | {metric} | — | {u[0]:g} {u[1]} | — | — |")

    if stock["meta"]["gpu_unlock"]["active"] == unlocked["meta"]["gpu_unlock"]["active"]:
        lines += [
            "",
            f"> **Atenção:** os dois runs têm `gpu_unlock.active={stock['meta']['gpu_unlock']['active']}`.",
            "> O baseline stock exige um boot sem `gpu_unlock.targets=` no kernel cmdline.",
        ]

    lines += [
        "",
        "Gain positivo = melhora. Para métricas de latência (segundos por imagem,",
        "tempo de transcrição), o sinal é invertido: menor é melhor.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark real workloads before/after the gpu_unlock boot-time unlock.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--label", default="run", help="run label, e.g. stock or unlocked (default: run)")
    parser.add_argument("--only", help="comma-separated subset: micro,whisper,sd,llamacpp,ollama")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--timeout", type=int, default=900, help="per-request timeout in seconds")
    parser.add_argument("--pull", action="store_true", help="pull the llama.cpp container image if missing")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("STOCK_JSON", "UNLOCKED_JSON"),
        type=Path,
        help="compare two result files and exit",
    )

    whisper = parser.add_argument_group("whisper")
    whisper_ip = container_ip("assistente-pessoal-stt")
    whisper.add_argument(
        "--whisper-url",
        default=os.environ.get("WHISPER_URL")
        or (f"http://{whisper_ip}:8080" if whisper_ip else None)
        or "http://127.0.0.1:8080",
    )
    whisper.add_argument("--whisper-audio", help="audio sample (default: first file in data/voices)")
    whisper.add_argument("--whisper-reps", type=int, default=3)

    sd = parser.add_argument_group("stable diffusion")
    sd_ip = container_ip("assistente-pessoal-studio-sd")
    sd.add_argument(
        "--sd-url",
        default=os.environ.get("SD_URL")
        or (f"http://{sd_ip}:7870" if sd_ip else None)
        or "http://127.0.0.1:7870",
    )
    sd.add_argument("--sd-prompt", default="a red apple on a wooden table, studio photo, sharp focus")
    sd.add_argument("--sd-steps", type=int, default=2)
    sd.add_argument("--sd-size", type=int, default=256)
    sd.add_argument("--sd-seed", type=int, default=42)
    sd.add_argument("--sd-reps", type=int, default=2)

    llama = parser.add_argument_group("llama.cpp")
    llama.add_argument("--llamacpp-bin", help="local llama-bench binary (default: PATH)")
    llama.add_argument("--llamacpp-model", help="local GGUF path (default: Ollama blob)")
    llama.add_argument("--llamacpp-ollama-model", default="qwen2.5:7b", help="Ollama model used to locate a GGUF blob")
    llama.add_argument("--llamacpp-image", default="ghcr.io/ggml-org/llama.cpp:full-cuda")
    llama.add_argument("--llamacpp-entrypoint", help="llama-bench path inside the image")
    llama.add_argument("--llamacpp-prompt", type=int, default=512)
    llama.add_argument("--llamacpp-gen", type=int, default=128)
    llama.add_argument("--llamacpp-reps", type=int, default=3)

    ollama = parser.add_argument_group("ollama")
    ollama.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    ollama.add_argument("--ollama-container", default=os.environ.get("OLLAMA_CONTAINER", "ollama-llm"))
    ollama.add_argument("--ollama-model", default="qwen2.5:7b")
    ollama.add_argument("--ollama-prompt-tokens", type=int, default=1024)
    ollama.add_argument("--ollama-predict", type=int, default=128)
    ollama.add_argument("--ollama-reps", type=int, default=3)

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.compare:
        stock_path, unlocked_path = args.compare
        for path in (stock_path, unlocked_path):
            if not path.is_file():
                print(f"error: file not found: {path}", file=sys.stderr)
                return 2
        print(compare_reports(stock_path, unlocked_path))
        return 0

    report = run_suite(args)
    json_path, md_path = write_outputs(report, args.out_dir)

    print()
    print(render_markdown(report))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    if not report["meta"]["gpu_unlock"]["active"]:
        print("note: gpu_unlock was not detected as active in this boot.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
