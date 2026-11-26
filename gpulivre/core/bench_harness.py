"""
GPULivre Core: OpenAI-compatible benchmark harness.

Mede servidores OpenAI-compatible com `urllib`/SSE. Tokens vêm
exclusivamente do bloco `usage` do servidor ou de um tokenizador real
injetado (`token_counter`). Chunks SSE nunca são usados como contador.

Clock e opener são injetáveis para testes sem rede e sem relógio real.
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

MIB = 1024 * 1024

# Flags de saída inválida/estado do host (spec 5.2).
FLAG_LOOP = "LOOP"
FLAG_EMPTY_OUTPUT = "EMPTY_OUTPUT"
FLAG_REQ_FAIL = "REQ_FAIL"
FLAG_THERMAL_THROTTLE = "THERMAL_THROTTLE"
FLAG_CLOCK_DROP = "CLOCK_DROP"
FLAG_MEM_CAP = "MEM_CAP"
FLAG_ON_BATTERY = "ON_BATTERY"

Clock = Callable[[], float]
Opener = Callable[..., object]
TokenCounter = Callable[[str], int]
Sampler = Callable[[], float]


class BenchmarkConfig(BaseModel):
    """Configuração declarativa de uma rodada de benchmark."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    model: str
    base_url: str
    api_key: Optional[str] = None

    timeout_s: float = Field(default=120.0, gt=0)
    samples: int = Field(default=3, ge=1)
    concurrency: int = Field(default=4, ge=1)
    max_tokens: int = Field(default=128, ge=1)

    short_prompt: str = "Responda com uma frase curta e objetiva."
    long_context_tokens: int = Field(default=2048, ge=16)
    prefill_contexts: list[int] = Field(
        default_factory=lambda: [4096, 32768, 131072, 262144],
        alias="prefill_lengths",
    )
    max_context_tokens: Optional[int] = None
    kv_bytes_per_token: Optional[float] = Field(default=None, gt=0)
    runtime_reserve_mb: float = Field(default=0.0, ge=0)

    needle_enabled: bool = True
    needle_context_tokens: int = Field(default=32768, ge=32)
    soak_enabled: bool = True
    soak_context_tokens: int = Field(default=32768, ge=16)
    soak_concurrency: int = Field(default=4, ge=1)
    soak_max_tokens: int = Field(default=16, ge=1)

    memory_cap_mb: Optional[float] = Field(default=None, gt=0)
    watchdog_interval_s: float = Field(default=0.1, gt=0)

    launch: Optional[list[str] | str] = None
    stop: Optional[list[str] | str] = None
    startup_timeout_s: float = Field(default=60.0, gt=0)
    health_interval_s: float = Field(default=0.5, gt=0)

    temperature_delta_c: float = 5.0
    clock_drop_threshold: float = 0.10
    thermal_limit_c: Optional[float] = None

    extra_body: dict = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class ChatResult(BaseModel):
    """Resultado de UMA requisição de chat medida."""

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: Optional[float] = None
    decode_tok_s: float = 0.0
    duration_s: float = 0.0

    @property
    def prefill_tok_s(self) -> Optional[float]:
        if self.ttft_s and self.ttft_s > 0:
            return self.prompt_tokens / self.ttft_s
        return None


class BenchmarkResult(BaseModel):
    """Agregado de uma amostra completa do runner."""

    name: str
    ok: bool = True
    metrics: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    needle_passed: bool = True
    soak_passed: bool = True
    memory_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    error: Optional[str] = None


class MissingUsageError(RuntimeError):
    """Servidor sem `usage` e sem tokenizador real injetado."""


class MemoryCapExceeded(RuntimeError):
    """A memória medida ultrapassou o cap do operador."""

    def __init__(self, used_bytes: float, cap_bytes: float):
        self.used_bytes = used_bytes
        self.cap_bytes = cap_bytes
        super().__init__(
            f"memory cap exceeded: {used_bytes / MIB:.1f} MB > {cap_bytes / MIB:.1f} MB"
        )


def _first_content_delta(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _int_tokens(value: object) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}{path}"


class OpenAICompatibleClient:
    """Cliente stdlib SSE para endpoints OpenAI-compatible."""

    def __init__(
        self,
        config: BenchmarkConfig,
        opener: Optional[Opener] = None,
        clock: Optional[Clock] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.config = config
        self.opener = opener or urllib.request.urlopen
        self.clock = clock or time.perf_counter
        self.token_counter = token_counter
        self._active: list = []
        self._lock = threading.Lock()

    # ---- infraestrutura -------------------------------------------------

    def _request(self, request: urllib.request.Request):
        return self.opener(request, timeout=self.config.timeout_s)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        headers.update(self.config.headers)
        if self.config.api_key:
            headers.setdefault("Authorization", f"Bearer {self.config.api_key}")
        return headers

    def health(self) -> bool:
        """GET /v1/models — health check barato."""
        request = urllib.request.Request(
            _endpoint(self.config.base_url, "/models"),
            method="GET",
            headers=self._headers(),
        )
        try:
            response = self._request(request)
        except (urllib.error.URLError, TimeoutError, OSError):
            return False
        with response:
            status = getattr(response, "status", None)
            code = getattr(response, "getcode", lambda: None)()
            if status is None:
                status = code
            if isinstance(status, int):
                return 200 <= status < 300
            return True

    def abort_active(self) -> None:
        """Fecha respostas SSE ativas (usado pelo watchdog)."""
        with self._lock:
            active = list(self._active)
            self._active.clear()
        for response in active:
            try:
                response.close()
            except Exception:
                pass

    def close(self) -> None:
        self.abort_active()

    @staticmethod
    def _data_payloads(response: object) -> Iterable[dict]:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if not body:
                continue
            if body == "[DONE]":
                return
            yield json.loads(body)

    # ---- medição --------------------------------------------------------

    def chat(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
    ) -> ChatResult:
        """Uma requisição stream=true com TTFT real e contagem por usage."""
        max_tokens = max_tokens or self.config.max_tokens
        payload: dict = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        payload.update(self.config.extra_body)
        payload.update(extra_body or {})
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        request = urllib.request.Request(
            _endpoint(self.config.base_url, "/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=self._headers(),
        )

        start = self.clock()
        ttft: Optional[float] = None
        text_parts: list[str] = []
        usage: dict = {}

        response = self._request(request)
        stream = None
        try:
            with response as stream:
                with self._lock:
                    self._active.append(stream)
                for event in self._data_payloads(stream):
                    content = _first_content_delta(event)
                    if content:
                        if ttft is None:
                            ttft = self.clock()
                        text_parts.append(content)
                    top_usage = event.get("usage")
                    if isinstance(top_usage, dict):
                        usage = top_usage
        finally:
            with self._lock:
                if stream in self._active:
                    self._active.remove(stream)

        end = self.clock()
        text = "".join(text_parts)
        duration_s = max(0.0, end - start)

        prompt_tokens = _int_tokens(usage.get("prompt_tokens"))
        completion_tokens = _int_tokens(usage.get("completion_tokens"))
        if prompt_tokens is None:
            if self.token_counter is None:
                raise MissingUsageError(
                    "server returned no usage and no real token_counter was provided"
                )
            prompt_tokens = self.token_counter(prompt)
        if completion_tokens is None:
            if self.token_counter is None:
                raise MissingUsageError(
                    "server returned no usage and no real token_counter was provided"
                )
            completion_tokens = self.token_counter(text)

        decode_tok_s = 0.0
        if ttft is not None and completion_tokens > 1:
            decode_window = end - ttft
            if decode_window > 0:
                decode_tok_s = (completion_tokens - 1) / decode_window

        return ChatResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft,
            decode_tok_s=decode_tok_s,
            duration_s=duration_s,
        )


def exact_needle_match(output: str, codes: list[str]) -> bool:
    """Correspondência exata: os códigos, nessa ordem, e nada mais."""
    return output.strip().split() == list(codes)


def looping(text: str, *, min_repeats: int = 4) -> bool:
    """Detecta saída em loop por repetição de n-gramas do output."""
    tokens = text.split()
    if len(tokens) < 4 * min_repeats:
        return False
    for size in (1, 2, 4, 8, 16, 32):
        if size * min_repeats > len(tokens):
            break
        windows = [
            tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)
        ]
        counts: dict[tuple[str, ...], int] = {}
        for window in windows:
            counts[window] = counts.get(window, 0) + 1
        if any(count >= min_repeats for count in counts.values()):
            return True
    return False


def build_needle_document(tokens: int, seed: int) -> tuple[str, list[str]]:
    """Documento com três códigos únicos em profundidades distintas."""
    if tokens < 32:
        raise ValueError("needle document needs at least 32 tokens")
    rng = random.Random(seed)
    codes: list[str] = []
    while len(codes) < 3:
        code = f"NEEDLE-{rng.randrange(100000, 1000000)}"
        if code not in codes:
            codes.append(code)

    filler_pool = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi"
    ).split()
    words = [filler_pool[i % len(filler_pool)] for i in range(tokens)]
    for position, code in zip((tokens // 5, tokens // 2, (4 * tokens) // 5), codes):
        words.insert(min(position + 1, len(words)), code)

    instruction = (
        "Neste documento ha tres codigos. "
        "Responda APENAS com os tres codigos, na ordem em que aparecem, "
        "separados por espaco simples, sem texto adicional."
    )
    return f"{instruction}\n\n{' '.join(words)}", codes


class MemoryWatchdog:
    """Amostra memória e interrompe imediatamente ao ultrapassar o cap."""

    def __init__(
        self,
        cap_mb: float,
        sampler: Sampler,
        on_trip: Callable[[], None],
        interval_s: float = 0.1,
    ):
        self.cap_bytes = cap_mb * MIB
        self.sampler = sampler
        self.on_trip = on_trip
        self.interval_s = interval_s
        self.tripped = False
        self.error: Optional[Exception] = None
        self.peak_bytes: float = 0.0
        self.last_bytes: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample_once(self) -> float:
        """Uma amostra; lança MemoryCapExceeded imediatamente se estourar."""
        used_bytes = float(self.sampler())
        self.last_bytes = used_bytes
        self.peak_bytes = max(self.peak_bytes, used_bytes)
        if used_bytes > self.cap_bytes:
            self.tripped = True
            try:
                self.on_trip()
            finally:
                raise MemoryCapExceeded(used_bytes, self.cap_bytes)
        if self.tripped:
            raise MemoryCapExceeded(self.last_bytes, self.cap_bytes)
        return used_bytes

    def start(self) -> "MemoryWatchdog":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="gpulivre-memory-watchdog", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.sample_once()
            except MemoryCapExceeded:
                return
            except Exception as exc:  # sampler quebrado não pode passar despercebido
                self.error = exc
                return

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.interval_s * 4))

    def __enter__(self) -> "MemoryWatchdog":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _numeric(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sized_prompt(tokens: int, salt: str) -> str:
    """Prompt de ~`tokens` palavras (contagem real vem do usage/counter)."""
    filler = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi"
    ).split()
    words = [filler[i % len(filler)] for i in range(tokens)]
    return (
        " Leia e processe o documento a seguir por completo. "
        f"Marcador unico de requisição: {salt}. "
        "Ao final responda apenas: OK.\n\n" + " ".join(words)
    )


def _output_flags(result: ChatResult) -> list[str]:
    flags = []
    if not result.text.strip():
        flags.append(FLAG_EMPTY_OUTPUT)
    if looping(result.text):
        flags.append(FLAG_LOOP)
    return flags


def _monitor_sampler(monitor) -> Sampler:
    def sample() -> float:
        status = monitor.get_status()
        used_mb = getattr(status, "used_vram_mb", None)
        if used_mb is None:
            raise RuntimeError("monitor did not report used VRAM")
        return float(used_mb) * MIB

    return sample


def _status_value(status: object, *names: str) -> object:
    for name in names:
        value = getattr(status, name, None)
        if value is not None:
            return value
        getter = getattr(status, "get", None)
        if callable(getter):
            try:
                value = getter(name)
            except Exception:
                value = None
            if value is not None:
                return value
    return None


def _telemetry_flags(
    before: object, after: object, config: BenchmarkConfig
) -> list[str]:
    flags: list[str] = []
    if before is None or after is None:
        return flags

    before_clock = _numeric(_status_value(before, "sm_clock_mhz", "clock_mhz"))
    after_clock = _numeric(_status_value(after, "sm_clock_mhz", "clock_mhz"))
    clock_dropped = False
    if before_clock and after_clock and after_clock < before_clock:
        clock_dropped = (before_clock - after_clock) / before_clock > config.clock_drop_threshold
        if clock_dropped:
            flags.append(FLAG_CLOCK_DROP)

    before_temp = _numeric(_status_value(before, "temperature_celsius"))
    after_temp = _numeric(_status_value(after, "temperature_celsius"))
    thermal = False
    if after_temp is not None and config.thermal_limit_c is not None:
        thermal = after_temp >= config.thermal_limit_c
    if (
        before_temp is not None
        and after_temp is not None
        and (after_temp - before_temp) >= config.temperature_delta_c
    ):
        thermal = thermal or clock_dropped
    if thermal:
        flags.append(FLAG_THERMAL_THROTTLE)

    on_battery = _status_value(before, "on_battery") or _status_value(after, "on_battery")
    if on_battery is True:
        flags.append(FLAG_ON_BATTERY)
    return flags


class _CommandRunner:
    def __init__(
        self,
        process_factory: Optional[Callable[..., object]] = None,
        command_runner: Optional[Callable[..., object]] = None,
    ):
        self.process_factory = process_factory or subprocess.Popen
        self.command_runner = command_runner or subprocess.run

    def launch(self, command: list[str] | str) -> object:
        shell = isinstance(command, str)
        return self.process_factory(command, shell=shell)

    def stop(self, command: list[str] | str, timeout_s: float) -> None:
        shell = isinstance(command, str)
        self.command_runner(command, shell=shell, check=False, timeout=timeout_s)


class BenchmarkRunner:
    """Orquestra startup, métricas, watchdog, needle e soak."""

    def __init__(
        self,
        monitor=None,
        opener: Optional[Opener] = None,
        clock: Optional[Clock] = None,
        token_counter: Optional[TokenCounter] = None,
        sampler: Optional[Sampler] = None,
        sleep: Callable[[float], None] = time.sleep,
        process_factory: Optional[Callable[..., object]] = None,
        command_runner: Optional[Callable[..., object]] = None,
    ):
        self.monitor = monitor
        self.opener = opener
        self.clock = clock or time.perf_counter
        self.token_counter = token_counter
        self.sampler = sampler
        self.sleep = sleep
        self._commands = _CommandRunner(process_factory, command_runner)

    # ---- clientes -------------------------------------------------------

    def client(self, config: BenchmarkConfig) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            config,
            opener=self.opener,
            clock=self.clock,
            token_counter=self.token_counter,
        )

    # ---- lifecycle ------------------------------------------------------

    def _launch(self, config: BenchmarkConfig, client: OpenAICompatibleClient) -> float:
        start = self.clock()
        self._commands.launch(config.launch)
        deadline = config.startup_timeout_s
        while True:
            if self.clock() - start >= deadline:
                raise RuntimeError(
                    f"server did not become healthy within {deadline:.0f}s"
                )
            if client.health():
                return self.clock() - start
            self.sleep(config.health_interval_s)

    # ---- métricas -------------------------------------------------------

    def _chat(self, client: OpenAICompatibleClient, *args, **kwargs) -> ChatResult:
        return client.chat(*args, **kwargs)

    def _run_needle(self, config: BenchmarkConfig, client: OpenAICompatibleClient) -> tuple[bool, list[str]]:
        prompt, codes = build_needle_document(config.needle_context_tokens, seed=7)
        result = client.chat(prompt, max_tokens=32)
        passed = exact_needle_match(result.text, codes)
        return passed, _output_flags(result)

    def _run_soak(self, config: BenchmarkConfig, client: OpenAICompatibleClient) -> tuple[bool, list[str]]:
        prompts = [
            _sized_prompt(config.soak_context_tokens, uuid.uuid4().hex)
            for _ in range(config.soak_concurrency)
        ]

        def one(prompt: str) -> ChatResult:
            return client.chat(prompt, max_tokens=config.soak_max_tokens)

        with ThreadPoolExecutor(max_workers=config.soak_concurrency) as pool:
            results = list(pool.map(one, prompts))

        passed = all(
            result.text.strip() and not _output_flags(result) for result in results
        )
        flags: list[str] = []
        for result in results:
            flags.extend(_output_flags(result))
        return passed, flags

    # ---- API pública ----------------------------------------------------

    def run_needle(
        self, config: BenchmarkConfig, client: Optional[OpenAICompatibleClient] = None
    ) -> bool:
        owned = client is None
        client = client or self.client(config)
        try:
            passed, _flags = self._run_needle(config, client)
            return passed
        finally:
            if owned:
                client.close()

    def run_full_kv_soak(
        self, config: BenchmarkConfig, client: Optional[OpenAICompatibleClient] = None
    ) -> bool:
        owned = client is None
        client = client or self.client(config)
        try:
            passed, _flags = self._run_soak(config, client)
            return passed
        finally:
            if owned:
                client.close()

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        metrics: dict[str, float] = {}
        flags: list[str] = []
        before = None
        after = None
        error: Optional[str] = None
        watchdog: Optional[MemoryWatchdog] = None
        needle_passed = not config.needle_enabled
        soak_passed = not config.soak_enabled
        stop_error: Optional[str] = None
        stop_lock = threading.Lock()
        stopped = False

        def stop_managed() -> None:
            nonlocal stopped, stop_error
            if not (config.launch and config.stop) or stopped:
                return
            with stop_lock:
                if stopped:
                    return
                stopped = True
            try:
                self._commands.stop(config.stop, timeout_s=config.timeout_s)
            except Exception as exc:
                stop_error = f"stop command failed: {exc}"

        def on_cap() -> None:
            client.abort_active()
            stop_managed()

        client = self.client(config)
        try:
            try:
                before = self.monitor.get_status() if self.monitor else None
            except Exception:
                before = None  # telemetria opcional não invalida a requisição

            if config.launch:
                metrics["startup_s"] = self._launch(config, client)

            sampler = self.sampler
            if sampler is None and self.monitor is not None:
                sampler = _monitor_sampler(self.monitor)
            if config.memory_cap_mb is not None and sampler is None:
                raise ValueError("memory_cap_mb requires a memory sampler or monitor")
            if sampler is not None:
                watchdog = MemoryWatchdog(
                    cap_mb=config.memory_cap_mb if config.memory_cap_mb is not None else float("inf"),
                    sampler=sampler,
                    on_trip=on_cap,
                    interval_s=config.watchdog_interval_s,
                )
                try:
                    watchdog.sample_once()
                except MemoryCapExceeded:
                    raise
                except Exception:
                    if config.memory_cap_mb is not None:
                        raise
                    watchdog = None  # footprint é opcional; cap não
                if watchdog is not None and config.memory_cap_mb is not None:
                    watchdog.start()

            # c1: decode curto, mediana de N amostras; TTFT curto.
            short_results = [
                self._chat(client, config.short_prompt, max_tokens=32)
                for _ in range(config.samples)
            ]
            for result in short_results:
                flags.extend(_output_flags(result))
            metrics["c1_decode_tok_s"] = statistics.median(
                [r.decode_tok_s for r in short_results]
            )
            metrics["ttft_short_ms"] = statistics.median(
                [1000.0 * (r.ttft_s or 0.0) for r in short_results]
            )

            # TTFT longo.
            long_prompt = _sized_prompt(
                config.long_context_tokens, "long-context-bench"
            )
            long_result = self._chat(client, long_prompt, max_tokens=config.max_tokens)
            flags.extend(_output_flags(long_result))
            metrics["ttft_long_ms"] = 1000.0 * (long_result.ttft_s or 0.0)

            # cN aggregate: throughput concorrente (tempo de parede).
            concurrency = config.concurrency
            cN_prompts = [
                _sized_prompt(config.long_context_tokens, uuid.uuid4().hex)
                for _ in range(concurrency)
            ]
            cN_start = self.clock()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                cN_results = list(
                    pool.map(
                        lambda prompt: self._chat(client, prompt, max_tokens=32),
                        cN_prompts,
                    )
                )
            cN_elapsed = max(1e-9, self.clock() - cN_start)
            for result in cN_results:
                flags.extend(_output_flags(result))
            total_tokens = sum(r.completion_tokens for r in cN_results)
            metrics[f"c{concurrency}_aggregate_tok_s"] = total_tokens / cN_elapsed
            metrics[f"c{concurrency}_per_stream_tok_s"] = (
                total_tokens / cN_elapsed / concurrency
            )

            # Cold prefill, escalado à VRAM disponível quando conhecida.
            contexts = list(config.prefill_contexts)
            if config.kv_bytes_per_token and before is not None:
                free_mb = _numeric(getattr(before, "free_vram_mb", None)) or 0.0
                cap_mb = min(free_mb, config.memory_cap_mb or free_mb)
                available_tokens = (cap_mb - config.runtime_reserve_mb) * MIB / config.kv_bytes_per_token
                contexts = [c for c in contexts if c <= available_tokens]
            if config.max_context_tokens is not None:
                contexts = [c for c in contexts if c <= config.max_context_tokens]
            for context in contexts:
                prompt = _sized_prompt(
                    context, f"cold-prefill-{context}-{uuid.uuid4().hex}"
                )
                result = self._chat(client, prompt, max_tokens=1)
                flags.extend(_output_flags(result))
                key = f"prefill_{context // 1024}k_tok_s"
                metrics[key] = result.prefill_tok_s or 0.0

            # Needle e full-KV soak.
            if config.needle_enabled:
                needle_passed, needle_flags = self._run_needle(config, client)
                flags.extend(needle_flags)
            if config.soak_enabled:
                soak_passed, soak_flags = self._run_soak(config, client)
                flags.extend(soak_flags)

            if watchdog is not None:
                try:
                    watchdog.sample_once()
                except MemoryCapExceeded:
                    raise
                except Exception:
                    if config.memory_cap_mb is not None:
                        raise
        except MemoryCapExceeded:
            flags.append(FLAG_MEM_CAP)
            error = "memory cap exceeded"
        except MissingUsageError as exc:
            flags.append(FLAG_MEM_CAP if watchdog and watchdog.tripped else FLAG_REQ_FAIL)
            error = str(exc)
        except Exception as exc:
            flags.append(FLAG_MEM_CAP if watchdog and watchdog.tripped else FLAG_REQ_FAIL)
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if watchdog is not None:
                watchdog.close()
                if watchdog.tripped and FLAG_MEM_CAP not in flags:
                    flags.append(FLAG_MEM_CAP)
                    error = error or "memory cap exceeded"
                if watchdog.error is not None:
                    flags.append(FLAG_REQ_FAIL)
                    error = error or f"memory sampler failed: {watchdog.error}"
            try:
                after = self.monitor.get_status() if self.monitor else None
            except Exception:
                after = None
            if watchdog is not None:
                if watchdog.last_bytes:
                    metrics["memory_mb"] = watchdog.last_bytes / MIB
                if watchdog.peak_bytes:
                    metrics["peak_memory_mb"] = watchdog.peak_bytes / MIB
            stop_managed()
            client.close()
            if stop_error:
                flags.append(FLAG_REQ_FAIL)
                error = error or stop_error

        flags.extend(_telemetry_flags(before, after, config))
        seen: set[str] = set()
        unique_flags = [f for f in flags if not (f in seen or seen.add(f))]
        ok = not unique_flags and needle_passed and soak_passed
        return BenchmarkResult(
            name=config.name,
            ok=ok,
            metrics=metrics,
            flags=unique_flags,
            needle_passed=needle_passed,
            soak_passed=soak_passed,
            memory_mb=metrics.get("memory_mb"),
            peak_memory_mb=metrics.get("peak_memory_mb"),
            error=error,
        )
