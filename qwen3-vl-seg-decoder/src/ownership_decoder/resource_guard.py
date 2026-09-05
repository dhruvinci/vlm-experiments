from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


class GuardedRunInterrupted(RuntimeError):
    """Raised after a termination signal is converted into kill-safe cleanup."""

    def __init__(self, signum: int):
        super().__init__(f"guarded run interrupted by signal {signum}")
        self.signum = signum


@contextmanager
def _termination_signals_as_exceptions():
    """Let the monitor clean up its process group before honoring termination."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}

    def raise_interrupted(signum, _frame) -> None:
        raise GuardedRunInterrupted(signum)

    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, raise_interrupted)
    try:
        yield
    finally:
        for handled_signal, handler in previous.items():
            signal.signal(handled_signal, handler)


@dataclass(frozen=True)
class ResourceSnapshot:
    host_available_bytes: int
    swap_free_bytes: int
    gpu_free_bytes: int | None
    gpu_total_bytes: int | None

    @property
    def gpu_used_fraction(self) -> float | None:
        if self.gpu_free_bytes is None or self.gpu_total_bytes in (None, 0):
            return None
        return 1.0 - (self.gpu_free_bytes / self.gpu_total_bytes)


@dataclass(frozen=True)
class ResourceLimits:
    min_host_available_bytes: int = 0
    min_swap_free_bytes: int = 0
    min_gpu_free_bytes: int = 0
    max_gpu_used_fraction: float = 1.0

    def __post_init__(self) -> None:
        if min(
            self.min_host_available_bytes,
            self.min_swap_free_bytes,
            self.min_gpu_free_bytes,
        ) < 0:
            raise ValueError("resource byte limits cannot be negative")
        if not 0.0 <= self.max_gpu_used_fraction <= 1.0:
            raise ValueError("max_gpu_used_fraction must be in [0, 1]")


@dataclass(frozen=True)
class ResourceViolation:
    resource: str
    observed: float | int | None
    limit: float | int
    comparison: str


@dataclass(frozen=True)
class GuardedRunResult:
    started: bool
    killed_for_limit: bool
    returncode: int | None
    violations: tuple[ResourceViolation, ...]
    minimum_host_available_bytes: int
    minimum_gpu_free_bytes: int | None
    maximum_gpu_used_fraction: float | None
    elapsed_seconds: float
    termination_reason: str


def assess_snapshot(
    snapshot: ResourceSnapshot,
    limits: ResourceLimits,
) -> tuple[ResourceViolation, ...]:
    violations: list[ResourceViolation] = []
    lower_bounds = (
        (
            "host_available_bytes",
            snapshot.host_available_bytes,
            limits.min_host_available_bytes,
        ),
        ("swap_free_bytes", snapshot.swap_free_bytes, limits.min_swap_free_bytes),
    )
    for resource, observed, limit in lower_bounds:
        if observed < limit:
            violations.append(ResourceViolation(resource, observed, limit, ">="))

    needs_gpu_telemetry = (
        limits.min_gpu_free_bytes > 0 or limits.max_gpu_used_fraction < 1.0
    )
    if needs_gpu_telemetry and (
        snapshot.gpu_free_bytes is None or snapshot.gpu_total_bytes in (None, 0)
    ):
        violations.append(ResourceViolation("gpu_telemetry", None, 1, "available"))
        return tuple(violations)

    if (
        snapshot.gpu_free_bytes is not None
        and snapshot.gpu_free_bytes < limits.min_gpu_free_bytes
    ):
        violations.append(
            ResourceViolation(
                "gpu_free_bytes",
                snapshot.gpu_free_bytes,
                limits.min_gpu_free_bytes,
                ">=",
            )
        )
    used_fraction = snapshot.gpu_used_fraction
    if used_fraction is not None and used_fraction > limits.max_gpu_used_fraction:
        violations.append(
            ResourceViolation(
                "gpu_used_fraction",
                used_fraction,
                limits.max_gpu_used_fraction,
                "<=",
            )
        )
    return tuple(violations)


def _read_meminfo(path: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, raw = line.split(":", 1)
        pieces = raw.strip().split()
        if pieces:
            values[key] = int(pieces[0]) * 1024
    if "MemAvailable" not in values or "SwapFree" not in values:
        raise RuntimeError("/proc/meminfo is missing MemAvailable or SwapFree")
    return values["MemAvailable"], values["SwapFree"]


def _read_nvidia_memory() -> tuple[int | None, int | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None
    first_gpu = completed.stdout.strip().splitlines()[0].split(",")
    if len(first_gpu) != 2:
        return None, None
    mib = 1024**2
    return int(first_gpu[0].strip()) * mib, int(first_gpu[1].strip()) * mib


def sample_resources() -> ResourceSnapshot:
    host_available, swap_free = _read_meminfo()
    gpu_free, gpu_total = _read_nvidia_memory()
    return ResourceSnapshot(
        host_available_bytes=host_available,
        swap_free_bytes=swap_free,
        gpu_free_bytes=gpu_free,
        gpu_total_bytes=gpu_total,
    )


def _telemetry_record(
    event: str,
    snapshot: ResourceSnapshot,
    violations: Sequence[ResourceViolation] = (),
) -> dict:
    return {
        "event": event,
        "unix_time": time.time(),
        "snapshot": asdict(snapshot),
        "violations": [asdict(violation) for violation in violations],
    }


def _terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float) -> int:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait()
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def systemd_scoped_command(
    command: Sequence[str],
    *,
    memory_max_bytes: int,
    systemd_run_path: str | None = None,
) -> list[str]:
    """Wrap a command in a kernel-enforced per-job cgroup memory ceiling."""

    if not command:
        raise ValueError("scoped command cannot be empty")
    if memory_max_bytes < 1:
        raise ValueError("memory_max_bytes must be positive")
    executable = systemd_run_path or shutil.which("systemd-run")
    if executable is None:
        raise RuntimeError("systemd-run is required for a hard child memory cap")
    memory_high_bytes = memory_max_bytes * 9 // 10
    return [
        executable,
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryHigh={memory_high_bytes}",
        "-p",
        f"MemoryMax={memory_max_bytes}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "TasksMax=128",
        "-p",
        "KillMode=control-group",
        "-p",
        "OOMPolicy=kill",
        "--",
        *command,
    ]


def run_guarded(
    command: Sequence[str],
    *,
    limits: ResourceLimits,
    log_path: Path,
    telemetry_path: Path,
    sample_fn: Callable[[], ResourceSnapshot] = sample_resources,
    poll_interval_seconds: float = 0.25,
    terminate_grace_seconds: float = 2.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    child_memory_max_bytes: int | None = None,
    maximum_runtime_seconds: float | None = None,
) -> GuardedRunResult:
    """Run a child under a parent-side RAM/VRAM circuit breaker.

    Child output is streamed directly to disk so the supervising process never retains
    an unbounded output buffer. A breached limit terminates the child's complete process
    group, first with SIGTERM and then SIGKILL after the configured grace period.
    """

    if not command:
        raise ValueError("guarded command cannot be empty")
    if poll_interval_seconds <= 0 or terminate_grace_seconds < 0:
        raise ValueError("poll interval must be positive and grace period non-negative")
    if maximum_runtime_seconds is not None and maximum_runtime_seconds <= 0:
        raise ValueError("maximum runtime must be positive when supplied")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    preflight = sample_fn()
    violations = assess_snapshot(preflight, limits)
    with telemetry_path.open("a", encoding="utf-8", buffering=1) as telemetry:
        telemetry.write(json.dumps(_telemetry_record("preflight", preflight, violations)) + "\n")
        if violations:
            return GuardedRunResult(
                started=False,
                killed_for_limit=False,
                returncode=None,
                violations=violations,
                minimum_host_available_bytes=preflight.host_available_bytes,
                minimum_gpu_free_bytes=preflight.gpu_free_bytes,
                maximum_gpu_used_fraction=preflight.gpu_used_fraction,
                elapsed_seconds=0.0,
                termination_reason="preflight_limit",
            )

        minimum_host = preflight.host_available_bytes
        minimum_gpu = preflight.gpu_free_bytes
        maximum_gpu_fraction = preflight.gpu_used_fraction
        launch_command = list(command)
        if child_memory_max_bytes is not None:
            launch_command = systemd_scoped_command(
                launch_command,
                memory_max_bytes=child_memory_max_bytes,
            )
        with log_path.open("ab", buffering=0) as log_file:
            started_at = time.monotonic()
            process = subprocess.Popen(
                launch_command,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            last_snapshot = preflight
            try:
                with _termination_signals_as_exceptions():
                    while True:
                        returncode = process.poll()
                        if returncode is not None:
                            elapsed = time.monotonic() - started_at
                            completed = _telemetry_record("completed", last_snapshot)
                            completed.update(
                                {
                                    "elapsed_seconds": elapsed,
                                    "returncode": returncode,
                                }
                            )
                            telemetry.write(json.dumps(completed) + "\n")
                            return GuardedRunResult(
                                started=True,
                                killed_for_limit=False,
                                returncode=returncode,
                                violations=(),
                                minimum_host_available_bytes=minimum_host,
                                minimum_gpu_free_bytes=minimum_gpu,
                                maximum_gpu_used_fraction=maximum_gpu_fraction,
                                elapsed_seconds=elapsed,
                                termination_reason=(
                                    "completed" if returncode == 0 else "child_exit"
                                ),
                            )

                        current = sample_fn()
                        last_snapshot = current
                        elapsed = time.monotonic() - started_at
                        minimum_host = min(minimum_host, current.host_available_bytes)
                        if current.gpu_free_bytes is not None:
                            minimum_gpu = (
                                current.gpu_free_bytes
                                if minimum_gpu is None
                                else min(minimum_gpu, current.gpu_free_bytes)
                            )
                        current_fraction = current.gpu_used_fraction
                        if current_fraction is not None:
                            maximum_gpu_fraction = (
                                current_fraction
                                if maximum_gpu_fraction is None
                                else max(maximum_gpu_fraction, current_fraction)
                            )
                        violations = assess_snapshot(current, limits)
                        if (
                            not violations
                            and maximum_runtime_seconds is not None
                            and elapsed >= maximum_runtime_seconds
                        ):
                            violations = (
                                ResourceViolation(
                                    "runtime_seconds",
                                    elapsed,
                                    maximum_runtime_seconds,
                                    "<=",
                                ),
                            )
                        event = (
                            "runtime_limit"
                            if violations and violations[0].resource == "runtime_seconds"
                            else "limit_breach"
                            if violations
                            else "sample"
                        )
                        record = _telemetry_record(event, current, violations)
                        record["elapsed_seconds"] = elapsed
                        telemetry.write(json.dumps(record) + "\n")
                        if violations:
                            returncode = _terminate_process_group(
                                process,
                                terminate_grace_seconds,
                            )
                            return GuardedRunResult(
                                started=True,
                                killed_for_limit=True,
                                returncode=returncode,
                                violations=violations,
                                minimum_host_available_bytes=minimum_host,
                                minimum_gpu_free_bytes=minimum_gpu,
                                maximum_gpu_used_fraction=maximum_gpu_fraction,
                                elapsed_seconds=time.monotonic() - started_at,
                                termination_reason=(
                                    "runtime_limit"
                                    if violations[0].resource == "runtime_seconds"
                                    else "resource_limit"
                                ),
                            )
                        time.sleep(poll_interval_seconds)
            except BaseException:
                returncode = process.poll()
                if returncode is None:
                    returncode = _terminate_process_group(
                        process,
                        terminate_grace_seconds,
                    )
                interrupted = _telemetry_record("interrupted", last_snapshot)
                interrupted.update(
                    {
                        "elapsed_seconds": time.monotonic() - started_at,
                        "returncode": returncode,
                    }
                )
                telemetry.write(json.dumps(interrupted) + "\n")
                raise
