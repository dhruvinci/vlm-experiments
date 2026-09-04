from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .remote_telemetry import RuntimeUsageSnapshot, probe_runtime_usage


RESTART_WORKER_EXIT_CODE = 75


class RemoteSupervisorError(RuntimeError):
    """The remote worker could not finish safely."""


@dataclass(frozen=True)
class SupervisorPolicy:
    max_restarts: int = 2
    poll_interval_seconds: float = 30.0
    max_runtime_seconds: float = 8 * 60 * 60
    terminate_grace_seconds: float = 20.0
    min_host_available_bytes: int = 16 * 1024**3
    min_workspace_free_bytes: int = 20 * 1024**3
    max_gpu_used_fraction: float = 0.90
    max_gpu_temperature_c: int = 90

    def __post_init__(self) -> None:
        if self.max_restarts < 0:
            raise ValueError("max_restarts cannot be negative")
        if self.poll_interval_seconds <= 0 or self.max_runtime_seconds <= 0:
            raise ValueError("supervisor polling and runtime limits must be positive")
        if self.terminate_grace_seconds < 0:
            raise ValueError("terminate_grace_seconds cannot be negative")
        if self.min_host_available_bytes < 1 or self.min_workspace_free_bytes < 1:
            raise ValueError("supervisor host and disk minima must be positive")
        if not 0.0 < self.max_gpu_used_fraction < 1.0:
            raise ValueError("supervisor GPU usage ceiling must be between zero and one")
        if self.max_gpu_temperature_c < 1:
            raise ValueError("supervisor GPU temperature ceiling must be positive")


@dataclass(frozen=True)
class SupervisorResult:
    attempt_count: int
    restart_count: int
    returncode: int
    elapsed_seconds: float


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _resource_action(
    usage: RuntimeUsageSnapshot,
    policy: SupervisorPolicy,
) -> tuple[str, str] | None:
    if usage.workspace_free_bytes < policy.min_workspace_free_bytes:
        return "fatal", "workspace_disk"
    if usage.host_available_bytes < policy.min_host_available_bytes:
        return "restart", "host_ram"
    if usage.gpu_used_fraction >= policy.max_gpu_used_fraction:
        return "restart", "vram"
    if usage.gpu_temperature_c >= policy.max_gpu_temperature_c:
        return "restart", "gpu_temperature"
    return None


def supervise_worker(
    worker_command: Sequence[str],
    *,
    output_root: str | Path,
    policy: SupervisorPolicy = SupervisorPolicy(),
    usage_probe: Callable[[Path], RuntimeUsageSnapshot] = probe_runtime_usage,
) -> SupervisorResult:
    """Run a model worker in a replaceable process with bounded retry and wall time."""

    if not worker_command:
        raise ValueError("worker command cannot be empty")
    if "--attempt-index" in worker_command:
        raise ValueError("supervisor owns --attempt-index")
    output = Path(output_root)
    supervisor_dir = output / "_supervisor"
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    events_path = supervisor_dir / "events.jsonl"
    heartbeat_path = supervisor_dir / "heartbeat.json"
    started = time.monotonic()
    restarts = 0
    attempts = 0
    last_usage: RuntimeUsageSnapshot | None = None

    while True:
        attempt_index = attempts
        attempts += 1
        command = [*worker_command, "--attempt-index", str(attempt_index)]
        log_path = supervisor_dir / f"worker_attempt_{attempt_index:02d}.log"
        resource_restart = False
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            elapsed = time.monotonic() - started
            _append_event(
                events_path,
                {
                    "event": "worker_started",
                    "at": datetime.now(timezone.utc).isoformat(),
                    "attempt_index": attempt_index,
                    "pid": process.pid,
                    "elapsed_seconds": elapsed,
                },
            )
            _atomic_json(
                heartbeat_path,
                {
                    "state": "running",
                    "attempt_index": attempt_index,
                    "pid": process.pid,
                    "elapsed_seconds": elapsed,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            while True:
                elapsed = time.monotonic() - started
                remaining = policy.max_runtime_seconds - elapsed
                if remaining <= 0:
                    _terminate_process_group(process, policy.terminate_grace_seconds)
                    _append_event(
                        events_path,
                        {
                            "event": "worker_runtime_ceiling",
                            "attempt_index": attempt_index,
                            "elapsed_seconds": elapsed,
                        },
                    )
                    _atomic_json(
                        heartbeat_path,
                        {
                            "state": "terminated_runtime_ceiling",
                            "attempt_index": attempt_index,
                            "elapsed_seconds": elapsed,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    raise RemoteSupervisorError("remote worker exceeded the runtime ceiling")
                try:
                    returncode = process.wait(
                        timeout=min(policy.poll_interval_seconds, remaining)
                    )
                    break
                except subprocess.TimeoutExpired:
                    try:
                        last_usage = usage_probe(output)
                    except Exception as error:
                        _terminate_process_group(process, policy.terminate_grace_seconds)
                        _append_event(
                            events_path,
                            {
                                "event": "worker_resource_fatal",
                                "attempt_index": attempt_index,
                                "reason": "telemetry_failure",
                                "message": str(error),
                            },
                        )
                        raise RemoteSupervisorError(
                            f"parent watchdog telemetry failed: {error}"
                        ) from error
                    action = _resource_action(last_usage, policy)
                    _atomic_json(
                        heartbeat_path,
                        {
                            "state": "running",
                            "attempt_index": attempt_index,
                            "pid": process.pid,
                            "elapsed_seconds": time.monotonic() - started,
                            "usage": last_usage.as_dict(),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    if action is not None:
                        disposition, reason = action
                        _terminate_process_group(process, policy.terminate_grace_seconds)
                        if disposition == "fatal":
                            _append_event(
                                events_path,
                                {
                                    "event": "worker_resource_fatal",
                                    "attempt_index": attempt_index,
                                    "reason": reason,
                                    "usage": last_usage.as_dict(),
                                },
                            )
                            raise RemoteSupervisorError(
                                f"parent watchdog fatal resource breach: {reason}"
                            )
                        if restarts >= policy.max_restarts:
                            _append_event(
                                events_path,
                                {
                                    "event": "worker_fatal",
                                    "attempt_index": attempt_index,
                                    "reason": f"resource_{reason}",
                                    "restart_count": restarts,
                                    "usage": last_usage.as_dict(),
                                },
                            )
                            raise RemoteSupervisorError(
                                f"worker exhausted restarts after resource breach: {reason}"
                            )
                        restarts += 1
                        _append_event(
                            events_path,
                            {
                                "event": "worker_resource_restart",
                                "attempt_index": attempt_index,
                                "reason": reason,
                                "restart_count": restarts,
                                "usage": last_usage.as_dict(),
                            },
                        )
                        resource_restart = True
                        break

        if resource_restart:
            continue

        elapsed = time.monotonic() - started
        try:
            last_usage = usage_probe(output)
        except Exception:
            # The worker already exited; its own durable telemetry and exit code remain
            # authoritative. The next live polling cycle will retry the parent probe.
            pass
        if returncode == 0:
            if not (output / "RUN_COMPLETE").is_file():
                _append_event(
                    events_path,
                    {
                        "event": "worker_fatal",
                        "attempt_index": attempt_index,
                        "returncode": returncode,
                        "reason": "missing_RUN_COMPLETE",
                    },
                )
                raise RemoteSupervisorError(
                    "worker exited zero without the campaign RUN_COMPLETE sentinel"
                )
            _append_event(
                events_path,
                {
                    "event": "supervisor_complete",
                    "attempt_index": attempt_index,
                    "returncode": returncode,
                    "elapsed_seconds": elapsed,
                },
            )
            _atomic_json(
                heartbeat_path,
                {
                    "state": "complete",
                    "attempt_index": attempt_index,
                    "elapsed_seconds": elapsed,
                    "usage": last_usage.as_dict() if last_usage is not None else None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return SupervisorResult(attempts, restarts, returncode, elapsed)

        if returncode == RESTART_WORKER_EXIT_CODE and restarts < policy.max_restarts:
            restarts += 1
            _append_event(
                events_path,
                {
                    "event": "worker_restart",
                    "attempt_index": attempt_index,
                    "returncode": returncode,
                    "restart_count": restarts,
                    "elapsed_seconds": elapsed,
                },
            )
            continue

        _append_event(
            events_path,
            {
                "event": "worker_fatal",
                "attempt_index": attempt_index,
                "returncode": returncode,
                "restart_count": restarts,
                "elapsed_seconds": elapsed,
            },
        )
        _atomic_json(
            heartbeat_path,
            {
                "state": "fatal",
                "attempt_index": attempt_index,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if returncode == RESTART_WORKER_EXIT_CODE:
            raise RemoteSupervisorError(
                f"worker exhausted {policy.max_restarts} permitted restarts"
            )
        raise RemoteSupervisorError(f"worker failed with non-retry exit code {returncode}")
