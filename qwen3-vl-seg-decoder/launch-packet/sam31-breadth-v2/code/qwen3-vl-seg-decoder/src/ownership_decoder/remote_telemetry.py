from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


GIB = 1024**3
MODEL_BASELINE_EVENTS = frozenset(
    {"predictor_loaded", "sam31_loaded", "sam3_image_loaded"}
)
CLEANUP_EVENTS = frozenset(
    {"session_closed", "image_agreement_frame_completed"}
)


class RuntimeWorkerRestartRequired(RuntimeError):
    """The current model process must exit so a fresh worker can resume."""


class RuntimeFatalSafetyError(RuntimeError):
    """The remote campaign must terminate without automatic replacement."""


@dataclass(frozen=True)
class RuntimeUsageSnapshot:
    unix_time: float
    host_available_bytes: int
    workspace_free_bytes: int
    gpu_used_bytes: int
    gpu_total_bytes: int
    gpu_temperature_c: int
    gpu_utilization_percent: int

    @property
    def gpu_used_fraction(self) -> float:
        if self.gpu_total_bytes <= 0:
            raise RuntimeFatalSafetyError("GPU telemetry reported non-positive capacity")
        return self.gpu_used_bytes / self.gpu_total_bytes

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["gpu_used_fraction"] = self.gpu_used_fraction
        return value


@dataclass(frozen=True)
class RuntimeSafetyPolicy:
    min_host_available_bytes: int = 16 * GIB
    min_workspace_free_bytes: int = 20 * GIB
    max_gpu_used_fraction: float = 0.90
    gpu_growth_warning_bytes: int = 1 * GIB
    gpu_growth_restart_bytes: int = 2 * GIB
    warning_persistence_items: int = 3
    restart_persistence_items: int = 5

    def __post_init__(self) -> None:
        if min(
            self.min_host_available_bytes,
            self.min_workspace_free_bytes,
            self.gpu_growth_warning_bytes,
            self.gpu_growth_restart_bytes,
            self.warning_persistence_items,
            self.restart_persistence_items,
        ) < 1:
            raise ValueError("runtime safety thresholds must be positive")
        if not 0.0 < self.max_gpu_used_fraction < 1.0:
            raise ValueError("max_gpu_used_fraction must be between zero and one")
        if self.gpu_growth_warning_bytes > self.gpu_growth_restart_bytes:
            raise ValueError("GPU growth warning cannot exceed the restart threshold")
        if self.warning_persistence_items > self.restart_persistence_items:
            raise ValueError("warning persistence cannot exceed restart persistence")


def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeFatalSafetyError("MemAvailable is absent from /proc/meminfo")


def probe_runtime_usage(workspace_path: Path) -> RuntimeUsageSnapshot:
    """Sample usage through nvidia-smi without importing Torch."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeFatalSafetyError("nvidia-smi runtime telemetry failed") from error
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeFatalSafetyError(
            f"runtime telemetry requires exactly one GPU; observed {len(lines)}"
        )
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4:
        raise RuntimeFatalSafetyError("nvidia-smi runtime telemetry schema is invalid")
    try:
        used_mib, total_mib, temperature, utilization = (int(value) for value in fields)
    except ValueError as error:
        raise RuntimeFatalSafetyError("nvidia-smi runtime telemetry values are invalid") from error
    mib = 1024**2
    return RuntimeUsageSnapshot(
        unix_time=time.time(),
        host_available_bytes=_mem_available_bytes(),
        workspace_free_bytes=shutil.disk_usage(workspace_path).free,
        gpu_used_bytes=used_mib * mib,
        gpu_total_bytes=total_mib * mib,
        gpu_temperature_c=temperature,
        gpu_utilization_percent=utilization,
    )


def _append_durable_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RuntimeTelemetryWriter:
    """Durably record progress and turn resource trends into fail-closed actions."""

    def __init__(
        self,
        path: str | Path,
        *,
        workspace_path: str | Path | None = None,
        policy: RuntimeSafetyPolicy = RuntimeSafetyPolicy(),
        sample_fn: Callable[[Path], RuntimeUsageSnapshot] = probe_runtime_usage,
    ) -> None:
        self.path = Path(path)
        self.workspace_path = Path(workspace_path) if workspace_path else self.path.parent
        self.policy = policy
        self.sample_fn = sample_fn
        self._gpu_baseline_bytes: int | None = None
        self._persistent_warning_count = 0
        self._persistent_restart_count = 0

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(event, dict) or not str(event.get("event", "")).strip():
            raise ValueError("telemetry event requires a non-empty event name")
        if any(key in event for key in ("usage", "safety_action")):
            raise ValueError("telemetry event contains a reserved field")
        snapshot = self.sample_fn(self.workspace_path)
        if event["event"] in MODEL_BASELINE_EVENTS:
            self._gpu_baseline_bytes = snapshot.gpu_used_bytes
            self._persistent_warning_count = 0
            self._persistent_restart_count = 0
        delta = (
            None
            if self._gpu_baseline_bytes is None
            else snapshot.gpu_used_bytes - self._gpu_baseline_bytes
        )
        record = {
            **event,
            "usage": snapshot.as_dict(),
            "gpu_baseline_bytes": self._gpu_baseline_bytes,
            "gpu_delta_from_baseline_bytes": delta,
        }

        error: RuntimeError | None = None
        if snapshot.workspace_free_bytes < self.policy.min_workspace_free_bytes:
            record["safety_action"] = "terminate"
            error = RuntimeFatalSafetyError("workspace disk is below the safety minimum")
        elif snapshot.host_available_bytes < self.policy.min_host_available_bytes:
            record["safety_action"] = "restart_worker"
            error = RuntimeWorkerRestartRequired("host RAM is below the safety minimum")
        elif snapshot.gpu_used_fraction >= self.policy.max_gpu_used_fraction:
            record["safety_action"] = "restart_worker"
            error = RuntimeWorkerRestartRequired("VRAM usage reached the safety ceiling")

        if event["event"] in CLEANUP_EVENTS and delta is not None:
            if delta >= self.policy.gpu_growth_warning_bytes:
                self._persistent_warning_count += 1
            else:
                self._persistent_warning_count = 0
            if delta >= self.policy.gpu_growth_restart_bytes:
                self._persistent_restart_count += 1
            else:
                self._persistent_restart_count = 0
            record["persistent_warning_count"] = self._persistent_warning_count
            record["persistent_restart_count"] = self._persistent_restart_count
            if self._persistent_warning_count >= self.policy.warning_persistence_items:
                record["safety_warning"] = "persistent_gpu_growth"
            if self._persistent_restart_count >= self.policy.restart_persistence_items:
                record["safety_action"] = "restart_worker"
                error = RuntimeWorkerRestartRequired(
                    "persistent post-cleanup GPU growth requires a fresh worker"
                )

        _append_durable_json(self.path, record)
        if error is not None:
            raise error
        return record
