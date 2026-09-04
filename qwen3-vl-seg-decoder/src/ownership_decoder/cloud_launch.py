from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


FULL_GPU_TYPES = (
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
)
MIG_GPU_TYPE = "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"
REMOTE_PACKET_ROOT = "/workspace/qwen38-campaign/mask-campaign/v2/packet"
REMOTE_CONTROL_PREFIX = "qwen38-campaign/mask-campaign/v2/control"
REMOTE_RUN_PREFIX = "qwen38-campaign/mask-campaign/runs"
REMOTE_PREFLIGHT_RUN_PREFIX = "qwen38-campaign/mask-campaign/preflight-runs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _gpu_name(pod: dict[str, Any]) -> str:
    candidates = (
        pod.get("gpu", {}).get("displayName"),
        pod.get("machine", {}).get("gpuDisplayName"),
        pod.get("machine", {}).get("gpuTypeId"),
    )
    return next((str(value) for value in candidates if value), "")


def _reported_rate(pod: dict[str, Any]) -> float:
    try:
        value = float(pod.get("adjustedCostPerHr") or pod.get("costPerHr") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value >= 0 else 0.0


@dataclass(frozen=True)
class MaskCloudLaunchContract:
    """Exact user-approved mutation and billing scope for one Pod only."""

    run_id: str
    packet_manifest_path: Path
    packet_sha256: str
    approved_by: str
    approved_at: datetime
    network_volume_id: str
    datacenter: str
    container_image: str
    spend_before_usd: float
    frozen_hourly_rate_usd: float
    terminate_at_usd: float
    max_runtime_seconds: float
    purpose: str = "full"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{7,79}", self.run_id):
            raise ValueError("run ID must be a stable lowercase identifier")
        if not self.packet_manifest_path.is_file():
            raise FileNotFoundError(f"packet manifest is missing: {self.packet_manifest_path}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.packet_sha256):
            raise ValueError("packet SHA-256 is invalid")
        if not self.approved_by.strip() or self.approved_at.tzinfo is None:
            raise ValueError("explicit approval identity and timezone are required")
        if not self.network_volume_id.strip() or not re.fullmatch(
            r"[A-Z]{2}-[A-Z]{2}-[0-9]+", self.datacenter
        ):
            raise ValueError("network volume and datacenter are invalid")
        if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", self.container_image):
            raise ValueError("container image must be pinned by SHA-256 digest")
        values = (
            self.spend_before_usd,
            self.frozen_hourly_rate_usd,
            self.terminate_at_usd,
            self.max_runtime_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("launch billing values must be finite")
        if self.spend_before_usd < 0 or self.frozen_hourly_rate_usd <= 0:
            raise ValueError("launch prior spend and hourly rate are invalid")
        if self.terminate_at_usd <= self.spend_before_usd:
            raise ValueError("launch spend cap must exceed prior campaign spend")
        if not 0 < self.max_runtime_seconds <= 8 * 60 * 60:
            raise ValueError("launch runtime must be positive and at most eight hours")
        if self.purpose not in {"mig", "full"}:
            raise ValueError("launch purpose must be mig or full")

    def verify_packet(self) -> None:
        if _sha256(self.packet_manifest_path) != self.packet_sha256:
            raise RuntimeError("approved launch packet SHA-256 no longer matches")

    def pod_payload(self, *, api_key: str) -> dict[str, Any]:
        if not api_key:
            raise ValueError("RunPod API key is required")
        gpu_types = [MIG_GPU_TYPE] if self.purpose == "mig" else list(FULL_GPU_TYPES)
        entrypoint = "preflight.sh" if self.purpose == "mig" else "launch.sh"
        return {
            "name": f"sam31-{self.purpose}-{self.run_id[-12:]}",
            "cloudType": "SECURE",
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": gpu_types,
            "gpuTypePriority": "custom",
            "dataCenterIds": [self.datacenter],
            "dataCenterPriority": "custom",
            "allowedCudaVersions": ["13.0"],
            "networkVolumeId": self.network_volume_id,
            "volumeMountPath": "/workspace",
            "containerDiskInGb": 50,
            "imageName": self.container_image,
            "interruptible": False,
            "locked": False,
            "env": {
                "MASK_CAMPAIGN_RUN_ID": self.run_id,
                "RUNPOD_API_KEY": api_key,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "supportPublicIp": True,
            "ports": ["22/tcp"],
            "dockerStartCmd": ["/bin/bash", f"{REMOTE_PACKET_ROOT}/{entrypoint}"],
        }

    def control(
        self,
        *,
        pod_id: str,
        billing_started_at: datetime,
        observed_hourly_rate_usd: float,
    ) -> dict[str, Any]:
        rate = max(self.frozen_hourly_rate_usd, observed_hourly_rate_usd)
        return {
            "pod_id": pod_id,
            "run_id": self.run_id,
            "billing_started_at": billing_started_at.isoformat(),
            "spend_before_usd": self.spend_before_usd,
            "frozen_hourly_rate_usd": rate,
            "terminate_at_usd": self.terminate_at_usd,
            "max_runtime_seconds": self.max_runtime_seconds,
            "packet_sha256": self.packet_sha256,
        }


def _terminate(request_fn: Callable[..., dict[str, Any]], pod_id: str) -> None:
    request_fn("DELETE", f"/pods/{pod_id}", None)


def launch_mask_pod_once(
    contract: MaskCloudLaunchContract,
    *,
    api_key: str,
    request_fn: Callable[[str, str, dict[str, Any] | None], dict[str, Any]],
    put_control_fn: Callable[[str, dict[str, Any]], None],
    state_path: str | Path,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Create exactly one Pod, publish its immutable control, or delete it."""

    state = Path(state_path)
    if state.exists():
        raise FileExistsError("replacement Pod launch is prohibited by existing state")
    contract.verify_packet()
    billing_started_at = now_fn()
    if billing_started_at.tzinfo is None:
        raise ValueError("billing start must include a timezone")
    pod: dict[str, Any] | None = None
    try:
        pod = request_fn("POST", "/pods", contract.pod_payload(api_key=api_key))
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RuntimeError("RunPod did not return a Pod ID")
        pod_id = str(pod["id"])
        hardware = _gpu_name(pod)
        if hardware:
            allowed = (
                hardware == MIG_GPU_TYPE
                if contract.purpose == "mig"
                else hardware in FULL_GPU_TYPES and "Max-Q" not in hardware
            )
            if not allowed:
                raise RuntimeError(f"RunPod returned unapproved hardware: {hardware}")
        observed_rate = _reported_rate(pod)
        control = contract.control(
            pod_id=pod_id,
            billing_started_at=billing_started_at,
            observed_hourly_rate_usd=observed_rate,
        )
        control_key = f"{REMOTE_CONTROL_PREFIX}/{contract.run_id}.json"
        put_control_fn(control_key, control)
        result = {
            "format": "ownership-mask-cloud-launch-v1",
            "pod_id": pod_id,
            "run_id": contract.run_id,
            "purpose": contract.purpose,
            "billing_started_at": billing_started_at.isoformat(),
            "packet_sha256": contract.packet_sha256,
            "approved_by": contract.approved_by,
            "approved_at": contract.approved_at.isoformat(),
            "network_volume_id": contract.network_volume_id,
            "datacenter": contract.datacenter,
            "gpu_name_if_reported": hardware or None,
            "hourly_rate_usd": max(contract.frozen_hourly_rate_usd, observed_rate),
            "spend_before_usd": contract.spend_before_usd,
            "terminate_at_usd": contract.terminate_at_usd,
            "max_runtime_seconds": contract.max_runtime_seconds,
            "control_key": control_key,
        }
        _atomic_json(state, result)
        return result
    except Exception:
        if isinstance(pod, dict) and pod.get("id"):
            try:
                _terminate(request_fn, str(pod["id"]))
            except Exception:
                # Preserve the originating failure. The caller's outer emergency
                # terminator must continue retrying the known Pod ID.
                pass
        raise


def monitor_mask_pod(
    contract: MaskCloudLaunchContract,
    *,
    pod_id: str,
    billing_started_at: datetime,
    get_pod_fn: Callable[[str], dict[str, Any]],
    terminate_fn: Callable[[str], None],
    read_remote_json_fn: Callable[[str], dict[str, Any]],
    remote_exists_fn: Callable[[str], bool],
    local_log: str | Path,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep_fn: Callable[[float], None] = time.sleep,
    interval_seconds: float = 60.0,
    startup_heartbeat_grace_seconds: float = 600.0,
    heartbeat_max_age_seconds: float = 180.0,
) -> str:
    """Independent local one-minute monitor; it never provisions a replacement."""

    if billing_started_at.tzinfo is None:
        raise ValueError("billing start must include a timezone")
    if (
        interval_seconds <= 0
        or startup_heartbeat_grace_seconds <= 0
        or heartbeat_max_age_seconds <= 0
    ):
        raise ValueError("monitor intervals must be positive")
    log = Path(local_log)
    run_prefix = (
        REMOTE_PREFLIGHT_RUN_PREFIX if contract.purpose == "mig" else REMOTE_RUN_PREFIX
    )
    prefix = f"{run_prefix}/{contract.run_id}"
    guard_key = f"{prefix}/pod-guard-heartbeat.json"
    supervisor_key = f"{prefix}/_supervisor/heartbeat.json"
    while True:
        now = now_fn()
        if now.tzinfo is None:
            raise ValueError("monitor clock must include a timezone")
        elapsed = max(0.0, (now - billing_started_at).total_seconds())
        pod = get_pod_fn(pod_id)
        status = str(pod.get("desiredStatus") or pod.get("status") or "UNKNOWN")
        rate = max(contract.frozen_hourly_rate_usd, _reported_rate(pod))
        spend = contract.spend_before_usd + rate * elapsed / 3600.0
        heartbeats: dict[str, Any] = {}
        for name, key in (("guard", guard_key), ("supervisor", supervisor_key)):
            try:
                heartbeats[name] = read_remote_json_fn(key)
            except Exception as error:
                heartbeats[name] = {"unavailable": type(error).__name__}
        guard_age_seconds: float | None = None
        if "unavailable" not in heartbeats["guard"]:
            raw_guard_time = heartbeats["guard"].get("at") or heartbeats["guard"].get(
                "updated_at"
            )
            try:
                guard_time = datetime.fromisoformat(str(raw_guard_time))
                if guard_time.tzinfo is None:
                    raise ValueError("naive heartbeat timestamp")
                guard_age_seconds = max(0.0, (now - guard_time).total_seconds())
            except (TypeError, ValueError):
                guard_age_seconds = None
        complete = remote_exists_fn(f"{prefix}/RUN_COMPLETE")
        fatal = remote_exists_fn(f"{prefix}/RUN_FATAL")
        if status.upper() in {"EXITED", "TERMINATED"}:
            decision = status.lower()
        elif complete:
            decision = "terminate_success"
        elif fatal:
            decision = "terminate_fatal"
        elif spend >= contract.terminate_at_usd:
            decision = "terminate_budget"
        elif elapsed >= contract.max_runtime_seconds:
            decision = "terminate_time"
        elif (
            elapsed > startup_heartbeat_grace_seconds
            and "unavailable" in heartbeats["guard"]
        ):
            decision = "terminate_missing_guard"
        elif elapsed > startup_heartbeat_grace_seconds and guard_age_seconds is None:
            decision = "terminate_invalid_guard"
        elif (
            guard_age_seconds is not None
            and guard_age_seconds > heartbeat_max_age_seconds
        ):
            decision = "terminate_stale_guard"
        else:
            decision = "continue"
        _append_jsonl(
            log,
            {
                "at": now.isoformat(),
                "pod_id": pod_id,
                "run_id": contract.run_id,
                "pod_status": status,
                "elapsed_seconds": elapsed,
                "hourly_rate_usd": rate,
                "cumulative_gpu_usd_estimate": spend,
                "heartbeats": heartbeats,
                "guard_heartbeat_age_seconds": guard_age_seconds,
                "run_complete": complete,
                "run_fatal": fatal,
                "decision": decision,
            },
        )
        if decision.startswith("terminate_"):
            terminate_fn(pod_id)
            return decision
        if decision != "continue":
            return decision
        sleep_fn(interval_seconds)
