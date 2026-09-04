from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REST_BASE = "https://rest.runpod.io/v1"


@dataclass(frozen=True)
class PodGuardContract:
    pod_id: str
    run_id: str
    billing_started_at: datetime
    spend_before_usd: float
    frozen_hourly_rate_usd: float
    terminate_at_usd: float
    max_runtime_seconds: float
    packet_sha256: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.pod_id):
            raise ValueError("Pod ID contains unsupported characters")
        if not self.run_id.strip():
            raise ValueError("run ID cannot be empty")
        if self.billing_started_at.tzinfo is None:
            raise ValueError("billing start must include a timezone")
        numbers = (
            self.spend_before_usd,
            self.frozen_hourly_rate_usd,
            self.terminate_at_usd,
            self.max_runtime_seconds,
        )
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Pod guard numeric values must be finite")
        if self.spend_before_usd < 0 or self.frozen_hourly_rate_usd <= 0:
            raise ValueError("Pod guard spend and rate are invalid")
        if self.terminate_at_usd <= self.spend_before_usd:
            raise ValueError("Pod guard termination threshold must exceed prior spend")
        if self.max_runtime_seconds <= 0:
            raise ValueError("Pod guard runtime ceiling must be positive")
        if not re.fullmatch(r"[0-9a-f]{64}", self.packet_sha256):
            raise ValueError("packet SHA-256 is invalid")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PodGuardContract":
        expected = {
            "pod_id",
            "run_id",
            "billing_started_at",
            "spend_before_usd",
            "frozen_hourly_rate_usd",
            "terminate_at_usd",
            "max_runtime_seconds",
            "packet_sha256",
        }
        if set(value) != expected:
            raise ValueError("Pod guard control keys are invalid")
        return cls(
            pod_id=str(value["pod_id"]),
            run_id=str(value["run_id"]),
            billing_started_at=datetime.fromisoformat(str(value["billing_started_at"])),
            spend_before_usd=float(value["spend_before_usd"]),
            frozen_hourly_rate_usd=float(value["frozen_hourly_rate_usd"]),
            terminate_at_usd=float(value["terminate_at_usd"]),
            max_runtime_seconds=float(value["max_runtime_seconds"]),
            packet_sha256=str(value["packet_sha256"]),
        )


def guard_decision(
    contract: PodGuardContract,
    *,
    now: datetime,
    reported_hourly_rate_usd: float,
    run_complete: bool,
    fatal: bool,
) -> tuple[str, dict[str, Any]]:
    if now.tzinfo is None:
        raise ValueError("Pod guard current time must include a timezone")
    rate = max(contract.frozen_hourly_rate_usd, reported_hourly_rate_usd)
    elapsed_seconds = max(
        0.0,
        (now - contract.billing_started_at).total_seconds(),
    )
    spend = contract.spend_before_usd + rate * elapsed_seconds / 3600.0
    if run_complete:
        decision = "terminate_success"
    elif fatal:
        decision = "terminate_fatal"
    elif spend >= contract.terminate_at_usd:
        decision = "terminate_budget"
    elif elapsed_seconds >= contract.max_runtime_seconds:
        decision = "terminate_time"
    else:
        decision = "continue"
    return decision, {
        "at": now.isoformat(),
        "pod_id": contract.pod_id,
        "run_id": contract.run_id,
        "elapsed_seconds": elapsed_seconds,
        "hourly_rate_usd": rate,
        "spend_before_usd": contract.spend_before_usd,
        "cumulative_gpu_usd_estimate": spend,
        "terminate_at_usd": contract.terminate_at_usd,
        "run_complete": run_complete,
        "fatal": fatal,
        "decision": decision,
    }


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


def run_pod_guard(
    contract: PodGuardContract,
    *,
    run_root: str | Path,
    heartbeat_path: str | Path,
    get_pod_fn: Callable[[str], dict[str, Any]],
    terminate_fn: Callable[[str], None],
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep_fn: Callable[[float], None] = time.sleep,
    interval_seconds: float = 30.0,
) -> str:
    if interval_seconds <= 0:
        raise ValueError("Pod guard interval must be positive")
    output = Path(run_root)
    heartbeat = Path(heartbeat_path)
    while True:
        pod_probe_error: str | None = None
        try:
            pod = get_pod_fn(contract.pod_id)
            reported_rate = float(
                pod.get("adjustedCostPerHr") or pod.get("costPerHr") or 0.0
            )
        except Exception as error:
            reported_rate = 0.0
            pod_probe_error = f"{type(error).__name__}: {error}"[:500]
        decision, snapshot = guard_decision(
            contract,
            now=now_fn(),
            reported_hourly_rate_usd=reported_rate,
            run_complete=(output / "RUN_COMPLETE").is_file(),
            fatal=(output / "RUN_FATAL").is_file(),
        )
        if pod_probe_error is not None:
            snapshot["pod_probe_error"] = pod_probe_error
        _atomic_json(heartbeat, snapshot)
        if decision != "continue":
            termination_attempt = 0
            while True:
                termination_attempt += 1
                try:
                    terminate_fn(contract.pod_id)
                except Exception as error:
                    snapshot["termination_attempt"] = termination_attempt
                    snapshot["termination_error"] = (
                        f"{type(error).__name__}: {error}"[:500]
                    )
                    _atomic_json(heartbeat, snapshot)
                    sleep_fn(interval_seconds)
                    continue
                snapshot["termination_attempt"] = termination_attempt
                snapshot.pop("termination_error", None)
                _atomic_json(heartbeat, snapshot)
                return decision
        sleep_fn(interval_seconds)


def _api_functions(api_key: str) -> tuple[Callable[[str], dict[str, Any]], Callable[[str], None]]:
    def request(method: str, pod_id: str) -> dict[str, Any]:
        encoded_id = urllib.parse.quote(pod_id, safe="")
        call = urllib.request.Request(
            f"{REST_BASE}/pods/{encoded_id}",
            method=method,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(call, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else {}

    def get_pod(pod_id: str) -> dict[str, Any]:
        return request("GET", pod_id)

    def terminate(pod_id: str) -> None:
        try:
            request("DELETE", pod_id)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise

    return get_pod, terminate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent RunPod budget/deadline guard")
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--heartbeat", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise EnvironmentError("RUNPOD_API_KEY is required")
    try:
        control = json.loads(args.control.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Pod guard control is invalid: {args.control}") from error
    contract = PodGuardContract.from_mapping(control)
    get_pod, terminate = _api_functions(api_key)
    result = run_pod_guard(
        contract,
        run_root=args.run_root,
        heartbeat_path=args.heartbeat,
        get_pod_fn=get_pod,
        terminate_fn=terminate,
        interval_seconds=args.interval_seconds,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
