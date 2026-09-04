from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .remote_supervisor import SupervisorPolicy, SupervisorResult, supervise_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise a replaceable SAM3.1 worker with a 30-second watchdog."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=8 * 60 * 60)
    parser.add_argument("--terminate-grace-seconds", type=float, default=20.0)
    parser.add_argument("--pod-id")
    parser.add_argument("--spend-before-usd", type=float, default=0.0)
    parser.add_argument("--hourly-rate-usd", type=float, default=2.09)
    parser.add_argument("--terminate-at-usd", type=float, default=21.50)
    parser.add_argument("--billing-started-at")
    parser.add_argument("worker_command", nargs=argparse.REMAINDER)
    return parser


def _terminate_runpod(pod_id: str, api_key: str) -> None:
    encoded_id = urllib.parse.quote(pod_id, safe="")
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{encoded_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            pass
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise


def _remaining_budget_seconds(
    *,
    spend_before_usd: float,
    hourly_rate_usd: float,
    terminate_at_usd: float,
    billing_started_at: str,
    now: datetime,
) -> float:
    if spend_before_usd < 0 or hourly_rate_usd <= 0:
        raise ValueError("managed Pod spend and hourly rate must be positive")
    if terminate_at_usd <= spend_before_usd:
        raise RuntimeError("managed Pod budget threshold is already exhausted")
    started = datetime.fromisoformat(billing_started_at)
    if started.tzinfo is None or now.tzinfo is None:
        raise ValueError("managed Pod billing timestamps must include timezones")
    elapsed = max(0.0, (now - started).total_seconds())
    budget_seconds = (
        (terminate_at_usd - spend_before_usd) / hourly_rate_usd * 3600.0
    )
    remaining = budget_seconds - elapsed
    if remaining <= 0:
        raise RuntimeError("managed Pod budget threshold was reached before worker start")
    return remaining


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., SupervisorResult] = supervise_worker,
    environment: Mapping[str, str] | None = None,
    terminator: Callable[[str, str], None] = _terminate_runpod,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.worker_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("supervisor requires a worker command after --")
    managed = bool(args.pod_id)
    values = environment if environment is not None else os.environ
    api_key = values.get("RUNPOD_API_KEY") if managed else None
    if managed and not api_key:
        raise EnvironmentError("RUNPOD_API_KEY is required for managed Pod termination")
    if managed and not args.billing_started_at:
        raise ValueError("--billing-started-at is required with --pod-id")
    try:
        runtime_ceiling = args.max_runtime_seconds
        if managed:
            runtime_ceiling = min(
                runtime_ceiling,
                _remaining_budget_seconds(
                    spend_before_usd=args.spend_before_usd,
                    hourly_rate_usd=args.hourly_rate_usd,
                    terminate_at_usd=args.terminate_at_usd,
                    billing_started_at=args.billing_started_at,
                    now=now_fn(),
                ),
            )
        policy = SupervisorPolicy(
            max_restarts=args.max_restarts,
            poll_interval_seconds=args.poll_seconds,
            max_runtime_seconds=runtime_ceiling,
            terminate_grace_seconds=args.terminate_grace_seconds,
        )
        result = runner(command, output_root=args.output_root, policy=policy)
    finally:
        if managed and api_key is not None:
            terminator(args.pod_id, api_key)
    print(
        json.dumps(
            {
                "status": "complete",
                "attempt_count": result.attempt_count,
                "restart_count": result.restart_count,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
    )
    return 0
