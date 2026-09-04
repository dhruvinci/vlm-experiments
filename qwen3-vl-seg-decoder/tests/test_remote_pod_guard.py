from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ownership_decoder.remote_pod_guard import (
    PodGuardContract,
    guard_decision,
    run_pod_guard,
)


def _contract() -> PodGuardContract:
    return PodGuardContract(
        pod_id="pod-123",
        run_id="run-1",
        billing_started_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        spend_before_usd=20.0,
        frozen_hourly_rate_usd=2.0,
        terminate_at_usd=21.5,
        max_runtime_seconds=8 * 60 * 60,
        packet_sha256="a" * 64,
    )


class RemotePodGuardTests(unittest.TestCase):
    def test_decision_uses_higher_reported_rate_and_enforces_budget(self) -> None:
        decision, snapshot = guard_decision(
            _contract(),
            now=datetime(2026, 9, 4, 0, 30, tzinfo=timezone.utc),
            reported_hourly_rate_usd=4.0,
            run_complete=False,
            fatal=False,
        )

        self.assertEqual(decision, "terminate_budget")
        self.assertEqual(snapshot["hourly_rate_usd"], 4.0)
        self.assertEqual(snapshot["cumulative_gpu_usd_estimate"], 22.0)

    def test_success_sentinel_terminates_and_writes_heartbeat_before_delete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "RUN_COMPLETE").write_text("complete")
            heartbeat = root / "guard-heartbeat.json"
            terminated = []

            result = run_pod_guard(
                _contract(),
                run_root=root,
                heartbeat_path=heartbeat,
                get_pod_fn=lambda pod_id: {"costPerHr": 2.0},
                terminate_fn=lambda pod_id: terminated.append(pod_id),
                now_fn=lambda: datetime(2026, 9, 4, 0, 1, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: self.fail("guard slept after completion"),
                interval_seconds=30,
            )

            self.assertEqual(result, "terminate_success")
            self.assertEqual(terminated, ["pod-123"])
            self.assertEqual(json.loads(heartbeat.read_text())["decision"], result)

    def test_fatal_bootstrap_sentinel_terminates_without_waiting_for_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "RUN_FATAL").write_text("bootstrap failed")
            terminated = []

            result = run_pod_guard(
                _contract(),
                run_root=root,
                heartbeat_path=root / "heartbeat.json",
                get_pod_fn=lambda pod_id: {"adjustedCostPerHr": 0.5},
                terminate_fn=lambda pod_id: terminated.append(pod_id),
                now_fn=lambda: datetime(2026, 9, 4, 0, 1, tzinfo=timezone.utc),
                sleep_fn=lambda _seconds: self.fail("guard slept after fatal sentinel"),
            )

            self.assertEqual(result, "terminate_fatal")
            self.assertEqual(terminated, ["pod-123"])

    def test_termination_is_retried_until_the_api_confirms_delete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "RUN_FATAL").write_text("bootstrap failed")
            attempts = []
            sleeps = []

            def terminate(pod_id: str) -> None:
                attempts.append(pod_id)
                if len(attempts) == 1:
                    raise OSError("transient network failure")

            result = run_pod_guard(
                _contract(),
                run_root=root,
                heartbeat_path=root / "heartbeat.json",
                get_pod_fn=lambda pod_id: {"costPerHr": 2.0},
                terminate_fn=terminate,
                now_fn=lambda: datetime(2026, 9, 4, 0, 1, tzinfo=timezone.utc),
                sleep_fn=sleeps.append,
                interval_seconds=30,
            )

            self.assertEqual(result, "terminate_fatal")
            self.assertEqual(attempts, ["pod-123", "pod-123"])
            self.assertEqual(sleeps, [30])
            heartbeat = json.loads((root / "heartbeat.json").read_text())
            self.assertEqual(heartbeat["termination_attempt"], 2)
            self.assertNotIn("termination_error", heartbeat)


if __name__ == "__main__":
    unittest.main()
