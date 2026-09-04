from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.remote_supervisor import (
    RemoteSupervisorError,
    SupervisorPolicy,
    supervise_worker,
)
from ownership_decoder.remote_telemetry import RuntimeUsageSnapshot


GIB = 1024**3


def safe_usage(_path: Path) -> RuntimeUsageSnapshot:
    return RuntimeUsageSnapshot(
        unix_time=100.0,
        host_available_bytes=64 * GIB,
        workspace_free_bytes=100 * GIB,
        gpu_used_bytes=20 * GIB,
        gpu_total_bytes=96 * GIB,
        gpu_temperature_c=55,
        gpu_utilization_percent=70,
    )


class RemoteSupervisorTests(unittest.TestCase):
    def test_retry_exit_code_starts_fresh_process_and_then_requires_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            program = (
                "import pathlib,sys; "
                "i=int(sys.argv[sys.argv.index('--attempt-index')+1]); "
                f"out=pathlib.Path({str(output)!r}); out.mkdir(parents=True,exist_ok=True); "
                "(out/'RUN_COMPLETE').write_text('ok') if i == 1 else None; "
                "raise SystemExit(0 if i == 1 else 75)"
            )

            result = supervise_worker(
                [sys.executable, "-c", program],
                output_root=output,
                policy=SupervisorPolicy(
                    max_restarts=2,
                    poll_interval_seconds=0.01,
                    max_runtime_seconds=5,
                    terminate_grace_seconds=0.05,
                ),
                usage_probe=safe_usage,
            )

            self.assertEqual(result.attempt_count, 2)
            self.assertEqual(result.restart_count, 1)
            self.assertEqual(result.returncode, 0)
            events = [
                json.loads(line)
                for line in (output / "_supervisor" / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["event"] for row in events],
                ["worker_started", "worker_restart", "worker_started", "supervisor_complete"],
            )

    def test_unknown_worker_failure_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.assertRaisesRegex(RemoteSupervisorError, "exit code 1"):
                supervise_worker(
                    [sys.executable, "-c", "raise SystemExit(1)"],
                    output_root=output,
                    policy=SupervisorPolicy(
                        max_restarts=2,
                        poll_interval_seconds=0.01,
                        max_runtime_seconds=5,
                        terminate_grace_seconds=0.05,
                    ),
                    usage_probe=safe_usage,
                )

            events = [
                json.loads(line)
                for line in (output / "_supervisor" / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["event"] for row in events], ["worker_started", "worker_fatal"])

    def test_zero_exit_without_campaign_sentinel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.assertRaisesRegex(RemoteSupervisorError, "RUN_COMPLETE"):
                supervise_worker(
                    [sys.executable, "-c", "raise SystemExit(0)"],
                    output_root=output,
                    policy=SupervisorPolicy(
                        poll_interval_seconds=0.01,
                        max_runtime_seconds=5,
                        terminate_grace_seconds=0.05,
                    ),
                    usage_probe=safe_usage,
                )

    def test_runtime_ceiling_terminates_worker_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.assertRaisesRegex(RemoteSupervisorError, "runtime ceiling"):
                supervise_worker(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    output_root=output,
                    policy=SupervisorPolicy(
                        poll_interval_seconds=0.01,
                        max_runtime_seconds=0.05,
                        terminate_grace_seconds=0.01,
                    ),
                    usage_probe=safe_usage,
                )

            heartbeat = json.loads(
                (output / "_supervisor" / "heartbeat.json").read_text()
            )
            self.assertEqual(heartbeat["state"], "terminated_runtime_ceiling")

    def test_parent_watchdog_kills_high_vram_worker_and_restarts_fresh_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            program = (
                "import pathlib,sys,time; "
                "i=int(sys.argv[sys.argv.index('--attempt-index')+1]); "
                f"out=pathlib.Path({str(output)!r}); out.mkdir(parents=True,exist_ok=True); "
                "(time.sleep(30) if i == 0 else (out/'RUN_COMPLETE').write_text('ok')); "
                "raise SystemExit(0)"
            )
            samples = iter(
                [
                    RuntimeUsageSnapshot(
                        unix_time=100,
                        host_available_bytes=64 * GIB,
                        workspace_free_bytes=100 * GIB,
                        gpu_used_bytes=90 * GIB,
                        gpu_total_bytes=96 * GIB,
                        gpu_temperature_c=55,
                        gpu_utilization_percent=99,
                    ),
                    *[safe_usage(output) for _ in range(10)],
                ]
            )

            result = supervise_worker(
                [sys.executable, "-c", program],
                output_root=output,
                policy=SupervisorPolicy(
                    max_restarts=2,
                    poll_interval_seconds=0.01,
                    max_runtime_seconds=5,
                    terminate_grace_seconds=0.01,
                ),
                usage_probe=lambda _: next(samples),
            )

            self.assertEqual(result.restart_count, 1)
            events = [
                json.loads(line)
                for line in (output / "_supervisor" / "events.jsonl").read_text().splitlines()
            ]
            self.assertIn("worker_resource_restart", [row["event"] for row in events])
            heartbeat = json.loads(
                (output / "_supervisor" / "heartbeat.json").read_text()
            )
            self.assertIn("usage", heartbeat)


if __name__ == "__main__":
    unittest.main()
