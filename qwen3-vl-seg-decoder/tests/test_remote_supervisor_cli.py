from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

from ownership_decoder.remote_supervisor import SupervisorResult
from ownership_decoder.remote_supervisor_cli import main


class RemoteSupervisorCliTests(unittest.TestCase):
    def test_cli_passes_bounded_policy_and_exact_worker_command(self) -> None:
        captured = []

        def runner(command, **kwargs):
            captured.append((command, kwargs))
            return SupervisorResult(2, 1, 0, 12.5)

        returncode = main(
            [
                "--output-root",
                "/volume/output",
                "--max-restarts",
                "2",
                "--poll-seconds",
                "30",
                "--max-runtime-seconds",
                "28800",
                "--terminate-grace-seconds",
                "20",
                "--",
                "python",
                "worker.py",
                "--config",
                "clip.json",
            ],
            runner=runner,
        )

        self.assertEqual(returncode, 0)
        command, kwargs = captured[0]
        self.assertEqual(command, ["python", "worker.py", "--config", "clip.json"])
        self.assertEqual(kwargs["output_root"], Path("/volume/output"))
        self.assertEqual(kwargs["policy"].max_runtime_seconds, 28800)
        self.assertEqual(kwargs["policy"].max_restarts, 2)

    def test_cli_requires_a_worker_command(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker command"):
            main(["--output-root", "/volume/output", "--"], runner=lambda *_a, **_k: None)

    def test_managed_pod_uses_remaining_budget_as_ceiling_and_always_terminates(self) -> None:
        captured = []
        terminated = []

        def runner(command, **kwargs):
            captured.append((command, kwargs))
            return SupervisorResult(1, 0, 0, 1.0)

        result = main(
            [
                "--output-root",
                "/volume/output",
                "--max-runtime-seconds",
                "28800",
                "--pod-id",
                "pod-123",
                "--spend-before-usd",
                "20.00",
                "--hourly-rate-usd",
                "2.00",
                "--terminate-at-usd",
                "21.50",
                "--billing-started-at",
                "2026-09-04T00:00:00+00:00",
                "--",
                "python",
                "worker.py",
            ],
            runner=runner,
            environment={"RUNPOD_API_KEY": "secret"},
            terminator=lambda pod_id, api_key: terminated.append((pod_id, api_key)),
            now_fn=lambda: datetime(2026, 9, 4, 0, 15, tzinfo=timezone.utc),
        )

        self.assertEqual(result, 0)
        self.assertEqual(terminated, [("pod-123", "secret")])
        self.assertAlmostEqual(captured[0][1]["policy"].max_runtime_seconds, 1800.0)

    def test_managed_pod_is_terminated_when_worker_raises(self) -> None:
        terminated = []

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            main(
                [
                    "--output-root",
                    "/volume/output",
                    "--pod-id",
                    "pod-123",
                    "--spend-before-usd",
                    "0",
                    "--hourly-rate-usd",
                    "2.09",
                    "--billing-started-at",
                    "2026-09-04T00:00:00+00:00",
                    "--",
                    "python",
                    "worker.py",
                ],
                runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("worker failed")
                ),
                environment={"RUNPOD_API_KEY": "secret"},
                terminator=lambda pod_id, api_key: terminated.append((pod_id, api_key)),
                now_fn=lambda: datetime(2026, 9, 4, 0, 1, tzinfo=timezone.utc),
            )

        self.assertEqual(terminated, [("pod-123", "secret")])


if __name__ == "__main__":
    unittest.main()
