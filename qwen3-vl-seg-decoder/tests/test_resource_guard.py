from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from ownership_decoder.resource_guard import (
    ResourceLimits,
    ResourceSnapshot,
    assess_snapshot,
    run_guarded,
    systemd_scoped_command,
)


GIB = 1024**3
MIB = 1024**2


def snapshot(*, host_gib: float = 8, gpu_free_mib: int = 4000) -> ResourceSnapshot:
    return ResourceSnapshot(
        host_available_bytes=int(host_gib * GIB),
        swap_free_bytes=2 * GIB,
        gpu_free_bytes=gpu_free_mib * MIB,
        gpu_total_bytes=6 * GIB,
    )


class ResourceGuardTests(unittest.TestCase):
    def test_systemd_scope_command_applies_an_absolute_child_memory_cap(self) -> None:
        wrapped = systemd_scoped_command(
            ["python", "worker.py", "--flag"],
            memory_max_bytes=6 * GIB,
            systemd_run_path="/usr/bin/systemd-run",
        )

        self.assertEqual(
            wrapped,
            [
                "/usr/bin/systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "-p",
                f"MemoryHigh={int(6 * GIB * 0.9)}",
                "-p",
                f"MemoryMax={6 * GIB}",
                "-p",
                "MemorySwapMax=0",
                "--",
                "python",
                "worker.py",
                "--flag",
            ],
        )

    def test_assess_snapshot_reports_every_crossed_limit(self) -> None:
        limits = ResourceLimits(
            min_host_available_bytes=3 * GIB,
            min_gpu_free_bytes=768 * MIB,
            max_gpu_used_fraction=0.85,
        )

        violations = assess_snapshot(snapshot(host_gib=2, gpu_free_mib=500), limits)

        self.assertEqual(
            {violation.resource for violation in violations},
            {"host_available_bytes", "gpu_free_bytes", "gpu_used_fraction"},
        )

    def test_guard_refuses_to_start_when_preflight_is_unsafe(self) -> None:
        with self.subTest("preflight"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                side_effect = tmp_path / "started"
                result = run_guarded(
                    [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(side_effect)!r}).touch()",
                    ],
                    limits=ResourceLimits(min_host_available_bytes=3 * GIB),
                    sample_fn=lambda: snapshot(host_gib=2),
                    log_path=tmp_path / "child.log",
                    telemetry_path=tmp_path / "telemetry.jsonl",
                    poll_interval_seconds=0.01,
                )

                self.assertFalse(result.started)
                self.assertFalse(result.killed_for_limit)
                self.assertIsNone(result.returncode)
                self.assertEqual(result.violations[0].resource, "host_available_bytes")
                self.assertFalse(side_effect.exists())

    def test_guard_kills_process_group_when_runtime_limit_is_crossed(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            samples = iter(
                [
                    snapshot(),  # preflight
                    *[snapshot() for _ in range(10)],  # allow the child to start
                    snapshot(host_gib=1),  # breach
                ]
            )
            last = snapshot(host_gib=1)

            def sampler() -> ResourceSnapshot:
                nonlocal last
                try:
                    last = next(samples)
                except StopIteration:
                    pass
                return last

            result = run_guarded(
                [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
                limits=ResourceLimits(min_host_available_bytes=3 * GIB),
                sample_fn=sampler,
                log_path=tmp_path / "child.log",
                telemetry_path=tmp_path / "telemetry.jsonl",
                poll_interval_seconds=0.01,
                terminate_grace_seconds=0.1,
            )

            self.assertTrue(result.started)
            self.assertTrue(result.killed_for_limit)
            self.assertIsNotNone(result.returncode)
            self.assertEqual(result.violations[0].resource, "host_available_bytes")
            self.assertIn("started", (tmp_path / "child.log").read_text())
            rows = [
                json.loads(line)
                for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[0]["event"], "preflight")
            self.assertEqual(rows[-1]["event"], "limit_breach")

    def test_guard_allows_a_safe_process_to_complete(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            result = run_guarded(
                [sys.executable, "-c", "print('complete')"],
                limits=ResourceLimits(min_host_available_bytes=3 * GIB),
                sample_fn=lambda: snapshot(),
                log_path=tmp_path / "child.log",
                telemetry_path=tmp_path / "telemetry.jsonl",
                poll_interval_seconds=0.01,
            )

            self.assertTrue(result.started)
            self.assertFalse(result.killed_for_limit)
            self.assertEqual(result.returncode, 0)
            self.assertEqual((tmp_path / "child.log").read_text().strip(), "complete")
