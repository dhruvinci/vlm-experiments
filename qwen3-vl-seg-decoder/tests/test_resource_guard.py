from __future__ import annotations

import json
import os
import signal
import sys
import time
import unittest
from pathlib import Path

from ownership_decoder.resource_guard import (
    GuardedRunInterrupted,
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
                "-p",
                "TasksMax=128",
                "-p",
                "KillMode=control-group",
                "-p",
                "OOMPolicy=kill",
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
            self.assertEqual(result.termination_reason, "completed")
            self.assertEqual((tmp_path / "child.log").read_text().strip(), "complete")

    def test_guard_tolerates_a_transient_gpu_telemetry_gap(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            safe = snapshot()
            missing_gpu = ResourceSnapshot(
                host_available_bytes=safe.host_available_bytes,
                swap_free_bytes=safe.swap_free_bytes,
                gpu_free_bytes=None,
                gpu_total_bytes=None,
            )
            samples = iter([safe, missing_gpu, safe])

            def sampler() -> ResourceSnapshot:
                return next(samples, safe)

            result = run_guarded(
                [sys.executable, "-c", "import time; time.sleep(0.05)"],
                limits=ResourceLimits(min_gpu_free_bytes=768 * MIB),
                sample_fn=sampler,
                log_path=tmp_path / "child.log",
                telemetry_path=tmp_path / "telemetry.jsonl",
                poll_interval_seconds=0.01,
                gpu_telemetry_grace_samples=2,
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse(result.killed_for_limit)
            rows = [
                json.loads(line)
                for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
            ]
            gaps = [row for row in rows if row["event"] == "telemetry_gap"]
            self.assertEqual(len(gaps), 1)
            self.assertEqual(gaps[0]["consecutive_gpu_telemetry_failures"], 1)

    def test_guard_kills_after_gpu_telemetry_grace_is_exhausted(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            safe = snapshot()
            missing_gpu = ResourceSnapshot(
                host_available_bytes=safe.host_available_bytes,
                swap_free_bytes=safe.swap_free_bytes,
                gpu_free_bytes=None,
                gpu_total_bytes=None,
            )
            samples = iter([safe, missing_gpu, missing_gpu])

            def sampler() -> ResourceSnapshot:
                return next(samples, missing_gpu)

            result = run_guarded(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                limits=ResourceLimits(min_gpu_free_bytes=768 * MIB),
                sample_fn=sampler,
                log_path=tmp_path / "child.log",
                telemetry_path=tmp_path / "telemetry.jsonl",
                poll_interval_seconds=0.01,
                terminate_grace_seconds=0.1,
                gpu_telemetry_grace_samples=1,
            )

            self.assertTrue(result.killed_for_limit)
            self.assertEqual(result.violations[0].resource, "gpu_telemetry")
            rows = [
                json.loads(line)
                for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[-1]["event"], "limit_breach")
            self.assertEqual(rows[-1]["consecutive_gpu_telemetry_failures"], 2)

    def test_guard_terminates_a_stalled_process_at_the_runtime_ceiling(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            result = run_guarded(
                [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                limits=ResourceLimits(min_host_available_bytes=3 * GIB),
                sample_fn=lambda: snapshot(),
                log_path=tmp_path / "child.log",
                telemetry_path=tmp_path / "telemetry.jsonl",
                poll_interval_seconds=0.01,
                terminate_grace_seconds=0.1,
                maximum_runtime_seconds=0.05,
            )

            self.assertTrue(result.started)
            self.assertTrue(result.killed_for_limit)
            self.assertEqual(result.termination_reason, "runtime_limit")
            self.assertEqual(result.violations[0].resource, "runtime_seconds")
            self.assertGreaterEqual(result.elapsed_seconds, 0.05)
            rows = [
                json.loads(line)
                for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[-1]["event"], "runtime_limit")

    def test_guard_cleans_up_child_when_supervisor_is_interrupted(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            pid_path = tmp_path / "child.pid"
            calls = 0

            def sampler() -> ResourceSnapshot:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return snapshot()
                deadline = time.monotonic() + 1.0
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                run_guarded(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os,time; "
                            f"open({str(pid_path)!r},'w').write(str(os.getpid())); "
                            "time.sleep(30)"
                        ),
                    ],
                    limits=ResourceLimits(min_host_available_bytes=3 * GIB),
                    sample_fn=sampler,
                    log_path=tmp_path / "child.log",
                    telemetry_path=tmp_path / "telemetry.jsonl",
                    poll_interval_seconds=0.01,
                    terminate_grace_seconds=0.1,
                )

            child_pid = int(pid_path.read_text())
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                child_alive = False
            else:
                child_alive = True
                os.kill(child_pid, signal.SIGKILL)
            self.assertFalse(child_alive)
            rows = [
                json.loads(line)
                for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[-1]["event"], "interrupted")

    def test_guard_converts_sigterm_to_kill_safe_cleanup(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            pid_path = tmp_path / "child.pid"
            calls = 0

            def sampler() -> ResourceSnapshot:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return snapshot()
                deadline = time.monotonic() + 1.0
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                os.kill(os.getpid(), signal.SIGTERM)
                raise AssertionError("SIGTERM handler did not interrupt the guard")

            with self.assertRaises(GuardedRunInterrupted):
                run_guarded(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os,time; "
                            f"open({str(pid_path)!r},'w').write(str(os.getpid())); "
                            "time.sleep(30)"
                        ),
                    ],
                    limits=ResourceLimits(min_host_available_bytes=3 * GIB),
                    sample_fn=sampler,
                    log_path=tmp_path / "child.log",
                    telemetry_path=tmp_path / "telemetry.jsonl",
                    poll_interval_seconds=0.01,
                    terminate_grace_seconds=0.1,
                )

            child_pid = int(pid_path.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            rows = [
                json.loads(line)
                for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[-1]["event"], "interrupted")
