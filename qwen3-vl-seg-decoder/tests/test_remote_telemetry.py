from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.remote_telemetry import (
    RuntimeFatalSafetyError,
    RuntimeSafetyPolicy,
    RuntimeTelemetryWriter,
    RuntimeUsageSnapshot,
    RuntimeWorkerRestartRequired,
)


GIB = 1024**3


def usage(*, gpu_used_gib: float = 12, host_gib: float = 60, disk_gib: float = 100):
    return RuntimeUsageSnapshot(
        unix_time=100.0,
        host_available_bytes=int(host_gib * GIB),
        workspace_free_bytes=int(disk_gib * GIB),
        gpu_used_bytes=int(gpu_used_gib * GIB),
        gpu_total_bytes=96 * GIB,
        gpu_temperature_c=55,
        gpu_utilization_percent=70,
    )


class RemoteTelemetryTests(unittest.TestCase):
    def test_records_model_baseline_and_each_event_as_durable_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            snapshots = iter([usage(gpu_used_gib=10), usage(gpu_used_gib=10.5)])
            writer = RuntimeTelemetryWriter(path, sample_fn=lambda _: next(snapshots))

            first = writer.record({"event": "predictor_loaded"})
            second = writer.record({"event": "segment_completed", "segment_index": 0})

            self.assertEqual(first["gpu_baseline_bytes"], 10 * GIB)
            self.assertEqual(second["gpu_delta_from_baseline_bytes"], int(0.5 * GIB))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["predictor_loaded", "segment_completed"])
            self.assertEqual(rows[1]["segment_index"], 0)

    def test_vram_pressure_is_recorded_then_requests_fresh_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            writer = RuntimeTelemetryWriter(
                path,
                sample_fn=lambda _: usage(gpu_used_gib=87),
            )

            with self.assertRaisesRegex(RuntimeWorkerRestartRequired, "VRAM"):
                writer.record({"event": "segment_completed"})

            row = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(row["safety_action"], "restart_worker")
            self.assertGreater(row["usage"]["gpu_used_fraction"], 0.9)

    def test_low_disk_is_fatal_and_is_recorded_before_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            writer = RuntimeTelemetryWriter(
                path,
                sample_fn=lambda _: usage(disk_gib=10),
            )

            with self.assertRaisesRegex(RuntimeFatalSafetyError, "disk"):
                writer.record({"event": "segment_completed"})

            row = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(row["safety_action"], "terminate")

    def test_one_gib_post_cleanup_growth_warns_after_three_consecutive_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            snapshots = iter(
                [usage(gpu_used_gib=10)]
                + [usage(gpu_used_gib=11.25) for _ in range(3)]
            )
            writer = RuntimeTelemetryWriter(path, sample_fn=lambda _: next(snapshots))
            writer.record({"event": "predictor_loaded"})

            results = [
                writer.record({"event": "session_closed", "clip_id": f"clip-{index}"})
                for index in range(3)
            ]

            self.assertNotIn("safety_warning", results[0])
            self.assertEqual(results[-1]["safety_warning"], "persistent_gpu_growth")

    def test_two_gib_post_cleanup_growth_requests_restart_after_five_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            snapshots = iter(
                [usage(gpu_used_gib=10)]
                + [usage(gpu_used_gib=12.5) for _ in range(5)]
            )
            writer = RuntimeTelemetryWriter(path, sample_fn=lambda _: next(snapshots))
            writer.record({"event": "predictor_loaded"})

            for index in range(4):
                writer.record({"event": "session_closed", "clip_id": f"clip-{index}"})
            with self.assertRaisesRegex(RuntimeWorkerRestartRequired, "persistent"):
                writer.record({"event": "session_closed", "clip_id": "clip-4"})

            row = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(row["safety_action"], "restart_worker")
            self.assertEqual(row["persistent_restart_count"], 5)

    def test_policy_rejects_incoherent_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeSafetyPolicy(
                gpu_growth_warning_bytes=3 * GIB,
                gpu_growth_restart_bytes=2 * GIB,
            )

    def test_campaign_specific_model_events_reset_the_leak_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            snapshots = iter(
                [usage(gpu_used_gib=10), usage(gpu_used_gib=11), usage(gpu_used_gib=8)]
            )
            writer = RuntimeTelemetryWriter(path, sample_fn=lambda _: next(snapshots))

            first = writer.record({"event": "sam31_loaded"})
            after_tracker = writer.record({"event": "session_closed"})
            second = writer.record({"event": "sam3_image_loaded"})

            self.assertEqual(first["gpu_baseline_bytes"], 10 * GIB)
            self.assertEqual(after_tracker["gpu_delta_from_baseline_bytes"], 1 * GIB)
            self.assertEqual(second["gpu_baseline_bytes"], 8 * GIB)
            self.assertEqual(second["gpu_delta_from_baseline_bytes"], 0)

    def test_image_frame_completion_participates_in_persistent_leak_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            snapshots = iter(
                [usage(gpu_used_gib=10)]
                + [usage(gpu_used_gib=12.5) for _ in range(5)]
            )
            writer = RuntimeTelemetryWriter(path, sample_fn=lambda _: next(snapshots))
            writer.record({"event": "sam3_image_loaded"})

            for index in range(4):
                writer.record(
                    {"event": "image_agreement_frame_completed", "frame_index": index}
                )
            with self.assertRaisesRegex(RuntimeWorkerRestartRequired, "persistent"):
                writer.record(
                    {"event": "image_agreement_frame_completed", "frame_index": 4}
                )


if __name__ == "__main__":
    unittest.main()
