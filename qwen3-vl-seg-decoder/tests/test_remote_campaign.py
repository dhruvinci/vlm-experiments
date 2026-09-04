from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.remote_campaign import (
    RemoteSam31CampaignSpec,
    run_remote_sam31_campaign,
)
from ownership_decoder.remote_preflight import (
    RemoteHardwareSnapshot,
    RemoteRuntimeApproval,
)


GIB = 1024**3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_valid_config(root: Path, clip_id: str = "test_clip") -> Path:
    clip = root / clip_id
    frames = clip / "frames"
    frames.mkdir(parents=True)
    frame = frames / "frame_000001.jpg"
    frame.write_bytes(b"frame")
    manifest = {
        "contract": {"clip_id": clip_id},
        "decode": {"frame_count": 1},
        "frames": [
            {
                "clip_frame_index": 0,
                "path": "frames/frame_000001.jpg",
                "sha256": sha256(frame),
                "height": 3,
                "width": 4,
            }
        ],
    }
    manifest_path = clip / "clip-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "schema_version": "1.0",
        "clip_id": clip_id,
        "clip_manifest_path": f"{clip_id}/clip-manifest.json",
        "clip_manifest_sha256": sha256(manifest_path),
        "seeds": [
            {
                "frame_index": 0,
                "actors": [
                    {
                        "actor_id": "A1",
                        "bbox": [0.0, 0.0, 0.49, 1.0],
                        "positive_points": [[0.1, 0.2], [0.3, 0.8]],
                        "negative_points": [[0.6, 0.2], [0.8, 0.8]],
                    },
                    {
                        "actor_id": "A2",
                        "bbox": [0.5, 0.0, 1.0, 1.0],
                        "positive_points": [[0.6, 0.2], [0.8, 0.8]],
                        "negative_points": [[0.1, 0.2], [0.3, 0.8]],
                    },
                ],
            }
        ],
        "propagations": [
            {"start_frame_idx": 0, "max_frame_num_to_track": 1, "reverse": False}
        ],
    }
    config_path = root / f"{clip_id}.json"
    config_path.write_text(json.dumps(config))
    return config_path


class EventSink:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def record(self, event: dict) -> dict:
        self.events.append(("telemetry", event["event"]))
        return event


class RemoteCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.config = write_valid_config(self.inputs)
        self.repo = self.root / "sam3"
        self.repo.mkdir()
        self.checkpoint = self.root / "sam3.1_multiplex.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.spec = RemoteSam31CampaignSpec(
            config_paths=(self.config,),
            input_root=self.inputs,
            output_root=self.root / "outputs",
            sam_repo_path=self.repo,
            sam_repo_revision="8f0b7f4d4e7eda2ed606ebde6702c93359ad01da",
            checkpoint_path=self.checkpoint,
            checkpoint_sha256=sha256(self.checkpoint),
            workspace_path=self.root,
            required_distribution_versions=(("torch", "2.12.1+cu130"),),
            attempt_index=0,
        )
        self.approval = RemoteRuntimeApproval(
            contract_sha256=self.spec.runtime_contract.sha256,
            checkpoint_sha256=self.spec.checkpoint_sha256,
            hardware=RemoteHardwareSnapshot(
                gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                gpu_total_bytes=96 * GIB,
                gpu_free_bytes=90 * GIB,
                compute_capability=(12, 0),
                driver_version=(580, 65, 6),
                host_available_bytes=64 * GIB,
                workspace_free_bytes=100 * GIB,
            ),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_all_clip_configs_validate_before_preflight_is_called(self) -> None:
        payload = json.loads(self.config.read_text())
        payload["clip_manifest_sha256"] = "0" * 64
        self.config.write_text(json.dumps(payload))
        touched = False

        def preflight(_):
            nonlocal touched
            touched = True
            return self.approval

        with self.assertRaisesRegex(Exception, "manifest checksum"):
            run_remote_sam31_campaign(self.spec, preflight_fn=preflight)

        self.assertFalse(touched)

    def test_preflight_report_is_committed_before_single_predictor_build(self) -> None:
        events: list[object] = []
        attempt_dir = self.spec.output_root / "_runtime" / "attempt_00"

        def preflight(_):
            events.append("preflight")
            return self.approval

        predictor = object()

        def build(contract, approval):
            self.assertTrue((attempt_dir / "preflight.json").is_file())
            events.append("build")
            return predictor

        def campaign(config_paths, **kwargs):
            events.append("campaign")
            self.assertEqual(kwargs["predictor_factory"](), predictor)
            kwargs["event_callback"]({"event": "segment_completed", "clip_id": "test_clip"})
            return {"format": "test-campaign", "clip_count": 1}

        manifest = run_remote_sam31_campaign(
            self.spec,
            preflight_fn=preflight,
            predictor_builder=build,
            campaign_runner=campaign,
            telemetry_factory=lambda *_args, **_kwargs: EventSink(events),
        )

        self.assertEqual(manifest["clip_count"], 1)
        self.assertEqual(events[:3], ["preflight", ("telemetry", "preflight_completed"), "campaign"])
        self.assertEqual(events.count("build"), 1)
        self.assertIn(("telemetry", "predictor_loaded"), events)
        report = json.loads((attempt_dir / "preflight.json").read_text())
        self.assertEqual(report["contract_sha256"], self.spec.runtime_contract.sha256)
        self.assertEqual(report["plan_sha256_by_clip"]["test_clip"], report["plans"][0]["plan_sha256"])

    def test_builder_failure_is_durably_recorded_and_propagated(self) -> None:
        def fail_build(*_args):
            raise RuntimeError("synthetic model failure")

        def campaign(_configs, **kwargs):
            kwargs["predictor_factory"]()
            raise AssertionError("unreachable")

        with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
            run_remote_sam31_campaign(
                self.spec,
                preflight_fn=lambda _: self.approval,
                predictor_builder=fail_build,
                campaign_runner=campaign,
                telemetry_factory=lambda *_args, **_kwargs: EventSink([]),
            )

        failure_path = self.spec.output_root / "_runtime" / "attempt_00" / "FAILURE.json"
        self.assertTrue(failure_path.is_file())
        failure = json.loads(failure_path.read_text())
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertIn("synthetic model failure", failure["message"])


if __name__ == "__main__":
    unittest.main()
