from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.tracking_campaign import (
    load_completed_tracking_campaign,
    run_tracking_campaign,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_clip(input_root: Path, clip_id: str, pixel: int) -> Path:
    clip_dir = input_root / clip_id
    frame_dir = clip_dir / "frames"
    frame_dir.mkdir(parents=True)
    frame_path = frame_dir / "frame_000001.jpg"
    frame_path.write_bytes(f"frame-{clip_id}".encode())
    manifest = {
        "contract": {"clip_id": clip_id},
        "decode": {"frame_count": 1},
        "frames": [
            {
                "clip_frame_index": 0,
                "path": "frames/frame_000001.jpg",
                "sha256": _sha256(frame_path),
                "height": 3,
                "width": 4,
            }
        ],
    }
    manifest_path = clip_dir / "clip-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "schema_version": "1.0",
        "clip_id": clip_id,
        "clip_manifest_path": f"{clip_id}/clip-manifest.json",
        "clip_manifest_sha256": _sha256(manifest_path),
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
    config_path = input_root / f"{clip_id}.json"
    config_path.write_text(json.dumps(config))
    return config_path


class CampaignPredictor:
    def __init__(self) -> None:
        self.closed = False

    def handle_request(self, request: dict) -> dict:
        if request["type"] == "start_session":
            return {"session_id": "session"}
        if request["type"] == "add_prompt":
            return {"is_success": True}
        if request["type"] == "close_session":
            return {"is_success": True}
        if request["type"] == "propagate_in_video":
            a1 = np.zeros((3, 4), dtype=bool)
            a2 = np.zeros((3, 4), dtype=bool)
            a1[:, :2] = True
            a2[:, 2:] = True
            return {
                "frames": [
                    {
                        "frame_index": 0,
                        "outputs": {
                            "out_obj_ids": np.array([1, 2]),
                            "out_probs": np.array([0.9, 0.8]),
                            "out_binary_masks": np.stack([a1, a2]),
                        },
                    }
                ]
            }
        raise AssertionError(request)

    def close(self) -> None:
        self.closed = True


class TrackingCampaignTests(unittest.TestCase):
    def test_campaign_loads_predictor_once_for_multiple_clips_and_resumes_without_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            inputs = root / "inputs"
            inputs.mkdir()
            configs = [
                _write_clip(inputs, "clip_b", 2),
                _write_clip(inputs, "clip_a", 1),
            ]
            built: list[CampaignPredictor] = []

            def factory() -> CampaignPredictor:
                predictor = CampaignPredictor()
                built.append(predictor)
                return predictor

            manifest = run_tracking_campaign(
                configs,
                input_root=inputs,
                output_root=root / "outputs",
                backend="sam3.1-tracker-only",
                revision="rev",
                predictor_factory=factory,
            )

            self.assertEqual(len(built), 1)
            self.assertTrue(built[0].closed)
            self.assertEqual([item["clip_id"] for item in manifest["clips"]], ["clip_a", "clip_b"])
            self.assertTrue((root / "outputs" / "RUN_COMPLETE").is_file())
            restored = load_completed_tracking_campaign(
                root / "outputs",
                config_paths=configs,
                input_root=inputs,
                expected_backend="sam3.1-tracker-only",
                expected_revision="rev",
            )
            self.assertEqual(restored, manifest)

            resumed = run_tracking_campaign(
                configs,
                input_root=inputs,
                output_root=root / "outputs",
                backend="sam3.1-tracker-only",
                revision="rev",
                predictor_factory=lambda: self.fail("resume rebuilt predictor"),
            )
            self.assertEqual(resumed, manifest)

    def test_every_config_is_validated_before_predictor_factory_is_called(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            inputs = root / "inputs"
            inputs.mkdir()
            valid = _write_clip(inputs, "clip_a", 1)
            invalid = _write_clip(inputs, "clip_b", 2)
            payload = json.loads(invalid.read_text())
            payload["clip_manifest_sha256"] = "0" * 64
            invalid.write_text(json.dumps(payload))
            built = False

            def factory() -> CampaignPredictor:
                nonlocal built
                built = True
                return CampaignPredictor()

            with self.assertRaisesRegex(Exception, "manifest checksum"):
                run_tracking_campaign(
                    [valid, invalid],
                    input_root=inputs,
                    output_root=root / "outputs",
                    backend="sam3.1-tracker-only",
                    revision="rev",
                    predictor_factory=factory,
                )

            self.assertFalse(built)

    def test_campaign_forwards_per_clip_and_per_segment_progress_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            inputs = root / "inputs"
            inputs.mkdir()
            config = _write_clip(inputs, "clip_a", 1)
            events: list[dict] = []

            run_tracking_campaign(
                [config],
                input_root=inputs,
                output_root=root / "outputs",
                backend="sam3.1-tracker-only",
                revision="rev",
                predictor_factory=CampaignPredictor,
                event_callback=events.append,
            )

            names = [event["event"] for event in events]
            self.assertEqual(names[0], "campaign_started")
            self.assertIn("clip_started", names)
            self.assertIn("segment_completed", names)
            self.assertIn("clip_completed", names)
            self.assertEqual(names[-1], "campaign_completed")
            self.assertEqual(events[0]["clip_count"], 1)
            self.assertTrue(all(event.get("clip_id") == "clip_a" for event in events[1:-1]))


if __name__ == "__main__":
    unittest.main()
