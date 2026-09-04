from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ownership_decoder.image_agreement import (
    ImageAgreementArtifactError,
    load_completed_image_agreement_campaign,
    run_image_agreement_campaign,
)
from ownership_decoder.tracking import (
    TrackingArtifactError,
    load_tracking_plan_config,
    run_tracking_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_plan(root: Path, clip_id: str = "clip_a", frame_count: int = 2) -> Path:
    clip = root / clip_id
    frames = clip / "frames"
    frames.mkdir(parents=True)
    records = []
    for index in range(frame_count):
        path = frames / f"frame_{index + 1:06d}.jpg"
        path.write_bytes(f"synthetic-{clip_id}-{index}".encode())
        records.append(
            {
                "clip_frame_index": index,
                "path": f"frames/{path.name}",
                "sha256": _sha256(path),
                "height": 8,
                "width": 10,
            }
        )
    manifest = {
        "contract": {"clip_id": clip_id},
        "decode": {"frame_count": frame_count},
        "frames": records,
    }
    manifest_path = clip / "clip-manifest.json"
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
                        "bbox": [0.0, 0.0, 0.45, 1.0],
                        "positive_points": [[0.1, 0.2], [0.3, 0.7]],
                        "negative_points": [[0.7, 0.2], [0.9, 0.7]],
                    },
                    {
                        "actor_id": "A2",
                        "bbox": [0.55, 0.0, 1.0, 1.0],
                        "positive_points": [[0.7, 0.2], [0.9, 0.7]],
                        "negative_points": [[0.1, 0.2], [0.3, 0.7]],
                    },
                ],
            }
        ],
        "propagations": [
            {
                "start_frame_idx": 0,
                "max_frame_num_to_track": frame_count,
                "reverse": False,
            }
        ],
    }
    config_path = root / f"{clip_id}.json"
    config_path.write_text(json.dumps(config))
    return config_path


class FakeTracker:
    def handle_request(self, request: dict) -> dict:
        if request["type"] == "start_session":
            return {"session_id": "tracker"}
        if request["type"] in {"add_prompt", "close_session"}:
            return {"is_success": True}
        if request["type"] == "propagate_in_video":
            frames = []
            for index in range(request["max_frame_num_to_track"]):
                a1 = np.zeros((8, 10), dtype=bool)
                a2 = np.zeros((8, 10), dtype=bool)
                a1[:, :4] = True
                a2[:, 6:] = True
                frames.append(
                    {
                        "frame_index": index,
                        "outputs": {
                            "out_obj_ids": np.array([1, 2]),
                            "out_probs": np.array([0.9, 0.8]),
                            "out_binary_masks": np.stack([a1, a2]),
                        },
                    }
                )
            return {"frames": frames}
        raise AssertionError(request)


class FakeImagePredictor:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls = []
        self.closed = False
        self.fail_at = fail_at

    def segment(self, image_path: Path, prompts: dict, *, expected_shape: tuple[int, int]):
        call_index = len(self.calls)
        self.calls.append((image_path, prompts, expected_shape))
        if self.fail_at == call_index:
            raise RuntimeError("synthetic image failure")
        logits_a1 = np.full(expected_shape, -2.0, dtype=np.float32)
        logits_a2 = np.full(expected_shape, -2.0, dtype=np.float32)
        logits_a1[:, :6] = 1.0
        logits_a2[:, 4:] = 2.0
        logits_a1[0, 4] = 3.0
        logits_a2[0, 4] = 3.0  # Exact ties stay unowned, not biased to A1.
        return {
            "logits_A1": logits_a1,
            "logits_A2": logits_a2,
            "raw_A1": logits_a1 > 0,
            "raw_A2": logits_a2 > 0,
            "score_A1": 0.9,
            "score_A2": 0.8,
            "selected_index_A1": 1,
            "selected_index_A2": 2,
        }

    def close(self) -> None:
        self.closed = True


def _prepare_tracking(root: Path, config: Path) -> None:
    plan = load_tracking_plan_config(config, input_root=root / "inputs")
    run_tracking_plan(
        FakeTracker(),
        plan,
        root / "tracking" / plan.clip_id,
        backend="sam3.1-tracker-only",
        revision="sam31-rev",
    )


class ImageAgreementTests(unittest.TestCase):
    def test_commits_exclusive_two_actor_masks_with_correlated_prompt_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs = root / "inputs"
            inputs.mkdir()
            config = _write_plan(inputs)
            _prepare_tracking(root, config)
            predictors = []
            events = []

            def factory():
                predictor = FakeImagePredictor()
                predictors.append(predictor)
                return predictor

            manifest = run_image_agreement_campaign(
                [config],
                input_root=inputs,
                tracker_root=root / "tracking",
                output_root=root / "agreement",
                tracker_backend="sam3.1-tracker-only",
                tracker_revision="sam31-rev",
                backend="sam3-tracker-image-pvs",
                revision="sam3-rev",
                predictor_factory=factory,
                minimum_prompt_area=8,
                event_callback=events.append,
            )

            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["localization_dependency"], "sam3.1-tracker")
            self.assertEqual(len(predictors), 1)
            self.assertTrue(predictors[0].closed)
            self.assertEqual(len(predictors[0].calls), 2)
            artifact = root / "agreement" / "clip_a" / "masks" / "frame_000000.npz"
            with np.load(artifact, allow_pickle=False) as values:
                self.assertFalse(np.any(values["A1"] & values["A2"]))
                self.assertFalse(values["A1"][0, 4])
                self.assertFalse(values["A2"][0, 4])
                self.assertTrue(values["A2"][1, 4])
            sidecar = json.loads(
                artifact.with_suffix(".npz.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["prompt"]["localization_source"], "sam3.1_tracker")
            self.assertEqual(sidecar["unresolved_tie_pixels"], 1)
            self.assertEqual(events[-1]["event"], "image_agreement_campaign_completed")

            resumed = run_image_agreement_campaign(
                [config],
                input_root=inputs,
                tracker_root=root / "tracking",
                output_root=root / "agreement",
                tracker_backend="sam3.1-tracker-only",
                tracker_revision="sam31-rev",
                backend="sam3-tracker-image-pvs",
                revision="sam3-rev",
                predictor_factory=lambda: self.fail("resume loaded model"),
                minimum_prompt_area=8,
            )
            self.assertEqual(resumed, manifest)

    def test_corrupted_completed_artifact_is_rejected_before_predictor_load(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs = root / "inputs"
            inputs.mkdir()
            config = _write_plan(inputs, frame_count=1)
            _prepare_tracking(root, config)
            output = root / "agreement"
            run_image_agreement_campaign(
                [config],
                input_root=inputs,
                tracker_root=root / "tracking",
                output_root=output,
                tracker_backend="sam3.1-tracker-only",
                tracker_revision="sam31-rev",
                backend="sam3-tracker-image-pvs",
                revision="sam3-rev",
                predictor_factory=FakeImagePredictor,
                minimum_prompt_area=8,
            )
            artifact = output / "clip_a" / "masks" / "frame_000000.npz"
            artifact.write_bytes(artifact.read_bytes() + b"corrupt")

            with self.assertRaises(ImageAgreementArtifactError):
                load_completed_image_agreement_campaign(
                    output,
                    config_paths=[config],
                    input_root=inputs,
                    tracker_root=root / "tracking",
                    tracker_backend="sam3.1-tracker-only",
                    tracker_revision="sam31-rev",
                    expected_backend="sam3-tracker-image-pvs",
                    expected_revision="sam3-rev",
                )

    def test_corrupted_tracker_source_invalidates_completed_agreement_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs = root / "inputs"
            inputs.mkdir()
            config = _write_plan(inputs, frame_count=1)
            _prepare_tracking(root, config)
            output = root / "agreement"
            run_image_agreement_campaign(
                [config],
                input_root=inputs,
                tracker_root=root / "tracking",
                output_root=output,
                tracker_backend="sam3.1-tracker-only",
                tracker_revision="sam31-rev",
                backend="sam3-tracker-image-pvs",
                revision="sam3-rev",
                predictor_factory=FakeImagePredictor,
                minimum_prompt_area=8,
            )
            source = root / "tracking" / "clip_a" / "masks" / "frame_000000.npz"
            source.write_bytes(source.read_bytes() + b"corrupt")

            with self.assertRaises(TrackingArtifactError):
                load_completed_image_agreement_campaign(
                    output,
                    config_paths=[config],
                    input_root=inputs,
                    tracker_root=root / "tracking",
                    tracker_backend="sam3.1-tracker-only",
                    tracker_revision="sam31-rev",
                    expected_backend="sam3-tracker-image-pvs",
                    expected_revision="sam3-rev",
                )

    def test_predictor_is_closed_when_a_frame_fails_and_partial_frame_remains_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs = root / "inputs"
            inputs.mkdir()
            config = _write_plan(inputs)
            _prepare_tracking(root, config)
            predictor = FakeImagePredictor(fail_at=1)

            with self.assertRaisesRegex(RuntimeError, "synthetic image failure"):
                run_image_agreement_campaign(
                    [config],
                    input_root=inputs,
                    tracker_root=root / "tracking",
                    output_root=root / "agreement",
                    tracker_backend="sam3.1-tracker-only",
                    tracker_revision="sam31-rev",
                    backend="sam3-tracker-image-pvs",
                    revision="sam3-rev",
                    predictor_factory=lambda: predictor,
                    minimum_prompt_area=8,
                )

            self.assertTrue(predictor.closed)
            self.assertTrue(
                (root / "agreement" / "clip_a" / "masks" / "frame_000000.npz").is_file()
            )
            self.assertFalse((root / "agreement" / "RUN_COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
