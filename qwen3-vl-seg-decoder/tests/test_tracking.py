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

from ownership_decoder.tracking import (
    ActorPrompt,
    FrameSpec,
    PropagationSegment,
    SeedPair,
    TrackingArtifactError,
    TrackingPlan,
    forward_propagation_chunks,
    load_tracking_plan_config,
    load_completed_tracking_run,
    run_tracking_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actor_prompt(actor_id: str, offset: float) -> ActorPrompt:
    return ActorPrompt(
        actor_id=actor_id,
        bbox=(0.05 + offset, 0.05, 0.45 + offset, 0.90),
        positive_points=((0.15 + offset, 0.25), (0.30 + offset, 0.70)),
        negative_points=((0.65 - offset, 0.25), (0.80 - offset, 0.70)),
    )


def _seed(frame_index: int = 0) -> SeedPair:
    return SeedPair(
        frame_index=frame_index,
        actors=(_actor_prompt("A1", 0.0), _actor_prompt("A2", 0.45)),
    )


def _frames(directory: Path, count: int = 3) -> tuple[FrameSpec, ...]:
    result = []
    for index in range(count):
        path = directory / f"frame_{index + 1:06d}.jpg"
        path.write_bytes(f"synthetic-frame-{index}".encode())
        result.append(
            FrameSpec(
                frame_index=index,
                path=path,
                sha256=_sha256(path),
                height=4,
                width=5,
            )
        )
    return tuple(result)


def _plan(directory: Path) -> TrackingPlan:
    return TrackingPlan(
        clip_id="test_clip",
        frames=_frames(directory),
        seeds=(_seed(),),
        propagations=(PropagationSegment(0, 3, reverse=False),),
    )


class FakePredictor:
    def __init__(self, *, fail_during_propagation: bool = False, nonfinite: bool = False):
        self.requests: list[dict] = []
        self.closed = False
        self.fail_during_propagation = fail_during_propagation
        self.nonfinite = nonfinite

    def handle_request(self, request: dict) -> dict:
        self.requests.append(request)
        request_type = request["type"]
        if request_type == "start_session":
            return {"session_id": "fake-session"}
        if request_type == "add_prompt":
            return {"is_success": True}
        if request_type == "close_session":
            self.closed = True
            return {"is_success": True}
        if request_type != "propagate_in_video":
            raise AssertionError(request_type)
        if self.fail_during_propagation:
            raise RuntimeError("synthetic propagation failure")

        start = request["start_frame_idx"]
        count = request["max_frame_num_to_track"]
        direction = -1 if request["reverse"] else 1
        frames = []
        for frame_index in (start + direction * step for step in range(count)):
            raw_a1 = np.zeros((4, 5), dtype=bool)
            raw_a2 = np.zeros((4, 5), dtype=bool)
            raw_a1[:3, :3] = True
            raw_a2[2:, 2:] = True
            scores = np.array([0.4, np.nan if self.nonfinite else 0.9], dtype=np.float32)
            frames.append(
                {
                    "frame_index": frame_index,
                    "outputs": {
                        # Return the objects in reverse order to exercise explicit mapping.
                        "out_obj_ids": np.array([2, 1], dtype=np.int64),
                        "out_probs": scores,
                        "out_binary_masks": np.stack([raw_a2, raw_a1]),
                    },
                }
            )
        return {"frames": frames}


class NeverPredictor:
    def handle_request(self, request: dict) -> dict:
        raise AssertionError(f"completed resume touched predictor: {request}")


class TrackingContractTests(unittest.TestCase):
    def test_actor_prompt_rejects_invalid_geometry_and_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "bbox"):
            ActorPrompt(
                actor_id="A1",
                bbox=(0.7, 0.1, 0.2, 0.8),
                positive_points=((0.3, 0.3), (0.4, 0.4)),
                negative_points=((0.8, 0.3), (0.9, 0.4)),
            )
        with self.assertRaisesRegex(ValueError, "inside"):
            ActorPrompt(
                actor_id="A1",
                bbox=(0.1, 0.1, 0.5, 0.8),
                positive_points=((0.3, 0.3), (0.9, 0.4)),
                negative_points=((0.8, 0.3), (0.9, 0.4)),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            ActorPrompt(
                actor_id="A1",
                bbox=(0.1, 0.1, 0.5, 0.8),
                positive_points=((0.3, 0.3), (float("nan"), 0.4)),
                negative_points=((0.8, 0.3), (0.9, 0.4)),
            )

    def test_plan_rejects_incomplete_temporal_coverage_and_duplicate_actors(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            frames = _frames(directory)
            with self.assertRaisesRegex(ValueError, "A1.*A2"):
                SeedPair(frame_index=0, actors=(_actor_prompt("A1", 0.0), _actor_prompt("A1", 0.45)))
            with self.assertRaisesRegex(ValueError, "coverage"):
                TrackingPlan(
                    clip_id="test_clip",
                    frames=frames,
                    seeds=(_seed(),),
                    propagations=(PropagationSegment(0, 2, reverse=False),),
                )

    def test_forward_chunks_bound_each_response_and_cover_clip_exactly(self) -> None:
        chunks = forward_propagation_chunks(24)

        self.assertEqual(
            [(chunk.start_frame_idx, chunk.max_frame_num_to_track) for chunk in chunks],
            [(0, 8), (8, 8), (16, 8)],
        )
        self.assertEqual(
            {index for chunk in chunks for index in chunk.frame_indices()},
            set(range(24)),
        )

    def test_strict_config_loader_binds_prompts_to_hashed_frame_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            clip = root / "test_clip"
            frames_dir = clip / "frames"
            frames_dir.mkdir(parents=True)
            frames = _frames(frames_dir)
            manifest = {
                "contract": {"clip_id": "test_clip"},
                "decode": {"frame_count": len(frames)},
                "frames": [
                    {
                        "clip_frame_index": frame.frame_index,
                        "path": f"frames/{frame.path.name}",
                        "sha256": frame.sha256,
                        "height": frame.height,
                        "width": frame.width,
                    }
                    for frame in frames
                ],
            }
            manifest_path = clip / "clip-manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            config = {
                "schema_version": "1.0",
                "clip_id": "test_clip",
                "clip_manifest_path": "test_clip/clip-manifest.json",
                "clip_manifest_sha256": _sha256(manifest_path),
                "seeds": [
                    {
                        "frame_index": 0,
                        "actors": [
                            {
                                "actor_id": actor.actor_id,
                                "bbox": list(actor.bbox),
                                "positive_points": [list(point) for point in actor.positive_points],
                                "negative_points": [list(point) for point in actor.negative_points],
                            }
                            for actor in _seed().ordered_actors()
                        ],
                    }
                ],
                "propagations": [
                    {
                        "start_frame_idx": 0,
                        "max_frame_num_to_track": 3,
                        "reverse": False,
                    }
                ],
            }
            config_path = root / "plan.json"
            config_path.write_text(json.dumps(config))

            loaded = load_tracking_plan_config(config_path, input_root=root)

            self.assertEqual(loaded.clip_id, "test_clip")
            self.assertEqual(loaded.sha256, _plan(frames_dir).sha256)

            config["appearance_hint"] = "forbidden semantic nudge"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unexpected.*appearance_hint"):
                load_tracking_plan_config(config_path, input_root=root)

    def test_config_loader_rejects_manifest_checksum_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            clip = root / "test_clip"
            frames_dir = clip / "frames"
            frames_dir.mkdir(parents=True)
            frames = _frames(frames_dir)
            manifest_path = clip / "clip-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "contract": {"clip_id": "test_clip"},
                        "decode": {"frame_count": 3},
                        "frames": [
                            {
                                "clip_frame_index": frame.frame_index,
                                "path": f"frames/{frame.path.name}",
                                "sha256": frame.sha256,
                                "height": frame.height,
                                "width": frame.width,
                            }
                            for frame in frames
                        ],
                    }
                )
            )
            config_path = root / "plan.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "clip_id": "test_clip",
                        "clip_manifest_path": "test_clip/clip-manifest.json",
                        "clip_manifest_sha256": "0" * 64,
                        "seeds": [
                            {
                                "frame_index": 0,
                                "actors": [
                                    {
                                        "actor_id": actor.actor_id,
                                        "bbox": list(actor.bbox),
                                        "positive_points": [list(p) for p in actor.positive_points],
                                        "negative_points": [list(p) for p in actor.negative_points],
                                    }
                                    for actor in _seed().ordered_actors()
                                ],
                            }
                        ],
                        "propagations": [
                            {"start_frame_idx": 0, "max_frame_num_to_track": 3, "reverse": False}
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(TrackingArtifactError, "manifest checksum"):
                load_tracking_plan_config(config_path, input_root=root)

    def test_run_maps_actor_ids_resolves_overlap_and_writes_atomic_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            plan = _plan(root / "frames") if (root / "frames").mkdir() is None else None
            output = root / "run"
            predictor = FakePredictor()

            manifest = run_tracking_plan(
                predictor,
                plan,
                output,
                backend="sam3.1-tracker-only",
                revision="test-revision",
            )

            self.assertTrue(predictor.closed)
            self.assertEqual(manifest["frame_count"], 3)
            self.assertTrue((output / "RUN_COMPLETE").exists())
            self.assertTrue((output / "journal.jsonl").exists())
            self.assertEqual(len(list((output / "masks").glob("*.npz"))), 3)
            with np.load(output / "masks" / "frame_000000.npz") as arrays:
                self.assertFalse(np.any(arrays["A1"] & arrays["A2"]))
                self.assertTrue(arrays["A1"][2, 2])
                self.assertFalse(arrays["A2"][2, 2])
                self.assertEqual(int(np.sum(arrays["raw_A1"] & arrays["raw_A2"])), 1)
            restored = load_completed_tracking_run(output, expected_plan=plan)
            self.assertEqual(restored, manifest)

    def test_completed_run_resumes_without_touching_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            frame_dir = root / "frames"
            frame_dir.mkdir()
            plan = _plan(frame_dir)
            output = root / "run"
            first = run_tracking_plan(
                FakePredictor(), plan, output, backend="sam3.1-tracker-only", revision="rev"
            )

            resumed = run_tracking_plan(
                NeverPredictor(), plan, output, backend="sam3.1-tracker-only", revision="rev"
            )

            self.assertEqual(resumed, first)

    def test_corrupted_completed_run_is_rejected_before_predictor_use(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            frame_dir = root / "frames"
            frame_dir.mkdir()
            plan = _plan(frame_dir)
            output = root / "run"
            run_tracking_plan(
                FakePredictor(), plan, output, backend="sam3.1-tracker-only", revision="rev"
            )
            artifact = output / "masks" / "frame_000001.npz"
            artifact.write_bytes(artifact.read_bytes() + b"corruption")

            with self.assertRaisesRegex(TrackingArtifactError, "checksum|size"):
                run_tracking_plan(
                    NeverPredictor(), plan, output, backend="sam3.1-tracker-only", revision="rev"
                )

    def test_session_is_closed_when_propagation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            frame_dir = root / "frames"
            frame_dir.mkdir()
            plan = _plan(frame_dir)
            predictor = FakePredictor(fail_during_propagation=True)

            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                run_tracking_plan(
                    predictor,
                    plan,
                    root / "run",
                    backend="sam3.1-tracker-only",
                    revision="rev",
                )

            self.assertTrue(predictor.closed)
            self.assertFalse((root / "run" / "RUN_COMPLETE").exists())

    def test_run_reports_seed_segment_cleanup_and_completion_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            frame_dir = root / "frames"
            frame_dir.mkdir()
            plan = _plan(frame_dir)
            events: list[dict] = []

            run_tracking_plan(
                FakePredictor(),
                plan,
                root / "run",
                backend="sam3.1-tracker-only",
                revision="rev",
                event_callback=events.append,
            )

            self.assertEqual(
                [event["event"] for event in events],
                [
                    "session_started",
                    "seed_prompted",
                    "seed_prompted",
                    "segment_started",
                    "segment_completed",
                    "session_closed",
                    "run_completed",
                ],
            )
            self.assertEqual(events[0]["clip_id"], "test_clip")
            self.assertEqual(events[3]["segment_index"], 0)
            self.assertEqual(events[4]["committed_frame_count"], 3)

    def test_nonfinite_scores_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            frame_dir = root / "frames"
            frame_dir.mkdir()
            plan = _plan(frame_dir)
            predictor = FakePredictor(nonfinite=True)

            with self.assertRaisesRegex(ValueError, "finite"):
                run_tracking_plan(
                    predictor,
                    plan,
                    root / "run",
                    backend="sam3.1-tracker-only",
                    revision="rev",
                )

            self.assertTrue(predictor.closed)


if __name__ == "__main__":
    unittest.main()
