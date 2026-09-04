from __future__ import annotations

import unittest

import numpy as np

from ownership_decoder.remote_canary import (
    run_sam31_tracker_canary,
    run_sam3_image_canary,
)
from ownership_decoder.tracking import (
    ActorPrompt,
    FrameSpec,
    PropagationSegment,
    SeedPair,
    TrackingPlan,
)


def _actor(actor: str, offset: float) -> ActorPrompt:
    return ActorPrompt(
        actor_id=actor,
        bbox=(0.0 + offset, 0.0, 0.45 + offset, 1.0),
        positive_points=((0.1 + offset, 0.2), (0.3 + offset, 0.8)),
        negative_points=((0.65 - offset, 0.2), (0.85 - offset, 0.8)),
    )


def _plan() -> TrackingPlan:
    return TrackingPlan(
        clip_id="canary",
        frames=(
            FrameSpec(0, __file_path__.parent / "frame_000000.jpg", "a" * 64, 4, 5),
            FrameSpec(1, __file_path__.parent / "frame_000001.jpg", "b" * 64, 4, 5),
        ),
        seeds=(SeedPair(1, (_actor("A1", 0.0), _actor("A2", 0.55))),),
        propagations=(PropagationSegment(0, 2),),
    )


from pathlib import Path

__file_path__ = Path(__file__)


class Predictor:
    def __init__(self, *, bad_ids: bool = False) -> None:
        self.requests = []
        self.bad_ids = bad_ids

    def handle_request(self, request: dict) -> dict:
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "canary-session"}
        if request["type"] == "add_prompt":
            return {"is_success": True}
        if request["type"] == "close_session":
            return {"is_success": True}
        if request["type"] == "propagate_in_video":
            a1 = np.zeros((4, 5), dtype=bool)
            a2 = np.zeros((4, 5), dtype=bool)
            a1[:, :2] = True
            a2[:, 3:] = True
            ids = np.array([1, 3] if self.bad_ids else [1, 2])
            return {
                "frames": [
                    {
                        "frame_index": 1,
                        "outputs": {
                            "out_obj_ids": ids,
                            "out_probs": np.array([0.9, 0.8]),
                            "out_binary_masks": np.stack([a1, a2]),
                        },
                    }
                ]
            }
        raise AssertionError(request)


class RemoteCanaryTests(unittest.TestCase):
    def test_tracker_canary_uses_seed_frame_exercises_both_actors_and_closes_session(self) -> None:
        predictor = Predictor()

        result = run_sam31_tracker_canary(predictor, _plan())

        self.assertEqual(result["frame_index"], 1)
        self.assertGreater(result["areas"]["A1"], 0)
        self.assertGreater(result["areas"]["A2"], 0)
        self.assertEqual(
            [request["type"] for request in predictor.requests],
            ["start_session", "add_prompt", "add_prompt", "propagate_in_video", "close_session"],
        )
        self.assertEqual(predictor.requests[3]["max_frame_num_to_track"], 1)

    def test_tracker_canary_closes_session_when_output_validation_fails(self) -> None:
        predictor = Predictor(bad_ids=True)

        with self.assertRaisesRegex(ValueError, "object IDs"):
            run_sam31_tracker_canary(predictor, _plan())

        self.assertEqual(predictor.requests[-1]["type"], "close_session")

    def test_image_canary_validates_both_actor_masks_without_retaining_logits(self) -> None:
        class ImagePredictor:
            def segment(self, image_path, prompts, *, expected_shape):
                logits_a1 = np.full(expected_shape, -1.0, dtype=np.float32)
                logits_a2 = np.full(expected_shape, -1.0, dtype=np.float32)
                logits_a1[:, :2] = 1.0
                logits_a2[:, 3:] = 1.0
                return {
                    "logits_A1": logits_a1,
                    "logits_A2": logits_a2,
                    "raw_A1": logits_a1 > 0,
                    "raw_A2": logits_a2 > 0,
                    "score_A1": 0.9,
                    "score_A2": 0.8,
                    "selected_index_A1": 0,
                    "selected_index_A2": 2,
                }

        prompts = {
            actor: {"box": [0, 0, 1, 1], "points": [], "labels": []}
            for actor in ("A1", "A2")
        }
        result = run_sam3_image_canary(
            ImagePredictor(),
            image_path=Path("frame.jpg"),
            prompts=prompts,
            expected_shape=(4, 5),
            clip_id="clip",
            frame_index=3,
        )

        self.assertEqual(result["frame_index"], 3)
        self.assertEqual(result["areas"], {"A1": 8, "A2": 8})
        self.assertNotIn("logits_A1", result)


if __name__ == "__main__":
    unittest.main()
