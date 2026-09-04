from __future__ import annotations

from typing import Any

import numpy as np

from .tracking import TrackingPlan, _normalize_predictor_frame
from .image_agreement import _normalize_prediction


_ACTOR_OBJECT_IDS = {"A1": 1, "A2": 2}


def run_sam31_tracker_canary(
    predictor: Any,
    plan: TrackingPlan,
) -> dict[str, Any]:
    """Exercise one real seeded frame before tracker bulk extraction."""

    seed = plan.seeds[0]
    session_id: str | None = None
    try:
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(plan.frame_directory),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": False,
            }
        )
        session_id = str(response["session_id"])
        if not session_id:
            raise RuntimeError("SAM3.1 canary returned an empty session ID")
        for actor in seed.ordered_actors():
            predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": seed.frame_index,
                    "obj_id": _ACTOR_OBJECT_IDS[actor.actor_id],
                    **actor.predictor_payload(),
                    "clear_old_points": True,
                    "rel_coordinates": True,
                }
            )
        response = predictor.handle_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_idx": seed.frame_index,
                "max_frame_num_to_track": 1,
                "reverse": False,
                "tqdm_disable": True,
            }
        )
        frames = response.get("frames") if isinstance(response, dict) else None
        if not isinstance(frames, list) or len(frames) != 1:
            raise ValueError("SAM3.1 canary must return exactly one frame")
        normalized = _normalize_predictor_frame(
            frames[0],
            expected_shape=plan.frame_shape,
            frame_count=len(plan.frames),
        )
        if normalized.frame_index != seed.frame_index:
            raise ValueError("SAM3.1 canary returned the wrong seed frame")
        areas = {
            "A1": int(np.count_nonzero(normalized.a1)),
            "A2": int(np.count_nonzero(normalized.a2)),
        }
        if min(areas.values()) < 1:
            raise ValueError("SAM3.1 canary produced an empty actor mask")
        return {
            "format": "sam31-tracker-canary-v1",
            "clip_id": plan.clip_id,
            "frame_index": normalized.frame_index,
            "areas": areas,
            "scores": {
                "A1": normalized.score_a1,
                "A2": normalized.score_a2,
            },
            "raw_overlap_pixels": int(
                np.count_nonzero(normalized.raw_a1 & normalized.raw_a2)
            ),
        }
    finally:
        if session_id is not None:
            predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": True,
                }
            )


def run_sam3_image_canary(
    predictor: Any,
    *,
    image_path: Any,
    prompts: dict[str, dict[str, Any]],
    expected_shape: tuple[int, int],
    clip_id: str,
    frame_index: int,
) -> dict[str, Any]:
    """Exercise one image after base SAM3 load and retain only compact diagnostics."""

    raw = predictor.segment(
        image_path,
        prompts,
        expected_shape=expected_shape,
    )
    prediction = _normalize_prediction(raw, expected_shape=expected_shape)
    areas = {
        "A1": int(np.count_nonzero(prediction["A1"])),
        "A2": int(np.count_nonzero(prediction["A2"])),
    }
    if min(areas.values()) < 1:
        raise ValueError("base SAM3 canary produced an empty actor mask")
    return {
        "format": "sam3-image-canary-v1",
        "clip_id": clip_id,
        "frame_index": frame_index,
        "areas": areas,
        "scores": {
            "A1": prediction["score_A1"],
            "A2": prediction["score_A2"],
        },
        "selected_indices": {
            "A1": prediction["selected_index_A1"],
            "A2": prediction["selected_index_A2"],
        },
        "raw_overlap_pixels": prediction["raw_overlap_pixels"],
        "unresolved_tie_pixels": prediction["unresolved_tie_pixels"],
    }
