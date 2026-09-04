from __future__ import annotations

import hashlib
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ownership_decoder.remote_preflight import (
    RemoteHardwareSnapshot,
    RemoteRuntimeApproval,
    RemoteRuntimeContract,
)
from ownership_decoder.sam31_tracker_adapter import (
    Sam31RuntimeBindings,
    TrackerOnlySam31Predictor,
    build_tracker_only_sam31,
    initialize_dimension_tracker_state,
    remap_tracker_only_checkpoint,
)


GIB = 1024**3


def numpy_point_tensors(points, labels):
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class Sam31TrackerAdapterTests(unittest.TestCase):
    def test_checkpoint_remap_combines_tracker_and_shared_detector_backbone(self) -> None:
        checkpoint = {
            "tracker.model.mask_decoder.weight": "tracker-value",
            "detector.backbone.vision_backbone.trunk.weight": "backbone-value",
            "detector.text_encoder.weight": "unused",
        }

        remapped = remap_tracker_only_checkpoint(
            checkpoint,
            {"mask_decoder.weight", "backbone.vision_backbone.trunk.weight"},
        )

        self.assertEqual(
            remapped,
            {
                "mask_decoder.weight": "tracker-value",
                "backbone.vision_backbone.trunk.weight": "backbone-value",
            },
        )

    def test_builder_rejects_unbound_approval_before_loading_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            contract = RemoteRuntimeContract(
                sam_repo_path=repo,
                sam_repo_revision="revision",
                checkpoint_path=checkpoint,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                workspace_path=root,
                required_distribution_versions=(("torch", "test"),),
            )
            approval = RemoteRuntimeApproval(
                contract_sha256="0" * 64,
                checkpoint_sha256=contract.checkpoint_sha256,
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
            touched = False

            def load_bindings():
                nonlocal touched
                touched = True
                raise AssertionError("runtime must not be imported")

            with self.assertRaisesRegex(Exception, "approval"):
                build_tracker_only_sam31(
                    contract,
                    approval,
                    runtime_bindings_factory=load_bindings,
                )

            self.assertFalse(touched)

    def test_builder_strictly_loads_tracker_only_model_and_owns_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            contract = RemoteRuntimeContract(
                sam_repo_path=repo,
                sam_repo_revision="revision",
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                workspace_path=root,
                required_distribution_versions=(("torch", "test"),),
            )
            approval = RemoteRuntimeApproval(
                contract_sha256=contract.sha256,
                checkpoint_sha256=checkpoint_sha,
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
            events: list[object] = []

            class Model:
                def state_dict(self):
                    return {
                        "mask_decoder.weight": FakeTensor((2, 3)),
                        "backbone.vision.weight": FakeTensor((4,)),
                    }

                def load_state_dict(self, state, strict):
                    events.append(("load_state", tuple(sorted(state)), strict))

                def forward_image(self, image, **kwargs):
                    return None

                def to(self, *, device):
                    events.append(("to", device))
                    return self

                def eval(self):
                    events.append("eval")
                    return self

            model = Model()
            checkpoint = {
                "model": {
                    "tracker.model.mask_decoder.weight": FakeTensor((2, 3)),
                    "detector.backbone.vision.weight": FakeTensor((4,)),
                }
            }

            def build_model(**kwargs):
                events.append(("build", kwargs))
                return model

            bindings = Sam31RuntimeBindings(
                build_model=build_model,
                load_checkpoint=lambda path: events.append(("checkpoint", path)) or checkpoint,
                inference_context_factory=nullcontext,
                point_tensor_factory=numpy_point_tensors,
                collect=lambda: events.append("collect"),
                release_cuda_cache=lambda: events.append("release_cuda"),
            )

            predictor = build_tracker_only_sam31(
                contract,
                approval,
                runtime_bindings_factory=lambda: bindings,
            )

            build_kwargs = next(value[1] for value in events if isinstance(value, tuple) and value[0] == "build")
            self.assertEqual(build_kwargs["device"], "cpu")
            self.assertFalse(build_kwargs["compile"])
            self.assertFalse(build_kwargs["use_fa3"])
            self.assertIn(("load_state", ("backbone.vision.weight", "mask_decoder.weight"), True), events)
            self.assertIn(("to", "cuda"), events)
            predictor.close()
            self.assertEqual(events[-2:], ["collect", "release_cuda"])

    def test_dimension_state_loader_keeps_normalized_video_tensors_on_cpu(self) -> None:
        calls: list[object] = []

        class Model:
            image_size = 1008

            def init_state(self, video_height, video_width, num_frames, **kwargs):
                calls.append((video_height, video_width, num_frames, kwargs))
                return {"height": video_height, "width": video_width, "count": num_frames}

        def frame_loader(**kwargs):
            calls.append(kwargs)
            return ["frame-0", "frame-1"], 2560, 1440

        state = initialize_dimension_tracker_state(
            Model(), Path("/tmp/frames"), frame_loader=frame_loader
        )

        self.assertEqual(state["images"], ["frame-0", "frame-1"])
        self.assertTrue(calls[0]["offload_video_to_cpu"])
        self.assertFalse(calls[1][3]["offload_state_to_cpu"])

    def test_propagation_response_is_bounded_to_current_chunk_not_history(self) -> None:
        class Model:
            def init_state(self, video_path, **kwargs):
                return {"video_path": video_path, "object_id": None}

            def add_new_points(self, inference_state, frame_idx, obj_id, **kwargs):
                inference_state["object_id"] = obj_id
                mask = np.full((1, 1, 3, 4), float(obj_id), dtype=np.float32)
                return frame_idx, [obj_id], None, mask

            def propagate_in_video(
                self,
                inference_state,
                start_frame_idx,
                max_frame_num_to_track,
                reverse,
                **kwargs,
            ):
                object_id = inference_state["object_id"]
                direction = -1 if reverse else 1
                for offset in range(max_frame_num_to_track):
                    index = start_frame_idx + direction * offset
                    mask = np.full((1, 1, 3, 4), float(object_id), dtype=np.float32)
                    yield index, [object_id], None, mask, np.asarray([object_id / 10])

        predictor = TrackerOnlySam31Predictor(
            Model(),
            inference_context_factory=nullcontext,
            point_tensor_factory=numpy_point_tensors,
        )
        session = predictor.handle_request(
            {"type": "start_session", "resource_path": "/tmp/frames"}
        )["session_id"]
        for object_id in (1, 2):
            predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session,
                    "frame_index": 0,
                    "obj_id": object_id,
                    "points": [[0.1, 0.2]],
                    "point_labels": [1],
                }
            )

        first = predictor.handle_request(
            {
                "type": "propagate_in_video",
                "session_id": session,
                "start_frame_idx": 0,
                "max_frame_num_to_track": 8,
                "reverse": False,
            }
        )
        second = predictor.handle_request(
            {
                "type": "propagate_in_video",
                "session_id": session,
                "start_frame_idx": 8,
                "max_frame_num_to_track": 8,
                "reverse": False,
            }
        )

        self.assertEqual([row["frame_index"] for row in first["frames"]], list(range(8)))
        self.assertEqual([row["frame_index"] for row in second["frames"]], list(range(8, 16)))
        self.assertLessEqual(len(first["frames"]), 8)
        self.assertLessEqual(len(second["frames"]), 8)
        self.assertEqual(second["frames"][0]["outputs"]["out_obj_ids"].tolist(), [1, 2])

    def test_close_requires_sessions_to_be_closed_and_blocks_future_requests(self) -> None:
        class Model:
            def init_state(self, video_path, **kwargs):
                return {"video_path": video_path}

        predictor = TrackerOnlySam31Predictor(
            Model(),
            inference_context_factory=nullcontext,
            point_tensor_factory=numpy_point_tensors,
        )
        session = predictor.handle_request(
            {"type": "start_session", "resource_path": "/tmp/frames"}
        )["session_id"]

        with self.assertRaisesRegex(RuntimeError, "active sessions"):
            predictor.close()
        predictor.handle_request({"type": "close_session", "session_id": session})
        predictor.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            predictor.handle_request({"type": "start_session", "resource_path": "/tmp/frames"})


if __name__ == "__main__":
    unittest.main()
