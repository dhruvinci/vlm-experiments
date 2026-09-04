from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ownership_decoder.remote_preflight import (
    RemoteHardwareSnapshot,
    RemoteRuntimeApproval,
    RemoteRuntimeContract,
    RemoteRuntimePreflightError,
    RequiredArtifact,
)
from ownership_decoder.sam3_image_adapter import (
    Sam3ImageRuntimeBindings,
    build_sam3_image_predictor,
    select_actor_multimasks,
)


GIB = 1024**3


def _hardware() -> RemoteHardwareSnapshot:
    return RemoteHardwareSnapshot(
        gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpu_total_bytes=96 * GIB,
        gpu_free_bytes=90 * GIB,
        compute_capability=(12, 0),
        driver_version=(580, 65, 6),
        host_available_bytes=64 * GIB,
        workspace_free_bytes=200 * GIB,
    )


def _contract(root: Path) -> tuple[RemoteRuntimeContract, Path]:
    repo = root / "sam3"
    repo.mkdir()
    checkpoint = root / "sam31.pt"
    checkpoint.write_bytes(b"sam31")
    model_dir = root / "sam3-model"
    model_dir.mkdir()
    artifacts = []
    for name, payload in (
        ("config.json", b"config"),
        ("model.safetensors", b"weights"),
        ("processor_config.json", b"processor"),
    ):
        path = model_dir / name
        path.write_bytes(payload)
        import hashlib

        artifacts.append(
            RequiredArtifact(
                path=path,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    return (
        RemoteRuntimeContract(
            sam_repo_path=repo,
            sam_repo_revision="revision",
            checkpoint_path=checkpoint,
            checkpoint_sha256="a" * 64,
            workspace_path=root,
            required_distribution_versions=(("torch", "2.12.1+cu130"),),
            additional_artifacts=tuple(artifacts),
        ),
        model_dir,
    )


class FakeBatch(dict):
    def __init__(self) -> None:
        super().__init__(pixel_values="pixels", original_sizes=[(4, 5)])
        self.moved_to = None

    def to(self, device: str):
        self.moved_to = device
        return self


class FakeProcessor:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeBatch()

    def post_process_masks(self, masks, original_sizes, **kwargs):
        self.calls.append(
            {
                "post_masks": np.asarray(masks),
                "original_sizes": original_sizes,
                **kwargs,
            }
        )
        return [self.logits]


class FakeModel:
    def __init__(self, iou_scores: np.ndarray) -> None:
        self.iou_scores = iou_scores
        self.device = None
        self.evaluated = False
        self.calls = []

    def to(self, *, device: str):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            pred_masks=np.zeros((1, 2, 3, 2, 2), dtype=np.float32),
            iou_scores=self.iou_scores,
        )


class Sam3ImageAdapterTests(unittest.TestCase):
    def test_selects_highest_iou_candidate_independently_for_each_actor(self) -> None:
        logits = np.zeros((2, 3, 4, 5), dtype=np.float32)
        logits[0, 2, :, :2] = 3.0
        logits[1, 1, :, 3:] = 4.0
        scores = np.array([[[0.1, 0.2, 0.9], [0.3, 0.8, 0.4]]], dtype=np.float32)

        selected = select_actor_multimasks(logits, scores, expected_shape=(4, 5))

        self.assertEqual(selected["selected_index_A1"], 2)
        self.assertEqual(selected["selected_index_A2"], 1)
        self.assertAlmostEqual(selected["score_A1"], 0.9, places=6)
        self.assertAlmostEqual(selected["score_A2"], 0.8, places=6)
        np.testing.assert_array_equal(selected["raw_A1"], logits[0, 2] > 0)
        np.testing.assert_array_equal(selected["raw_A2"], logits[1, 1] > 0)

    def test_rejects_unbound_approval_before_loading_transformers_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, model_dir = _contract(Path(raw))
            approval = RemoteRuntimeApproval(
                contract_sha256="0" * 64,
                checkpoint_sha256=contract.checkpoint_sha256,
                hardware=_hardware(),
            )
            loaded = False

            def factory():
                nonlocal loaded
                loaded = True
                raise AssertionError("heavy runtime loaded")

            with self.assertRaises(RemoteRuntimePreflightError):
                build_sam3_image_predictor(
                    contract,
                    approval,
                    model_directory=model_dir,
                    runtime_bindings_factory=factory,
                )

            self.assertFalse(loaded)

    def test_builds_local_only_sdpa_model_and_runs_two_actor_multimask_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, model_dir = _contract(Path(raw))
            approval = RemoteRuntimeApproval(
                contract_sha256=contract.sha256,
                checkpoint_sha256=contract.checkpoint_sha256,
                hardware=_hardware(),
                additional_artifact_sha256=tuple(
                    (str(item.path.resolve()), item.sha256)
                    for item in contract.additional_artifacts
                ),
            )
            logits = np.zeros((2, 3, 4, 5), dtype=np.float32)
            logits[0, 0, :, :3] = 2.0
            logits[1, 2, :, 2:] = 5.0
            processor = FakeProcessor(logits)
            model = FakeModel(
                np.array([[[0.9, 0.2, 0.1], [0.1, 0.2, 0.95]]], dtype=np.float32)
            )
            load_calls = []
            cleanup_calls = []
            bindings = Sam3ImageRuntimeBindings(
                load_model=lambda path: load_calls.append(("model", path)) or model,
                load_processor=lambda path: load_calls.append(("processor", path)) or processor,
                inference_context_factory=nullcontext,
                load_rgb_image=lambda path: f"image:{path.name}",
                collect=lambda: cleanup_calls.append("collect"),
                release_cuda_cache=lambda: cleanup_calls.append("cuda"),
            )

            predictor = build_sam3_image_predictor(
                contract,
                approval,
                model_directory=model_dir,
                runtime_bindings_factory=lambda: bindings,
            )
            prompts = {
                "A1": {
                    "box": [0.0, 0.0, 2.0, 3.0],
                    "points": [[1.0, 1.0], [1.0, 2.0], [4.0, 1.0], [4.0, 2.0]],
                    "labels": [1, 1, 0, 0],
                },
                "A2": {
                    "box": [2.0, 0.0, 4.0, 3.0],
                    "points": [[4.0, 1.0], [4.0, 2.0], [1.0, 1.0], [1.0, 2.0]],
                    "labels": [1, 1, 0, 0],
                },
            }

            result = predictor.segment(
                Path("frame.jpg"), prompts, expected_shape=(4, 5)
            )

            self.assertEqual(load_calls, [("processor", model_dir), ("model", model_dir)])
            self.assertEqual(model.device, "cuda")
            self.assertTrue(model.evaluated)
            call = processor.calls[0]
            self.assertEqual(call["input_boxes"], [[prompts["A1"]["box"], prompts["A2"]["box"]]])
            self.assertEqual(call["input_points"], [[prompts["A1"]["points"], prompts["A2"]["points"]]])
            self.assertTrue(model.calls[0]["multimask_output"])
            self.assertFalse(processor.calls[1]["binarize"])
            self.assertEqual(result["selected_index_A1"], 0)
            self.assertEqual(result["selected_index_A2"], 2)

            predictor.close()
            self.assertEqual(
                cleanup_calls,
                ["collect", "cuda", "collect", "cuda", "collect", "cuda"],
            )
            with self.assertRaisesRegex(RuntimeError, "closed"):
                predictor.segment(Path("frame.jpg"), prompts, expected_shape=(4, 5))

    def test_missing_bound_base_model_file_fails_before_runtime_load(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, model_dir = _contract(Path(raw))
            incomplete = RemoteRuntimeContract(
                **{
                    **contract.__dict__,
                    "additional_artifacts": contract.additional_artifacts[:2],
                }
            )
            approval = RemoteRuntimeApproval(
                contract_sha256=incomplete.sha256,
                checkpoint_sha256=incomplete.checkpoint_sha256,
                hardware=_hardware(),
                additional_artifact_sha256=tuple(
                    (str(item.path.resolve()), item.sha256)
                    for item in incomplete.additional_artifacts
                ),
            )

            with self.assertRaisesRegex(RemoteRuntimePreflightError, "processor_config"):
                build_sam3_image_predictor(
                    incomplete,
                    approval,
                    model_directory=model_dir,
                    runtime_bindings_factory=lambda: self.fail("loaded runtime"),
                )


if __name__ == "__main__":
    unittest.main()
