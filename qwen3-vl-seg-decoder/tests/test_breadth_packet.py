from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.breadth_packet import (
    BreadthPacketContract,
    MaskBreadthPacketContract,
    build_breadth_launch_manifest,
    build_mask_breadth_launch_manifest,
    verify_mask_breadth_packet,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root: Path) -> tuple[Path, Path, Path]:
    input_root = root / "inputs"
    clip = input_root / "clip_a"
    frames = clip / "frames"
    frames.mkdir(parents=True)
    frame = frames / "frame_000001.jpg"
    frame.write_bytes(b"frame")
    clip_manifest = {
        "contract": {"clip_id": "clip_a"},
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
    clip_manifest_path = clip / "clip-manifest.json"
    clip_manifest_path.write_text(json.dumps(clip_manifest))
    config = {
        "schema_version": "1.0",
        "clip_id": "clip_a",
        "clip_manifest_path": "clip_a/clip-manifest.json",
        "clip_manifest_sha256": sha256(clip_manifest_path),
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
    config_path = root / "clip_a.json"
    config_path.write_text(json.dumps(config))

    from ownership_decoder.tracking import load_tracking_plan_config

    plan = load_tracking_plan_config(config_path, input_root=input_root)
    review_root = root / "reviews"
    review_root.mkdir()
    image = review_root / "clip_a.png"
    image.write_bytes(b"review-image")
    image.with_suffix(".png.json").write_text(
        json.dumps(
            {
                "format": "ownership-seed-prompt-review-v1",
                "clip_id": "clip_a",
                "plan_sha256": plan.sha256,
                "image_sha256": sha256(image),
                "seed_frame_indices": [0],
            }
        )
    )
    runtime = root / "runtime.py"
    runtime.write_text("print('runtime')\n")
    return config_path, review_root, runtime


class BreadthPacketTests(unittest.TestCase):
    def test_manifest_binds_inputs_reviews_runtime_and_scientific_review_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, reviews, runtime = make_fixture(root)
            output = root / "packet" / "manifest.json"
            contract = BreadthPacketContract(
                expected_clip_ids=("clip_a",),
                expected_frames_per_clip=1,
                sam_repo_revision="8f0b7f4d4e7eda2ed606ebde6702c93359ad01da",
                checkpoint_sha256="0" * 64,
                checkpoint_size_bytes=3_502_755_717,
                dependency_versions=(
                    ("torch", "2.12.1+cu130"),
                    ("torchvision", "0.27.1+cu130"),
                    ("numpy", "1.26.4"),
                ),
            )

            manifest = build_breadth_launch_manifest(
                contract,
                config_paths=(config,),
                input_root=root / "inputs",
                review_root=reviews,
                runtime_paths=(runtime,),
                output_path=output,
            )

            self.assertEqual(manifest["totals"], {"clips": 1, "frames": 1, "seed_frames": 1})
            self.assertEqual(manifest["clips"][0]["review_image_sha256"], sha256(reviews / "clip_a.png"))
            self.assertEqual(manifest["runtime_files"][0]["sha256"], sha256(runtime))
            self.assertEqual(manifest["human_review"]["contact_ownership_labels"], "pending")
            self.assertEqual(manifest["prompt_contract"], "geometry_only_no_semantic_or_appearance_nudges")
            self.assertEqual(
                (output.with_suffix(".json.sha256")).read_text().strip(),
                sha256(output),
            )

    def test_review_plan_mismatch_is_rejected_before_manifest_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, reviews, runtime = make_fixture(root)
            sidecar = reviews / "clip_a.png.json"
            payload = json.loads(sidecar.read_text())
            payload["plan_sha256"] = "f" * 64
            sidecar.write_text(json.dumps(payload))
            output = root / "manifest.json"
            contract = BreadthPacketContract(
                expected_clip_ids=("clip_a",),
                expected_frames_per_clip=1,
                sam_repo_revision="rev",
                checkpoint_sha256="0" * 64,
                checkpoint_size_bytes=1,
                dependency_versions=(("torch", "test"),),
            )

            with self.assertRaisesRegex(ValueError, "review.*plan"):
                build_breadth_launch_manifest(
                    contract,
                    config_paths=(config,),
                    input_root=root / "inputs",
                    review_root=reviews,
                    runtime_paths=(runtime,),
                    output_path=output,
                )

            self.assertFalse(output.exists())

    def test_v2_manifest_binds_both_models_environment_and_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, reviews, runtime = make_fixture(root)
            environment = root / "requirements.txt"
            environment.write_text("numpy==1.26.4\n")
            model_directory = root / "sam3"
            model_directory.mkdir()
            artifact_records = []
            for name in (
                "config.json",
                "merges.txt",
                "model.safetensors",
                "processor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ):
                path = model_directory / name
                path.write_bytes(name.encode())
                artifact_records.append(
                    {
                        "relative_path": name,
                        "sha256": sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            artifact_manifest = root / "sam3-model-artifacts.json"
            artifact_manifest.write_text(
                json.dumps(
                    {
                        "format": "sam3-model-artifacts-v1",
                        "model_revision": "sam3-hf-revision",
                        "artifacts": artifact_records,
                    }
                )
            )
            sam31_checkpoint = root / "sam3.1_multiplex.pt"
            sam31_checkpoint.write_bytes(b"sam31")
            output = root / "manifest.json"
            contract = MaskBreadthPacketContract(
                expected_clip_ids=("clip_a",),
                expected_frames_per_clip=1,
                sam_repo_revision="8f0b7f4d4e7eda2ed606ebde6702c93359ad01da",
                sam31_checkpoint_sha256=sha256(sam31_checkpoint),
                sam31_checkpoint_size_bytes=sam31_checkpoint.stat().st_size,
                sam3_model_revision="sam3-hf-revision",
                container_image_digest=(
                    "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:"
                    + "1" * 64
                ),
                dependency_versions=(("numpy", "1.26.4"),),
            )

            manifest = build_mask_breadth_launch_manifest(
                contract,
                config_paths=(config,),
                input_root=root / "inputs",
                review_root=reviews,
                runtime_paths=(runtime,),
                environment_paths=(environment,),
                sam31_checkpoint_path=sam31_checkpoint,
                sam3_artifact_manifest_path=artifact_manifest,
                sam3_model_directory=model_directory,
                output_path=output,
            )

            self.assertEqual(manifest["format"], "sam31-sam3-breadth-launch-packet-v2")
            self.assertEqual(manifest["execution_order"], ["sam3.1_tracker", "sam3_image"])
            self.assertEqual(manifest["models"]["base_sam3"]["artifact_count"], 8)
            self.assertEqual(
                manifest["models"]["base_sam3"]["revision"],
                "sam3-hf-revision",
            )
            self.assertEqual(manifest["environment_files"][0]["sha256"], sha256(environment))
            self.assertFalse(manifest["human_review"]["pseudo_labels_usable_for_training"])
            self.assertEqual(output.with_suffix(".json.sha256").read_text().strip(), sha256(output))
            verified = verify_mask_breadth_packet(
                root,
                input_root=root / "inputs",
            )
            self.assertEqual(verified["verified_clips"], 1)
            environment.write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "environment.*mismatch"):
                verify_mask_breadth_packet(root, input_root=root / "inputs")

    def test_v2_rejects_tampered_base_model_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, reviews, runtime = make_fixture(root)
            model_directory = root / "sam3"
            model_directory.mkdir()
            records = []
            for name in (
                "config.json",
                "merges.txt",
                "model.safetensors",
                "processor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ):
                path = model_directory / name
                path.write_bytes(name.encode())
                records.append(
                    {
                        "relative_path": name,
                        "sha256": sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            artifact_manifest = root / "artifacts.json"
            artifact_manifest.write_text(
                json.dumps(
                    {
                        "format": "sam3-model-artifacts-v1",
                        "model_revision": "revision",
                        "artifacts": records,
                    }
                )
            )
            (model_directory / "model.safetensors").write_bytes(b"tampered")
            environment = root / "requirements.txt"
            environment.write_text("numpy==1.26.4\n")
            sam31_checkpoint = root / "sam3.1_multiplex.pt"
            sam31_checkpoint.write_bytes(b"sam31")
            output = root / "manifest.json"
            contract = MaskBreadthPacketContract(
                expected_clip_ids=("clip_a",),
                expected_frames_per_clip=1,
                sam_repo_revision="repo-revision",
                sam31_checkpoint_sha256=sha256(sam31_checkpoint),
                sam31_checkpoint_size_bytes=sam31_checkpoint.stat().st_size,
                sam3_model_revision="revision",
                container_image_digest="image@sha256:" + "1" * 64,
                dependency_versions=(("numpy", "1.26.4"),),
            )

            with self.assertRaisesRegex(ValueError, "SAM3 artifact.*mismatch"):
                build_mask_breadth_launch_manifest(
                    contract,
                    config_paths=(config,),
                    input_root=root / "inputs",
                    review_root=reviews,
                    runtime_paths=(runtime,),
                    environment_paths=(environment,),
                    sam31_checkpoint_path=sam31_checkpoint,
                    sam3_artifact_manifest_path=artifact_manifest,
                    sam3_model_directory=model_directory,
                    output_path=output,
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
