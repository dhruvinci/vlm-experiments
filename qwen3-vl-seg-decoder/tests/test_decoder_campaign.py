from __future__ import annotations

import sys
import hashlib
import json
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.decoder_campaign import (
    DEFAULT_STATIC_SCREEN,
    DecoderFoldRunSpec,
    fold_run_spec_from_dict,
    fold_run_spec_to_dict,
    run_decoder_fold,
    validate_fold_run_spec,
)
from ownership_decoder.breadth_labels import (
    CandidateFrameInput,
    finalize_review_manifest,
    freeze_reviewed_label_package,
    write_candidate_review_package,
    write_review_template,
)
from ownership_decoder.fold_visualization import render_completed_fold_diagnostics


def fold_spec(**overrides) -> DecoderFoldRunSpec:
    values = {
        "run_name": "l11__holdout-d",
        "spatial_arm": "l11",
        "train_clips": ("a", "b"),
        "validation_clip": "c",
        "heldout_clip": "d",
        "label_manifests": {
            value: Path(f"/{value}/label-manifest.json") for value in "abcd"
        },
        "qwen_breadth_root": Path("/qwen"),
        "input_root": Path("/inputs"),
        "output_root": Path("/outputs"),
        "device": "cuda",
    }
    values.update(overrides)
    return DecoderFoldRunSpec(**values)


class DecoderCampaignContractTests(unittest.TestCase):
    def test_one_fold_runs_end_to_end_and_resumes_from_hash_bound_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs_root = root / "inputs"
            qwen_root = root / "qwen"
            candidates = []
            for clip_number, clip_id in enumerate(("a", "b", "c", "d")):
                clip_root = inputs_root / clip_id
                frame_path = clip_root / "frames/frame_000001.png"
                frame_path.parent.mkdir(parents=True)
                Image.new("RGB", (40, 30), (30 + clip_number, 40, 50)).save(frame_path)
                source_sha = hashlib.sha256(frame_path.read_bytes()).hexdigest()
                (clip_root / "clip-manifest.json").write_text(
                    json.dumps(
                        {
                            "frames": [
                                {
                                    "clip_frame_index": 0,
                                    "path": "frames/frame_000001.png",
                                    "sha256": source_sha,
                                }
                            ]
                        }
                    )
                )
                a1 = np.zeros((30, 40), dtype=bool)
                a2 = np.zeros((30, 40), dtype=bool)
                a1[4:27, 2:20] = True
                a2[4:27, 20:38] = True
                tracker = root / "raw-masks" / clip_id / "tracker.npz"
                image = root / "raw-masks" / clip_id / "image.npz"
                tracker.parent.mkdir(parents=True)
                np.savez_compressed(tracker, A1=a1, A2=a2)
                np.savez_compressed(image, A1=a1, A2=a2)
                spatial = qwen_root / clip_id / "spatial/full/layer_11/frame_000000.safetensors"
                spatial.parent.mkdir(parents=True)
                save_file(
                    {
                        "hidden": torch.randn((48, 1152), dtype=torch.bfloat16),
                        "grid_thw": torch.tensor([[1, 6, 8]], dtype=torch.int64),
                    },
                    spatial,
                    metadata={"campaign": '{"stage":"spatial_full"}'},
                )
                candidates.append(
                    CandidateFrameInput(
                        clip_id=clip_id,
                        frame_index=0,
                        source_path=frame_path,
                        source_sha256=source_sha,
                        tracker_mask_path=tracker,
                        tracker_mask_sha256=hashlib.sha256(tracker.read_bytes()).hexdigest(),
                        image_mask_path=image,
                        image_mask_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                        qwen_spatial_path=spatial,
                        qwen_spatial_sha256=hashlib.sha256(spatial.read_bytes()).hexdigest(),
                        output_hw=(6, 8),
                    )
                )
            candidate_root = root / "candidates"
            write_candidate_review_package(
                candidates,
                candidate_root,
                preview_width=80,
                dilation_radius=3,
            )
            review_root = root / "review"
            review = write_review_template(
                candidate_root / "candidate-manifest.json",
                review_root,
            )
            for record in review["records"]:
                owner = review_root / record["contact_owner_path"]
                values = np.zeros((6, 8), dtype=np.uint8)
                values[2:4, 3] = 1
                values[2:4, 4] = 2
                Image.fromarray(values, mode="L").save(owner)
            finalize_review_manifest(
                candidate_root / "candidate-manifest.json",
                review_root / "review-manifest.json",
                reviewer="fixture reviewer",
                attested=True,
            )
            labels_root = root / "labels"
            campaign = freeze_reviewed_label_package(
                candidate_root / "candidate-manifest.json",
                review_root / "review-manifest.json",
                labels_root,
            )
            label_manifests = {
                record["clip_id"]: labels_root / record["label_manifest_path"]
                for record in campaign["clips"]
            }
            spec = DecoderFoldRunSpec(
                run_name="smoke",
                spatial_arm="l11",
                train_clips=("a", "b"),
                validation_clip="c",
                heldout_clip="d",
                label_manifests=label_manifests,
                qwen_breadth_root=qwen_root,
                input_root=inputs_root,
                output_root=root / "runs",
                width=8,
                residual_blocks=0,
                learning_rate=0.001,
                max_epochs=1,
                patience=1,
                gradient_accumulation=1,
                device="cpu",
                use_amp=False,
            )

            first = run_decoder_fold(spec)
            resumed = run_decoder_fold(spec)

            self.assertEqual(first, resumed)
            self.assertEqual(first["sample_counts"], {"train": 2, "validation": 1, "test": 1})
            self.assertIn("macro_actor_iou", first["test_metrics"])
            self.assertTrue((root / "runs/smoke/RUN_COMPLETE").is_file())
            diagnostics = render_completed_fold_diagnostics(
                spec,
                root / "diagnostics/smoke",
                panel_size=(80, 64),
            )
            resumed_diagnostics = render_completed_fold_diagnostics(
                spec,
                root / "diagnostics/smoke",
                panel_size=(80, 64),
            )
            self.assertEqual(diagnostics, resumed_diagnostics)
            self.assertEqual(diagnostics["frame_count"], 1)
            self.assertTrue((root / "diagnostics/smoke/images/frame_000000.png").is_file())
            self.assertTrue((root / "diagnostics/smoke/tensors/frame_000000.npz").is_file())

            for clip_id in ("a", "b", "c", "d"):
                semantic_root = (
                    qwen_root
                    / clip_id
                    / "semantic/video/4fps/action_relational/off"
                )
                semantic_root.mkdir(parents=True)
                for actor, offset in (("A1", 1.0), ("A2", 2.0)):
                    save_file(
                        {
                            "marker_states": torch.full(
                                (64, 5120),
                                offset,
                                dtype=torch.bfloat16,
                            )
                        },
                        semantic_root / f"{actor}.safetensors",
                        metadata={
                            "campaign": json.dumps(
                                {
                                    "stage": "semantic_video",
                                    "actor": actor,
                                    "condition": "action_relational",
                                    "context": "4fps",
                                    "thinking_mode": "off",
                                }
                            )
                        },
                    )
            semantic = run_decoder_fold(
                DecoderFoldRunSpec(
                    **{
                        **spec.__dict__,
                        "run_name": "semantic-smoke",
                        "semantic_condition": "action_relational",
                        "language_layer": 25,
                    }
                )
            )
            self.assertEqual(
                set(semantic["semantic_controls"]),
                {"real", "shuffled_clip", "random_matched", "zero", "mean"},
            )
            self.assertAlmostEqual(
                semantic["swap_metrics"]["actor_probability_swap_error"],
                0.0,
                places=6,
            )

    def test_default_screen_is_small_and_contains_preregistered_baselines(self) -> None:
        self.assertEqual(
            DEFAULT_STATIC_SCREEN,
            ("rgb", "l11", "p12", "merged", "l11_merged", "l05_l11_l18_l26"),
        )

    def test_fold_contract_separates_train_validation_and_heldout_clips(self) -> None:
        validated = validate_fold_run_spec(fold_spec())

        self.assertEqual(validated.heldout_clip, "d")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_fold_run_spec(fold_spec(train_clips=("a", "d")))

    def test_semantic_condition_and_language_layer_are_all_or_nothing(self) -> None:
        with self.assertRaisesRegex(ValueError, "together"):
            validate_fold_run_spec(fold_spec(semantic_condition="action_relational"))
        with self.assertRaisesRegex(ValueError, "together"):
            validate_fold_run_spec(fold_spec(language_layer=25))

        validated = validate_fold_run_spec(
            fold_spec(
                semantic_condition="action_relational",
                language_layer=25,
            )
        )
        self.assertEqual(validated.language_layer, 25)

    def test_fold_contract_rejects_missing_or_extra_label_manifests(self) -> None:
        with self.assertRaisesRegex(ValueError, "label manifest"):
            validate_fold_run_spec(
                fold_spec(label_manifests={value: Path(f"/{value}") for value in "abc"})
            )

    def test_fold_contract_rejects_unknown_arm_and_unsafe_training_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "spatial arm"):
            validate_fold_run_spec(fold_spec(spatial_arm="invented"))
        with self.assertRaisesRegex(ValueError, "width"):
            validate_fold_run_spec(fold_spec(width=1024))

    def test_fold_spec_json_round_trip_preserves_paths_and_controls(self) -> None:
        original = fold_spec(
            semantic_condition="action_relational",
            language_layer=25,
            qwen_download_manifest=Path("/qwen/download-manifest.json"),
        )

        restored = fold_run_spec_from_dict(fold_run_spec_to_dict(original))

        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
