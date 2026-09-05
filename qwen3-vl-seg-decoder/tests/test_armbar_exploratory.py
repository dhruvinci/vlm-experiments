from __future__ import annotations

import unittest
import hashlib
import json
import sys
import tempfile
from pathlib import Path


from ownership_decoder.armbar_exploratory import (
    ARMBAR_FIXED_SUBSTITUTIONS,
    ArmbarJobSpec,
    armbar_job_spec_from_dict,
    armbar_job_spec_to_dict,
    run_armbar_job,
    validate_armbar_job_spec,
)
from ownership_decoder.armbar_controller import (
    ArmbarCampaignSpec,
    run_armbar_exploratory_campaign,
)


def job_spec(**overrides) -> ArmbarJobSpec:
    values = {
        "run_name": "screen__l11",
        "spatial_arm": "l11",
        "split": "screen",
        "label_manifest": Path("/labels/label-manifest.json"),
        "cache_root": Path("/cache"),
        "frame_manifest": Path("/packet/inputs/frame-manifest.json"),
        "frame_project_root": Path("/project"),
        "output_root": Path("/output"),
        "device": "cuda",
    }
    values.update(overrides)
    return ArmbarJobSpec(**values)


class ArmbarExploratoryContractTests(unittest.TestCase):
    def test_fixed_substitution_registry_contains_temporal_nulls_and_known_remaps(self) -> None:
        by_name = {item.name: item for item in ARMBAR_FIXED_SUBSTITUTIONS}

        self.assertIn("action_ordered_4fps_off", by_name)
        self.assertIn("action_reversed_4fps_off", by_name)
        self.assertIn("action_shuffled_4fps_off", by_name)
        self.assertTrue(by_name["action_2fps_remapped"].flip_actors)
        self.assertTrue(by_name["action_five_frame_remapped"].flip_actors)
        self.assertFalse(by_name["action_ordered_4fps_off"].flip_actors)

    def test_job_spec_round_trip_preserves_resource_and_semantic_contract(self) -> None:
        original = job_spec(
            run_name="final__action__l25",
            split="final",
            semantic_condition="action_relational",
            language_layer=25,
            semantic_context="4fps",
            thinking_mode="off",
            training_control="real",
            cuda_memory_fraction=0.60,
        )

        restored = armbar_job_spec_from_dict(armbar_job_spec_to_dict(original))

        self.assertEqual(restored, original)

    def test_job_spec_rejects_semantic_leakage_and_unsafe_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "together"):
            validate_armbar_job_spec(
                job_spec(semantic_condition="action_relational")
            )
        with self.assertRaisesRegex(ValueError, "static"):
            validate_armbar_job_spec(job_spec(training_control="random_matched"))
        with self.assertRaisesRegex(ValueError, "split"):
            validate_armbar_job_spec(job_spec(split="invented"))
        with self.assertRaisesRegex(ValueError, "memory fraction"):
            validate_armbar_job_spec(job_spec(cuda_memory_fraction=0.99))

    def test_static_job_trains_on_screen_split_and_resumes_from_bound_result(self) -> None:
        import numpy as np
        import torch
        from PIL import Image
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            labels_root = root / "labels"
            cache_root = root / "cache"
            records = []
            split_records = (
                (0, "train", "train"),
                (1, "train", "validation"),
                (2, "test", "test"),
            )
            for frame_index, subset, screen_subset in split_records:
                labels = np.array(
                    [
                        [0, 0, 0, 0],
                        [1, 1, 2, 2],
                        [1, 1, 2, 2],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.uint8,
                )
                contact = np.zeros((4, 4), dtype=np.uint8)
                if subset == "test":
                    contact[1, 2] = 255
                label_path = labels_root / "labels" / f"frame_{frame_index:06d}.png"
                contact_path = labels_root / "contact" / f"frame_{frame_index:06d}.png"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                contact_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(labels, mode="L").save(label_path)
                Image.fromarray(contact, mode="L").save(contact_path)
                records.append(
                    {
                        "frame_index": frame_index,
                        "subset": subset,
                        "screen_subset": screen_subset,
                        "label_path": label_path.relative_to(labels_root).as_posix(),
                        "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                        "contact_path": contact_path.relative_to(labels_root).as_posix(),
                        "contact_sha256": hashlib.sha256(contact_path.read_bytes()).hexdigest(),
                    }
                )
                spatial_path = (
                    cache_root
                    / "spatial/full/layer_11"
                    / f"frame_{frame_index:06d}.safetensors"
                )
                spatial_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {
                        "hidden": torch.randn((16, 1152), dtype=torch.bfloat16),
                        "grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.int64),
                    },
                    spatial_path,
                    metadata={"campaign": json.dumps({"stage": "spatial_full"})},
                )
            label_manifest = labels_root / "label-manifest.json"
            label_manifest.write_text(
                json.dumps(
                    {
                        "status": "conservative_pseudo_labels_with_manual_final_contact_truth",
                        "records": records,
                    }
                )
            )
            frame_manifest = root / "frame-manifest.json"
            frame_manifest.write_text(json.dumps({"frames": []}))
            spec = ArmbarJobSpec(
                run_name="static-smoke",
                spatial_arm="l11",
                split="screen",
                label_manifest=label_manifest,
                cache_root=cache_root,
                frame_manifest=frame_manifest,
                frame_project_root=root,
                output_root=root / "runs",
                width=8,
                residual_blocks=0,
                max_epochs=1,
                patience=1,
                gradient_accumulation=1,
                device="cpu",
                use_amp=False,
            )

            first = run_armbar_job(spec)
            resumed = run_armbar_job(spec)

            self.assertEqual(first, resumed)
            self.assertEqual(first["supervision_status"], "exploratory_legacy_pseudo_labels")
            self.assertEqual(
                first["sample_counts"],
                {"train": 1, "evaluation": 1},
            )
            self.assertEqual(first["evaluation_subset"], "validation")
            self.assertEqual(set(first["per_frame_evaluation"]), {"1"})
            self.assertTrue((root / "runs/static-smoke/RUN_COMPLETE").is_file())


class ArmbarExploratoryControllerTests(unittest.TestCase):
    def test_controller_screens_on_validation_then_runs_paired_final_controls(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            label_manifest = root / "label-manifest.json"
            label_manifest.write_text(
                json.dumps(
                    {
                        "status": "conservative_pseudo_labels_with_manual_final_contact_truth",
                        "records": [
                            {"frame_index": 0},
                            {"frame_index": 1},
                            {"frame_index": 2},
                        ],
                    }
                )
            )
            frame_manifest = root / "frame-manifest.json"
            frame_manifest.write_text(json.dumps({"frames": []}))
            cache_root = root / "cache"
            cache_root.mkdir()
            seen = []

            def metrics(iou: float, contact_margin: float = 0.10) -> dict[str, float]:
                return {
                    "accuracy": 0.8,
                    "a1_iou": iou,
                    "a1_dice": iou,
                    "a2_iou": iou,
                    "a2_dice": iou,
                    "macro_actor_iou": iou,
                    "background_stability": 0.95,
                    "contact_pixel_count": 27.0,
                    "contact_accuracy": 0.8,
                    "contact_margin": contact_margin,
                    "positive_contact_margin_fraction": 0.8,
                    "contact_region_count": 1.0,
                    "positive_contact_region_fraction": 1.0,
                }

            def fake_runner(job):
                seen.append(job)
                if job.split == "screen":
                    validation_iou = {
                        "l11": 0.61,
                        "merged": 0.58,
                    }.get(job.spatial_arm, 0.50)
                    if job.semantic_condition is not None:
                        validation_iou = 0.66 if job.language_layer == 60 else 0.62
                    evaluation = metrics(validation_iou)
                elif job.semantic_condition == "action_relational":
                    if job.training_control == "random_matched":
                        evaluation = metrics(0.59, -0.02)
                    elif job.training_control in {"zero", "mean"}:
                        evaluation = metrics(0.55, -0.08)
                    else:
                        evaluation = metrics(0.67, 0.18)
                elif job.semantic_condition is not None:
                    evaluation = metrics(0.62, 0.05)
                else:
                    evaluation = metrics(0.61, 0.02)
                substitutions = None
                if (
                    job.split == "final"
                    and job.semantic_condition == "action_relational"
                    and job.training_control == "real"
                ):
                    substitutions = {
                        item.name: metrics(
                            0.67
                            if item.name == "action_ordered_4fps_off"
                            else 0.60,
                            0.18
                            if item.name == "action_ordered_4fps_off"
                            else 0.01,
                        )
                        for item in ARMBAR_FIXED_SUBSTITUTIONS
                    }
                return {
                    "run_name": job.run_name,
                    "training": {"best_validation_metrics": evaluation},
                    "evaluation_metrics": evaluation,
                    "semantic_controls": None,
                    "swap_metrics": None,
                    "fixed_substitutions": substitutions,
                }

            output = root / "output"
            result = run_armbar_exploratory_campaign(
                ArmbarCampaignSpec(
                    label_manifest=label_manifest,
                    cache_root=cache_root,
                    frame_manifest=frame_manifest,
                    frame_project_root=root,
                    output_root=output,
                    python_executable=Path(sys.executable),
                    static_arms=("rgb", "l11", "merged"),
                    semantic_layers=(25, 60),
                    seeds=(7, 71),
                    device="cpu",
                ),
                job_runner=fake_runner,
            )

            self.assertEqual(result["selected_static_arm"], "l11")
            self.assertEqual(result["selected_action_language_layer"], 60)
            self.assertEqual(result["supervision_status"], "exploratory_legacy_pseudo_labels")
            self.assertFalse(result["north_star_eligible"])
            self.assertEqual(result["final_aggregates"]["static"]["run_count"], 2)
            self.assertEqual(result["final_aggregates"]["action_real"]["run_count"], 2)
            self.assertEqual(result["final_aggregates"]["action_random_matched"]["run_count"], 2)
            self.assertTrue(result["exploratory_signal"]["passed"])
            self.assertEqual(len(seen), 15)
            self.assertTrue(all(job.cuda_memory_fraction == 0.60 for job in seen))
            self.assertTrue((output / "RUN_COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
