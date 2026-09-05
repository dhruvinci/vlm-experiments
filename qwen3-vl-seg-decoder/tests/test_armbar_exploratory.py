from __future__ import annotations

import unittest
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


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
from ownership_decoder.armbar_visualization import render_armbar_contact_diagnostics
from ownership_decoder import armbar_controller, armbar_controller_cli, armbar_exploratory


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
    def test_condition_delta_cancels_identity_state_per_actor(self) -> None:
        import torch
        from safetensors.torch import save_file

        self.assertTrue(hasattr(armbar_exploratory, "load_armbar_semantic_pair"))
        with tempfile.TemporaryDirectory() as raw:
            cache_root = Path(raw)
            identity = torch.arange(64 * 5120, dtype=torch.float32).reshape(64, 5120)
            identity = (identity.remainder(97) / 97).to(torch.bfloat16)
            action = (identity.float() + 0.25).to(torch.bfloat16)
            for condition, states in (
                ("identity_only", identity),
                ("action_relational", action),
            ):
                for actor, offset in (("A1", 0.0), ("A2", 0.125)):
                    path = (
                        cache_root
                        / "semantic/video/4fps"
                        / condition
                        / "off"
                        / f"{actor}.safetensors"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(
                        {"marker_states": (states.float() + offset).to(torch.bfloat16)},
                        path,
                        metadata={
                            "campaign": json.dumps(
                                {
                                    "actor": actor,
                                    "stage": "semantic_video",
                                    "condition": condition,
                                    "context": "4fps",
                                    "thinking_mode": "off",
                                }
                            )
                        },
                    )

            observed = armbar_exploratory.load_armbar_semantic_pair(
                cache_root,
                condition="action_delta",
                context="4fps",
                thinking_mode="off",
                language_layer=12,
            )
            expected = torch.stack(
                (
                    action[12].float() - identity[12].float(),
                    (action[12].float() + 0.125).to(torch.bfloat16).float()
                    - (identity[12].float() + 0.125).to(torch.bfloat16).float(),
                )
            ).to(torch.bfloat16)

            self.assertTrue(torch.equal(observed, expected))

    def test_job_dataset_uses_condition_delta_instead_of_raw_action_pair(self) -> None:
        import numpy as np
        import torch
        from PIL import Image
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            labels_root = root / "labels"
            cache_root = root / "cache"
            records = []
            for frame_index, subset, screen_subset in (
                (0, "train", "train"),
                (1, "train", "validation"),
                (2, "test", "test"),
            ):
                labels = np.array(
                    [[0, 0], [1, 2]],
                    dtype=np.uint8,
                )
                label_path = labels_root / "labels" / f"frame_{frame_index:06d}.png"
                contact_path = labels_root / "contact" / f"frame_{frame_index:06d}.png"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                contact_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(labels, mode="L").save(label_path)
                Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(contact_path)
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
                        "hidden": torch.randn((4, 1152), dtype=torch.bfloat16),
                        "grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
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
            identity = torch.zeros((64, 5120), dtype=torch.bfloat16)
            action = torch.zeros_like(identity)
            action[12] = 0.5
            for condition, states in (
                ("identity_only", identity),
                ("action_relational", action),
            ):
                for actor, sign in (("A1", 1.0), ("A2", -1.0)):
                    path = (
                        cache_root
                        / "semantic/video/4fps"
                        / condition
                        / "off"
                        / f"{actor}.safetensors"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(
                        {"marker_states": states * sign},
                        path,
                        metadata={
                            "campaign": json.dumps(
                                {
                                    "actor": actor,
                                    "stage": "semantic_video",
                                    "condition": condition,
                                    "context": "4fps",
                                    "thinking_mode": "off",
                                }
                            )
                        },
                    )
            spec = ArmbarJobSpec(
                run_name="delta-dataset",
                spatial_arm="l11",
                split="screen",
                label_manifest=label_manifest,
                cache_root=cache_root,
                frame_manifest=root / "unused-frame-manifest.json",
                frame_project_root=root,
                output_root=root / "runs",
                semantic_condition="action_delta",
                language_layer=12,
                width=8,
                residual_blocks=0,
                max_epochs=1,
                patience=1,
                gradient_accumulation=1,
                device="cpu",
                use_amp=False,
            )

            train, validation, subset = armbar_exploratory._build_job_datasets(spec)

            self.assertEqual(subset, "validation")
            self.assertTrue(torch.all(train[0].actor_states[0] == 0.5))
            self.assertTrue(torch.all(train[0].actor_states[1] == -0.5))
            self.assertTrue(torch.equal(train[0].actor_states, validation[0].actor_states))

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
            frame_entries = []
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
                frame_entries.append(
                    {
                        "frame_index": frame_index,
                        "path": label_path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
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
            frame_manifest.write_text(json.dumps({"frames": frame_entries}))
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

            final_spec = replace(spec, run_name="static-final", split="final")
            run_armbar_job(final_spec)
            diagnostic = render_armbar_contact_diagnostics(
                final_spec,
                root / "diagnostics",
                panel_size=(64, 96),
            )

            self.assertEqual(diagnostic["contact_frame_count"], 1)
            self.assertEqual(diagnostic["records"][0]["frame_index"], 2)
            self.assertTrue((root / "diagnostics/RUN_COMPLETE").is_file())


class ArmbarExploratoryControllerTests(unittest.TestCase):
    def test_cli_requires_audit_with_slot_cancellation_reference(self) -> None:
        with self.assertRaises(SystemExit):
            armbar_controller_cli.main(
                [
                    "--label-manifest",
                    "/labels.json",
                    "--cache-root",
                    "/cache",
                    "--frame-manifest",
                    "/frames.json",
                    "--frame-project-root",
                    "/project",
                    "--output",
                    "/output",
                    "--slot-cancellation-reference",
                    "/reference/campaign-result.json",
                ]
            )

    def test_cli_dispatches_fixed_layer_slot_cancellation_campaign(self) -> None:
        from unittest.mock import patch

        returned = {
            "supervision_status": "exploratory_legacy_pseudo_labels",
            "north_star_eligible": False,
            "fixed_language_layers": {"action": 12, "contact": 45},
            "slot_cancellation_signal": {"passed": False},
        }
        with patch.object(
            armbar_controller_cli,
            "run_armbar_delta_campaign",
            return_value=returned,
            create=True,
        ) as run_delta:
            status = armbar_controller_cli.main(
                [
                    "--label-manifest",
                    "/labels.json",
                    "--cache-root",
                    "/cache",
                    "--frame-manifest",
                    "/frames.json",
                    "--frame-project-root",
                    "/project",
                    "--output",
                    "/output",
                    "--slot-cancellation-reference",
                    "/reference/campaign-result.json",
                    "--slot-cancellation-audit",
                    "/audits/representation-audit.json",
                    "--action-delta-layer",
                    "12",
                    "--contact-delta-layer",
                    "45",
                    "--device",
                    "cpu",
                ]
            )

        self.assertEqual(status, 0)
        _, kwargs = run_delta.call_args
        self.assertEqual(kwargs["reference_campaign_result"], Path("/reference/campaign-result.json"))
        self.assertEqual(
            kwargs["layer_selection_audit"],
            Path("/audits/representation-audit.json"),
        )
        self.assertEqual(kwargs["action_layer"], 12)
        self.assertEqual(kwargs["contact_layer"], 45)

    def test_delta_controller_pairs_raw_and_slot_cancelled_conditions_by_seed(self) -> None:
        self.assertTrue(hasattr(armbar_controller, "run_armbar_delta_campaign"))
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
            reference = root / "reference/campaign-result.json"
            reference.parent.mkdir()
            reference.write_text(
                json.dumps(
                    {
                        "format": "armbar-exploratory-campaign-result-v1",
                        "selected_static_arm": "l11_merged",
                        "final_aggregates": {
                            "static": {
                                "run_count": 2,
                                "run_names": ["static-7", "static-71"],
                                "metrics": {
                                    "macro_actor_iou": {
                                        "mean": 0.61,
                                        "population_std": 0.01,
                                        "values": [0.60, 0.62],
                                    },
                                    "contact_accuracy": {
                                        "mean": 0.10,
                                        "population_std": 0.10,
                                        "values": [0.0, 0.20],
                                    },
                                    "contact_margin": {
                                        "mean": -0.50,
                                        "population_std": 0.10,
                                        "values": [-0.60, -0.40],
                                    },
                                },
                            }
                        },
                    }
                )
            )
            (reference.parent / "RUN_COMPLETE").write_text(
                json.dumps(
                    {"result_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
                )
            )
            audit = root / "representation-audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "format": "breadth-semantic-geometry-audit-v1",
                        "artifact_count": 24,
                        "selected_layers": {
                            "action_delta": 12,
                            "contact_delta": 45,
                        },
                    }
                )
            )
            audit.with_suffix(".json.sha256").write_text(
                hashlib.sha256(audit.read_bytes()).hexdigest() + "\n"
            )
            seen = []

            def fake_runner(job):
                seen.append(job)
                real_iou = {
                    "action_relational": 0.60,
                    "action_delta": 0.64,
                    "contact_ownership": 0.59,
                    "contact_delta": 0.66,
                }[job.semantic_condition]
                iou = real_iou if job.training_control == "real" else real_iou - 0.03
                margin = {
                    "action_relational": -0.55,
                    "action_delta": -0.20,
                    "contact_ownership": -0.40,
                    "contact_delta": 0.15,
                }[job.semantic_condition]
                metrics = {
                    "macro_actor_iou": iou,
                    "contact_accuracy": 0.75 if margin > 0 else 0.0,
                    "contact_margin": margin,
                    "background_stability": 0.98,
                }
                return {
                    "run_name": job.run_name,
                    "evaluation_metrics": metrics,
                }

            output = root / "delta"
            result = armbar_controller.run_armbar_delta_campaign(
                ArmbarCampaignSpec(
                    label_manifest=label_manifest,
                    cache_root=cache_root,
                    frame_manifest=frame_manifest,
                    frame_project_root=root,
                    output_root=output,
                    python_executable=Path(sys.executable),
                    seeds=(7, 71),
                    device="cpu",
                ),
                reference_campaign_result=reference,
                layer_selection_audit=audit,
                action_layer=12,
                contact_layer=45,
                job_runner=fake_runner,
            )

            self.assertEqual(len(seen), 16)
            self.assertEqual(
                {(job.semantic_condition, job.language_layer) for job in seen},
                {
                    ("action_relational", 12),
                    ("action_delta", 12),
                    ("contact_ownership", 45),
                    ("contact_delta", 45),
                },
            )
            self.assertEqual(
                {job.training_control for job in seen},
                {"real", "random_matched"},
            )
            self.assertEqual(result["aggregates"]["action_delta_real"]["run_count"], 2)
            for value in result["paired_deltas"]["action_delta_minus_raw_iou"]:
                self.assertAlmostEqual(value, 0.04)
            for value in result["paired_deltas"]["contact_delta_minus_raw_iou"]:
                self.assertAlmostEqual(value, 0.07)
            self.assertFalse(result["north_star_eligible"])
            self.assertEqual(
                result["layer_selection_audit_sha256"],
                hashlib.sha256(audit.read_bytes()).hexdigest(),
            )
            self.assertTrue((output / "RUN_COMPLETE").is_file())

    def test_delta_controller_rejects_layers_not_bound_by_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            audit = root / "representation-audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "format": "breadth-semantic-geometry-audit-v1",
                        "artifact_count": 24,
                        "selected_layers": {
                            "action_delta": 12,
                            "contact_delta": 45,
                        },
                    }
                )
            )
            audit.with_suffix(".json.sha256").write_text(
                hashlib.sha256(audit.read_bytes()).hexdigest() + "\n"
            )

            with self.assertRaisesRegex(ValueError, "selected layers"):
                armbar_controller.run_armbar_delta_campaign(
                    ArmbarCampaignSpec(
                        label_manifest=root / "missing-labels.json",
                        cache_root=root / "missing-cache",
                        frame_manifest=root / "missing-frames.json",
                        frame_project_root=root,
                        output_root=root / "output",
                        python_executable=Path(sys.executable),
                        device="cpu",
                    ),
                    reference_campaign_result=root / "missing-reference.json",
                    layer_selection_audit=audit,
                    action_layer=13,
                    contact_layer=45,
                )

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
