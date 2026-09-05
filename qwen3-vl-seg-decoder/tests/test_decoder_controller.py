from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.decoder_controller import (
    LocalDecoderCampaignSpec,
    run_local_decoder_campaign,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(iou: float, contact: float) -> dict[str, float]:
    return {
        "macro_actor_iou": iou,
        "contact_accuracy": contact,
        "positive_contact_region_fraction": 0.80,
        "background_stability": 0.95,
        "contact_pixel_count": 10.0,
        "contact_region_count": 1.0,
    }


class LocalDecoderControllerTests(unittest.TestCase):
    def test_nested_controller_selects_only_on_validation_then_evaluates_north_star(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            label_root = root / "labels"
            clips = []
            for clip_id in ("a", "b", "c", "d"):
                manifest = label_root / "clips" / clip_id / "label-manifest.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text(
                    json.dumps(
                        {
                            "clip_id": clip_id,
                            "records": [{"frame_index": 0}],
                        }
                    )
                )
                clips.append(
                    {
                        "clip_id": clip_id,
                        "frame_count": 1,
                        "label_manifest_path": manifest.relative_to(label_root).as_posix(),
                        "label_manifest_sha256": _sha256(manifest),
                    }
                )
            campaign_path = label_root / "campaign-manifest.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "format": "reviewed-ownership-label-campaign-v1",
                        "training_eligible": True,
                        "clip_count": 4,
                        "clips": clips,
                    }
                )
            )
            campaign_path.with_suffix(".json.sha256").write_text(_sha256(campaign_path))
            qwen_root = root / "qwen"
            input_root = root / "inputs"
            qwen_root.mkdir()
            input_root.mkdir()
            qwen_manifest = root / "download-manifest.json"
            qwen_manifest.write_text("{}")
            seen = []

            def fake_runner(fold):
                seen.append(fold)
                if fold.semantic_condition is None:
                    validation_iou = 0.70 if fold.spatial_arm == "l11" else 0.50
                    test = _metrics(0.62 if fold.spatial_arm == "l11" else 0.55, 0.60)
                    controls = None
                    swap = None
                else:
                    validation_iou = (
                        0.80
                        if fold.semantic_condition == "action_relational"
                        and fold.language_layer == 60
                        else 0.70
                    )
                    candidate_iou = {
                        "action_relational": 0.70,
                        "identity_only": 0.64,
                        "contact_ownership": 0.65,
                    }[fold.semantic_condition]
                    test = _metrics(candidate_iou, 0.80)
                    controls = {
                        "real": test,
                        "shuffled_clip": _metrics(0.63, 0.63),
                        "random_matched": _metrics(0.65, 0.65),
                        "zero": _metrics(0.55, 0.55),
                        "mean": _metrics(0.54, 0.54),
                    }
                    swap = {
                        "actor_pixel_count": 20.0,
                        "actor_prediction_flip_fraction": 0.90,
                        "background_pixel_count": 30.0,
                        "background_probability_delta": 0.005,
                        "actor_probability_value_count": 60.0,
                        "actor_probability_swap_error": 0.002,
                    }
                return {
                    "run_name": fold.run_name,
                    "training": {
                        "best_validation_metrics": _metrics(validation_iou, 0.75)
                    },
                    "test_metrics": test,
                    "semantic_controls": controls,
                    "swap_metrics": swap,
                }

            result = run_local_decoder_campaign(
                LocalDecoderCampaignSpec(
                    reviewed_label_campaign=campaign_path,
                    qwen_breadth_root=qwen_root,
                    qwen_download_manifest=qwen_manifest,
                    input_root=input_root,
                    output_root=root / "runs",
                    python_executable=Path(sys.executable),
                ),
                job_runner=fake_runner,
                cache_verifier=lambda *args, **kwargs: {
                    "format": "fixture-qwen-cache-verification-v1",
                    "verified_artifact_count": 0,
                    "verified_bytes": 0,
                    "inventory_sha256": "0" * 64,
                },
            )

            self.assertEqual(
                set(result["selected_static_arm_by_heldout"].values()),
                {"l11"},
            )
            self.assertEqual(
                set(result["selected_action_layer_by_heldout"].values()),
                {60},
            )
            self.assertEqual(len(result["selected_static_run_by_heldout"]), 4)
            self.assertEqual(len(result["selected_action_run_by_heldout"]), 4)
            self.assertIn("shuffled_clip", result["action_semantic_controls"])
            self.assertEqual(
                result["action_over_static_paired_uncertainty"][
                    "bootstrap_resample_count"
                ],
                256,
            )
            self.assertTrue(result["north_star"]["passed"])
            self.assertEqual(
                result["qwen_cache_verification"]["inventory_sha256"],
                "0" * 64,
            )
            self.assertEqual(len(seen), 40)
            self.assertTrue(
                all(fold.cuda_memory_fraction == 0.60 for fold in seen)
            )
            self.assertTrue((root / "runs/RUN_COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
