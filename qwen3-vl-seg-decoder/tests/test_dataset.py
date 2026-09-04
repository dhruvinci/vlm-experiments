from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.cache import CacheContractError
from ownership_decoder.data import (
    ActorStateControlDataset,
    FrameSampleSpec,
    OwnershipDataset,
    OwnershipSample,
    SpatialSource,
    build_specs_from_label_manifest,
    load_rgb_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class OwnershipDatasetTests(unittest.TestCase):
    def _write_spatial(self, directory: Path) -> Path:
        path = directory / "full.safetensors"
        save_file(
            {
                "hidden": torch.zeros((6, 1152), dtype=torch.bfloat16),
                "grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            },
            path,
            metadata={"campaign": '{"stage":"spatial_full"}'},
        )
        return path

    def _write_actor(self, directory: Path, actor: str) -> Path:
        path = directory / f"{actor}.safetensors"
        marker_states = torch.full(
            (3, 5120),
            fill_value=1 if actor == "A1" else 2,
            dtype=torch.bfloat16,
        )
        save_file(
            {"marker_states": marker_states},
            path,
            metadata={
                "campaign": (
                    '{"stage":"semantic_video","actor":"'
                    + actor
                    + '","condition":"action_relational","context":"4fps","thinking_mode":"off"}'
                )
            },
        )
        return path

    def _write_mask(self, directory: Path, name: str, values: np.ndarray) -> Path:
        path = directory / name
        Image.fromarray(values.astype(np.uint8), mode="L").save(path)
        return path

    def test_constructor_does_not_touch_tensor_files(self) -> None:
        missing = Path("/definitely/missing/cache.safetensors")
        spec = FrameSampleSpec(
            clip_id="clip",
            frame_index=7,
            spatial={"layer_11": SpatialSource(missing, "full")},
        )

        dataset = OwnershipDataset([spec])

        self.assertEqual(len(dataset), 1)
        with self.assertRaises(CacheContractError):
            _ = dataset[0]

    def test_item_loads_validated_spatial_labels_contact_and_actor_states(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            spatial_path = self._write_spatial(directory)
            label_path = self._write_mask(
                directory,
                "label.png",
                np.array([[0, 1, 2], [255, 2, 1]], dtype=np.uint8),
            )
            contact_path = self._write_mask(
                directory,
                "contact.png",
                np.array([[0, 0, 255], [0, 0, 0]], dtype=np.uint8),
            )
            a1_path = self._write_actor(directory, "A1")
            a2_path = self._write_actor(directory, "A2")
            spec = FrameSampleSpec(
                clip_id="clip",
                frame_index=7,
                spatial={"layer_11": SpatialSource(spatial_path, "full")},
                label_path=label_path,
                label_sha256=_sha256(label_path),
                contact_path=contact_path,
                contact_sha256=_sha256(contact_path),
                actor_state_paths=(a1_path, a2_path),
                language_layer=1,
            )

            sample = OwnershipDataset([spec])[0]

            self.assertEqual(sample.clip_id, "clip")
            self.assertEqual(sample.frame_index, 7)
            self.assertEqual(tuple(sample.spatial["layer_11"].shape), (1152, 2, 3))
            torch.testing.assert_close(
                sample.labels,
                torch.tensor([[0, 1, 2], [255, 2, 1]], dtype=torch.long),
            )
            torch.testing.assert_close(
                sample.contact,
                torch.tensor([[False, False, True], [False, False, False]]),
            )
            self.assertEqual(tuple(sample.actor_states.shape), (2, 5120))
            self.assertEqual(float(sample.actor_states[0, 0]), 1.0)
            self.assertEqual(float(sample.actor_states[1, 0]), 2.0)

    def test_label_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            spatial_path = self._write_spatial(directory)
            label_path = self._write_mask(directory, "label.png", np.zeros((2, 3), dtype=np.uint8))
            spec = FrameSampleSpec(
                clip_id="clip",
                frame_index=7,
                spatial={"layer_11": SpatialSource(spatial_path, "full")},
                label_path=label_path,
                label_sha256="0" * 64,
            )

            with self.assertRaisesRegex(CacheContractError, "checksum"):
                _ = OwnershipDataset([spec])[0]

    def test_label_manifest_builds_deterministic_cache_paths_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            labels_root = directory / "labels-root"
            labels_root.mkdir()
            manifest_path = labels_root / "label-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "grid_hw": [160, 90],
                        "records": [
                            {
                                "frame_index": 4,
                                "subset": "train",
                                "screen_subset": "validation",
                                "label_path": "labels/frame_000004.png",
                                "label_sha256": "a" * 64,
                                "contact_path": "contact/frame_000004.png",
                                "contact_sha256": "b" * 64,
                            }
                        ],
                    }
                )
            )
            cache_root = directory / "cache"
            a1_path, a2_path = directory / "A1.safetensors", directory / "A2.safetensors"

            specs = build_specs_from_label_manifest(
                manifest_path,
                cache_root=cache_root,
                clip_id="armbar",
                full_layers=(5, 11),
                pooled_layers=(12,),
                include_merged=True,
                actor_state_paths=(a1_path, a2_path),
                language_layer=25,
                rgb_records={4: (directory / "frame.jpg", "c" * 64)},
            )

            self.assertEqual(len(specs), 1)
            spec = specs[0]
            self.assertEqual(spec.subset, "train")
            self.assertEqual(spec.screen_subset, "validation")
            self.assertEqual(
                spec.spatial["layer_05"].path,
                cache_root / "spatial/full/layer_05/frame_000004.safetensors",
            )
            self.assertEqual(
                spec.spatial["merged"].path,
                cache_root / "merged-vision/frame_000004.safetensors",
            )
            self.assertEqual(
                spec.spatial["pooled_12"].path,
                cache_root / "spatial/pooled/layer_12/frame_000004.safetensors",
            )
            self.assertEqual(spec.rgb_path, directory / "frame.jpg")
            self.assertEqual(spec.rgb_sha256, "c" * 64)

            with self.assertRaisesRegex((ValueError, FileNotFoundError), "manifest|sidecar"):
                build_specs_from_label_manifest(
                    manifest_path,
                    cache_root=cache_root,
                    clip_id="armbar",
                    full_layers=(11,),
                    require_reviewed=True,
                )

    def test_rgb_is_hash_checked_resized_and_normalized_without_changing_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            rgb_values = np.zeros((4, 2, 3), dtype=np.uint8)
            rgb_values[:2, :, 0] = 255
            rgb_values[2:, :, 2] = 255
            rgb_path = directory / "frame.jpg"
            Image.fromarray(rgb_values, mode="RGB").save(rgb_path, quality=100, subsampling=0)
            spec = FrameSampleSpec(
                clip_id="clip",
                frame_index=0,
                spatial={},
                rgb_path=rgb_path,
                rgb_sha256=_sha256(rgb_path),
            )

            sample = OwnershipDataset([spec], rgb_output_hw=(2, 1))[0]

            self.assertEqual(tuple(sample.rgb.shape), (3, 2, 1))
            self.assertTrue(torch.all((sample.rgb >= 0) & (sample.rgb <= 1)))
            self.assertGreater(float(sample.rgb[0, 0, 0]), float(sample.rgb[2, 0, 0]))
            self.assertGreater(float(sample.rgb[2, 1, 0]), float(sample.rgb[0, 1, 0]))

    def test_rgb_can_follow_each_samples_label_grid_without_large_intermediate_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            rgb_path = directory / "frame.png"
            label_path = directory / "label.png"
            Image.new("RGB", (80, 60), (20, 40, 60)).save(rgb_path)
            Image.fromarray(np.zeros((3, 5), dtype=np.uint8), mode="L").save(label_path)
            spec = FrameSampleSpec(
                clip_id="clip",
                frame_index=0,
                spatial={},
                label_path=label_path,
                label_sha256=_sha256(label_path),
                rgb_path=rgb_path,
                rgb_sha256=_sha256(rgb_path),
            )

            sample = OwnershipDataset([spec], rgb_output_hw=None)[0]

            self.assertEqual(tuple(sample.rgb.shape), (3, 3, 5))

    def test_rgb_frame_manifest_resolves_project_relative_paths_by_frame_index(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            project_root = Path(raw_directory)
            manifest_path = project_root / "frame-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "frame_index": 4,
                                "path": "frames/frame_000005.jpg",
                                "sha256": "c" * 64,
                            }
                        ]
                    }
                )
            )

            records = load_rgb_records(manifest_path, project_root=project_root)

            self.assertEqual(
                records[4],
                (project_root / "frames/frame_000005.jpg", "c" * 64),
            )

    def test_rgb_loader_accepts_frozen_breadth_clip_frame_index(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            project_root = Path(raw_directory)
            manifest_path = project_root / "clip-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "clip_frame_index": 3,
                                "path": "frames/frame_000004.jpg",
                                "sha256": "d" * 64,
                            }
                        ]
                    }
                )
            )

            records = load_rgb_records(manifest_path, project_root=project_root)

            self.assertEqual(
                records[3],
                (project_root / "frames/frame_000004.jpg", "d" * 64),
            )

    def test_actor_state_controls_are_deterministic_and_do_not_mutate_source(self) -> None:
        actor_states = torch.stack((torch.arange(8), torch.arange(8) + 20)).float()
        source = [
            OwnershipSample(
                clip_id="clip",
                frame_index=0,
                spatial={},
                labels=torch.zeros((1, 1), dtype=torch.long),
                contact=torch.zeros((1, 1), dtype=torch.bool),
                actor_states=actor_states,
            )
        ]

        swapped = ActorStateControlDataset(source, control="swapped")[0].actor_states
        zero = ActorStateControlDataset(source, control="zero")[0].actor_states
        mean = ActorStateControlDataset(source, control="mean")[0].actor_states
        random_one = ActorStateControlDataset(source, control="random_matched", seed=7)[0].actor_states
        random_two = ActorStateControlDataset(source, control="random_matched", seed=7)[0].actor_states

        torch.testing.assert_close(swapped, actor_states.flip(0))
        torch.testing.assert_close(zero, torch.zeros_like(actor_states))
        torch.testing.assert_close(mean[0], actor_states.mean(dim=0))
        torch.testing.assert_close(mean[0], mean[1])
        torch.testing.assert_close(random_one, random_two)
        torch.testing.assert_close(random_one.norm(dim=-1), actor_states.norm(dim=-1))
        torch.testing.assert_close(source[0].actor_states, actor_states)


if __name__ == "__main__":
    unittest.main()
