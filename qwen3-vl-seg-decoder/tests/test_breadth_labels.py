from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.breadth_labels import (
    REVIEW_ATTESTATION,
    CandidateFrameInput,
    collect_breadth_candidate_inputs,
    finalize_review_manifest,
    freeze_reviewed_label_package,
    read_qwen_full_grid_hw,
    write_review_template,
    verify_reviewed_label_manifest,
    write_candidate_review_package,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BreadthLabelPackageTests(unittest.TestCase):
    def _write_qwen_fixture(self, path: Path, *, grid_hw: tuple[int, int]) -> None:
        height, width = grid_hw
        hidden_bytes = height * width * 1152 * 2
        header = {
            "grid_thw": {
                "dtype": "I64",
                "shape": [1, 3],
                "data_offsets": [0, 24],
            },
            "hidden": {
                "dtype": "BF16",
                "shape": [height * width, 1152],
                "data_offsets": [24, 24 + hidden_bytes],
            },
            "__metadata__": {"campaign": '{"stage":"spatial_full"}'},
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            struct.pack("<Q", len(encoded))
            + encoded
            + struct.pack("<qqq", 1, height, width)
            + bytes(hidden_bytes)
        )

    def test_qwen_grid_is_read_from_safetensors_without_loading_hidden_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "frame.safetensors"
            self._write_qwen_fixture(path, grid_hw=(2, 3))

            self.assertEqual(read_qwen_full_grid_hw(path), (2, 3))

    def test_collector_binds_verified_remote_masks_to_matching_qwen_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "clip.json"
            config.write_text("{}")
            source = root / "source.png"
            Image.new("RGB", (40, 30), (1, 2, 3)).save(source)
            frame = SimpleNamespace(frame_index=0, path=source, sha256=_sha256(source))
            plan = SimpleNamespace(clip_id="clip_a", frames=(frame,))
            mask_root = root / "mask-run"
            mask_root.mkdir()
            campaign_manifest = mask_root / "campaign-manifest.json"
            campaign_manifest.write_text(
                json.dumps(
                    {
                        "format": "ownership-remote-mask-campaign-v1",
                        "tracker_revision": "tracker-revision",
                        "image_revision": "image-revision",
                    }
                )
            )
            (mask_root / "RUN_COMPLETE").write_text(
                json.dumps({"manifest_sha256": _sha256(campaign_manifest)})
            )
            masks = np.zeros((30, 40), dtype=bool)
            for stage in ("sam31-tracking", "sam3-image-agreement"):
                artifact = mask_root / stage / "clip_a" / "masks" / "frame_000000.npz"
                artifact.parent.mkdir(parents=True)
                np.savez_compressed(artifact, A1=masks, A2=masks)
            qwen = root / "qwen" / "clip_a" / "spatial/full/layer_11/frame_000000.safetensors"
            self._write_qwen_fixture(qwen, grid_hw=(2, 3))

            with (
                patch(
                    "ownership_decoder.breadth_labels.load_tracking_plan_config",
                    return_value=plan,
                ),
                patch("ownership_decoder.breadth_labels.load_completed_tracking_campaign"),
                patch("ownership_decoder.breadth_labels.load_completed_image_agreement_campaign"),
            ):
                collected = collect_breadth_candidate_inputs(
                    [config],
                    input_root=root,
                    mask_campaign_root=mask_root,
                    qwen_breadth_root=root / "qwen",
                    spatial_layer=11,
                )

            self.assertEqual(len(collected), 1)
            self.assertEqual(collected[0].output_hw, (2, 3))
            self.assertEqual(collected[0].source_sha256, _sha256(source))
            self.assertEqual(collected[0].qwen_spatial_sha256, _sha256(qwen))

    def _candidate_input(self, root: Path, clip_id: str, frame_index: int) -> CandidateFrameInput:
        source = root / f"{clip_id}-{frame_index}.png"
        Image.new("RGB", (40, 30), (45, 55, 65)).save(source)
        tracker = root / f"{clip_id}-{frame_index}-tracker.npz"
        image = root / f"{clip_id}-{frame_index}-image.npz"
        a1 = np.zeros((30, 40), dtype=bool)
        a2 = np.zeros((30, 40), dtype=bool)
        a1[6:25, 4:20] = True
        a2[7:26, 20:36] = True
        # Give the independent image decoder slightly different boundaries.
        np.savez_compressed(tracker, A1=a1, A2=a2)
        image_a1 = np.roll(a1, -1, axis=1)
        image_a2 = np.roll(a2, 1, axis=1)
        np.savez_compressed(
            image,
            A1=image_a1,
            A2=image_a2,
        )
        qwen_spatial = root / f"{clip_id}-{frame_index}.safetensors"
        qwen_spatial.write_bytes(b"frozen-qwen-spatial-fixture")
        return CandidateFrameInput(
            clip_id=clip_id,
            frame_index=frame_index,
            source_path=source,
            source_sha256=_sha256(source),
            tracker_mask_path=tracker,
            tracker_mask_sha256=_sha256(tracker),
            image_mask_path=image,
            image_mask_sha256=_sha256(image),
            qwen_spatial_path=qwen_spatial,
            qwen_spatial_sha256=_sha256(qwen_spatial),
            output_hw=(6, 8),
        )

    def _write_review(self, candidate_root: Path, review_root: Path) -> Path:
        candidate_manifest = candidate_root / "candidate-manifest.json"
        payload = json.loads(candidate_manifest.read_text())
        records = []
        for record in payload["records"]:
            owner = review_root / record["clip_id"] / f"frame_{record['frame_index']:06d}.png"
            owner.parent.mkdir(parents=True, exist_ok=True)
            values = np.zeros(tuple(record["grid_hw"]), dtype=np.uint8)
            values[2:4, 3] = 1
            values[2:4, 4] = 2
            Image.fromarray(values, mode="L").save(owner)
            records.append(
                {
                    "clip_id": record["clip_id"],
                    "frame_index": record["frame_index"],
                    "decision": "approved",
                    "contact_owner_path": owner.relative_to(review_root).as_posix(),
                    "contact_owner_sha256": _sha256(owner),
                    "notes": "contact cells checked against the source frame",
                }
            )
        review_path = review_root / "review-manifest.json"
        review_path.write_text(
            json.dumps(
                {
                    "format": "ownership-contact-review-v1",
                    "candidate_manifest_sha256": _sha256(candidate_manifest),
                    "reviewer": "human-reviewer",
                    "reviewed_at": "2026-09-04T12:00:00+00:00",
                    "attestation": REVIEW_ATTESTATION,
                    "records": records,
                }
            )
        )
        return review_path

    def test_candidate_package_is_hash_bound_and_never_training_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "candidates"

            manifest = write_candidate_review_package(
                [self._candidate_input(root, "clip_a", 0)],
                output,
                preview_width=160,
                dilation_radius=3,
            )

            self.assertEqual(manifest["format"], "ownership-label-candidates-v1")
            self.assertFalse(manifest["training_eligible"])
            self.assertEqual(manifest["frame_count"], 1)
            record = manifest["records"][0]
            self.assertEqual(record["source_sha256"], _sha256(root / "clip_a-0.png"))
            self.assertTrue((output / record["label_path"]).is_file())
            self.assertTrue((output / record["contact_proposal_path"]).is_file())
            self.assertTrue((output / record["preview_path"]).is_file())
            self.assertEqual(
                _sha256(output / "candidate-manifest.json"),
                (output / "candidate-manifest.json.sha256").read_text().strip(),
            )

    def test_freeze_requires_exact_human_attestation_and_complete_frame_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate_root = root / "candidates"
            write_candidate_review_package(
                [
                    self._candidate_input(root, "clip_a", 0),
                    self._candidate_input(root, "clip_a", 1),
                ],
                candidate_root,
                preview_width=160,
                dilation_radius=3,
            )
            review_root = root / "review"
            review_root.mkdir()
            review_path = self._write_review(candidate_root, review_root)
            review = json.loads(review_path.read_text())
            review["attestation"] = "looks fine"
            review["records"].pop()
            review_path.write_text(json.dumps(review))

            with self.assertRaisesRegex(ValueError, "attestation"):
                freeze_reviewed_label_package(
                    candidate_root / "candidate-manifest.json",
                    review_path,
                    root / "final",
                )

            review["attestation"] = REVIEW_ATTESTATION
            review_path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, "inventory"):
                freeze_reviewed_label_package(
                    candidate_root / "candidate-manifest.json",
                    review_path,
                    root / "final",
                )

    def test_reviewed_package_is_the_only_training_eligible_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate_root = root / "candidates"
            write_candidate_review_package(
                [
                    self._candidate_input(root, "clip_a", 0),
                    self._candidate_input(root, "clip_a", 1),
                ],
                candidate_root,
                preview_width=160,
                dilation_radius=3,
            )
            review_root = root / "review"
            review_root.mkdir()
            review_path = self._write_review(candidate_root, review_root)

            campaign = freeze_reviewed_label_package(
                candidate_root / "candidate-manifest.json",
                review_path,
                root / "final",
            )

            self.assertTrue(campaign["training_eligible"])
            self.assertEqual(campaign["approved_frame_count"], 2)
            clip_manifest = root / "final" / campaign["clips"][0]["label_manifest_path"]
            verified = verify_reviewed_label_manifest(
                clip_manifest,
                expected_clip_id="clip_a",
                expected_frame_count=2,
            )
            self.assertTrue(verified["training_eligible"])
            self.assertTrue(all(record["contact_patch_count"] == 4 for record in verified["records"]))
            label = np.asarray(
                Image.open(clip_manifest.parent / verified["records"][0]["label_path"]),
                dtype=np.uint8,
            )
            self.assertEqual(int(label[2, 3]), 1)
            self.assertEqual(int(label[2, 4]), 2)

    def test_review_with_no_explicit_contact_truth_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate_root = root / "candidates"
            write_candidate_review_package(
                [self._candidate_input(root, "clip_a", 0)],
                candidate_root,
                preview_width=160,
                dilation_radius=3,
            )
            review_root = root / "review"
            review_root.mkdir()
            review_path = self._write_review(candidate_root, review_root)
            review = json.loads(review_path.read_text())
            owner = review_root / review["records"][0]["contact_owner_path"]
            Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L").save(owner)
            review["records"][0]["contact_owner_sha256"] = _sha256(owner)
            review_path.write_text(json.dumps(review))

            with self.assertRaisesRegex(ValueError, "contact truth"):
                freeze_reviewed_label_package(
                    candidate_root / "candidate-manifest.json",
                    review_path,
                    root / "final",
                )

    def test_review_rejects_contact_truth_outside_proposed_review_band(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidates"
            write_candidate_review_package(
                [self._candidate_input(root, "clip_a", 0)],
                candidates,
                preview_width=160,
                dilation_radius=3,
            )
            template = write_review_template(
                candidates / "candidate-manifest.json",
                root / "review",
            )
            owner = root / "review" / template["records"][0]["contact_owner_path"]
            values = np.asarray(Image.open(owner), dtype=np.uint8).copy()
            values[values == 255] = 0
            proposal_record = json.loads(
                (candidates / "candidate-manifest.json").read_text()
            )["records"][0]
            proposal = np.asarray(
                Image.open(candidates / proposal_record["contact_proposal_path"]),
                dtype=np.uint8,
            )
            inside = np.argwhere(proposal != 0)[0]
            values[tuple(inside)] = 1
            outside = np.argwhere(proposal == 0)[0]
            values[tuple(outside)] = 1
            Image.fromarray(values, mode="L").save(owner)

            with self.assertRaisesRegex(ValueError, "outside"):
                finalize_review_manifest(
                    candidates / "candidate-manifest.json",
                    root / "review/review-manifest.json",
                    reviewer="human-reviewer",
                    attested=True,
                )

    def test_explicit_no_contact_frame_is_allowed_when_clip_has_other_contact_truth(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidates"
            write_candidate_review_package(
                [
                    self._candidate_input(root, "clip_a", 0),
                    self._candidate_input(root, "clip_a", 1),
                ],
                candidates,
                preview_width=160,
                dilation_radius=3,
            )
            template = write_review_template(
                candidates / "candidate-manifest.json",
                root / "review",
            )
            first = template["records"][0]
            first_owner = root / "review" / first["contact_owner_path"]
            Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L").save(first_owner)
            review_path = root / "review/review-manifest.json"
            review = json.loads(review_path.read_text())
            review["records"][0]["decision"] = "no_contact"
            second_owner = root / "review" / review["records"][1]["contact_owner_path"]
            second_values = np.zeros((6, 8), dtype=np.uint8)
            second_values[2, 3] = 1
            second_values[2, 4] = 2
            Image.fromarray(second_values, mode="L").save(second_owner)
            review_path.write_text(json.dumps(review))

            finalized = finalize_review_manifest(
                candidates / "candidate-manifest.json",
                review_path,
                reviewer="human-reviewer",
                attested=True,
            )
            campaign = freeze_reviewed_label_package(
                candidates / "candidate-manifest.json",
                review_path,
                root / "labels",
            )

            self.assertEqual(finalized["records"][0]["decision"], "approved_no_contact")
            manifest = root / "labels" / campaign["clips"][0]["label_manifest_path"]
            verified = verify_reviewed_label_manifest(manifest)
            self.assertEqual(verified["records"][0]["contact_patch_count"], 0)
            self.assertGreater(verified["records"][1]["contact_patch_count"], 0)

    def test_review_template_stays_pending_until_human_finalizes_edited_masks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidates"
            write_candidate_review_package(
                [self._candidate_input(root, "clip_a", 0)],
                candidates,
                preview_width=160,
                dilation_radius=3,
            )

            template = write_review_template(
                candidates / "candidate-manifest.json",
                root / "review",
            )

            self.assertEqual(template["records"][0]["decision"], "pending")
            with self.assertRaisesRegex(ValueError, "attest"):
                finalize_review_manifest(
                    candidates / "candidate-manifest.json",
                    root / "review" / "review-manifest.json",
                    reviewer="human-reviewer",
                    attested=False,
                )
            owner = root / "review" / template["records"][0]["contact_owner_path"]
            values = np.zeros((6, 8), dtype=np.uint8)
            values[2, 3] = 1
            values[2, 4] = 2
            Image.fromarray(values, mode="L").save(owner)

            finalized = finalize_review_manifest(
                candidates / "candidate-manifest.json",
                root / "review" / "review-manifest.json",
                reviewer="human-reviewer",
                attested=True,
            )

            self.assertEqual(finalized["attestation"], REVIEW_ATTESTATION)
            self.assertEqual(finalized["records"][0]["decision"], "approved")
            self.assertEqual(finalized["records"][0]["contact_owner_sha256"], _sha256(owner))


if __name__ == "__main__":
    unittest.main()
