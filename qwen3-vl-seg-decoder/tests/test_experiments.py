from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.data import FrameSampleSpec
from ownership_decoder.experiments import (
    STATIC_ARMS,
    build_leave_one_clip_out_folds,
    build_nested_leave_one_clip_out_folds,
    split_armbar_specs,
)


class ExperimentDesignTests(unittest.TestCase):
    def test_armbar_split_keeps_validation_and_test_out_of_screen_training(self) -> None:
        specs = [
            FrameSampleSpec("armbar", 0, {}, subset="train", screen_subset="train"),
            FrameSampleSpec("armbar", 1, {}, subset="train", screen_subset="validation"),
            FrameSampleSpec("armbar", 2, {}, subset="test", screen_subset="test"),
        ]

        split = split_armbar_specs(specs)

        self.assertEqual([item.frame_index for item in split.screen_train], [0])
        self.assertEqual([item.frame_index for item in split.validation], [1])
        self.assertEqual([item.frame_index for item in split.final_train], [0, 1])
        self.assertEqual([item.frame_index for item in split.test], [2])
        self.assertTrue(
            set(item.frame_index for item in split.screen_train).isdisjoint(
                item.frame_index for item in split.validation + split.test
            )
        )

    def test_static_arm_matrix_includes_required_input_ablations(self) -> None:
        self.assertEqual(STATIC_ARMS["rgb"].input_channels, {"rgb": 3})
        self.assertEqual(STATIC_ARMS["l11"].full_layers, (11,))
        self.assertEqual(STATIC_ARMS["p12"].pooled_layers, (12,))
        self.assertEqual(STATIC_ARMS["p12"].input_channels, {"pooled_12": 1152})
        self.assertTrue(STATIC_ARMS["merged"].include_merged)
        self.assertEqual(STATIC_ARMS["l05_l11_l18_l26"].full_layers, (5, 11, 18, 26))
        self.assertTrue(STATIC_ARMS["l05_l11_l18_l26_merged"].include_merged)

    def test_leave_one_clip_out_folds_never_mix_heldout_clip_into_training(self) -> None:
        clip_ids = ("back", "guard", "half", "mount")

        folds = build_leave_one_clip_out_folds(clip_ids)

        self.assertEqual([fold.heldout_clip for fold in folds], sorted(clip_ids))
        self.assertEqual(len(folds), 4)
        for fold in folds:
            self.assertNotIn(fold.heldout_clip, fold.train_clips)
            self.assertEqual(set(fold.train_clips) | {fold.heldout_clip}, set(clip_ids))

    def test_leave_one_clip_out_rejects_duplicate_or_too_few_clips(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            build_leave_one_clip_out_folds(("a", "a", "b"))
        with self.assertRaisesRegex(ValueError, "at least three"):
            build_leave_one_clip_out_folds(("a", "b"))

    def test_nested_outer_folds_reserve_a_whole_validation_clip(self) -> None:
        clip_ids = ("back", "guard", "half", "mount")

        folds = build_nested_leave_one_clip_out_folds(clip_ids)

        self.assertEqual(len(folds), 4)
        self.assertEqual({fold.validation_clip for fold in folds}, set(clip_ids))
        for fold in folds:
            self.assertNotEqual(fold.heldout_clip, fold.validation_clip)
            self.assertNotIn(fold.heldout_clip, fold.train_clips)
            self.assertNotIn(fold.validation_clip, fold.train_clips)
            self.assertEqual(
                set(fold.train_clips) | {fold.validation_clip, fold.heldout_clip},
                set(clip_ids),
            )

    def test_nested_folds_require_four_unique_clips(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least four"):
            build_nested_leave_one_clip_out_folds(("a", "b", "c"))


if __name__ == "__main__":
    unittest.main()
