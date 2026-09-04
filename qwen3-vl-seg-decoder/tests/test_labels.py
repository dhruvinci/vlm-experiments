from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.labels import (
    apply_reviewed_contact_ownership,
    build_agreement_ownership_labels,
    label_summary,
    validate_label_ready,
)


class OwnershipLabelTests(unittest.TestCase):
    def test_only_cross_backend_actor_agreement_becomes_actor_truth(self) -> None:
        shape = (12, 12)
        sam3_a1 = np.zeros(shape, dtype=bool)
        sam31_a1 = np.zeros(shape, dtype=bool)
        sam3_a2 = np.zeros(shape, dtype=bool)
        sam31_a2 = np.zeros(shape, dtype=bool)
        sam3_a1[:6, :6] = True
        sam31_a1[:6, :6] = True
        sam3_a2[6:, 6:] = True
        sam31_a2[6:, 6:] = True

        labels = build_agreement_ownership_labels(
            sam3_a1,
            sam3_a2,
            sam31_a1,
            sam31_a2,
            output_hw=(2, 2),
            erosion_radius=1,
        )

        np.testing.assert_array_equal(
            labels,
            np.array([[1, 0], [0, 2]], dtype=np.uint8),
        )

    def test_disagreement_overlap_and_possible_foreground_stay_ignored(self) -> None:
        shape = (10, 10)
        sam3_a1 = np.zeros(shape, dtype=bool)
        sam31_a1 = np.zeros(shape, dtype=bool)
        sam3_a2 = np.zeros(shape, dtype=bool)
        sam31_a2 = np.zeros(shape, dtype=bool)
        sam3_a1[:5, :5] = True
        # Cross-backend disagreement: SAM 3.1 assigns the same cell to A2.
        sam31_a2[:5, :5] = True
        possible = np.zeros(shape, dtype=bool)
        possible[:5, :] = True

        labels = build_agreement_ownership_labels(
            sam3_a1,
            sam3_a2,
            sam31_a1,
            sam31_a2,
            possible_foreground=possible,
            output_hw=(2, 2),
            erosion_radius=1,
        )

        self.assertEqual(int(labels[0, 0]), 255)
        self.assertEqual(int(labels[0, 1]), 255)
        self.assertEqual(int(labels[1, 0]), 0)
        self.assertEqual(int(labels[1, 1]), 0)

    def test_contact_correction_requires_explicit_review_and_overrides_candidate(self) -> None:
        labels = np.array([[0, 255], [1, 2]], dtype=np.uint8)
        owner = np.array([[0, 2], [0, 1]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "reviewed"):
            apply_reviewed_contact_ownership(labels, owner, reviewed=False)

        corrected, contact = apply_reviewed_contact_ownership(labels, owner, reviewed=True)

        np.testing.assert_array_equal(corrected, np.array([[0, 2], [1, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(contact, np.array([[False, True], [False, True]]))

    def test_ready_validation_requires_actor_truth_at_every_contact_pixel(self) -> None:
        labels = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        contact = np.array([[False, False], [True, False]])
        validate_label_ready(labels, contact, require_both_actors=True)

        invalid = contact.copy()
        invalid[0, 0] = True
        with self.assertRaisesRegex(ValueError, "contact"):
            validate_label_ready(labels, invalid, require_both_actors=True)

    def test_summary_exposes_coverage_and_review_state(self) -> None:
        labels = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        contact = np.array([[False, True], [True, False]])

        summary = label_summary(labels, contact, contact_reviewed=True)

        self.assertEqual(summary["pixel_counts"], {"0": 1, "1": 1, "2": 1, "255": 1})
        self.assertEqual(summary["labeled_fraction"], 0.75)
        self.assertEqual(summary["contact_patch_count"], 2)
        self.assertTrue(summary["contact_reviewed"])


if __name__ == "__main__":
    unittest.main()
