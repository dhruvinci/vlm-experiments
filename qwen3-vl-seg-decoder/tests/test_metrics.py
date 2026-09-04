from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.metrics import (
    OwnershipMetricAccumulator,
    SwapMetricAccumulator,
    ownership_metrics,
    swap_response_metrics,
)


class OwnershipMetricsTests(unittest.TestCase):
    def test_contact_regions_are_owner_specific_connected_components(self) -> None:
        logits = torch.zeros((1, 3, 3, 5))
        labels = torch.zeros((1, 3, 5), dtype=torch.long)
        contact = torch.zeros((1, 3, 5), dtype=torch.bool)
        labels[0, 0, :2] = 1
        contact[0, 0, :2] = True
        logits[0, 1, 0, :2] = 4.0
        labels[0, 2, 3:] = 2
        contact[0, 2, 3:] = True
        logits[0, 1, 2, 3:] = 4.0

        metrics = ownership_metrics(logits, labels, contact)

        self.assertEqual(metrics["contact_region_count"], 2.0)
        self.assertAlmostEqual(metrics["positive_contact_region_fraction"], 0.5)

    def test_metrics_ignore_unlabeled_pixels_and_measure_contact_owner(self) -> None:
        logits = torch.tensor(
            [[
                [[8.0, 0.0, 0.0, 8.0]],
                [[0.0, 8.0, 6.0, 0.0]],
                [[0.0, 0.0, 2.0, 0.0]],
            ]]
        )
        labels = torch.tensor([[0, 1, 2, 255]]).reshape(1, 1, 4)
        contact = torch.tensor([[False, False, True, False]]).reshape(1, 1, 4)

        metrics = ownership_metrics(logits, labels, contact)

        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["a1_iou"], 0.5)
        self.assertAlmostEqual(metrics["a2_iou"], 0.0)
        self.assertAlmostEqual(metrics["macro_actor_iou"], 0.25)
        self.assertAlmostEqual(metrics["background_stability"], 1.0)
        self.assertEqual(metrics["contact_pixel_count"], 1.0)
        self.assertEqual(metrics["contact_region_count"], 1.0)
        self.assertAlmostEqual(metrics["contact_accuracy"], 0.0)
        self.assertLess(metrics["contact_margin"], 0.0)
        self.assertAlmostEqual(metrics["positive_contact_margin_fraction"], 0.0)
        self.assertAlmostEqual(metrics["positive_contact_region_fraction"], 0.0)

    def test_contact_margin_uses_each_pixels_labeled_actor_as_owner(self) -> None:
        logits = torch.tensor(
            [[
                [[0.0, 0.0]],
                [[4.0, 1.0]],
                [[1.0, 4.0]],
            ]]
        )
        labels = torch.tensor([[[1, 2]]])
        contact = torch.tensor([[[True, True]]])

        metrics = ownership_metrics(logits, labels, contact)

        self.assertAlmostEqual(metrics["contact_accuracy"], 1.0)
        self.assertGreater(metrics["contact_margin"], 0.0)
        self.assertAlmostEqual(metrics["positive_contact_margin_fraction"], 1.0)
        self.assertAlmostEqual(metrics["positive_contact_region_fraction"], 1.0)

    def test_contact_pixels_without_actor_truth_are_rejected(self) -> None:
        logits = torch.zeros((1, 3, 1, 1))
        labels = torch.zeros((1, 1, 1), dtype=torch.long)
        contact = torch.ones((1, 1, 1), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "actor-owned"):
            ownership_metrics(logits, labels, contact)

    def test_streaming_accumulator_supports_different_frame_grids(self) -> None:
        accumulator = OwnershipMetricAccumulator()
        first_logits = torch.tensor([[[[8.0, 0.0]], [[0.0, 8.0]], [[0.0, 0.0]]]])
        first_labels = torch.tensor([[[0, 1]]])
        second_logits = torch.tensor([[[[0.0]], [[0.0]], [[8.0]]]])
        second_labels = torch.tensor([[[2]]])

        accumulator.update(first_logits, first_labels, torch.zeros_like(first_labels, dtype=torch.bool))
        accumulator.update(second_logits, second_labels, torch.ones_like(second_labels, dtype=torch.bool))
        metrics = accumulator.compute()

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_actor_iou"], 1.0)
        self.assertAlmostEqual(metrics["background_stability"], 1.0)
        self.assertAlmostEqual(metrics["contact_accuracy"], 1.0)
        self.assertEqual(metrics["contact_region_count"], 1.0)

    def test_swap_response_measures_actor_flip_and_background_stability(self) -> None:
        logits = torch.tensor(
            [[
                [[0.0, 0.0, 9.0]],
                [[8.0, 1.0, 0.0]],
                [[1.0, 8.0, 0.0]],
            ]]
        )
        swapped = logits[:, [0, 2, 1]]

        metrics = swap_response_metrics(logits, swapped)

        self.assertAlmostEqual(metrics["actor_prediction_flip_fraction"], 1.0)
        self.assertAlmostEqual(metrics["background_probability_delta"], 0.0)
        self.assertAlmostEqual(metrics["actor_probability_swap_error"], 0.0)

    def test_swap_accumulator_weights_different_grids_by_pixels(self) -> None:
        accumulator = SwapMetricAccumulator()
        ordinary = torch.tensor([[[[0.0, 0.0]], [[8.0, 8.0]], [[1.0, 1.0]]]])
        perfect = ordinary[:, [0, 2, 1]]
        failed = torch.tensor([[[[0.0]], [[8.0]], [[1.0]]]])

        accumulator.update(ordinary, perfect)
        accumulator.update(failed, failed)
        metrics = accumulator.compute()

        self.assertAlmostEqual(metrics["actor_prediction_flip_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["background_probability_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
