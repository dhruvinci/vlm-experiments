from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.losses import balanced_class_weights, ownership_loss, semantic_ownership_loss


class OwnershipLossTests(unittest.TestCase):
    def test_balanced_weights_are_inverse_frequency_and_mean_normalized(self) -> None:
        labels = torch.tensor([[[0, 0, 0, 0, 1, 1, 2, 255]]])

        weights = balanced_class_weights(labels)

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))

    def test_correct_logits_have_lower_combined_loss_than_swapped_logits(self) -> None:
        labels = torch.tensor([[[0, 1], [2, 255]]])
        correct = torch.tensor(
            [[
                [[8.0, 0.0], [0.0, 0.0]],
                [[0.0, 8.0], [0.0, 0.0]],
                [[0.0, 0.0], [8.0, 0.0]],
            ]],
            requires_grad=True,
        )
        wrong = correct.detach()[:, [0, 2, 1]].clone().requires_grad_(True)

        correct_losses = ownership_loss(correct, labels, dice_weight=0.5)
        wrong_losses = ownership_loss(wrong, labels, dice_weight=0.5)
        correct_losses["total"].backward()

        self.assertLess(
            float(correct_losses["total"].detach()),
            float(wrong_losses["total"].detach()),
        )
        self.assertTrue(torch.isfinite(correct.grad).all())
        self.assertEqual(set(correct_losses), {"total", "cross_entropy", "dice"})

    def test_all_ignored_batch_is_rejected(self) -> None:
        logits = torch.zeros((1, 3, 2, 2))
        labels = torch.full((1, 2, 2), 255)

        with self.assertRaisesRegex(ValueError, "labeled pixel"):
            ownership_loss(logits, labels)

    def test_semantic_loss_adds_contact_and_swap_equivariance_terms(self) -> None:
        logits = torch.tensor(
            [[
                [[0.0, 0.0]],
                [[4.0, 1.0]],
                [[1.0, 4.0]],
            ]],
            requires_grad=True,
        )
        labels = torch.tensor([[[1, 2]]])
        contact = torch.tensor([[[False, True]]])
        exact_swap = logits[:, [0, 2, 1]].detach().clone().requires_grad_(True)
        incorrect_swap = logits.detach().clone().requires_grad_(True)

        exact = semantic_ownership_loss(logits, labels, contact, swapped_logits=exact_swap)
        incorrect = semantic_ownership_loss(logits, labels, contact, swapped_logits=incorrect_swap)

        self.assertAlmostEqual(float(exact["swap_equivariance"].detach()), 0.0)
        self.assertGreater(float(incorrect["swap_equivariance"].detach()), 0.0)
        self.assertGreater(float(exact["contact_cross_entropy"].detach()), 0.0)
        self.assertEqual(
            set(exact),
            {"total", "cross_entropy", "dice", "contact_cross_entropy", "swap_equivariance"},
        )


if __name__ == "__main__":
    unittest.main()
