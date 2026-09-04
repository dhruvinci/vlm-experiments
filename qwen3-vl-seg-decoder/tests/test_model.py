from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.model import OwnershipDecoder, SemanticOwnershipDecoder


class OwnershipDecoderTests(unittest.TestCase):
    def test_semantic_decoder_is_exactly_equivariant_to_actor_query_swap(self) -> None:
        torch.manual_seed(4)
        model = SemanticOwnershipDecoder(
            input_channels={"merged": 12},
            semantic_dim=20,
            width=16,
            residual_blocks=1,
        )
        spatial = {"merged": torch.randn(2, 12, 3, 2)}
        actor_states = torch.randn(2, 2, 20)

        logits = model(spatial, actor_states=actor_states, output_size=(6, 4))
        swapped = model(spatial, actor_states=actor_states.flip(1), output_size=(6, 4))
        logits.square().mean().backward()

        self.assertEqual(tuple(logits.shape), (2, 3, 6, 4))
        torch.testing.assert_close(swapped[:, 0], logits[:, 0])
        torch.testing.assert_close(swapped[:, 1], logits[:, 2])
        torch.testing.assert_close(swapped[:, 2], logits[:, 1])
        self.assertIsNotNone(model.query_projection.weight.grad)
        self.assertTrue(torch.isfinite(model.query_projection.weight.grad).all())

    def test_static_decoder_fuses_different_grids_and_backpropagates(self) -> None:
        model = OwnershipDecoder(
            input_channels={"layer_11": 8, "merged": 12},
            width=16,
            residual_blocks=2,
        )
        spatial = {
            "layer_11": torch.randn(2, 8, 6, 4),
            "merged": torch.randn(2, 12, 3, 2),
        }

        logits = model(spatial, output_size=(6, 4))
        loss = logits.square().mean()
        loss.backward()

        self.assertEqual(tuple(logits.shape), (2, 3, 6, 4))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_production_shape_stays_under_parameter_budget(self) -> None:
        model = OwnershipDecoder(
            input_channels={
                "layer_05": 1152,
                "layer_11": 1152,
                "layer_18": 1152,
                "layer_26": 1152,
                "merged": 5120,
            },
            width=192,
            residual_blocks=3,
        )

        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertLessEqual(parameter_count, 12_000_000)

    def test_missing_or_unconfigured_sources_are_rejected(self) -> None:
        model = OwnershipDecoder(input_channels={"layer_11": 8}, width=8, residual_blocks=1)

        with self.assertRaisesRegex(ValueError, "source set"):
            model({"merged": torch.randn(1, 12, 3, 2)})


if __name__ == "__main__":
    unittest.main()
