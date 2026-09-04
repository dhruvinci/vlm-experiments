from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.data import OwnershipSample
from ownership_decoder.model import OwnershipDecoder, SemanticOwnershipDecoder
from ownership_decoder.losses import semantic_ownership_loss
from ownership_decoder.training import (
    TrainingConfig,
    _class_weights,
    evaluate_query_swap,
    train_decoder,
)


def _learnable_samples() -> list[OwnershipSample]:
    labels = [
        torch.tensor([[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, 1, 1], [0, 0, 2, 2]]),
        torch.tensor([[2, 2, 0, 0], [2, 1, 1, 0], [2, 1, 0, 0], [2, 2, 1, 1]]),
    ]
    samples = []
    for frame_index, ownership in enumerate(labels):
        features = torch.nn.functional.one_hot(ownership, num_classes=3).permute(2, 0, 1).float()
        features = torch.cat((features, torch.ones((1, 4, 4))), dim=0)
        samples.append(
            OwnershipSample(
                clip_id="synthetic",
                frame_index=frame_index,
                spatial={"features": features},
                labels=ownership.long(),
                contact=torch.zeros_like(ownership, dtype=torch.bool),
                actor_states=None,
            )
        )
    return samples


class DecoderTrainingTests(unittest.TestCase):
    def test_class_weights_use_label_only_loader_when_available(self) -> None:
        class LabelOnlyDataset:
            def __len__(self) -> int:
                return 2

            def load_labels(self, index: int) -> torch.Tensor:
                labels = (
                    torch.tensor([[0, 1], [2, 255]])
                    if index == 0
                    else torch.tensor([[0, 0], [1, 2]])
                )
                return labels.long()

            def __getitem__(self, index: int) -> OwnershipSample:
                raise AssertionError("feature tensors must not be loaded for class weights")

        weights = _class_weights(LabelOnlyDataset())

        self.assertEqual(tuple(weights.shape), (3,))
        self.assertTrue(torch.isfinite(weights).all())

    def test_semantic_query_swap_evaluation_streams_exact_equivariance(self) -> None:
        samples = _learnable_samples()
        actor_states = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        )
        for sample in samples:
            sample.actor_states = actor_states
        model = SemanticOwnershipDecoder(
            input_channels={"features": 4},
            semantic_dim=6,
            width=8,
            residual_blocks=0,
        )

        metrics = evaluate_query_swap(
            model,
            samples,
            device=torch.device("cpu"),
            use_amp=False,
        )

        self.assertAlmostEqual(metrics["background_probability_delta"], 0.0, places=6)
        self.assertAlmostEqual(metrics["actor_probability_swap_error"], 0.0, places=6)

    def test_static_decoder_uses_reviewed_contact_loss(self) -> None:
        samples = _learnable_samples()
        samples[0].contact[0, 2] = True
        model = OwnershipDecoder(input_channels={"features": 4}, width=8, residual_blocks=0)

        with patch(
            "ownership_decoder.training.semantic_ownership_loss",
            wraps=semantic_ownership_loss,
        ) as contact_loss:
            train_decoder(
                model,
                samples,
                samples,
                config=TrainingConfig(
                    learning_rate=0.01,
                    weight_decay=0.0,
                    max_epochs=1,
                    patience=1,
                    gradient_accumulation=2,
                    seed=7,
                    device="cpu",
                    use_amp=False,
                ),
            )

        self.assertTrue(contact_loss.called)
        self.assertTrue(any(call.args[2].any() for call in contact_loss.call_args_list))
        self.assertTrue(
            all(call.kwargs.get("swapped_logits") is None for call in contact_loss.call_args_list)
        )

    def test_semantic_decoder_receives_actor_states_during_training(self) -> None:
        samples = _learnable_samples()
        actor_states = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        )
        for sample in samples:
            sample.actor_states = actor_states
        model = SemanticOwnershipDecoder(
            input_channels={"features": 4},
            semantic_dim=6,
            width=8,
            residual_blocks=0,
        )

        result = train_decoder(
            model,
            samples,
            samples,
            config=TrainingConfig(
                learning_rate=0.05,
                weight_decay=0.0,
                max_epochs=60,
                patience=60,
                gradient_accumulation=2,
                seed=7,
                device="cpu",
                use_amp=False,
            ),
        )

        self.assertGreaterEqual(result.best_metrics["macro_actor_iou"], 0.95)

    def test_rgb_only_samples_use_the_same_training_path(self) -> None:
        samples = _learnable_samples()
        for sample in samples:
            sample.rgb = sample.spatial["features"][:3]
            sample.spatial = {}
        model = OwnershipDecoder(input_channels={"rgb": 3}, width=8, residual_blocks=0)

        result = train_decoder(
            model,
            samples,
            samples,
            config=TrainingConfig(
                learning_rate=0.05,
                weight_decay=0.0,
                max_epochs=40,
                patience=40,
                gradient_accumulation=2,
                seed=7,
                device="cpu",
                use_amp=False,
            ),
        )

        self.assertGreaterEqual(result.best_metrics["macro_actor_iou"], 0.95)

    def test_two_frame_problem_overfits_above_gate(self) -> None:
        samples = _learnable_samples()
        model = OwnershipDecoder(
            input_channels={"features": 4},
            width=8,
            residual_blocks=0,
        )
        config = TrainingConfig(
            learning_rate=0.05,
            weight_decay=0.0,
            max_epochs=40,
            patience=40,
            gradient_accumulation=2,
            seed=7,
            device="cpu",
            use_amp=False,
        )

        result = train_decoder(model, samples, samples, config=config)

        self.assertGreaterEqual(result.best_metrics["macro_actor_iou"], 0.95)
        self.assertGreaterEqual(result.best_metrics["background_stability"], 0.95)
        self.assertGreaterEqual(result.best_epoch, 0)

    def test_training_resumes_from_latest_valid_epoch(self) -> None:
        samples = _learnable_samples()
        with tempfile.TemporaryDirectory() as raw_directory:
            checkpoint_directory = Path(raw_directory)
            first_model = OwnershipDecoder(
                input_channels={"features": 4},
                width=8,
                residual_blocks=0,
            )
            first = train_decoder(
                first_model,
                samples,
                samples,
                config=TrainingConfig(
                    learning_rate=0.01,
                    weight_decay=0.0,
                    max_epochs=1,
                    patience=5,
                    gradient_accumulation=2,
                    seed=7,
                    device="cpu",
                    use_amp=False,
                    checkpoint_directory=checkpoint_directory,
                ),
            )
            resumed_model = OwnershipDecoder(
                input_channels={"features": 4},
                width=8,
                residual_blocks=0,
            )
            resumed = train_decoder(
                resumed_model,
                samples,
                samples,
                config=TrainingConfig(
                    learning_rate=0.01,
                    weight_decay=0.0,
                    max_epochs=3,
                    patience=5,
                    gradient_accumulation=2,
                    seed=7,
                    device="cpu",
                    use_amp=False,
                    checkpoint_directory=checkpoint_directory,
                ),
            )

            self.assertEqual(len(first.history), 1)
            self.assertEqual(len(resumed.history), 3)
            self.assertTrue((checkpoint_directory / "epoch_0002.pt").exists())
            self.assertLessEqual(len(list(checkpoint_directory.glob("epoch_*.pt"))), 2)


if __name__ == "__main__":
    unittest.main()
