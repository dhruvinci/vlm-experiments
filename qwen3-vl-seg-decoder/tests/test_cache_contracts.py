from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.cache import CacheContractError, load_actor_state_pair, load_spatial_map


class SpatialCacheContractTests(unittest.TestCase):
    def _write_artifact(
        self,
        directory: Path,
        *,
        hidden: torch.Tensor,
        grid_thw: tuple[int, int, int],
        stage: str,
    ) -> Path:
        path = directory / "artifact.safetensors"
        save_file(
            {
                "hidden": hidden,
                "grid_thw": torch.tensor([grid_thw], dtype=torch.int64),
            },
            path,
            metadata={"campaign": f'{{"stage":"{stage}"}}'},
        )
        return path

    def test_full_tokens_are_reconstructed_in_row_major_grid_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            hidden = torch.zeros((6, 1152), dtype=torch.bfloat16)
            hidden[:, 0] = torch.arange(6, dtype=torch.bfloat16)
            path = self._write_artifact(
                directory,
                hidden=hidden,
                grid_thw=(1, 2, 3),
                stage="spatial_full",
            )

            spatial = load_spatial_map(path, kind="full")

            self.assertEqual(tuple(spatial.shape), (1152, 2, 3))
            torch.testing.assert_close(
                spatial[0].float(),
                torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            )

    def test_grid_token_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = self._write_artifact(
                directory,
                hidden=torch.zeros((5, 1152), dtype=torch.bfloat16),
                grid_thw=(1, 2, 3),
                stage="spatial_full",
            )

            with self.assertRaisesRegex(CacheContractError, "token count"):
                load_spatial_map(path, kind="full")

    def test_merged_tokens_use_the_two_by_two_effective_grid(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            hidden = torch.zeros((6, 5120), dtype=torch.bfloat16)
            hidden[:, 0] = torch.arange(6, dtype=torch.bfloat16)
            path = self._write_artifact(
                directory,
                hidden=hidden,
                grid_thw=(1, 4, 6),
                stage="merged_vision",
            )

            spatial = load_spatial_map(path, kind="merged")

            self.assertEqual(tuple(spatial.shape), (5120, 2, 3))
            torch.testing.assert_close(
                spatial[0].float(),
                torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            )

    def test_non_finite_hidden_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            hidden = torch.zeros((6, 1152), dtype=torch.bfloat16)
            hidden[3, 7] = float("nan")
            path = self._write_artifact(
                directory,
                hidden=hidden,
                grid_thw=(1, 2, 3),
                stage="spatial_full",
            )

            with self.assertRaisesRegex(CacheContractError, "non-finite"):
                load_spatial_map(path, kind="full")


class SemanticCacheContractTests(unittest.TestCase):
    def _write_actor(self, directory: Path, actor: str, *, condition: str = "action_relational") -> Path:
        path = directory / f"{actor}.safetensors"
        states = torch.zeros((4, 5120), dtype=torch.bfloat16)
        states[:, 0] = torch.arange(4, dtype=torch.bfloat16) + (10 if actor == "A2" else 0)
        save_file(
            {"marker_states": states},
            path,
            metadata={
                "campaign": (
                    '{"stage":"semantic_video","actor":"'
                    + actor
                    + '","condition":"'
                    + condition
                    + '","context":"4fps","thinking_mode":"off"}'
                )
            },
        )
        return path

    def test_selected_language_layer_is_loaded_in_a1_a2_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            a1 = self._write_actor(directory, "A1")
            a2 = self._write_actor(directory, "A2")

            pair = load_actor_state_pair(a1, a2, language_layer=2)

            self.assertEqual(tuple(pair.shape), (2, 5120))
            self.assertEqual(pair.dtype, torch.bfloat16)
            self.assertEqual(float(pair[0, 0]), 2.0)
            self.assertEqual(float(pair[1, 0]), 12.0)

    def test_mismatched_semantic_conditions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            a1 = self._write_actor(directory, "A1", condition="action_relational")
            a2 = self._write_actor(directory, "A2", condition="identity_only")

            with self.assertRaisesRegex(CacheContractError, "semantic metadata mismatch"):
                load_actor_state_pair(a1, a2, language_layer=2)


if __name__ == "__main__":
    unittest.main()
