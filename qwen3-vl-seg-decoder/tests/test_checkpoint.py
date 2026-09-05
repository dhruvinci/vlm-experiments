from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.checkpoint import (
    CheckpointError,
    latest_valid_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


class AtomicCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_state_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = directory / "epoch_0003.pt"
            state = {"weight": torch.arange(4), "epoch": 3}

            manifest = save_checkpoint(path, state, metadata={"validation_iou": 0.61})
            restored, restored_manifest = load_checkpoint(path)

            torch.testing.assert_close(restored["weight"], state["weight"])
            self.assertEqual(restored["epoch"], 3)
            self.assertEqual(restored_manifest, manifest)
            self.assertEqual(restored_manifest["metadata"]["validation_iou"], 0.61)
            self.assertEqual(len(restored_manifest["sha256"]), 64)

    def test_corrupted_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = directory / "epoch_0001.pt"
            save_checkpoint(path, {"weight": torch.arange(4)}, metadata={})
            path.write_bytes(path.read_bytes() + b"corruption")

            with self.assertRaisesRegex(CheckpointError, "checksum|size"):
                load_checkpoint(path)

    def test_latest_valid_checkpoint_skips_corrupted_newer_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            older = directory / "epoch_0001.pt"
            newer = directory / "epoch_0002.pt"
            save_checkpoint(older, {"epoch": 1}, metadata={})
            save_checkpoint(newer, {"epoch": 2}, metadata={})
            newer.write_bytes(b"broken")

            selected = latest_valid_checkpoint(directory)

            self.assertEqual(selected, older)

    def test_save_recovers_from_an_uncommitted_checkpoint_after_preserving_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = directory / "epoch_0002.pt"
            path.write_bytes(b"interrupted checkpoint payload")
            unfinished_sidecar = directory / ".epoch_0002.pt.json.crash.tmp"
            unfinished_sidecar.write_text("unfinished manifest")

            save_checkpoint(path, {"epoch": 2}, metadata={"resumed": True})

            restored, manifest = load_checkpoint(path)
            self.assertEqual(restored["epoch"], 2)
            self.assertTrue(manifest["metadata"]["resumed"])
            quarantines = list(
                (directory / "failures" / "orphaned-checkpoints").glob(
                    "epoch_0002.pt.*"
                )
            )
            self.assertEqual(len(quarantines), 1)
            self.assertEqual(
                (quarantines[0] / "epoch_0002.pt").read_bytes(),
                b"interrupted checkpoint payload",
            )
            self.assertEqual(
                (quarantines[0] / unfinished_sidecar.name).read_text(),
                "unfinished manifest",
            )

    def test_save_still_refuses_to_overwrite_a_verified_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "epoch_0002.pt"
            save_checkpoint(path, {"epoch": 2}, metadata={})

            with self.assertRaisesRegex(CheckpointError, "verified checkpoint"):
                save_checkpoint(path, {"epoch": 2}, metadata={})

    def test_pruning_keeps_named_best_and_latest_complete_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths = [directory / f"epoch_{epoch:04d}.pt" for epoch in range(3)]
            for epoch, path in enumerate(paths):
                save_checkpoint(path, {"epoch": epoch}, metadata={})

            removed = prune_checkpoints(directory, keep_epochs={0}, keep_latest=1)

            self.assertEqual(removed, [paths[1]])
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[2].exists())
            self.assertFalse(paths[1].exists())
            self.assertFalse(paths[1].with_suffix(".pt.json").exists())


if __name__ == "__main__":
    unittest.main()
