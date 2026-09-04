from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.visualization import render_ownership_diagnostic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OwnershipVisualizationTests(unittest.TestCase):
    def test_renderer_writes_deterministic_six_panel_contact_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[:, :16, 0] = 180
            rgb[:, 16:, 2] = 180
            rgb_path = root / "frame.jpg"
            Image.fromarray(rgb, mode="RGB").save(rgb_path)
            labels = np.zeros((6, 8), dtype=np.uint8)
            labels[:, :4] = 1
            labels[:, 4:] = 2
            contact = np.zeros((6, 8), dtype=bool)
            contact[2:4, 3:5] = True
            logits = np.zeros((3, 6, 8), dtype=np.float32)
            logits[1, :, :4] = 4.0
            logits[2, :, 4:] = 4.0
            first = root / "first.png"
            second = root / "second.png"

            report = render_ownership_diagnostic(
                rgb_path=rgb_path,
                labels=labels,
                logits=logits,
                contact=contact,
                output_path=first,
                title="clip / frame 0",
                panel_size=(160, 120),
            )
            render_ownership_diagnostic(
                rgb_path=rgb_path,
                labels=labels,
                logits=logits,
                contact=contact,
                output_path=second,
                title="clip / frame 0",
                panel_size=(160, 120),
            )

            self.assertEqual(report["format"], "ownership-diagnostic-v1")
            self.assertEqual(report["contact_bbox_grid"], [3, 2, 5, 4])
            self.assertEqual(_sha256(first), _sha256(second))
            with Image.open(first) as rendered:
                self.assertEqual(rendered.size, (480, 278))

    def test_renderer_rejects_missing_contact_or_nonfinite_logits(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rgb_path = root / "frame.png"
            Image.new("RGB", (8, 6)).save(rgb_path)
            labels = np.zeros((6, 8), dtype=np.uint8)
            contact = np.zeros((6, 8), dtype=bool)
            logits = np.zeros((3, 6, 8), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "contact"):
                render_ownership_diagnostic(
                    rgb_path=rgb_path,
                    labels=labels,
                    logits=logits,
                    contact=contact,
                    output_path=root / "missing.png",
                )
            contact[2, 3] = True
            logits[0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                render_ownership_diagnostic(
                    rgb_path=rgb_path,
                    labels=labels,
                    logits=logits,
                    contact=contact,
                    output_path=root / "nan.png",
                )


if __name__ == "__main__":
    unittest.main()
