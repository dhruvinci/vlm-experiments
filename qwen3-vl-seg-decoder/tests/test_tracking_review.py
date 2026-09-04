from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.tracking import ActorPrompt, FrameSpec, PropagationSegment, SeedPair, TrackingPlan
from ownership_decoder.tracking_review import render_seed_prompt_review


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrackingReviewTests(unittest.TestCase):
    def test_renderer_creates_plan_bound_atomic_prompt_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            frame_path = root / "frame_000001.jpg"
            Image.new("RGB", (120, 80), "gray").save(frame_path)
            frame = FrameSpec(0, frame_path, _sha256(frame_path), 80, 120)
            a1 = ActorPrompt(
                "A1", (0.05, 0.05, 0.45, 0.95), ((0.15, 0.25), (0.30, 0.70)),
                ((0.65, 0.25), (0.80, 0.70)),
            )
            a2 = ActorPrompt(
                "A2", (0.50, 0.05, 0.95, 0.95), ((0.65, 0.25), (0.80, 0.70)),
                ((0.15, 0.25), (0.30, 0.70)),
            )
            plan = TrackingPlan(
                "test_clip", (frame,), (SeedPair(0, (a1, a2)),),
                (PropagationSegment(0, 1),),
            )
            output = root / "review.png"

            sidecar = render_seed_prompt_review(plan, output, max_panel_width=100)

            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".png.json").is_file())
            with Image.open(output) as rendered:
                self.assertEqual(rendered.width, 100)
                self.assertGreater(rendered.height, 60)
            self.assertEqual(sidecar["plan_sha256"], plan.sha256)
            self.assertEqual(sidecar["image_sha256"], _sha256(output))
            self.assertEqual(json.loads(output.with_suffix(".png.json").read_text()), sidecar)


if __name__ == "__main__":
    unittest.main()
