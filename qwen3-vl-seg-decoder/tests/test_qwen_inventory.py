from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.qwen_inventory import verify_qwen_breadth_cache


class QwenBreadthInventoryTests(unittest.TestCase):
    def test_required_cache_is_bound_to_verified_download_record(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            breadth = root / "breadth"
            artifact = breadth / "clip/spatial/full/layer_11/frame_000000.safetensors"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"representation")
            manifest = root / "download-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "path": "campaign/downloads/run/breadth/clip/spatial/full/layer_11/frame_000000.safetensors",
                                "size_bytes": artifact.stat().st_size,
                                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                "verified": True,
                            }
                        ]
                    }
                )
            )

            report = verify_qwen_breadth_cache(
                manifest,
                qwen_breadth_root=breadth,
                frame_indices_by_clip={"clip": (0,)},
                spatial_arms=("l11",),
                semantic_conditions=(),
                rehash=True,
            )

            self.assertEqual(report["verified_artifact_count"], 1)
            self.assertEqual(report["verified_bytes"], len(b"representation"))

    def test_corrupt_or_unverified_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            breadth = root / "breadth"
            artifact = breadth / "clip/spatial/full/layer_11/frame_000000.safetensors"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"changed")
            manifest = root / "download-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "path": "breadth/clip/spatial/full/layer_11/frame_000000.safetensors",
                                "size_bytes": artifact.stat().st_size,
                                "sha256": "0" * 64,
                                "verified": True,
                            }
                        ]
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_qwen_breadth_cache(
                    manifest,
                    qwen_breadth_root=breadth,
                    frame_indices_by_clip={"clip": (0,)},
                    spatial_arms=("l11",),
                    semantic_conditions=(),
                    rehash=True,
                )


if __name__ == "__main__":
    unittest.main()
