from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.diagnostic_controller import (
    DiagnosticCampaignSpec,
    run_diagnostic_campaign,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DiagnosticControllerTests(unittest.TestCase):
    def test_selected_static_and_action_folds_render_serially_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            decoder_root = root / "decoder"
            work_items = decoder_root / "work-items"
            work_items.mkdir(parents=True)
            clips = ("a", "b", "c", "d")
            static = {clip: f"static-{clip}" for clip in clips}
            action = {clip: f"action-{clip}" for clip in clips}
            for mode, mapping in (("static", static), ("action", action)):
                for clip, run_name in mapping.items():
                    (work_items / f"{run_name}.json").write_text(
                        json.dumps(
                            {
                                "run_name": run_name,
                                "heldout_clip": clip,
                                "semantic_condition": (
                                    None if mode == "static" else "action_relational"
                                ),
                                "device": "cpu",
                            }
                        )
                    )
            result_path = decoder_root / "campaign-result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "format": "ownership-local-decoder-campaign-result-v1",
                        "clip_ids": list(clips),
                        "selected_static_run_by_heldout": static,
                        "selected_action_run_by_heldout": action,
                    }
                )
            )
            (decoder_root / "RUN_COMPLETE").write_text(
                json.dumps({"result_sha256": _sha256(result_path)})
            )
            seen = []

            def fake_runner(work_item: Path, destination: Path) -> dict:
                payload = json.loads(work_item.read_text())
                seen.append(payload["run_name"])
                destination.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "format": "ownership-fold-diagnostics-v1",
                    "run_name": payload["run_name"],
                    "heldout_clip": payload["heldout_clip"],
                    "frame_count": 1,
                    "records": [{"frame_index": 0}],
                }
                manifest_path = destination / "diagnostic-manifest.json"
                manifest_path.write_text(json.dumps(manifest))
                (destination / "RUN_COMPLETE").write_text(
                    json.dumps({"manifest_sha256": _sha256(manifest_path)})
                )
                return manifest

            spec = DiagnosticCampaignSpec(
                decoder_output_root=decoder_root,
                output_root=root / "diagnostics",
                python_executable=Path(sys.executable),
            )
            first = run_diagnostic_campaign(spec, job_runner=fake_runner)
            resumed = run_diagnostic_campaign(spec, job_runner=fake_runner)

            self.assertEqual(first, resumed)
            self.assertEqual(len(seen), 8)
            self.assertEqual(first["diagnostic_count"], 8)
            self.assertTrue((root / "diagnostics/RUN_COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
