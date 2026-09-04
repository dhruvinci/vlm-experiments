from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.remote_mask_cli import build_parser, entrypoint, main
from ownership_decoder.remote_preflight import RemoteRuntimePreflightError
from ownership_decoder.remote_telemetry import RuntimeWorkerRestartRequired


def _write_artifact_manifest(root: Path) -> tuple[Path, Path]:
    model = root / "model"
    model.mkdir()
    records = []
    for name in ("config.json", "model.safetensors", "processor_config.json"):
        path = model / name
        path.write_bytes(name.encode())
        records.append(
            {
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = root / "artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "sam3-model-artifacts-v1",
                "model_revision": "3c879f39826c281e95690f02c7821c4de09afae7",
                "artifacts": records,
            }
        )
    )
    return model, manifest


def _base(root: Path, model: Path, manifest: Path) -> list[str]:
    return [
        "--config",
        str(root / "clip.json"),
        "--input-root",
        str(root / "inputs"),
        "--output-root",
        str(root / "outputs"),
        "--sam-repo",
        str(root / "repo"),
        "--sam-revision",
        "repo-rev",
        "--sam31-checkpoint",
        str(root / "sam31.pt"),
        "--sam31-checkpoint-sha256",
        "a" * 64,
        "--sam3-model-directory",
        str(model),
        "--sam3-artifact-manifest",
        str(manifest),
        "--workspace",
        str(root),
    ]


class RemoteMaskCliTests(unittest.TestCase):
    def test_parser_defaults_pin_blackwell_core_and_transformers_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model, manifest = _write_artifact_manifest(root)
            args = build_parser().parse_args(_base(root, model, manifest))

            self.assertIn("torch==2.12.1+cu130", args.require_distribution)
            self.assertIn("torchvision==0.27.1+cu130", args.require_distribution)
            self.assertIn("sam3==0.1.0", args.require_distribution)
            self.assertIn("numpy==1.26.4", args.require_distribution)
            self.assertIn("transformers==5.16.1", args.require_distribution)
            self.assertIn("tokenizers==0.23.1", args.require_distribution)
            self.assertIn("portalocker==4.3.0", args.require_distribution)
            self.assertIn("huggingface-hub==1.28.0", args.require_distribution)
            self.assertIn("typer==0.27.1", args.require_distribution)
            self.assertEqual(len(args.require_distribution), 37)

    def test_main_loads_sorted_hash_bound_artifacts_and_passes_exact_spec(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model, manifest = _write_artifact_manifest(root)
            captured = []

            result = main(
                [*_base(root, model, manifest), "--attempt-index", "2"],
                runner=lambda spec: captured.append(spec)
                or {"format": "ok", "frame_count": 96},
            )

            self.assertEqual(result, 0)
            spec = captured[0]
            self.assertEqual(spec.attempt_index, 2)
            self.assertEqual(
                [item.path.name for item in spec.sam3_model_artifacts],
                ["config.json", "model.safetensors", "processor_config.json"],
            )
            self.assertEqual(
                spec.sam3_model_revision,
                "3c879f39826c281e95690f02c7821c4de09afae7",
            )

    def test_manifest_rejects_path_escape_duplicate_or_unexpected_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model, manifest = _write_artifact_manifest(root)
            payload = json.loads(manifest.read_text())
            for mutation, message in (
                (lambda value: value["artifacts"][0].update(relative_path="../escape"), "escape"),
                (lambda value: value["artifacts"].append(dict(value["artifacts"][0])), "duplicate"),
                (lambda value: value.update(extra="forbidden"), "keys"),
            ):
                with self.subTest(message=message):
                    changed = json.loads(json.dumps(payload))
                    mutation(changed)
                    manifest.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, message):
                        main(_base(root, model, manifest), runner=lambda spec: {})
            manifest.write_text(json.dumps(payload))

    def test_entrypoint_maps_oom_and_safety_failure_to_supervisor_codes(self) -> None:
        self.assertEqual(
            entrypoint(
                [],
                main_fn=lambda argv: (_ for _ in ()).throw(
                    RuntimeWorkerRestartRequired("pressure")
                ),
            ),
            75,
        )
        self.assertEqual(
            entrypoint(
                [],
                main_fn=lambda argv: (_ for _ in ()).throw(
                    RemoteRuntimePreflightError("wrong GPU")
                ),
            ),
            70,
        )


if __name__ == "__main__":
    unittest.main()
