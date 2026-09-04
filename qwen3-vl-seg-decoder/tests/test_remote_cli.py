from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ownership_decoder.remote_cli import build_parser, entrypoint, main
from ownership_decoder.remote_preflight import RemoteRuntimePreflightError
from ownership_decoder.remote_telemetry import (
    RuntimeFatalSafetyError,
    RuntimeWorkerRestartRequired,
)


class RemoteCliTests(unittest.TestCase):
    def test_parser_builds_explicit_campaign_spec_with_blackwell_torch_pin(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--config",
                "a.json",
                "--config",
                "b.json",
                "--input-root",
                "/volume/inputs",
                "--output-root",
                "/volume/outputs",
                "--sam-repo",
                "/volume/sam3",
                "--sam-revision",
                "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da",
                "--checkpoint",
                "/volume/sam3.1_multiplex.pt",
                "--checkpoint-sha256",
                "0" * 64,
                "--workspace",
                "/volume",
                "--attempt-index",
                "2",
            ]
        )

        self.assertEqual(args.config, [Path("a.json"), Path("b.json")])
        self.assertEqual(args.attempt_index, 2)
        self.assertIn("torch==2.12.1+cu130", args.require_distribution)
        self.assertIn("torchvision==0.27.1+cu130", args.require_distribution)
        self.assertIn("numpy==1.26.4", args.require_distribution)

    def test_main_rejects_malformed_or_duplicate_distribution_pins(self) -> None:
        base = [
            "--config",
            "a.json",
            "--input-root",
            "/input",
            "--output-root",
            "/output",
            "--sam-repo",
            "/sam",
            "--sam-revision",
            "rev",
            "--checkpoint",
            "/checkpoint",
            "--checkpoint-sha256",
            "0" * 64,
            "--workspace",
            "/workspace",
        ]
        with self.assertRaisesRegex(ValueError, "name==version"):
            main([*base, "--require-distribution", "broken"], runner=lambda _: {})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            main(
                [
                    *base,
                    "--require-distribution",
                    "torch==other",
                ],
                runner=lambda _: {},
            )

    def test_main_passes_exact_paths_hashes_and_attempt_to_runner(self) -> None:
        captured = []
        result = main(
            [
                "--config",
                "a.json",
                "--input-root",
                "/input",
                "--output-root",
                "/output",
                "--sam-repo",
                "/sam",
                "--sam-revision",
                "rev",
                "--checkpoint",
                "/checkpoint",
                "--checkpoint-sha256",
                "a" * 64,
                "--workspace",
                "/workspace",
                "--attempt-index",
                "1",
                "--require-distribution",
                "Pillow==12.3.0",
            ],
            runner=lambda spec: captured.append(spec) or {"format": "ok"},
        )

        self.assertEqual(result, 0)
        self.assertEqual(captured[0].checkpoint_sha256, "a" * 64)
        self.assertEqual(captured[0].attempt_index, 1)
        self.assertEqual(
            captured[0].required_distribution_versions,
            (
                ("Pillow", "12.3.0"),
                ("numpy", "1.26.4"),
                ("torch", "2.12.1+cu130"),
                ("torchvision", "0.27.1+cu130"),
            ),
        )

    def test_entrypoint_maps_recoverable_oom_to_fresh_worker_exit_code(self) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        for error in (
            RuntimeWorkerRestartRequired("pressure"),
            OutOfMemoryError("CUDA out of memory"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertEqual(
                    entrypoint([], main_fn=lambda _: (_ for _ in ()).throw(error)),
                    75,
                )

    def test_entrypoint_maps_preflight_and_fatal_safety_to_non_retry_exit_code(self) -> None:
        for error in (
            RemoteRuntimePreflightError("wrong GPU"),
            RuntimeFatalSafetyError("low disk"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertEqual(
                    entrypoint([], main_fn=lambda _: (_ for _ in ()).throw(error)),
                    70,
                )


if __name__ == "__main__":
    unittest.main()
