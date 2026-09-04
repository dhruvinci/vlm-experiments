from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.cloud_controller import (
    build_parser,
    load_secret_environment,
    require_remote_staging,
)


class CloudControllerTests(unittest.TestCase):
    def test_cli_defaults_match_the_cache_free_staging_inventory(self) -> None:
        args = build_parser().parse_args(
            [
                "--purpose", "mig",
                "--run-id", "sam31-mask-mig-20260904",
                "--packet-manifest", "/tmp/manifest.json",
                "--packet-sha256", "a" * 64,
                "--approved-by", "user",
                "--approved-at", "2026-09-04T00:00:00+00:00",
                "--env-file", "/tmp/.env",
                "--container-image", "image@sha256:" + "b" * 64,
                "--spend-before-usd", "6.3",
                "--hourly-rate-usd", "0.59",
                "--terminate-at-usd", "7.48",
                "--max-runtime-seconds", "7200",
                "--state-path", "/tmp/state.json",
                "--monitor-log", "/tmp/monitor.jsonl",
            ]
        )
        self.assertEqual(args.expected_staging_artifacts, 717)
        self.assertEqual(args.expected_staging_bytes, 7_100_837_883)

    def test_secret_environment_accepts_colon_and_equals_without_extra_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                "RUNPOD_API_KEY: api-secret\n"
                "RUNPOD_S3_ACCESS_KEY=access-secret\n"
                "RUNPOD_S3_SECRET_KEY: 'storage-secret'\n"
                "UNRELATED: do-not-load\n",
                encoding="utf-8",
            )
            values = load_secret_environment(
                path,
                required=(
                    "RUNPOD_API_KEY",
                    "RUNPOD_S3_ACCESS_KEY",
                    "RUNPOD_S3_SECRET_KEY",
                ),
            )
            self.assertEqual(set(values), {
                "RUNPOD_API_KEY",
                "RUNPOD_S3_ACCESS_KEY",
                "RUNPOD_S3_SECRET_KEY",
            })
            self.assertEqual(values["RUNPOD_API_KEY"], "api-secret")

    def test_remote_staging_must_bind_exact_packet_and_nonempty_inventory(self) -> None:
        payload = {
            "format": "ownership-mask-staging-complete-v1",
            "packet_sha256": "a" * 64,
            "artifact_count": 790,
            "total_bytes": 7_102_514_080,
            "inventory_sha256": "b" * 64,
        }
        self.assertEqual(
            require_remote_staging(payload, packet_sha256="a" * 64)["artifact_count"],
            790,
        )
        with self.assertRaisesRegex(RuntimeError, "packet"):
            require_remote_staging(payload, packet_sha256="c" * 64)
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            require_remote_staging(
                {**payload, "artifact_count": 0}, packet_sha256="a" * 64
            )


if __name__ == "__main__":
    unittest.main()
