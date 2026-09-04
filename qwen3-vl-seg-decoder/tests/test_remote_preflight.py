from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ownership_decoder.remote_preflight import (
    RemoteHardwareSnapshot,
    RequiredArtifact,
    RemoteRuntimeContract,
    RemoteRuntimePreflightError,
    perform_remote_preflight,
)


GIB = 1024**3


def safe_blackwell_snapshot() -> RemoteHardwareSnapshot:
    return RemoteHardwareSnapshot(
        gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpu_total_bytes=96 * GIB,
        gpu_free_bytes=92 * GIB,
        compute_capability=(12, 0),
        driver_version=(580, 65, 6),
        host_available_bytes=64 * GIB,
        workspace_free_bytes=180 * GIB,
    )


class RemotePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repo = self.root / "sam3"
        self.repo.mkdir()
        self.checkpoint = self.root / "sam3.1_multiplex.pt"
        self.checkpoint.write_bytes(b"small-test-checkpoint")
        self.checkpoint_sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.contract = RemoteRuntimeContract(
            sam_repo_path=self.repo,
            sam_repo_revision="8f0b7f4d4e7eda2ed606ebde6702c93359ad01da",
            checkpoint_path=self.checkpoint,
            checkpoint_sha256=self.checkpoint_sha,
            workspace_path=self.root,
            required_distribution_versions=(
                ("torch", "2.12.1"),
                ("numpy", "2.3.3"),
            ),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_wrong_gpu_fails_before_repo_dependencies_or_checkpoint_are_touched(self) -> None:
        snapshot = safe_blackwell_snapshot()
        wrong_gpu = RemoteHardwareSnapshot(
            **{**snapshot.as_dict(), "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU"}
        )
        touched: list[str] = []

        with self.assertRaisesRegex(RemoteRuntimePreflightError, "GPU model"):
            perform_remote_preflight(
                self.contract,
                hardware_probe=lambda _: wrong_gpu,
                python_version_probe=lambda: touched.append("python") or (3, 12, 1),
                repo_revision_probe=lambda _: touched.append("repo") or "should-not-run",
                distribution_version_probe=lambda _: touched.append("dependency") or "0",
                checkpoint_hash_probe=lambda _: touched.append("checkpoint") or "0" * 64,
            )

        self.assertEqual(touched, [])

    def test_max_q_and_insufficient_96gb_capacity_are_rejected(self) -> None:
        safe = safe_blackwell_snapshot()
        cases = (
            RemoteHardwareSnapshot(
                **{**safe.as_dict(), "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q"}
            ),
            RemoteHardwareSnapshot(
                **{**safe.as_dict(), "gpu_total_bytes": 24 * GIB, "gpu_free_bytes": 23 * GIB}
            ),
        )
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                with self.assertRaises(RemoteRuntimePreflightError):
                    perform_remote_preflight(self.contract, hardware_probe=lambda _: snapshot)

    def test_low_free_vram_host_ram_disk_compute_or_driver_are_each_rejected(self) -> None:
        safe = safe_blackwell_snapshot().as_dict()
        cases = {
            "free VRAM": {"gpu_free_bytes": 79 * GIB},
            "host RAM": {"host_available_bytes": 31 * GIB},
            "workspace disk": {"workspace_free_bytes": 49 * GIB},
            "compute capability": {"compute_capability": (11, 0)},
            "driver": {"driver_version": (579, 99, 99)},
        }
        for expected_message, changed in cases.items():
            with self.subTest(expected_message=expected_message):
                snapshot = RemoteHardwareSnapshot(**{**safe, **changed})
                with self.assertRaisesRegex(RemoteRuntimePreflightError, expected_message):
                    perform_remote_preflight(self.contract, hardware_probe=lambda _: snapshot)

    def test_safe_runtime_is_fully_verified_and_returns_contract_bound_approval(self) -> None:
        calls: list[str] = []

        approval = perform_remote_preflight(
            self.contract,
            hardware_probe=lambda _: safe_blackwell_snapshot(),
            python_version_probe=lambda: (3, 12, 1),
            repo_revision_probe=lambda _: calls.append("repo") or self.contract.sam_repo_revision,
            distribution_version_probe=lambda name: calls.append(name)
            or dict(self.contract.required_distribution_versions)[name],
            checkpoint_hash_probe=lambda _: calls.append("checkpoint") or self.checkpoint_sha,
        )

        self.assertEqual(calls, ["repo", "torch", "numpy", "checkpoint"])
        self.assertEqual(approval.contract_sha256, self.contract.sha256)
        self.assertEqual(approval.checkpoint_sha256, self.checkpoint_sha)
        self.assertEqual(approval.hardware.gpu_name, safe_blackwell_snapshot().gpu_name)

    def test_checkpoint_hash_mismatch_fails_after_cheap_checks(self) -> None:
        with self.assertRaisesRegex(RemoteRuntimePreflightError, "checkpoint SHA-256"):
            perform_remote_preflight(
                self.contract,
                hardware_probe=lambda _: safe_blackwell_snapshot(),
                python_version_probe=lambda: (3, 12, 1),
                repo_revision_probe=lambda _: self.contract.sam_repo_revision,
                distribution_version_probe=lambda name: dict(
                    self.contract.required_distribution_versions
                )[name],
                checkpoint_hash_probe=lambda _: "0" * 64,
            )

    def test_wrong_python_fails_before_repo_dependencies_or_checkpoint_are_touched(self) -> None:
        touched: list[str] = []

        with self.assertRaisesRegex(RemoteRuntimePreflightError, "Python"):
            perform_remote_preflight(
                self.contract,
                hardware_probe=lambda _: safe_blackwell_snapshot(),
                python_version_probe=lambda: (3, 11, 9),
                repo_revision_probe=lambda _: touched.append("repo") or "should-not-run",
                distribution_version_probe=lambda _: touched.append("dependency") or "0",
                checkpoint_hash_probe=lambda _: touched.append("checkpoint") or "0" * 64,
            )

        self.assertEqual(touched, [])

    def test_additional_model_assets_are_size_and_hash_bound_to_approval(self) -> None:
        asset = self.root / "base-sam.safetensors"
        asset.write_bytes(b"base-model")
        asset_sha = hashlib.sha256(asset.read_bytes()).hexdigest()
        contract = replace(
            self.contract,
            additional_artifacts=(
                RequiredArtifact(
                    path=asset,
                    sha256=asset_sha,
                    size_bytes=asset.stat().st_size,
                ),
            ),
        )
        hashed: list[Path] = []

        approval = perform_remote_preflight(
            contract,
            hardware_probe=lambda _: safe_blackwell_snapshot(),
            python_version_probe=lambda: (3, 12, 1),
            repo_revision_probe=lambda _: contract.sam_repo_revision,
            distribution_version_probe=lambda name: dict(
                contract.required_distribution_versions
            )[name],
            checkpoint_hash_probe=lambda path: hashed.append(path)
            or (self.checkpoint_sha if path == self.checkpoint else asset_sha),
        )

        self.assertEqual(hashed, [self.checkpoint, asset])
        self.assertEqual(
            approval.additional_artifact_sha256,
            ((str(asset.resolve()), asset_sha),),
        )


if __name__ == "__main__":
    unittest.main()
