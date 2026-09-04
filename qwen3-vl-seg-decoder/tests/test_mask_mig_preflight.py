from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.mask_mig_preflight import (
    MaskMigPreflightSpec,
    run_mask_mig_preflight,
)
from ownership_decoder.remote_preflight import RemoteHardwareSnapshot, RequiredArtifact


GIB = 1024**3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> MaskMigPreflightSpec:
    packet = root / "manifest.json"
    packet.write_text('{"format":"packet"}\n')
    repo = root / "repo"
    repo.mkdir()
    checkpoint = root / "sam31.pt"
    checkpoint.write_bytes(b"sam31")
    base = root / "sam3.safetensors"
    base.write_bytes(b"sam3")
    return MaskMigPreflightSpec(
        packet_manifest_path=packet,
        packet_sha256=_sha(packet),
        sam_repo_path=repo,
        sam_repo_revision="repo-revision",
        sam31_checkpoint_path=checkpoint,
        sam31_checkpoint_sha256=_sha(checkpoint),
        sam31_checkpoint_size_bytes=checkpoint.stat().st_size,
        additional_artifacts=(
            RequiredArtifact(path=base, sha256=_sha(base), size_bytes=base.stat().st_size),
        ),
        workspace_path=root,
        required_distribution_versions=(
            ("sam3", "0.1.0"),
            ("torch", "2.12.1+cu130"),
        ),
        output_path=root / "MIG_PREFLIGHT_COMPLETE.json",
    )


def _hardware() -> RemoteHardwareSnapshot:
    return RemoteHardwareSnapshot(
        gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpu_total_bytes=24 * GIB,
        gpu_free_bytes=23 * GIB,
        compute_capability=(12, 0),
        driver_version=(580, 65, 6),
        host_available_bytes=64 * GIB,
        workspace_free_bytes=220 * GIB,
    )


class MaskMigPreflightTests(unittest.TestCase):
    def test_exact_mig_environment_and_artifacts_produce_bound_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            spec = _fixture(Path(raw))
            calls = []

            result = run_mask_mig_preflight(
                spec,
                hardware_probe=lambda workspace: calls.append(("hardware", workspace)) or _hardware(),
                python_version_probe=lambda: (3, 12, 11),
                repo_revision_probe=lambda repo: "repo-revision",
                distribution_version_probe=lambda name: {
                    "sam3": "0.1.0",
                    "torch": "2.12.1+cu130",
                }[name],
                hash_probe=_sha,
                runtime_smoke_fn=lambda: {"finite": True, "peak_cuda_bytes": 4096},
            )

            self.assertEqual(result["format"], "ownership-mask-mig-preflight-v1")
            self.assertEqual(result["packet_sha256"], spec.packet_sha256)
            self.assertEqual(result["runtime_smoke"]["finite"], True)
            self.assertEqual(result["artifact_count"], 2)
            self.assertEqual(json.loads(spec.output_path.read_text()), result)

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                run_mask_mig_preflight(
                    spec,
                    hardware_probe=lambda _workspace: _hardware(),
                    python_version_probe=lambda: (3, 12, 11),
                    repo_revision_probe=lambda _repo: "repo-revision",
                    distribution_version_probe=lambda name: {
                        "sam3": "0.1.0",
                        "torch": "2.12.1+cu130",
                    }[name],
                    hash_probe=_sha,
                    runtime_smoke_fn=lambda: {"finite": True},
                )

    def test_96gb_or_maxq_hardware_is_rejected_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            spec = _fixture(Path(raw))
            for bad in (
                RemoteHardwareSnapshot(**{**_hardware().__dict__, "gpu_total_bytes": 96 * GIB}),
                RemoteHardwareSnapshot(
                    **{
                        **_hardware().__dict__,
                        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                    }
                ),
            ):
                hashes = []
                with self.assertRaisesRegex(RuntimeError, "MIG|Max-Q"):
                    run_mask_mig_preflight(
                        spec,
                        hardware_probe=lambda _workspace, value=bad: value,
                        python_version_probe=lambda: (3, 12, 11),
                        repo_revision_probe=lambda _repo: "repo-revision",
                        distribution_version_probe=lambda _name: "unused",
                        hash_probe=lambda path: hashes.append(path) or _sha(path),
                        runtime_smoke_fn=lambda: {"finite": True},
                    )
                self.assertEqual(hashes, [])

    def test_wrong_packet_or_dependency_fails_without_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            spec = _fixture(root)
            spec.packet_manifest_path.write_text("mutated")

            with self.assertRaisesRegex(RuntimeError, "packet SHA-256"):
                run_mask_mig_preflight(
                    spec,
                    hardware_probe=lambda _workspace: _hardware(),
                    python_version_probe=lambda: (3, 12, 11),
                    repo_revision_probe=lambda _repo: "repo-revision",
                    distribution_version_probe=lambda _name: "wrong",
                    hash_probe=_sha,
                    runtime_smoke_fn=lambda: {"finite": True},
                )

            self.assertFalse(spec.output_path.exists())


if __name__ == "__main__":
    unittest.main()
