from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .remote_preflight import (
    RemoteHardwareSnapshot,
    RequiredArtifact,
    probe_distribution_version,
    probe_python_version,
    probe_remote_hardware,
    probe_repo_revision,
    sha256_file,
)


GIB = 1024**3


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class MaskMigPreflightSpec:
    packet_manifest_path: Path
    packet_sha256: str
    sam_repo_path: Path
    sam_repo_revision: str
    sam31_checkpoint_path: Path
    sam31_checkpoint_sha256: str
    sam31_checkpoint_size_bytes: int
    additional_artifacts: tuple[RequiredArtifact, ...]
    workspace_path: Path
    required_distribution_versions: tuple[tuple[str, str], ...]
    output_path: Path

    def __post_init__(self) -> None:
        for value, label in (
            (self.packet_sha256, "packet"),
            (self.sam31_checkpoint_sha256, "SAM3.1 checkpoint"),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} SHA-256 must contain 64 characters")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{label} SHA-256 must be hexadecimal") from error
        if not self.sam_repo_revision.strip() or not self.additional_artifacts:
            raise ValueError("SAM revision and base-model artifacts are required")
        if self.sam31_checkpoint_size_bytes < 1:
            raise ValueError("SAM3.1 checkpoint size must be positive")
        names = [name.lower().replace("_", "-") for name, _ in self.required_distribution_versions]
        if not names or len(names) != len(set(names)):
            raise ValueError("MIG dependency names must be non-empty and unique")


def _validate_mig_hardware(snapshot: RemoteHardwareSnapshot) -> None:
    if "Max-Q" in snapshot.gpu_name:
        raise RuntimeError(f"Max-Q hardware is prohibited: {snapshot.gpu_name}")
    if "RTX PRO 6000 Blackwell" not in snapshot.gpu_name:
        raise RuntimeError(f"MIG preflight requires RTX PRO 6000 Blackwell: {snapshot.gpu_name}")
    if not 20 * GIB <= snapshot.gpu_total_bytes <= 30 * GIB:
        raise RuntimeError(
            "MIG preflight requires the 24 GB slice; "
            f"observed {snapshot.gpu_total_bytes} bytes"
        )
    if snapshot.gpu_free_bytes < 16 * GIB:
        raise RuntimeError("MIG preflight has insufficient free VRAM")
    if snapshot.compute_capability < (12, 0):
        raise RuntimeError("MIG preflight compute capability is below SM120")
    if snapshot.driver_version < (580, 65, 6):
        raise RuntimeError("MIG preflight NVIDIA driver is too old")
    if snapshot.host_available_bytes < 16 * GIB:
        raise RuntimeError("MIG preflight host has insufficient available RAM")
    if snapshot.workspace_free_bytes < 200 * GIB:
        raise RuntimeError("MIG preflight must leave at least 200 GiB on the volume")


def default_blackwell_runtime_smoke() -> dict[str, Any]:
    """Import exact APIs and execute a tiny BF16 SDPA kernel, never model weights."""

    import gc
    import importlib.metadata
    import importlib.util

    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    from sam3.model_builder import build_sam3_multiplex_video_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in MIG runtime smoke")
    if torch.cuda.get_device_capability(0) < (12, 0):
        raise RuntimeError("runtime Torch does not observe SM120")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    query = torch.randn((1, 2, 32, 64), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        output = functional.scaled_dot_product_attention(query, query, query)
    finite = bool(torch.isfinite(output).all().item())
    peak = int(torch.cuda.max_memory_allocated())
    del output, query
    gc.collect()
    torch.cuda.empty_cache()
    if not finite or peak > 512 * 1024**2:
        raise RuntimeError(
            f"bounded Blackwell BF16 SDPA smoke failed: finite={finite}, peak={peak}"
        )
    return {
        "finite": finite,
        "peak_cuda_bytes": peak,
        "torch_cuda": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "sam3_version": importlib.metadata.version("sam3"),
        "sam3_tracker_model_api": Sam3TrackerModel.__name__,
        "sam3_tracker_processor_api": Sam3TrackerProcessor.__name__,
        "sam31_builder_api": build_sam3_multiplex_video_model.__name__,
        "flash_attention_distribution_present": bool(
            importlib.util.find_spec("flash_attn")
        ),
    }


def run_mask_mig_preflight(
    spec: MaskMigPreflightSpec,
    *,
    hardware_probe: Callable[[Path], RemoteHardwareSnapshot] = probe_remote_hardware,
    python_version_probe: Callable[[], tuple[int, ...]] = probe_python_version,
    repo_revision_probe: Callable[[Path], str] = probe_repo_revision,
    distribution_version_probe: Callable[[str], str] = probe_distribution_version,
    hash_probe: Callable[[Path], str] = sha256_file,
    runtime_smoke_fn: Callable[[], dict[str, Any]] = default_blackwell_runtime_smoke,
) -> dict[str, Any]:
    """Prove storage, package and Blackwell compatibility without loading a model."""

    if spec.output_path.exists():
        raise FileExistsError(f"refusing to overwrite MIG sentinel: {spec.output_path}")
    hardware = hardware_probe(spec.workspace_path)
    _validate_mig_hardware(hardware)
    if tuple(python_version_probe()[:2]) != (3, 12):
        raise RuntimeError("MIG preflight requires Python 3.12")
    if not spec.packet_manifest_path.is_file() or hash_probe(
        spec.packet_manifest_path
    ) != spec.packet_sha256:
        raise RuntimeError("MIG packet SHA-256 mismatch")
    if not spec.sam_repo_path.is_dir() or repo_revision_probe(
        spec.sam_repo_path
    ) != spec.sam_repo_revision:
        raise RuntimeError("MIG SAM repository revision mismatch")
    for name, expected in spec.required_distribution_versions:
        observed = distribution_version_probe(name)
        if observed != expected:
            raise RuntimeError(
                f"MIG dependency mismatch for {name}: observed={observed}, expected={expected}"
            )
    smoke = runtime_smoke_fn()
    if smoke.get("finite") is not True:
        raise RuntimeError("MIG runtime smoke did not prove finite BF16 output")

    artifacts = (
        RequiredArtifact(
            path=spec.sam31_checkpoint_path,
            sha256=spec.sam31_checkpoint_sha256,
            size_bytes=spec.sam31_checkpoint_size_bytes,
        ),
        *spec.additional_artifacts,
    )
    records = []
    for artifact in artifacts:
        if not artifact.path.is_file():
            raise RuntimeError(f"MIG model artifact is missing: {artifact.path}")
        if artifact.path.stat().st_size != artifact.size_bytes:
            raise RuntimeError(f"MIG model artifact size mismatch: {artifact.path}")
        digest = hash_probe(artifact.path)
        if digest != artifact.sha256:
            raise RuntimeError(f"MIG model artifact SHA-256 mismatch: {artifact.path}")
        records.append(
            {
                "path": str(artifact.path.resolve()),
                "size_bytes": artifact.size_bytes,
                "sha256": digest,
            }
        )
    result = {
        "format": "ownership-mask-mig-preflight-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "packet_sha256": spec.packet_sha256,
        "sam_repo_revision": spec.sam_repo_revision,
        "hardware": json.loads(json.dumps(hardware.as_dict())),
        "python_version": list(python_version_probe()),
        "dependency_versions": dict(spec.required_distribution_versions),
        "runtime_smoke": smoke,
        "artifact_count": len(records),
        "artifacts": records,
        "model_weights_loaded": False,
    }
    _atomic_json(spec.output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    from .remote_mask_cli import (
        DEFAULT_DISTRIBUTION_PINS,
        _load_artifact_manifest,
        _parse_distribution_pins,
    )

    parser = argparse.ArgumentParser(
        description="Run the model-free 24 GB Blackwell MIG mask campaign gate"
    )
    parser.add_argument("--packet-manifest", required=True, type=Path)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--sam-repo", required=True, type=Path)
    parser.add_argument("--sam-revision", required=True)
    parser.add_argument("--sam31-checkpoint", required=True, type=Path)
    parser.add_argument("--sam31-checkpoint-sha256", required=True)
    parser.add_argument("--sam31-checkpoint-size-bytes", required=True, type=int)
    parser.add_argument("--sam3-model-directory", required=True, type=Path)
    parser.add_argument("--sam3-artifact-manifest", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--require-distribution",
        action="append",
        default=list(DEFAULT_DISTRIBUTION_PINS),
    )
    args = parser.parse_args(argv)
    _, artifacts = _load_artifact_manifest(
        args.sam3_artifact_manifest,
        model_directory=args.sam3_model_directory,
    )
    result = run_mask_mig_preflight(
        MaskMigPreflightSpec(
            packet_manifest_path=args.packet_manifest,
            packet_sha256=args.packet_sha256,
            sam_repo_path=args.sam_repo,
            sam_repo_revision=args.sam_revision,
            sam31_checkpoint_path=args.sam31_checkpoint,
            sam31_checkpoint_sha256=args.sam31_checkpoint_sha256,
            sam31_checkpoint_size_bytes=args.sam31_checkpoint_size_bytes,
            additional_artifacts=artifacts,
            workspace_path=args.workspace,
            required_distribution_versions=_parse_distribution_pins(
                args.require_distribution
            ),
            output_path=args.output,
        )
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "format": result["format"],
                "artifact_count": result["artifact_count"],
            },
            sort_keys=True,
        )
    )
    return 0
