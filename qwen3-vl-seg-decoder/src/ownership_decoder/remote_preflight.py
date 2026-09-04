from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


GIB = 1024**3


class RemoteRuntimePreflightError(RuntimeError):
    """Raised before model import when the remote runtime is not approved."""


@dataclass(frozen=True)
class RemoteHardwareSnapshot:
    gpu_name: str
    gpu_total_bytes: int
    gpu_free_bytes: int
    compute_capability: tuple[int, int]
    driver_version: tuple[int, ...]
    host_available_bytes: int
    workspace_free_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RequiredArtifact:
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError("required artifact SHA-256 must contain 64 characters")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("required artifact SHA-256 must be hexadecimal") from error
        if self.size_bytes < 1:
            raise ValueError("required artifact size must be positive")


@dataclass(frozen=True)
class RemoteRuntimeContract:
    sam_repo_path: Path
    sam_repo_revision: str
    checkpoint_path: Path
    checkpoint_sha256: str
    workspace_path: Path
    required_distribution_versions: tuple[tuple[str, str], ...]
    additional_artifacts: tuple[RequiredArtifact, ...] = ()
    required_gpu_name_fragment: str = "RTX PRO 6000 Blackwell"
    forbidden_gpu_name_fragments: tuple[str, ...] = ("Max-Q",)
    min_gpu_total_bytes: int = 90 * GIB
    min_gpu_free_bytes: int = 80 * GIB
    min_compute_capability: tuple[int, int] = (12, 0)
    min_driver_version: tuple[int, ...] = (580, 65, 6)
    min_host_available_bytes: int = 32 * GIB
    min_workspace_free_bytes: int = 50 * GIB
    required_python_version: tuple[int, ...] = (3, 12)

    def __post_init__(self) -> None:
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must contain 64 hexadecimal characters")
        try:
            int(self.checkpoint_sha256, 16)
        except ValueError as error:
            raise ValueError("checkpoint_sha256 must be hexadecimal") from error
        if not self.sam_repo_revision.strip():
            raise ValueError("sam_repo_revision cannot be empty")
        if not self.required_python_version or any(
            value < 0 for value in self.required_python_version
        ):
            raise ValueError("required_python_version must be a non-negative prefix")
        names = [name for name, _ in self.required_distribution_versions]
        if not names or len(names) != len(set(names)):
            raise ValueError("required distribution names must be non-empty and unique")
        artifact_paths = [str(artifact.path.resolve()) for artifact in self.additional_artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("required artifact paths must be unique")
        byte_limits = (
            self.min_gpu_total_bytes,
            self.min_gpu_free_bytes,
            self.min_host_available_bytes,
            self.min_workspace_free_bytes,
        )
        if any(value < 1 for value in byte_limits):
            raise ValueError("remote runtime byte limits must be positive")

    def as_dict(self) -> dict:
        value = asdict(self)
        for field in ("sam_repo_path", "checkpoint_path", "workspace_path"):
            value[field] = str(value[field].resolve())
        value["additional_artifacts"] = [
            {
                "path": str(artifact.path.resolve()),
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in self.additional_artifacts
        ]
        return value

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RemoteRuntimeApproval:
    contract_sha256: str
    checkpoint_sha256: str
    hardware: RemoteHardwareSnapshot
    additional_artifact_sha256: tuple[tuple[str, str], ...] = ()

    def require_contract(self, contract: RemoteRuntimeContract) -> None:
        if self.contract_sha256 != contract.sha256:
            raise RemoteRuntimePreflightError(
                "runtime approval does not match the requested contract"
            )
        expected_artifacts = tuple(
            (str(artifact.path.resolve()), artifact.sha256)
            for artifact in contract.additional_artifacts
        )
        if self.additional_artifact_sha256 != expected_artifacts:
            raise RemoteRuntimePreflightError(
                "runtime approval does not contain every contract-bound artifact"
            )


def _parse_version(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(piece) for piece in value.strip().split("."))
    except ValueError as error:
        raise RemoteRuntimePreflightError(f"invalid numeric version: {value!r}") from error


def _read_mem_available() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RemoteRuntimePreflightError("MemAvailable is absent from /proc/meminfo")


def probe_remote_hardware(workspace_path: Path) -> RemoteHardwareSnapshot:
    """Inspect hardware without importing Torch or any SAM module."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RemoteRuntimePreflightError("nvidia-smi hardware probe failed") from error
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RemoteRuntimePreflightError(
            f"exactly one GPU is required; nvidia-smi returned {len(lines)}"
        )
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 5:
        raise RemoteRuntimePreflightError("nvidia-smi returned an unexpected schema")
    name, total_mib, free_mib, driver, capability = fields
    mib = 1024**2
    try:
        compute_capability = _parse_version(capability)
        if len(compute_capability) != 2:
            raise ValueError
        gpu_total_bytes = int(total_mib) * mib
        gpu_free_bytes = int(free_mib) * mib
    except (ValueError, RemoteRuntimePreflightError) as error:
        raise RemoteRuntimePreflightError("nvidia-smi returned invalid GPU values") from error
    return RemoteHardwareSnapshot(
        gpu_name=name,
        gpu_total_bytes=gpu_total_bytes,
        gpu_free_bytes=gpu_free_bytes,
        compute_capability=(compute_capability[0], compute_capability[1]),
        driver_version=_parse_version(driver),
        host_available_bytes=_read_mem_available(),
        workspace_free_bytes=shutil.disk_usage(workspace_path).free,
    )


def probe_repo_revision(repo_path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RemoteRuntimePreflightError("SAM repository revision probe failed") from error
    return completed.stdout.strip()


def probe_distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RemoteRuntimePreflightError(
            f"required distribution is not installed: {name}"
        ) from error


def probe_python_version() -> tuple[int, ...]:
    return tuple(sys.version_info[:3])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_hardware(
    snapshot: RemoteHardwareSnapshot, contract: RemoteRuntimeContract
) -> None:
    if contract.required_gpu_name_fragment not in snapshot.gpu_name:
        raise RemoteRuntimePreflightError(
            f"GPU model must contain {contract.required_gpu_name_fragment!r}; "
            f"observed {snapshot.gpu_name!r}"
        )
    if any(fragment in snapshot.gpu_name for fragment in contract.forbidden_gpu_name_fragments):
        raise RemoteRuntimePreflightError(
            f"GPU model is forbidden by contract: {snapshot.gpu_name!r}"
        )
    checks = (
        (snapshot.gpu_total_bytes, contract.min_gpu_total_bytes, "96 GB GPU capacity"),
        (snapshot.gpu_free_bytes, contract.min_gpu_free_bytes, "free VRAM"),
        (snapshot.host_available_bytes, contract.min_host_available_bytes, "host RAM"),
        (snapshot.workspace_free_bytes, contract.min_workspace_free_bytes, "workspace disk"),
    )
    for observed, minimum, label in checks:
        if observed < minimum:
            raise RemoteRuntimePreflightError(
                f"insufficient {label}: observed={observed}, required>={minimum}"
            )
    if snapshot.compute_capability < contract.min_compute_capability:
        raise RemoteRuntimePreflightError(
            "compute capability is too old: "
            f"observed={snapshot.compute_capability}, "
            f"required>={contract.min_compute_capability}"
        )
    if snapshot.driver_version < contract.min_driver_version:
        raise RemoteRuntimePreflightError(
            f"driver is too old: observed={snapshot.driver_version}, "
            f"required>={contract.min_driver_version}"
        )


def perform_remote_preflight(
    contract: RemoteRuntimeContract,
    *,
    hardware_probe: Callable[[Path], RemoteHardwareSnapshot] = probe_remote_hardware,
    python_version_probe: Callable[[], tuple[int, ...]] = probe_python_version,
    repo_revision_probe: Callable[[Path], str] = probe_repo_revision,
    distribution_version_probe: Callable[[str], str] = probe_distribution_version,
    checkpoint_hash_probe: Callable[[Path], str] = sha256_file,
) -> RemoteRuntimeApproval:
    """Approve a remote model load only after cheap hardware gates and exact hashes."""

    hardware = hardware_probe(contract.workspace_path)
    _validate_hardware(hardware, contract)

    python_version = python_version_probe()
    required_python = contract.required_python_version
    if tuple(python_version[: len(required_python)]) != required_python:
        raise RemoteRuntimePreflightError(
            f"Python version mismatch: observed={python_version}, "
            f"required prefix={required_python}"
        )

    if not contract.sam_repo_path.is_dir():
        raise RemoteRuntimePreflightError(
            f"SAM repository is missing: {contract.sam_repo_path}"
        )
    if not contract.workspace_path.is_dir():
        raise RemoteRuntimePreflightError(
            f"workspace is missing: {contract.workspace_path}"
        )
    if not contract.checkpoint_path.is_file():
        raise RemoteRuntimePreflightError(
            f"SAM3.1 checkpoint is missing: {contract.checkpoint_path}"
        )
    for artifact in contract.additional_artifacts:
        if not artifact.path.is_file():
            raise RemoteRuntimePreflightError(
                f"required model artifact is missing: {artifact.path}"
            )
        observed_size = artifact.path.stat().st_size
        if observed_size != artifact.size_bytes:
            raise RemoteRuntimePreflightError(
                f"required model artifact size mismatch for {artifact.path}: "
                f"observed={observed_size}, expected={artifact.size_bytes}"
            )

    revision = repo_revision_probe(contract.sam_repo_path)
    if revision != contract.sam_repo_revision:
        raise RemoteRuntimePreflightError(
            f"SAM repository revision mismatch: observed={revision!r}, "
            f"expected={contract.sam_repo_revision!r}"
        )
    for name, expected in contract.required_distribution_versions:
        observed = distribution_version_probe(name)
        if observed != expected:
            raise RemoteRuntimePreflightError(
                f"distribution version mismatch for {name}: "
                f"observed={observed!r}, expected={expected!r}"
            )

    checkpoint_sha256 = checkpoint_hash_probe(contract.checkpoint_path)
    if checkpoint_sha256 != contract.checkpoint_sha256:
        raise RemoteRuntimePreflightError(
            "checkpoint SHA-256 mismatch: "
            f"observed={checkpoint_sha256}, expected={contract.checkpoint_sha256}"
        )
    additional_hashes: list[tuple[str, str]] = []
    for artifact in contract.additional_artifacts:
        observed = checkpoint_hash_probe(artifact.path)
        if observed != artifact.sha256:
            raise RemoteRuntimePreflightError(
                f"required model artifact SHA-256 mismatch for {artifact.path}: "
                f"observed={observed}, expected={artifact.sha256}"
            )
        additional_hashes.append((str(artifact.path.resolve()), observed))
    return RemoteRuntimeApproval(
        contract_sha256=contract.sha256,
        checkpoint_sha256=checkpoint_sha256,
        hardware=hardware,
        additional_artifact_sha256=tuple(additional_hashes),
    )
