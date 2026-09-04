from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .tracking import TrackingPlan, load_tracking_plan_config


REQUIRED_BASE_SAM3_ARTIFACTS = {
    "config.json",
    "merges.txt",
    "model.safetensors",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class BreadthPacketContract:
    expected_clip_ids: tuple[str, ...]
    expected_frames_per_clip: int
    sam_repo_revision: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    dependency_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.expected_clip_ids or len(set(self.expected_clip_ids)) != len(
            self.expected_clip_ids
        ):
            raise ValueError("expected clip IDs must be non-empty and unique")
        if self.expected_frames_per_clip < 1 or self.checkpoint_size_bytes < 1:
            raise ValueError("frame count and checkpoint size must be positive")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint SHA-256 must contain 64 characters")
        try:
            int(self.checkpoint_sha256, 16)
        except ValueError as error:
            raise ValueError("checkpoint SHA-256 must be hexadecimal") from error
        if not self.sam_repo_revision.strip() or not self.dependency_versions:
            raise ValueError("SAM revision and dependency versions are required")
        names = [name for name, _ in self.dependency_versions]
        if len(names) != len(set(names)):
            raise ValueError("dependency names must be unique")


@dataclass(frozen=True)
class MaskBreadthPacketContract:
    expected_clip_ids: tuple[str, ...]
    expected_frames_per_clip: int
    sam_repo_revision: str
    sam31_checkpoint_sha256: str
    sam31_checkpoint_size_bytes: int
    sam3_model_revision: str
    container_image_digest: str
    dependency_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.expected_clip_ids or len(set(self.expected_clip_ids)) != len(
            self.expected_clip_ids
        ):
            raise ValueError("expected clip IDs must be non-empty and unique")
        if self.expected_frames_per_clip < 1 or self.sam31_checkpoint_size_bytes < 1:
            raise ValueError("frame and checkpoint sizes must be positive")
        if not self.sam_repo_revision.strip() or not self.sam3_model_revision.strip():
            raise ValueError("both SAM model revisions are required")
        if len(self.sam31_checkpoint_sha256) != 64:
            raise ValueError("SAM3.1 checkpoint SHA-256 must contain 64 characters")
        try:
            int(self.sam31_checkpoint_sha256, 16)
        except ValueError as error:
            raise ValueError("SAM3.1 checkpoint SHA-256 must be hexadecimal") from error
        digest_marker = "@sha256:"
        if digest_marker not in self.container_image_digest:
            raise ValueError("container image must be pinned by SHA-256 digest")
        container_sha = self.container_image_digest.rsplit(digest_marker, 1)[1]
        if len(container_sha) != 64:
            raise ValueError("container image SHA-256 must contain 64 characters")
        try:
            int(container_sha, 16)
        except ValueError as error:
            raise ValueError("container image SHA-256 must be hexadecimal") from error
        if not self.dependency_versions:
            raise ValueError("dependency versions are required")
        names = [name.lower().replace("_", "-") for name, _ in self.dependency_versions]
        if len(names) != len(set(names)):
            raise ValueError("dependency names must be unique")


def _packet_relative(path: Path, root: Path, *, kind: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{kind} must be inside the frozen packet: {path}") from error


def _review_record(
    review_root: Path,
    plan: TrackingPlan,
    *,
    packet_root: Path | None = None,
) -> dict[str, Any]:
    image = review_root / f"{plan.clip_id}.png"
    sidecar_path = image.with_suffix(".png.json")
    if not image.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"seed review artifact is incomplete for {plan.clip_id}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"seed review sidecar is invalid: {sidecar_path}") from error
    if sidecar.get("format") != "ownership-seed-prompt-review-v1":
        raise ValueError(f"seed review format is invalid for {plan.clip_id}")
    if sidecar.get("clip_id") != plan.clip_id:
        raise ValueError(f"seed review clip does not match plan for {plan.clip_id}")
    if sidecar.get("plan_sha256") != plan.sha256:
        raise ValueError(f"seed review plan checksum mismatch for {plan.clip_id}")
    image_sha256 = _sha256(image)
    if sidecar.get("image_sha256") != image_sha256:
        raise ValueError(f"seed review image checksum mismatch for {plan.clip_id}")
    expected_seeds = [seed.frame_index for seed in plan.seeds]
    if sidecar.get("seed_frame_indices") != expected_seeds:
        raise ValueError(f"seed review frame inventory mismatch for {plan.clip_id}")
    image_record = (
        image.name
        if packet_root is None
        else _packet_relative(image, packet_root, kind="seed review image")
    )
    sidecar_record = (
        sidecar_path.name
        if packet_root is None
        else _packet_relative(sidecar_path, packet_root, kind="seed review sidecar")
    )
    return {
        "review_image": image_record,
        "review_image_sha256": image_sha256,
        "review_sidecar": sidecar_record,
        "review_sidecar_sha256": _sha256(sidecar_path),
    }


def build_breadth_launch_manifest(
    contract: BreadthPacketContract,
    *,
    config_paths: Sequence[str | Path],
    input_root: str | Path,
    review_root: str | Path,
    runtime_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the complete geometry-only breadth packet without copying large frames."""

    output = Path(output_path)
    sidecar_output = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar_output.exists():
        raise FileExistsError(f"refusing to overwrite frozen launch packet: {output}")
    root = Path(input_root)
    reviews = Path(review_root)
    plans = sorted(
        [
            (
                Path(path),
                load_tracking_plan_config(path, input_root=root),
            )
            for path in config_paths
        ],
        key=lambda item: item[1].clip_id,
    )
    observed_ids = tuple(plan.clip_id for _, plan in plans)
    expected_ids = tuple(sorted(contract.expected_clip_ids))
    if observed_ids != expected_ids:
        raise ValueError(
            f"breadth clip inventory mismatch: observed={observed_ids}, expected={expected_ids}"
        )
    if any(len(plan.frames) != contract.expected_frames_per_clip for _, plan in plans):
        raise ValueError("breadth clip frame count does not match the packet contract")
    sources = [Path(path) for path in runtime_paths]
    if not sources or any(not path.is_file() for path in sources):
        raise FileNotFoundError("every runtime source path must be an existing file")

    clip_records: list[dict[str, Any]] = []
    for config_path, plan in plans:
        review = _review_record(reviews, plan)
        clip_records.append(
            {
                "clip_id": plan.clip_id,
                "config_name": config_path.name,
                "config_sha256": _sha256(config_path),
                "plan_sha256": plan.sha256,
                "clip_manifest_name": plan.frames[0].path.parent.parent.name
                + "/clip-manifest.json",
                "frame_count": len(plan.frames),
                "frame_inventory_sha256": _canonical_sha256(
                    [frame.sha256 for frame in plan.frames]
                ),
                "seed_frame_indices": [seed.frame_index for seed in plan.seeds],
                "propagation_segments": [asdict(segment) for segment in plan.propagations],
                **review,
            }
        )

    runtime_records = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(sources, key=lambda candidate: str(candidate))
    ]
    manifest = {
        "format": "sam31-breadth-launch-packet-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_contract": "geometry_only_no_semantic_or_appearance_nudges",
        "scientific_purpose": (
            "generate independent temporal SAM3.1 actor masks for breadth pseudo-label "
            "agreement and leave-one-clip-out decoder evaluation"
        ),
        "model": {
            "backend": "sam3.1-tracker-only",
            "sam_repo_revision": contract.sam_repo_revision,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "checkpoint_size_bytes": contract.checkpoint_size_bytes,
        },
        "environment": {
            "python_version_prefix": "3.12",
            "cuda_index": "cu130",
            "dependency_versions": dict(contract.dependency_versions),
            "gpu_required": "RTX PRO 6000 Blackwell 96GB Server/Workstation",
            "gpu_forbidden": "Max-Q",
        },
        "clips": clip_records,
        "totals": {
            "clips": len(clip_records),
            "frames": sum(record["frame_count"] for record in clip_records),
            "seed_frames": sum(len(record["seed_frame_indices"]) for record in clip_records),
        },
        "runtime_files": runtime_records,
        "human_review": {
            "seed_geometry": "pending_user_optional_geometry_check",
            "contact_ownership_labels": "pending",
            "pseudo_labels_usable_for_training": False,
        },
        "termination_contract": {
            "worker_restarts_max": 2,
            "watchdog_poll_seconds": 30,
            "worker_runtime_ceiling_seconds": 28800,
            "vram_restart_fraction": 0.90,
            "replacement_pod_allowed": False,
        },
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(output, encoded)
    _atomic_bytes(sidecar_output, (_sha256(output) + "\n").encode("utf-8"))
    return manifest


def _validated_file_records(
    paths: Sequence[str | Path],
    *,
    kind: str,
    packet_root: Path | None = None,
) -> list[dict[str, Any]]:
    resolved = [Path(path) for path in paths]
    if not resolved or any(not path.is_file() for path in resolved):
        raise FileNotFoundError(f"every {kind} path must be an existing file")
    return [
        {
            "path": (
                str(path)
                if packet_root is None
                else _packet_relative(path, packet_root, kind=kind)
            ),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(resolved, key=lambda candidate: str(candidate))
    ]


def _validated_sam3_artifacts(
    manifest_path: Path,
    *,
    model_directory: Path,
    expected_revision: str,
) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"SAM3 artifact manifest is invalid: {manifest_path}") from error
    if set(payload) != {"format", "model_revision", "artifacts"}:
        raise ValueError("SAM3 artifact manifest keys are invalid")
    if payload["format"] != "sam3-model-artifacts-v1":
        raise ValueError("SAM3 artifact manifest format is unsupported")
    if payload["model_revision"] != expected_revision:
        raise ValueError("SAM3 artifact manifest revision mismatch")
    if not isinstance(payload["artifacts"], list):
        raise ValueError("SAM3 artifact inventory must be a list")
    root = model_directory.resolve()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(payload["artifacts"]):
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"SAM3 artifact {index} schema is invalid")
        relative = Path(str(record["relative_path"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(f"SAM3 artifact path is invalid: {relative}")
        name = relative.as_posix()
        if name in seen:
            raise ValueError(f"duplicate SAM3 artifact path: {name}")
        seen.add(name)
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise FileNotFoundError(f"SAM3 artifact is missing: {name}")
        expected_size = int(record["size_bytes"])
        expected_sha = str(record["sha256"])
        if artifact.stat().st_size != expected_size or _sha256(artifact) != expected_sha:
            raise ValueError(f"SAM3 artifact size or SHA-256 mismatch: {name}")
        records.append(
            {
                "relative_path": name,
                "size_bytes": expected_size,
                "sha256": expected_sha,
            }
        )
    if seen != REQUIRED_BASE_SAM3_ARTIFACTS:
        missing = sorted(REQUIRED_BASE_SAM3_ARTIFACTS - seen)
        unexpected = sorted(seen - REQUIRED_BASE_SAM3_ARTIFACTS)
        raise ValueError(
            f"SAM3 artifact inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    records.sort(key=lambda record: record["relative_path"])
    return records, _sha256(manifest_path)


def build_mask_breadth_launch_manifest(
    contract: MaskBreadthPacketContract,
    *,
    config_paths: Sequence[str | Path],
    input_root: str | Path,
    review_root: str | Path,
    runtime_paths: Sequence[str | Path],
    environment_paths: Sequence[str | Path],
    sam31_checkpoint_path: str | Path,
    sam3_artifact_manifest_path: str | Path,
    sam3_model_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the two-model breadth campaign before any remote compute starts."""

    output = Path(output_path)
    packet_root = output.parent.resolve()
    sidecar_output = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar_output.exists():
        raise FileExistsError(f"refusing to overwrite frozen launch packet: {output}")
    plans = sorted(
        [
            (Path(path), load_tracking_plan_config(path, input_root=input_root))
            for path in config_paths
        ],
        key=lambda item: item[1].clip_id,
    )
    observed_ids = tuple(plan.clip_id for _, plan in plans)
    if observed_ids != tuple(sorted(contract.expected_clip_ids)):
        raise ValueError(
            f"breadth clip inventory mismatch: observed={observed_ids}, "
            f"expected={tuple(sorted(contract.expected_clip_ids))}"
        )
    if any(len(plan.frames) != contract.expected_frames_per_clip for _, plan in plans):
        raise ValueError("breadth clip frame count does not match the packet contract")

    reviews = Path(review_root)
    clip_records = []
    for config_path, plan in plans:
        clip_records.append(
            {
                "clip_id": plan.clip_id,
                "config_path": _packet_relative(
                    config_path, packet_root, kind="tracking config"
                ),
                "config_sha256": _sha256(config_path),
                "plan_sha256": plan.sha256,
                "frame_count": len(plan.frames),
                "frame_inventory_sha256": _canonical_sha256(
                    [frame.sha256 for frame in plan.frames]
                ),
                "seed_frame_indices": [seed.frame_index for seed in plan.seeds],
                "propagation_segments": [asdict(value) for value in plan.propagations],
                **_review_record(reviews, plan, packet_root=packet_root),
            }
        )

    sam31_checkpoint = Path(sam31_checkpoint_path)
    if not sam31_checkpoint.is_file():
        raise FileNotFoundError(f"SAM3.1 checkpoint is missing: {sam31_checkpoint}")
    if (
        sam31_checkpoint.stat().st_size != contract.sam31_checkpoint_size_bytes
        or _sha256(sam31_checkpoint) != contract.sam31_checkpoint_sha256
    ):
        raise ValueError("SAM3.1 checkpoint size or SHA-256 mismatch")

    artifact_manifest = Path(sam3_artifact_manifest_path)
    artifact_manifest_record = _packet_relative(
        artifact_manifest, packet_root, kind="SAM3 artifact manifest"
    )
    sam3_artifacts, artifact_manifest_sha = _validated_sam3_artifacts(
        artifact_manifest,
        model_directory=Path(sam3_model_directory),
        expected_revision=contract.sam3_model_revision,
    )
    runtime_records = _validated_file_records(
        runtime_paths, kind="runtime source", packet_root=packet_root
    )
    environment_records = _validated_file_records(
        environment_paths, kind="environment", packet_root=packet_root
    )
    manifest = {
        "format": "sam31-sam3-breadth-launch-packet-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "container_image": contract.container_image_digest,
        "prompt_contract": "geometry_only_no_semantic_or_appearance_nudges",
        "execution_order": ["sam3.1_tracker", "sam3_image"],
        "resident_model_limit": 1,
        "models": {
            "sam3.1_tracker": {
                "official_repository_revision": contract.sam_repo_revision,
                "checkpoint_sha256": contract.sam31_checkpoint_sha256,
                "checkpoint_size_bytes": contract.sam31_checkpoint_size_bytes,
            },
            "base_sam3": {
                "revision": contract.sam3_model_revision,
                "artifact_manifest_path": artifact_manifest_record,
                "artifact_manifest_sha256": artifact_manifest_sha,
                "artifact_count": len(sam3_artifacts),
                "artifacts": sam3_artifacts,
            },
        },
        "environment": {
            "python_version_prefix": "3.12",
            "cuda_index": "cu130",
            "dependency_versions": dict(contract.dependency_versions),
            "gpu_required": "RTX PRO 6000 Blackwell 96GB Server/Workstation",
            "gpu_forbidden": "Max-Q",
            "flash_attention": False,
            "torch_compile": False,
        },
        "clips": clip_records,
        "totals": {
            "clips": len(clip_records),
            "frames": sum(record["frame_count"] for record in clip_records),
            "seed_frames": sum(len(record["seed_frame_indices"]) for record in clip_records),
        },
        "runtime_files": runtime_records,
        "environment_files": environment_records,
        "scientific_design": {
            "purpose": (
                "create a 96-frame breadth ownership set for held-out-clip decoder "
                "evaluation"
            ),
            "agreement_definition": "exclusive SAM3.1 tracker masks intersect base-SAM3 image masks",
            "independence_limit": (
                "base-SAM3 contours are independently decoded but localization remains "
                "correlated through SAM3.1-derived geometry prompts"
            ),
        },
        "human_review": {
            "seed_geometry": "pending_user_optional_geometry_check",
            "contact_ownership_labels": "pending",
            "pseudo_labels_usable_for_training": False,
        },
        "termination_contract": {
            "worker_restarts_max": 2,
            "watchdog_poll_seconds": 30,
            "worker_runtime_ceiling_seconds": 28800,
            "vram_restart_fraction": 0.90,
            "replacement_pod_allowed": False,
        },
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(output, encoded)
    _atomic_bytes(sidecar_output, (_sha256(output) + "\n").encode("utf-8"))
    return manifest


def _safe_packet_child(root: Path, relative_value: Any, *, context: str) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} path is unsafe: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{context} path escapes the packet: {relative}")
    return resolved


def _verify_packet_file_record(
    packet_root: Path,
    record: Any,
    *,
    context: str,
) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"{context} record schema is invalid")
    path = _safe_packet_child(packet_root, record["path"], context=context)
    if not path.is_file():
        raise FileNotFoundError(f"{context} file is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]) or _sha256(path) != record["sha256"]:
        raise ValueError(f"{context} size or SHA-256 mismatch: {path}")
    return path


def verify_mask_breadth_packet(
    packet_root: str | Path,
    *,
    input_root: str | Path,
) -> dict[str, Any]:
    """Recompute every portable packet/input binding before either Pod does work."""

    root = Path(packet_root).resolve()
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.json.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("mask packet manifest or checksum sidecar is missing")
    expected_manifest_sha = sidecar_path.read_text(encoding="utf-8").strip()
    observed_manifest_sha = _sha256(manifest_path)
    if expected_manifest_sha != observed_manifest_sha:
        raise ValueError("mask packet manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("mask packet manifest is unreadable") from error
    if manifest.get("format") != "sam31-sam3-breadth-launch-packet-v2":
        raise ValueError("mask packet format is invalid")

    verified_files = 0
    for field, context in (
        ("runtime_files", "runtime source"),
        ("environment_files", "environment file"),
    ):
        records = manifest.get(field)
        if not isinstance(records, list) or not records:
            raise ValueError(f"mask packet {field} inventory is invalid")
        seen_paths: set[str] = set()
        for record in records:
            path = _verify_packet_file_record(root, record, context=context)
            identity = str(path)
            if identity in seen_paths:
                raise ValueError(f"duplicate {context}: {path}")
            seen_paths.add(identity)
            verified_files += 1

    try:
        base_model = manifest["models"]["base_sam3"]
        artifact_manifest_relative = base_model["artifact_manifest_path"]
        artifact_manifest_sha = base_model["artifact_manifest_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("base SAM3 packet contract is incomplete") from error
    artifact_manifest = _safe_packet_child(
        root, artifact_manifest_relative, context="SAM3 artifact manifest"
    )
    if not artifact_manifest.is_file() or _sha256(artifact_manifest) != artifact_manifest_sha:
        raise ValueError("SAM3 artifact manifest SHA-256 mismatch")
    try:
        artifact_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("SAM3 artifact manifest is unreadable") from error
    if (
        artifact_payload.get("model_revision") != base_model.get("revision")
        or artifact_payload.get("artifacts") != base_model.get("artifacts")
    ):
        raise ValueError("SAM3 artifact manifest content mismatch")
    verified_files += 1

    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("mask packet clip inventory is invalid")
    observed_clip_ids = []
    total_frames = 0
    total_seeds = 0
    for record in clips:
        if not isinstance(record, dict):
            raise ValueError("mask packet clip record is invalid")
        config = _safe_packet_child(root, record.get("config_path"), context="tracking config")
        if not config.is_file() or _sha256(config) != record.get("config_sha256"):
            raise ValueError(f"tracking config SHA-256 mismatch: {config}")
        plan = load_tracking_plan_config(config, input_root=input_root)
        if plan.clip_id != record.get("clip_id") or plan.sha256 != record.get("plan_sha256"):
            raise ValueError(f"tracking plan mismatch: {plan.clip_id}")
        if len(plan.frames) != record.get("frame_count"):
            raise ValueError(f"tracking frame count mismatch: {plan.clip_id}")
        inventory_sha = _canonical_sha256([frame.sha256 for frame in plan.frames])
        if inventory_sha != record.get("frame_inventory_sha256"):
            raise ValueError(f"tracking frame inventory mismatch: {plan.clip_id}")
        if [seed.frame_index for seed in plan.seeds] != record.get("seed_frame_indices"):
            raise ValueError(f"tracking seed inventory mismatch: {plan.clip_id}")
        review_image = _safe_packet_child(
            root, record.get("review_image"), context="seed review image"
        )
        review_sidecar = _safe_packet_child(
            root, record.get("review_sidecar"), context="seed review sidecar"
        )
        if not review_image.is_file() or _sha256(review_image) != record.get(
            "review_image_sha256"
        ):
            raise ValueError(f"seed review image mismatch: {plan.clip_id}")
        if not review_sidecar.is_file() or _sha256(review_sidecar) != record.get(
            "review_sidecar_sha256"
        ):
            raise ValueError(f"seed review sidecar mismatch: {plan.clip_id}")
        try:
            review = json.loads(review_sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"seed review sidecar is invalid: {plan.clip_id}") from error
        if (
            review.get("clip_id") != plan.clip_id
            or review.get("plan_sha256") != plan.sha256
            or review.get("image_sha256") != record.get("review_image_sha256")
        ):
            raise ValueError(f"seed review lineage mismatch: {plan.clip_id}")
        observed_clip_ids.append(plan.clip_id)
        total_frames += len(plan.frames)
        total_seeds += len(plan.seeds)
        verified_files += 3

    if len(observed_clip_ids) != len(set(observed_clip_ids)):
        raise ValueError("mask packet contains duplicate clip IDs")
    expected_totals = {
        "clips": len(observed_clip_ids),
        "frames": total_frames,
        "seed_frames": total_seeds,
    }
    if manifest.get("totals") != expected_totals:
        raise ValueError("mask packet totals mismatch")
    return {
        "format": "ownership-mask-packet-verification-v1",
        "packet_sha256": observed_manifest_sha,
        "verified_files": verified_files,
        "verified_clips": len(observed_clip_ids),
        "verified_frames": total_frames,
    }
