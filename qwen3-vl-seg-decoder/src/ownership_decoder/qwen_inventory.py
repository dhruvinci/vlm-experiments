from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .experiments import STATIC_ARMS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _breadth_relative(value: object) -> str | None:
    normalized = str(value).replace("\\", "/")
    marker = "breadth/"
    if marker not in normalized:
        return None
    return normalized.split(marker, 1)[1]


def verify_qwen_breadth_cache(
    download_manifest_path: str | Path,
    *,
    qwen_breadth_root: str | Path,
    frame_indices_by_clip: Mapping[str, Sequence[int]],
    spatial_arms: Sequence[str],
    semantic_conditions: Sequence[str],
    semantic_context: str = "4fps",
    thinking_mode: str = "off",
    rehash: bool = True,
) -> dict[str, object]:
    """Verify exactly the Qwen artifacts consumed by the decoder campaign."""

    if not frame_indices_by_clip:
        raise ValueError("Qwen cache verification requires at least one clip")
    if not spatial_arms:
        raise ValueError("Qwen cache verification requires at least one spatial arm")
    unknown_arms = sorted(set(spatial_arms) - set(STATIC_ARMS))
    if unknown_arms:
        raise ValueError(f"unknown Qwen spatial arms: {unknown_arms}")
    normalized_frames: dict[str, tuple[int, ...]] = {}
    for clip_id, raw_indices in frame_indices_by_clip.items():
        indices = tuple(int(value) for value in raw_indices)
        if not clip_id or not indices or len(indices) != len(set(indices)) or any(
            value < 0 for value in indices
        ):
            raise ValueError(f"invalid Qwen frame inventory for clip: {clip_id}")
        normalized_frames[str(clip_id)] = tuple(sorted(indices))

    expected: set[str] = set()
    for clip_id, indices in normalized_frames.items():
        for arm_name in spatial_arms:
            arm = STATIC_ARMS[arm_name]
            for frame_index in indices:
                filename = f"frame_{frame_index:06d}.safetensors"
                expected.update(
                    f"{clip_id}/spatial/full/layer_{layer:02d}/{filename}"
                    for layer in arm.full_layers
                )
                expected.update(
                    f"{clip_id}/spatial/pooled/layer_{layer:02d}/{filename}"
                    for layer in arm.pooled_layers
                )
                if arm.include_merged:
                    expected.add(f"{clip_id}/merged-vision/{filename}")
        for condition in semantic_conditions:
            for actor in ("A1", "A2"):
                expected.add(
                    f"{clip_id}/semantic/video/{semantic_context}/{condition}/"
                    f"{thinking_mode}/{actor}.safetensors"
                )
    manifest_path = Path(download_manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Qwen download manifest is invalid: {manifest_path}") from error
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Qwen download manifest records must be a list")
    by_relative: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Qwen download record must be an object")
        relative = _breadth_relative(record.get("path"))
        if relative is None:
            continue
        if relative in by_relative:
            raise ValueError(f"duplicate Qwen breadth download record: {relative}")
        by_relative[relative] = record
    missing = sorted(expected - set(by_relative))
    if missing:
        raise ValueError(f"Qwen download manifest is missing required artifacts: {missing[:5]}")

    root = Path(qwen_breadth_root).resolve()
    verified_records = []
    verified_bytes = 0
    for relative in sorted(expected):
        record = by_relative[relative]
        if record.get("verified") is not True:
            raise ValueError(f"Qwen download record was not verified: {relative}")
        expected_size = int(record.get("size_bytes", -1))
        expected_sha = str(record.get("sha256", ""))
        if expected_size < 1 or len(expected_sha) != 64:
            raise ValueError(f"Qwen download record contract is invalid: {relative}")
        try:
            int(expected_sha, 16)
        except ValueError as error:
            raise ValueError(f"Qwen download SHA-256 is malformed: {relative}") from error
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise FileNotFoundError(f"Qwen cache artifact is missing: {relative}")
        if artifact.stat().st_size != expected_size:
            raise ValueError(f"Qwen cache size mismatch: {relative}")
        if rehash and _sha256(artifact) != expected_sha:
            raise ValueError(f"Qwen cache SHA-256 mismatch: {relative}")
        verified_records.append(
            {"path": relative, "size_bytes": expected_size, "sha256": expected_sha}
        )
        verified_bytes += expected_size
    return {
        "format": "ownership-qwen-breadth-cache-verification-v1",
        "download_manifest_sha256": _sha256(manifest_path),
        "rehash_performed": bool(rehash),
        "clip_count": len(normalized_frames),
        "frame_count": sum(len(values) for values in normalized_frames.values()),
        "verified_artifact_count": len(verified_records),
        "verified_bytes": verified_bytes,
        "inventory_sha256": _canonical_sha256(verified_records),
    }
