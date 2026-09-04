from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .decoder_campaign import (
    DecoderFoldRunSpec,
    _build_clip_specs,
    validate_fold_run_spec,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_completed_manifest(output_root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_root / "diagnostic-manifest.json"
    completion_path = output_root / "RUN_COMPLETE"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("completed fold diagnostic metadata is invalid") from error
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("fold diagnostic manifest checksum mismatch")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"fold diagnostic provenance changed: {key}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != manifest.get("frame_count"):
        raise RuntimeError("fold diagnostic record inventory is invalid")
    for record in records:
        for path_key, sha_key in (("image_path", "image_sha256"), ("tensor_path", "tensor_sha256")):
            path = output_root / str(record[path_key])
            if not path.is_file() or _sha256(path) != record[sha_key]:
                raise RuntimeError(f"fold diagnostic artifact checksum mismatch: {path}")
    return manifest


def render_completed_fold_diagnostics(
    spec: DecoderFoldRunSpec,
    output_root: str | Path,
    *,
    panel_size: tuple[int, int] = (480, 320),
) -> dict[str, Any]:
    """Render all held-out predictions for one completed fold, one frame at a time."""

    validate_fold_run_spec(spec)
    fold_root = spec.output_root / spec.run_name
    result_path = fold_root / "result.json"
    completion_path = fold_root / "RUN_COMPLETE"
    if not result_path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"decoder fold is not complete: {spec.run_name}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("decoder fold result metadata is invalid") from error
    result_sha = _sha256(result_path)
    if completion.get("result_sha256") != result_sha:
        raise RuntimeError("decoder fold result checksum mismatch")
    if (
        result.get("run_name") != spec.run_name
        or result.get("heldout_clip") != spec.heldout_clip
        or result.get("spatial_arm") != spec.spatial_arm
        or result.get("semantic_condition") != spec.semantic_condition
        or result.get("language_layer") != spec.language_layer
    ):
        raise RuntimeError("decoder fold result does not match the visualization spec")
    checkpoint_value = result.get("training", {}).get("checkpoint_path")
    checkpoint_sha = result.get("training", {}).get("checkpoint_sha256")
    if not checkpoint_value or not checkpoint_sha:
        raise RuntimeError("decoder fold has no best checkpoint")
    checkpoint_path = Path(str(checkpoint_value))
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("decoder best checkpoint checksum mismatch")

    destination = Path(output_root)
    expected = {
        "format": "ownership-fold-diagnostics-v1",
        "run_name": spec.run_name,
        "heldout_clip": spec.heldout_clip,
        "fold_result_sha256": result_sha,
        "checkpoint_sha256": checkpoint_sha,
    }
    if (destination / "RUN_COMPLETE").exists():
        return _load_completed_manifest(destination, expected)
    destination.mkdir(parents=True, exist_ok=True)

    import torch

    from .checkpoint import load_checkpoint
    from .data import OwnershipDataset, load_rgb_records
    from .experiments import STATIC_ARMS
    from .model import OwnershipDecoder, SemanticOwnershipDecoder
    from .training import _batched_sample, _forward_decoder
    from .visualization import render_ownership_diagnostic

    device = torch.device(spec.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fold visualization requested but CUDA is unavailable")
    arm = STATIC_ARMS[spec.spatial_arm]
    if spec.semantic_condition is None:
        model = OwnershipDecoder(
            input_channels=arm.input_channels,
            width=spec.width,
            residual_blocks=spec.residual_blocks,
        )
    else:
        model = SemanticOwnershipDecoder(
            input_channels=arm.input_channels,
            semantic_dim=5120,
            width=spec.width,
            residual_blocks=spec.residual_blocks,
        )
    state, _ = load_checkpoint(checkpoint_path)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    clip_specs = _build_clip_specs(spec)[spec.heldout_clip]
    dataset = OwnershipDataset(clip_specs, rgb_output_hw=None)
    clip_root = spec.input_root / spec.heldout_clip
    rgb_records = load_rgb_records(
        clip_root / "clip-manifest.json",
        project_root=clip_root,
    )
    records = []
    amp_enabled = spec.use_amp and device.type == "cuda"
    try:
        with torch.inference_mode():
            for index, frame_spec in enumerate(clip_specs):
                frame_index = frame_spec.frame_index
                image_name = f"frame_{frame_index:06d}.png"
                tensor_name = f"frame_{frame_index:06d}.npz"
                image_path = destination / "images" / image_name
                tensor_path = destination / "tensors" / tensor_name
                record_path = destination / "records" / f"frame_{frame_index:06d}.json"
                if record_path.exists():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    if (
                        record.get("frame_index") != frame_index
                        or not image_path.is_file()
                        or not tensor_path.is_file()
                        or _sha256(image_path) != record.get("image_sha256")
                        or _sha256(tensor_path) != record.get("tensor_sha256")
                    ):
                        raise RuntimeError(f"partial diagnostic record is corrupt: {frame_index}")
                    records.append(record)
                    continue
                if image_path.exists() or tensor_path.exists():
                    raise RuntimeError(f"uncommitted diagnostic artifact exists: {frame_index}")
                sample = dataset[index]
                spatial, labels, contact, actor_states = _batched_sample(sample, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = _forward_decoder(
                        model,
                        spatial,
                        actor_states,
                        tuple(labels.shape[-2:]),
                    )
                logits_cpu = logits[0].float().cpu()
                probabilities = torch.softmax(logits_cpu, dim=0).numpy()
                rgb_path, rgb_sha = rgb_records[frame_index]
                if _sha256(rgb_path) != rgb_sha:
                    raise RuntimeError(f"diagnostic RGB checksum mismatch: {frame_index}")
                diagnostic = render_ownership_diagnostic(
                    rgb_path=rgb_path,
                    labels=labels[0].cpu().numpy(),
                    logits=logits_cpu.numpy(),
                    contact=contact[0].cpu().numpy(),
                    output_path=image_path,
                    title=f"{spec.run_name} / {spec.heldout_clip} / frame {frame_index}",
                    panel_size=panel_size,
                )
                _atomic_npz(
                    tensor_path,
                    probabilities=probabilities.astype(np.float16),
                    prediction=probabilities.argmax(axis=0).astype(np.uint8),
                    a2_minus_a1=(probabilities[2] - probabilities[1]).astype(np.float16),
                )
                record = {
                    "frame_index": frame_index,
                    "source_sha256": rgb_sha,
                    "image_path": image_path.relative_to(destination).as_posix(),
                    "image_sha256": diagnostic["output_sha256"],
                    "tensor_path": tensor_path.relative_to(destination).as_posix(),
                    "tensor_sha256": _sha256(tensor_path),
                    "contact_bbox_grid": diagnostic["contact_bbox_grid"],
                    "a2_minus_a1_min": diagnostic["a2_minus_a1_min"],
                    "a2_minus_a1_max": diagnostic["a2_minus_a1_max"],
                }
                _atomic_json(record_path, record)
                records.append(record)
                del sample, spatial, labels, contact, actor_states, logits, logits_cpu
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest = {
        **expected,
        "frame_count": len(records),
        "records": records,
    }
    manifest_path = destination / "diagnostic-manifest.json"
    if manifest_path.exists():
        raise RuntimeError("refusing to overwrite fold diagnostic manifest")
    _atomic_json(manifest_path, manifest)
    _atomic_json(destination / "RUN_COMPLETE", {"manifest_sha256": _sha256(manifest_path)})
    return _load_completed_manifest(destination, expected)
