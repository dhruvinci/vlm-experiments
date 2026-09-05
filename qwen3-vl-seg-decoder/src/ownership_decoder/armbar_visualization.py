from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .armbar_exploratory import (
    ArmbarJobSpec,
    _build_job_datasets,
    _fixed_substitution_pairs,
    validate_armbar_job_spec,
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


def _load_completed(destination: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest_path = destination / "diagnostic-manifest.json"
    completion_path = destination / "RUN_COMPLETE"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("armbar diagnostic is incomplete or invalid") from error
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("armbar diagnostic manifest checksum mismatch")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"armbar diagnostic provenance changed: {key}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != manifest.get("contact_frame_count"):
        raise RuntimeError("armbar diagnostic record inventory is inconsistent")
    for record in records:
        for path_key, sha_key in (
            ("image_path", "image_sha256"),
            ("tensor_path", "tensor_sha256"),
        ):
            path = destination / str(record[path_key])
            if not path.is_file() or _sha256(path) != record.get(sha_key):
                raise RuntimeError(f"armbar diagnostic artifact checksum mismatch: {path}")
    return manifest


def render_armbar_contact_diagnostics(
    spec: ArmbarJobSpec,
    output_root: str | Path,
    *,
    substitution: str | None = None,
    panel_size: tuple[int, int] = (360, 640),
) -> dict[str, Any]:
    """Render held-out contact frames for one completed armbar checkpoint."""

    validate_armbar_job_spec(spec)
    if spec.split != "final":
        raise ValueError("armbar contact diagnostics require the final held-out split")
    job_root = spec.output_root / spec.run_name
    result_path = job_root / "result.json"
    completion_path = job_root / "RUN_COMPLETE"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"armbar job is incomplete: {spec.run_name}") from error
    result_sha = _sha256(result_path)
    if completion.get("result_sha256") != result_sha:
        raise RuntimeError("armbar job result checksum mismatch")
    if (
        result.get("run_name") != spec.run_name
        or result.get("spatial_arm") != spec.spatial_arm
        or result.get("semantic_condition") != spec.semantic_condition
        or result.get("language_layer") != spec.language_layer
        or result.get("training_control") != spec.training_control
    ):
        raise RuntimeError("armbar result does not match the diagnostic work item")
    checkpoint_value = result.get("training", {}).get("checkpoint_path")
    checkpoint_sha = result.get("training", {}).get("checkpoint_sha256")
    if not checkpoint_value or not checkpoint_sha:
        raise RuntimeError("armbar result has no best checkpoint")
    checkpoint_path = Path(str(checkpoint_value))
    if not checkpoint_path.is_file():
        raise RuntimeError("armbar best checkpoint is missing")
    if _sha256(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("armbar best checkpoint checksum mismatch")
    if substitution is not None and not (
        spec.semantic_condition == "action_relational"
        and spec.training_control == "real"
    ):
        raise ValueError("fixed substitutions require a real action-conditioned job")

    destination = Path(output_root)
    expected = {
        "format": "armbar-contact-diagnostics-v1",
        "run_name": spec.run_name,
        "job_result_sha256": result_sha,
        "checkpoint_sha256": checkpoint_sha,
        "substitution": substitution,
    }
    if (destination / "RUN_COMPLETE").exists():
        return _load_completed(destination, expected)
    destination.mkdir(parents=True, exist_ok=True)

    import torch

    from .checkpoint import load_checkpoint
    from .data import ActorStateControlDataset, load_rgb_records
    from .experiments import STATIC_ARMS
    from .metrics import ownership_metrics
    from .model import OwnershipDecoder, SemanticOwnershipDecoder
    from .training import _batched_sample, _forward_decoder
    from .visualization import render_ownership_diagnostic

    device = torch.device(spec.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA armbar diagnostics requested but CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(spec.cuda_memory_fraction, device=0)
        torch.cuda.reset_peak_memory_stats(device)
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
    _, evaluation_base, evaluation_subset = _build_job_datasets(spec)
    if evaluation_subset != "test":
        raise RuntimeError("armbar diagnostic did not resolve the test split")
    if spec.semantic_condition is None:
        dataset = evaluation_base
    elif substitution is None:
        dataset = ActorStateControlDataset(
            evaluation_base,
            control=spec.training_control,
            seed=spec.seed,
        )
    else:
        pairs = _fixed_substitution_pairs(spec)
        if substitution not in pairs:
            raise ValueError(f"unknown armbar fixed substitution: {substitution}")
        dataset = ActorStateControlDataset(
            evaluation_base,
            control="shuffled_clip",
            replacement_actor_states=pairs[substitution],
        )
    rgb_records = load_rgb_records(
        spec.frame_manifest,
        project_root=spec.frame_project_root,
    )
    amp_enabled = spec.use_amp and device.type == "cuda"
    records = []
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                if sample.contact is None or not sample.contact.any():
                    del sample
                    continue
                frame_index = sample.frame_index
                image_path = destination / "images" / f"frame_{frame_index:06d}.png"
                tensor_path = destination / "tensors" / f"frame_{frame_index:06d}.npz"
                if image_path.exists() or tensor_path.exists():
                    raise RuntimeError(f"uncommitted armbar diagnostic exists: {frame_index}")
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
                logits_cpu = logits.float().cpu()
                probabilities = torch.softmax(logits_cpu[0], dim=0).numpy()
                rgb_path, rgb_sha = rgb_records[frame_index]
                if _sha256(rgb_path) != rgb_sha:
                    raise RuntimeError(f"armbar diagnostic RGB checksum mismatch: {frame_index}")
                title_suffix = substitution or spec.training_control
                diagnostic = render_ownership_diagnostic(
                    rgb_path=rgb_path,
                    labels=labels[0].cpu().numpy(),
                    logits=logits_cpu[0].numpy(),
                    contact=contact[0].cpu().numpy(),
                    output_path=image_path,
                    title=f"{spec.run_name} / {title_suffix} / frame {frame_index}",
                    panel_size=panel_size,
                )
                _atomic_npz(
                    tensor_path,
                    probabilities=probabilities.astype(np.float16),
                    prediction=probabilities.argmax(axis=0).astype(np.uint8),
                    a2_minus_a1=(probabilities[2] - probabilities[1]).astype(np.float16),
                )
                frame_metrics = ownership_metrics(
                    logits_cpu,
                    labels.cpu(),
                    contact.cpu(),
                )
                records.append(
                    {
                        "frame_index": frame_index,
                        "source_sha256": rgb_sha,
                        "image_path": image_path.relative_to(destination).as_posix(),
                        "image_sha256": diagnostic["output_sha256"],
                        "tensor_path": tensor_path.relative_to(destination).as_posix(),
                        "tensor_sha256": _sha256(tensor_path),
                        "contact_bbox_grid": diagnostic["contact_bbox_grid"],
                        "metrics": frame_metrics,
                    }
                )
                del (
                    sample,
                    spatial,
                    labels,
                    contact,
                    actor_states,
                    logits,
                    logits_cpu,
                )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not records:
        raise RuntimeError("armbar test split has no contact evidence to render")
    manifest = {
        **expected,
        "contact_frame_count": len(records),
        "records": records,
    }
    manifest_path = destination / "diagnostic-manifest.json"
    if manifest_path.exists():
        raise RuntimeError("refusing to overwrite armbar diagnostic manifest")
    _atomic_json(manifest_path, manifest)
    _atomic_json(destination / "RUN_COMPLETE", {"manifest_sha256": _sha256(manifest_path)})
    return _load_completed(destination, expected)
