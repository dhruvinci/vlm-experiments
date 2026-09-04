from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .experiments import STATIC_ARMS


DEFAULT_STATIC_SCREEN = (
    "rgb",
    "l11",
    "p12",
    "merged",
    "l11_merged",
    "l05_l11_l18_l26",
)
SEMANTIC_CONDITIONS = {"identity_only", "action_relational", "contact_ownership"}


@dataclass(frozen=True)
class DecoderFoldRunSpec:
    run_name: str
    spatial_arm: str
    train_clips: tuple[str, ...]
    validation_clip: str
    heldout_clip: str
    label_manifests: Mapping[str, Path]
    qwen_breadth_root: Path
    input_root: Path
    output_root: Path
    qwen_download_manifest: Path | None = None
    semantic_condition: str | None = None
    language_layer: int | None = None
    semantic_context: str = "4fps"
    thinking_mode: str = "off"
    width: int = 96
    residual_blocks: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 10
    gradient_accumulation: int = 4
    seed: int = 7
    device: str = "cuda"
    use_amp: bool = True


def validate_fold_run_spec(spec: DecoderFoldRunSpec) -> DecoderFoldRunSpec:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec.run_name):
        raise ValueError("decoder run name must be a safe non-empty path component")
    if spec.spatial_arm not in STATIC_ARMS:
        raise ValueError(f"unknown decoder spatial arm: {spec.spatial_arm}")
    if len(spec.train_clips) < 1 or len(set(spec.train_clips)) != len(spec.train_clips):
        raise ValueError("decoder training clips must be non-empty and unique")
    split_groups = (set(spec.train_clips), {spec.validation_clip}, {spec.heldout_clip})
    if any(not next(iter(group), "") for group in split_groups):
        raise ValueError("decoder split clip IDs cannot be empty")
    if any(split_groups[left] & split_groups[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("decoder train, validation, and heldout clips must be disjoint")
    expected_clips = set(spec.train_clips) | {spec.validation_clip, spec.heldout_clip}
    if set(spec.label_manifests) != expected_clips:
        raise ValueError("decoder label manifest inventory must exactly match the fold clips")
    has_condition = spec.semantic_condition is not None
    has_layer = spec.language_layer is not None
    if has_condition != has_layer:
        raise ValueError("semantic condition and language layer must be supplied together")
    if has_condition and spec.semantic_condition not in SEMANTIC_CONDITIONS:
        raise ValueError(f"unsupported semantic condition: {spec.semantic_condition}")
    if has_layer and not 0 <= int(spec.language_layer) <= 63:
        raise ValueError("language layer must be in [0, 63]")
    if not spec.semantic_context.strip() or not spec.thinking_mode.strip():
        raise ValueError("semantic context and thinking mode cannot be empty")
    if not 8 <= spec.width <= 256:
        raise ValueError("decoder width must remain in the memory-safe range [8, 256]")
    if not 0 <= spec.residual_blocks <= 4:
        raise ValueError("decoder residual blocks must remain in [0, 4]")
    if spec.learning_rate <= 0 or spec.weight_decay < 0:
        raise ValueError("decoder optimizer settings are invalid")
    if min(spec.max_epochs, spec.patience, spec.gradient_accumulation) < 1:
        raise ValueError("decoder epoch, patience, and accumulation settings must be positive")
    if spec.device not in {"cuda", "cpu"}:
        raise ValueError("decoder device must be cuda or cpu")
    return spec


def fold_run_spec_to_dict(spec: DecoderFoldRunSpec) -> dict[str, Any]:
    validate_fold_run_spec(spec)
    return {
        "run_name": spec.run_name,
        "spatial_arm": spec.spatial_arm,
        "train_clips": list(spec.train_clips),
        "validation_clip": spec.validation_clip,
        "heldout_clip": spec.heldout_clip,
        "label_manifests": {
            clip_id: str(path) for clip_id, path in sorted(spec.label_manifests.items())
        },
        "qwen_breadth_root": str(spec.qwen_breadth_root),
        "input_root": str(spec.input_root),
        "output_root": str(spec.output_root),
        "qwen_download_manifest": (
            str(spec.qwen_download_manifest)
            if spec.qwen_download_manifest is not None
            else None
        ),
        "semantic_condition": spec.semantic_condition,
        "language_layer": spec.language_layer,
        "semantic_context": spec.semantic_context,
        "thinking_mode": spec.thinking_mode,
        "width": spec.width,
        "residual_blocks": spec.residual_blocks,
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "max_epochs": spec.max_epochs,
        "patience": spec.patience,
        "gradient_accumulation": spec.gradient_accumulation,
        "seed": spec.seed,
        "device": spec.device,
        "use_amp": spec.use_amp,
    }


def fold_run_spec_from_dict(value: Mapping[str, Any]) -> DecoderFoldRunSpec:
    required = {
        "run_name",
        "spatial_arm",
        "train_clips",
        "validation_clip",
        "heldout_clip",
        "label_manifests",
        "qwen_breadth_root",
        "input_root",
        "output_root",
        "qwen_download_manifest",
        "semantic_condition",
        "language_layer",
        "semantic_context",
        "thinking_mode",
        "width",
        "residual_blocks",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "gradient_accumulation",
        "seed",
        "device",
        "use_amp",
    }
    if set(value) != required:
        raise ValueError("decoder fold work-item schema is invalid")
    labels = value["label_manifests"]
    if not isinstance(labels, Mapping):
        raise ValueError("decoder fold label manifests must be an object")
    qwen_manifest = value["qwen_download_manifest"]
    spec = DecoderFoldRunSpec(
        run_name=str(value["run_name"]),
        spatial_arm=str(value["spatial_arm"]),
        train_clips=tuple(str(item) for item in value["train_clips"]),
        validation_clip=str(value["validation_clip"]),
        heldout_clip=str(value["heldout_clip"]),
        label_manifests={str(key): Path(str(path)) for key, path in labels.items()},
        qwen_breadth_root=Path(str(value["qwen_breadth_root"])),
        input_root=Path(str(value["input_root"])),
        output_root=Path(str(value["output_root"])),
        qwen_download_manifest=(Path(str(qwen_manifest)) if qwen_manifest is not None else None),
        semantic_condition=(
            str(value["semantic_condition"])
            if value["semantic_condition"] is not None
            else None
        ),
        language_layer=(
            int(value["language_layer"])
            if value["language_layer"] is not None
            else None
        ),
        semantic_context=str(value["semantic_context"]),
        thinking_mode=str(value["thinking_mode"]),
        width=int(value["width"]),
        residual_blocks=int(value["residual_blocks"]),
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        max_epochs=int(value["max_epochs"]),
        patience=int(value["patience"]),
        gradient_accumulation=int(value["gradient_accumulation"]),
        seed=int(value["seed"]),
        device=str(value["device"]),
        use_amp=bool(value["use_amp"]),
    )
    return validate_fold_run_spec(spec)


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


def _spec_payload(spec: DecoderFoldRunSpec) -> dict[str, Any]:
    label_records = {}
    for clip_id, path_value in sorted(spec.label_manifests.items()):
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"reviewed label manifest is missing: {path}")
        label_records[clip_id] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    qwen_manifest_record = None
    if spec.qwen_download_manifest is not None:
        qwen_manifest = Path(spec.qwen_download_manifest)
        if not qwen_manifest.is_file():
            raise FileNotFoundError(f"Qwen download manifest is missing: {qwen_manifest}")
        qwen_manifest_record = {
            "path": str(qwen_manifest.resolve()),
            "sha256": _sha256(qwen_manifest),
        }
    return {
        "format": "ownership-decoder-fold-spec-v1",
        "run_name": spec.run_name,
        "spatial_arm": spec.spatial_arm,
        "train_clips": list(spec.train_clips),
        "validation_clip": spec.validation_clip,
        "heldout_clip": spec.heldout_clip,
        "label_manifests": label_records,
        "qwen_breadth_root": str(spec.qwen_breadth_root.resolve()),
        "qwen_download_manifest": qwen_manifest_record,
        "input_root": str(spec.input_root.resolve()),
        "semantic_condition": spec.semantic_condition,
        "language_layer": spec.language_layer,
        "semantic_context": spec.semantic_context,
        "thinking_mode": spec.thinking_mode,
        "model": {
            "width": spec.width,
            "residual_blocks": spec.residual_blocks,
        },
        "training": {
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "max_epochs": spec.max_epochs,
            "patience": spec.patience,
            "gradient_accumulation": spec.gradient_accumulation,
            "seed": spec.seed,
            "device": spec.device,
            "use_amp": spec.use_amp,
            "batch_size": 1,
        },
    }


def _build_clip_specs(spec: DecoderFoldRunSpec) -> dict[str, tuple[Any, ...]]:
    from .data import build_specs_from_label_manifest, load_rgb_records

    arm = STATIC_ARMS[spec.spatial_arm]
    result = {}
    for clip_id, label_manifest in sorted(spec.label_manifests.items()):
        rgb_records = None
        if arm.use_rgb:
            clip_root = spec.input_root / clip_id
            rgb_records = load_rgb_records(
                clip_root / "clip-manifest.json",
                project_root=clip_root,
            )
        actor_paths = None
        if spec.semantic_condition is not None:
            semantic_root = (
                spec.qwen_breadth_root
                / clip_id
                / "semantic"
                / "video"
                / spec.semantic_context
                / spec.semantic_condition
                / spec.thinking_mode
            )
            actor_paths = (semantic_root / "A1.safetensors", semantic_root / "A2.safetensors")
            if any(not path.is_file() for path in actor_paths):
                raise FileNotFoundError(
                    f"semantic actor-state pair is incomplete for {clip_id}: {semantic_root}"
                )
        result[clip_id] = build_specs_from_label_manifest(
            label_manifest,
            cache_root=spec.qwen_breadth_root / clip_id,
            clip_id=clip_id,
            full_layers=arm.full_layers,
            pooled_layers=arm.pooled_layers,
            include_merged=arm.include_merged,
            actor_state_paths=actor_paths,
            language_layer=spec.language_layer,
            rgb_records=rgb_records,
            require_reviewed=True,
        )
    return result


def _load_completed_fold(run_root: Path, expected_spec: dict[str, Any]) -> dict[str, Any]:
    spec_path = run_root / "run-spec.json"
    result_path = run_root / "result.json"
    completion_path = run_root / "RUN_COMPLETE"
    if not spec_path.is_file() or not result_path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"decoder fold is incomplete: {run_root}")
    if json.loads(spec_path.read_text(encoding="utf-8")) != expected_spec:
        raise RuntimeError("decoder fold run specification changed")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("result_sha256") != _sha256(result_path):
        raise RuntimeError("decoder fold result SHA-256 mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("run_spec_sha256") != _sha256(spec_path):
        raise RuntimeError("decoder fold result is not bound to its run specification")
    return result


def run_decoder_fold(spec: DecoderFoldRunSpec) -> dict[str, Any]:
    """Train and evaluate exactly one fold; process isolation is handled by the controller."""

    validate_fold_run_spec(spec)
    run_root = spec.output_root / spec.run_name
    expected_spec = _spec_payload(spec)
    completion_path = run_root / "RUN_COMPLETE"
    if completion_path.exists():
        return _load_completed_fold(run_root, expected_spec)
    run_root.mkdir(parents=True, exist_ok=True)
    spec_path = run_root / "run-spec.json"
    if spec_path.exists():
        try:
            observed_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("partial decoder fold has an invalid run specification") from error
        if observed_spec != expected_spec:
            raise RuntimeError("partial decoder fold run specification changed")
    else:
        _atomic_json(spec_path, expected_spec)

    import torch

    from .cache import load_actor_state_pair
    from .data import ActorStateControlDataset, OwnershipDataset
    from .model import OwnershipDecoder, SemanticOwnershipDecoder
    from .training import (
        TrainingConfig,
        evaluate_decoder,
        evaluate_query_swap,
        train_decoder,
    )

    if spec.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA decoder fold requested but no CUDA device is available")
    clip_specs = _build_clip_specs(spec)

    def dataset_for(clips: tuple[str, ...]) -> OwnershipDataset:
        samples = tuple(
            sample_spec
            for clip_id in clips
            for sample_spec in clip_specs[clip_id]
        )
        return OwnershipDataset(samples, rgb_output_hw=None)

    train_dataset = dataset_for(spec.train_clips)
    validation_dataset = dataset_for((spec.validation_clip,))
    test_dataset = dataset_for((spec.heldout_clip,))
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
    device = torch.device(spec.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        training = train_decoder(
            model,
            train_dataset,
            validation_dataset,
            config=TrainingConfig(
                learning_rate=spec.learning_rate,
                weight_decay=spec.weight_decay,
                max_epochs=spec.max_epochs,
                patience=spec.patience,
                gradient_accumulation=spec.gradient_accumulation,
                seed=spec.seed,
                device=spec.device,
                use_amp=spec.use_amp,
                checkpoint_directory=run_root / "checkpoints",
            ),
        )
        test_metrics = evaluate_decoder(
            model,
            test_dataset,
            device=device,
            use_amp=spec.use_amp,
        )
        semantic_controls = None
        swap_metrics = None
        if spec.semantic_condition is not None:
            semantic_controls = {"real": test_metrics}
            donor_clip = sorted(spec.train_clips)[0]
            donor_spec = clip_specs[donor_clip][0]
            if donor_spec.actor_state_paths is None or spec.language_layer is None:
                raise RuntimeError("semantic shuffled-clip control has no donor states")
            donor_states = load_actor_state_pair(
                donor_spec.actor_state_paths[0],
                donor_spec.actor_state_paths[1],
                language_layer=spec.language_layer,
            )
            semantic_controls["shuffled_clip"] = evaluate_decoder(
                model,
                ActorStateControlDataset(
                    test_dataset,
                    control="shuffled_clip",
                    replacement_actor_states=donor_states,
                ),
                device=device,
                use_amp=spec.use_amp,
            )
            for control in ("random_matched", "zero", "mean"):
                semantic_controls[control] = evaluate_decoder(
                    model,
                    ActorStateControlDataset(test_dataset, control=control, seed=spec.seed),
                    device=device,
                    use_amp=spec.use_amp,
                )
            swap_metrics = evaluate_query_swap(
                model,
                test_dataset,
                device=device,
                use_amp=spec.use_amp,
            )
        checkpoint_path = training.checkpoint_path
        result = {
            "format": "ownership-decoder-fold-result-v1",
            "run_spec_sha256": _sha256(spec_path),
            "run_name": spec.run_name,
            "spatial_arm": spec.spatial_arm,
            "semantic_condition": spec.semantic_condition,
            "language_layer": spec.language_layer,
            "train_clips": list(spec.train_clips),
            "validation_clip": spec.validation_clip,
            "heldout_clip": spec.heldout_clip,
            "sample_counts": {
                "train": len(train_dataset),
                "validation": len(validation_dataset),
                "test": len(test_dataset),
            },
            "training": {
                "best_epoch": training.best_epoch,
                "stopped_epoch": training.stopped_epoch,
                "best_validation_metrics": training.best_metrics,
                "history": training.history,
                "peak_vram_bytes": training.peak_vram_bytes,
                "peak_host_rss_bytes": training.peak_host_rss_bytes,
                "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else None,
                "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
            },
            "test_metrics": test_metrics,
            "semantic_controls": semantic_controls,
            "swap_metrics": swap_metrics,
        }
        result_path = run_root / "result.json"
        if result_path.exists():
            raise RuntimeError("refusing to overwrite decoder fold result")
        _atomic_json(result_path, result)
        _atomic_json(completion_path, {"result_sha256": _sha256(result_path)})
        return _load_completed_fold(run_root, expected_spec)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
