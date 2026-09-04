from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .decoder_campaign import (
    DEFAULT_STATIC_SCREEN,
    DecoderFoldRunSpec,
    fold_run_spec_to_dict,
)
from .evaluation import (
    aggregate_clip_metrics,
    evaluate_north_star,
    summarize_paired_clip_improvement,
)
from .experiments import build_nested_leave_one_clip_out_folds
from .resource_guard import ResourceLimits, run_guarded
from .qwen_inventory import verify_qwen_breadth_cache


GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class LocalDecoderCampaignSpec:
    reviewed_label_campaign: Path
    qwen_breadth_root: Path
    qwen_download_manifest: Path
    input_root: Path
    output_root: Path
    python_executable: Path = Path(sys.executable)
    static_arms: tuple[str, ...] = DEFAULT_STATIC_SCREEN
    semantic_layers: tuple[int, ...] = (25, 60)
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
    child_memory_max_bytes: int = 4 * GIB
    min_host_available_bytes: int = 2 * GIB
    min_gpu_free_bytes: int = 512 * MIB
    max_gpu_used_fraction: float = 0.92
    attempts_per_fold: int = 2


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


def _load_label_inventory(spec: LocalDecoderCampaignSpec) -> dict[str, Path]:
    path = spec.reviewed_label_campaign
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("reviewed label campaign manifest or checksum is missing")
    if sidecar.read_text(encoding="utf-8").strip() != _sha256(path):
        raise ValueError("reviewed label campaign manifest SHA-256 mismatch")
    try:
        campaign = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("reviewed label campaign manifest is invalid") from error
    if (
        campaign.get("format") != "reviewed-ownership-label-campaign-v1"
        or campaign.get("training_eligible") is not True
    ):
        raise ValueError("reviewed label campaign is not eligible for training")
    records = campaign.get("clips")
    if not isinstance(records, list) or len(records) < 4:
        raise ValueError("decoder campaign requires at least four reviewed clips")
    root = path.parent.resolve()
    result: dict[str, Path] = {}
    for record in records:
        clip_id = str(record.get("clip_id", ""))
        relative = Path(str(record.get("label_manifest_path", "")))
        if not clip_id or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("reviewed label clip record is unsafe")
        manifest_path = (root / relative).resolve()
        if not manifest_path.is_relative_to(root) or not manifest_path.is_file():
            raise FileNotFoundError(f"reviewed clip label manifest is missing: {clip_id}")
        if _sha256(manifest_path) != record.get("label_manifest_sha256"):
            raise ValueError(f"reviewed clip label manifest SHA-256 mismatch: {clip_id}")
        if clip_id in result:
            raise ValueError(f"reviewed label campaign duplicates clip: {clip_id}")
        result[clip_id] = manifest_path
    if campaign.get("clip_count") != len(result):
        raise ValueError("reviewed label campaign clip count is inconsistent")
    return result


def _validate_campaign_spec(spec: LocalDecoderCampaignSpec) -> None:
    if len(spec.static_arms) < 2 or len(set(spec.static_arms)) != len(spec.static_arms):
        raise ValueError("static decoder screen must contain at least two unique arms")
    if "rgb" not in spec.static_arms or "l11" not in spec.static_arms:
        raise ValueError("static decoder screen must include RGB and preregistered L11 baselines")
    if not spec.semantic_layers or len(set(spec.semantic_layers)) != len(spec.semantic_layers):
        raise ValueError("semantic language layers must be non-empty and unique")
    if any(layer < 0 or layer > 63 for layer in spec.semantic_layers):
        raise ValueError("semantic language layers must be in [0, 63]")
    if spec.attempts_per_fold not in {1, 2}:
        raise ValueError("decoder fold attempts must be one or two")
    if spec.child_memory_max_bytes < GIB:
        raise ValueError("decoder child memory cap is implausibly low")
    if not spec.python_executable.is_file():
        raise FileNotFoundError(f"decoder Python executable is missing: {spec.python_executable}")
    if not spec.qwen_download_manifest.is_file():
        raise FileNotFoundError("Qwen download manifest is required")
    if not spec.qwen_breadth_root.is_dir() or not spec.input_root.is_dir():
        raise FileNotFoundError("Qwen breadth cache or frozen input root is missing")


def _fold_spec(
    campaign: LocalDecoderCampaignSpec,
    *,
    label_manifests: dict[str, Path],
    run_name: str,
    spatial_arm: str,
    train_clips: tuple[str, ...],
    validation_clip: str,
    heldout_clip: str,
    semantic_condition: str | None = None,
    language_layer: int | None = None,
) -> DecoderFoldRunSpec:
    return DecoderFoldRunSpec(
        run_name=run_name,
        spatial_arm=spatial_arm,
        train_clips=train_clips,
        validation_clip=validation_clip,
        heldout_clip=heldout_clip,
        label_manifests=label_manifests,
        qwen_breadth_root=campaign.qwen_breadth_root,
        input_root=campaign.input_root,
        output_root=campaign.output_root / "fold-runs",
        qwen_download_manifest=campaign.qwen_download_manifest,
        semantic_condition=semantic_condition,
        language_layer=language_layer,
        width=campaign.width,
        residual_blocks=campaign.residual_blocks,
        learning_rate=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
        max_epochs=campaign.max_epochs,
        patience=campaign.patience,
        gradient_accumulation=campaign.gradient_accumulation,
        seed=campaign.seed,
        device=campaign.device,
        use_amp=campaign.use_amp,
    )


def _read_fold_result(fold: DecoderFoldRunSpec) -> dict[str, Any]:
    root = fold.output_root / fold.run_name
    result_path = root / "result.json"
    completion_path = root / "RUN_COMPLETE"
    if not result_path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"decoder worker did not complete: {fold.run_name}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"decoder worker metadata is invalid: {fold.run_name}") from error
    if completion.get("result_sha256") != _sha256(result_path):
        raise RuntimeError(f"decoder worker result SHA-256 mismatch: {fold.run_name}")
    return result


def _guarded_job_runner(campaign: LocalDecoderCampaignSpec) -> Callable[[DecoderFoldRunSpec], dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/run_decoder_fold.py"

    def run(fold: DecoderFoldRunSpec) -> dict[str, Any]:
        work_item = campaign.output_root / "work-items" / f"{fold.run_name}.json"
        payload = fold_run_spec_to_dict(fold)
        if work_item.exists():
            if json.loads(work_item.read_text(encoding="utf-8")) != payload:
                raise RuntimeError(f"decoder work item changed: {fold.run_name}")
        else:
            _atomic_json(work_item, payload)
        if (fold.output_root / fold.run_name / "RUN_COMPLETE").exists():
            return _read_fold_result(fold)
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(project_root / "src"),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        limits = ResourceLimits(
            min_host_available_bytes=campaign.min_host_available_bytes,
            min_gpu_free_bytes=(campaign.min_gpu_free_bytes if campaign.device == "cuda" else 0),
            max_gpu_used_fraction=(campaign.max_gpu_used_fraction if campaign.device == "cuda" else 1.0),
        )
        last_result = None
        for attempt in range(1, campaign.attempts_per_fold + 1):
            last_result = run_guarded(
                [str(campaign.python_executable), str(worker), "--spec", str(work_item)],
                limits=limits,
                log_path=campaign.output_root / "logs" / f"{fold.run_name}.attempt-{attempt}.log",
                telemetry_path=(
                    campaign.output_root
                    / "telemetry"
                    / f"{fold.run_name}.attempt-{attempt}.jsonl"
                ),
                cwd=project_root,
                env=environment,
                child_memory_max_bytes=campaign.child_memory_max_bytes,
            )
            if last_result.returncode == 0:
                return _read_fold_result(fold)
        assert last_result is not None
        raise RuntimeError(
            f"decoder fold failed after {campaign.attempts_per_fold} isolated attempts: "
            f"{fold.run_name}; violations={[value.resource for value in last_result.violations]}"
        )

    return run


def _best_validation_result(
    results: list[tuple[Any, dict[str, Any]]],
) -> tuple[Any, dict[str, Any]]:
    if not results:
        raise ValueError("model selection requires at least one validation result")
    return max(
        results,
        key=lambda item: (
            float(item[1]["training"]["best_validation_metrics"]["macro_actor_iou"]),
            -results.index(item),
        ),
    )


def _aggregate_swap(by_clip: dict[str, dict[str, float]]) -> dict[str, float]:
    actor_pixels = sum(value["actor_pixel_count"] for value in by_clip.values())
    background_pixels = sum(value["background_pixel_count"] for value in by_clip.values())
    probability_values = sum(
        value["actor_probability_value_count"] for value in by_clip.values()
    )
    return {
        "actor_pixel_count": actor_pixels,
        "actor_prediction_flip_fraction": sum(
            value["actor_prediction_flip_fraction"] * value["actor_pixel_count"]
            for value in by_clip.values()
        )
        / max(1.0, actor_pixels),
        "background_pixel_count": background_pixels,
        "background_probability_delta": sum(
            value["background_probability_delta"] * value["background_pixel_count"]
            for value in by_clip.values()
        )
        / max(1.0, background_pixels),
        "actor_probability_value_count": probability_values,
        "actor_probability_swap_error": sum(
            value["actor_probability_swap_error"]
            * value["actor_probability_value_count"]
            for value in by_clip.values()
        )
        / max(1.0, probability_values),
    }


def run_local_decoder_campaign(
    spec: LocalDecoderCampaignSpec,
    *,
    job_runner: Callable[[DecoderFoldRunSpec], dict[str, Any]] | None = None,
    cache_verifier: Callable[..., dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Run nested clip-held-out static and semantic decoder experiments serially."""

    _validate_campaign_spec(spec)
    labels = _load_label_inventory(spec)
    folds = build_nested_leave_one_clip_out_folds(tuple(labels))
    spec.output_root.mkdir(parents=True, exist_ok=True)
    result_path = spec.output_root / "campaign-result.json"
    completion_path = spec.output_root / "RUN_COMPLETE"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("result_sha256") != _sha256(result_path):
            raise RuntimeError("local decoder campaign result SHA-256 mismatch")
        return json.loads(result_path.read_text(encoding="utf-8"))
    frame_indices_by_clip = {}
    for clip_id, label_manifest in labels.items():
        try:
            label_payload = json.loads(label_manifest.read_text(encoding="utf-8"))
            frame_indices_by_clip[clip_id] = tuple(
                int(record["frame_index"]) for record in label_payload["records"]
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"could not read reviewed frame inventory for {clip_id}") from error
    verify_cache = cache_verifier or verify_qwen_breadth_cache
    cache_report = verify_cache(
        spec.qwen_download_manifest,
        qwen_breadth_root=spec.qwen_breadth_root,
        frame_indices_by_clip=frame_indices_by_clip,
        spatial_arms=spec.static_arms,
        semantic_conditions=("action_relational", "identity_only", "contact_ownership"),
        semantic_context="4fps",
        thinking_mode="off",
        rehash=True,
    )
    _atomic_json(spec.output_root / "cache-verification.json", dict(cache_report))
    run_job = job_runner or _guarded_job_runner(spec)

    static_all: dict[str, dict[str, dict[str, Any]]] = {
        arm: {} for arm in spec.static_arms
    }
    selected_static: dict[str, tuple[str, dict[str, Any]]] = {}
    for fold in folds:
        fold_results = []
        for arm in spec.static_arms:
            run = _fold_spec(
                spec,
                label_manifests=labels,
                run_name=f"static__{arm}__holdout-{fold.heldout_clip}",
                spatial_arm=arm,
                train_clips=fold.train_clips,
                validation_clip=fold.validation_clip,
                heldout_clip=fold.heldout_clip,
            )
            result = run_job(run)
            static_all[arm][fold.heldout_clip] = result
            fold_results.append((arm, result))
        selected_static[fold.heldout_clip] = _best_validation_result(fold_results)

    selected_action: dict[str, tuple[int, dict[str, Any]]] = {}
    semantic_results: dict[str, dict[str, dict[str, Any]]] = {
        condition: {} for condition in ("action_relational", "identity_only", "contact_ownership")
    }
    for fold in folds:
        spatial_arm = selected_static[fold.heldout_clip][0]
        layer_results = []
        for layer in spec.semantic_layers:
            run = _fold_spec(
                spec,
                label_manifests=labels,
                run_name=(
                    f"semantic__action_relational__l{layer:02d}__{spatial_arm}__"
                    f"holdout-{fold.heldout_clip}"
                ),
                spatial_arm=spatial_arm,
                train_clips=fold.train_clips,
                validation_clip=fold.validation_clip,
                heldout_clip=fold.heldout_clip,
                semantic_condition="action_relational",
                language_layer=layer,
            )
            layer_results.append((layer, run_job(run)))
        chosen_layer, chosen_action = _best_validation_result(layer_results)
        selected_action[fold.heldout_clip] = (chosen_layer, chosen_action)
        semantic_results["action_relational"][fold.heldout_clip] = chosen_action
        for condition in ("identity_only", "contact_ownership"):
            run = _fold_spec(
                spec,
                label_manifests=labels,
                run_name=(
                    f"semantic__{condition}__l{chosen_layer:02d}__{spatial_arm}__"
                    f"holdout-{fold.heldout_clip}"
                ),
                spatial_arm=spatial_arm,
                train_clips=fold.train_clips,
                validation_clip=fold.validation_clip,
                heldout_clip=fold.heldout_clip,
                semantic_condition=condition,
                language_layer=chosen_layer,
            )
            semantic_results[condition][fold.heldout_clip] = run_job(run)

    static_aggregates = {
        arm: aggregate_clip_metrics(
            {clip: result["test_metrics"] for clip, result in by_clip.items()}
        )
        for arm, by_clip in static_all.items()
    }
    baseline = aggregate_clip_metrics(
        {
            clip: result["test_metrics"]
            for clip, (_, result) in selected_static.items()
        }
    )
    semantic_aggregates = {
        condition: aggregate_clip_metrics(
            {clip: result["test_metrics"] for clip, result in by_clip.items()}
        )
        for condition, by_clip in semantic_results.items()
    }
    controls = {
        control: aggregate_clip_metrics(
            {
                clip: result["semantic_controls"][control]
                for clip, result in semantic_results["action_relational"].items()
            }
        )
        for control in ("real", "random_matched", "zero", "mean")
    }
    swap = _aggregate_swap(
        {
            clip: result["swap_metrics"]
            for clip, result in semantic_results["action_relational"].items()
        }
    )
    north_star = evaluate_north_star(
        candidate=semantic_aggregates["action_relational"],
        strongest_baseline=baseline,
        semantic_controls=controls,
        swap_metrics=swap,
    )
    paired_uncertainty = summarize_paired_clip_improvement(
        {
            clip: result["test_metrics"]
            for clip, result in semantic_results["action_relational"].items()
        },
        {
            clip: result["test_metrics"]
            for clip, (_, result) in selected_static.items()
        },
        random_seed=spec.seed,
    )
    result = {
        "format": "ownership-local-decoder-campaign-result-v1",
        "label_campaign_sha256": _sha256(spec.reviewed_label_campaign),
        "qwen_download_manifest_sha256": _sha256(spec.qwen_download_manifest),
        "qwen_cache_verification": cache_report,
        "clip_ids": sorted(labels),
        "static_screen": list(spec.static_arms),
        "semantic_layers": list(spec.semantic_layers),
        "selected_static_arm_by_heldout": {
            clip: arm for clip, (arm, _) in selected_static.items()
        },
        "selected_static_run_by_heldout": {
            clip: result["run_name"]
            for clip, (_, result) in selected_static.items()
        },
        "selected_action_layer_by_heldout": {
            clip: layer for clip, (layer, _) in selected_action.items()
        },
        "selected_action_run_by_heldout": {
            clip: result["run_name"]
            for clip, (_, result) in selected_action.items()
        },
        "static_aggregates": static_aggregates,
        "selected_static_aggregate": baseline,
        "semantic_aggregates": semantic_aggregates,
        "action_semantic_controls": controls,
        "action_swap_metrics": swap,
        "action_over_static_paired_uncertainty": paired_uncertainty,
        "north_star": {
            "passed": north_star.passed,
            "gates": north_star.gates,
            "evidence": north_star.evidence,
        },
        "scientific_status": (
            "decoder_north_star_passed"
            if north_star.passed
            else "scientifically_valid_negative_or_partial_result"
        ),
    }
    if result_path.exists():
        raise RuntimeError("refusing to overwrite local decoder campaign result")
    _atomic_json(result_path, result)
    _atomic_json(completion_path, {"result_sha256": _sha256(result_path)})
    return result
