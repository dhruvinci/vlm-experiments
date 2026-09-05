from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Mapping, Sequence

from .armbar_exploratory import (
    ARMBAR_SUPERVISION_STATUS,
    ArmbarJobSpec,
    armbar_job_spec_to_dict,
)
from .experiments import STATIC_ARMS
from .resource_guard import ResourceLimits, run_guarded


GIB = 1024**3
MIB = 1024**2
DEFAULT_ARMBAR_STATIC_ARMS = (
    "rgb",
    "l11",
    "p12",
    "merged",
    "l11_merged",
    "l05_l11_l18_l26",
)


@dataclass(frozen=True)
class ArmbarCampaignSpec:
    label_manifest: Path
    cache_root: Path
    frame_manifest: Path
    frame_project_root: Path
    output_root: Path
    python_executable: Path = Path(sys.executable)
    static_arms: tuple[str, ...] = DEFAULT_ARMBAR_STATIC_ARMS
    semantic_layers: tuple[int, ...] = (25, 60)
    seeds: tuple[int, ...] = (7, 71, 701)
    width: int = 96
    residual_blocks: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 10
    gradient_accumulation: int = 4
    device: str = "cuda"
    use_amp: bool = True
    cuda_memory_fraction: float = 0.60
    child_memory_max_bytes: int = 4 * GIB
    min_host_available_bytes: int = 4 * GIB
    min_swap_free_bytes: int = 3 * GIB
    min_gpu_free_bytes: int = 1536 * MIB
    max_gpu_used_fraction: float = 0.75
    maximum_job_runtime_seconds: float = 30 * 60
    resource_poll_interval_seconds: float = 1.0
    terminate_grace_seconds: float = 10.0
    attempts_per_job: int = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _validate_campaign_spec(spec: ArmbarCampaignSpec) -> None:
    if len(spec.static_arms) < 2 or len(set(spec.static_arms)) != len(spec.static_arms):
        raise ValueError("armbar static screen needs at least two unique arms")
    if "rgb" not in spec.static_arms or "l11" not in spec.static_arms:
        raise ValueError("armbar static screen must retain RGB and L11 baselines")
    unknown_arms = sorted(set(spec.static_arms) - set(STATIC_ARMS))
    if unknown_arms:
        raise ValueError(f"unknown armbar static arms: {unknown_arms}")
    if not spec.semantic_layers or len(set(spec.semantic_layers)) != len(spec.semantic_layers):
        raise ValueError("armbar semantic layers must be non-empty and unique")
    if any(layer < 0 or layer > 63 for layer in spec.semantic_layers):
        raise ValueError("armbar semantic layers must be in [0, 63]")
    if not spec.seeds or len(set(spec.seeds)) != len(spec.seeds):
        raise ValueError("armbar final seeds must be non-empty and unique")
    if spec.attempts_per_job not in {1, 2}:
        raise ValueError("armbar job attempts must be one or two")
    if spec.child_memory_max_bytes < GIB:
        raise ValueError("armbar child memory cap is implausibly low")
    if not 0.05 <= spec.cuda_memory_fraction <= 0.90:
        raise ValueError("armbar CUDA memory fraction is outside the safe range")
    if min(
        spec.min_host_available_bytes,
        spec.min_swap_free_bytes,
        spec.min_gpu_free_bytes,
    ) < 0:
        raise ValueError("armbar resource reserves cannot be negative")
    if not 0.0 < spec.max_gpu_used_fraction <= 1.0:
        raise ValueError("armbar maximum GPU usage fraction is invalid")
    if (
        spec.maximum_job_runtime_seconds <= 0
        or spec.resource_poll_interval_seconds <= 0
        or spec.terminate_grace_seconds < 0
    ):
        raise ValueError("armbar runtime guard settings are invalid")
    if spec.device not in {"cpu", "cuda"}:
        raise ValueError("armbar campaign device must be CPU or CUDA")
    for role, path in (
        ("label manifest", spec.label_manifest),
        ("frame manifest", spec.frame_manifest),
        ("Python executable", spec.python_executable),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"armbar {role} is missing: {path}")
    if not spec.cache_root.is_dir() or not spec.frame_project_root.is_dir():
        raise FileNotFoundError("armbar cache or frame project root is missing")
    try:
        labels = json.loads(spec.label_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("armbar label manifest is invalid") from error
    if labels.get("status") != "conservative_pseudo_labels_with_manual_final_contact_truth":
        raise ValueError("armbar controller only accepts the declared legacy pseudo-label set")


def _campaign_contract(spec: ArmbarCampaignSpec) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    implementation_files = (
        "armbar_controller.py",
        "armbar_exploratory.py",
        "cache.py",
        "data.py",
        "losses.py",
        "metrics.py",
        "model.py",
        "resource_guard.py",
        "training.py",
    )
    implementation_sha256 = {
        name: _sha256(source_root / name) for name in implementation_files
    }
    cache_manifest = spec.cache_root / "download-manifest.json"
    return {
        "format": "armbar-exploratory-campaign-spec-v1",
        "supervision_status": ARMBAR_SUPERVISION_STATUS,
        "north_star_eligible": False,
        "label_manifest": str(spec.label_manifest.resolve()),
        "label_manifest_sha256": _sha256(spec.label_manifest),
        "cache_root": str(spec.cache_root.resolve()),
        "cache_download_manifest_sha256": (
            _sha256(cache_manifest) if cache_manifest.is_file() else None
        ),
        "frame_manifest": str(spec.frame_manifest.resolve()),
        "frame_manifest_sha256": _sha256(spec.frame_manifest),
        "frame_project_root": str(spec.frame_project_root.resolve()),
        "static_arms": list(spec.static_arms),
        "semantic_layers": list(spec.semantic_layers),
        "seeds": list(spec.seeds),
        "architecture": {
            "width": spec.width,
            "residual_blocks": spec.residual_blocks,
        },
        "implementation_sha256": implementation_sha256,
        "optimizer": {
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "max_epochs": spec.max_epochs,
            "patience": spec.patience,
            "gradient_accumulation": spec.gradient_accumulation,
        },
        "execution": {
            "device": spec.device,
            "use_amp": spec.use_amp,
            "cuda_memory_fraction": spec.cuda_memory_fraction,
            "child_memory_max_bytes": spec.child_memory_max_bytes,
            "min_host_available_bytes": spec.min_host_available_bytes,
            "min_swap_free_bytes": spec.min_swap_free_bytes,
            "min_gpu_free_bytes": spec.min_gpu_free_bytes,
            "max_gpu_used_fraction": spec.max_gpu_used_fraction,
            "maximum_job_runtime_seconds": spec.maximum_job_runtime_seconds,
            "resource_poll_interval_seconds": spec.resource_poll_interval_seconds,
            "terminate_grace_seconds": spec.terminate_grace_seconds,
            "attempts_per_job": spec.attempts_per_job,
            "python_executable": str(spec.python_executable.resolve()),
        },
    }


def _job_spec(
    campaign: ArmbarCampaignSpec,
    *,
    run_name: str,
    spatial_arm: str,
    split: str,
    seed: int,
    semantic_condition: str | None = None,
    language_layer: int | None = None,
    training_control: str = "real",
) -> ArmbarJobSpec:
    return ArmbarJobSpec(
        run_name=run_name,
        spatial_arm=spatial_arm,
        split=split,
        label_manifest=campaign.label_manifest,
        cache_root=campaign.cache_root,
        frame_manifest=campaign.frame_manifest,
        frame_project_root=campaign.frame_project_root,
        output_root=campaign.output_root / "jobs",
        semantic_condition=semantic_condition,
        language_layer=language_layer,
        training_control=training_control,
        width=campaign.width,
        residual_blocks=campaign.residual_blocks,
        learning_rate=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
        max_epochs=campaign.max_epochs,
        patience=campaign.patience,
        gradient_accumulation=campaign.gradient_accumulation,
        seed=seed,
        device=campaign.device,
        use_amp=campaign.use_amp,
        cuda_memory_fraction=campaign.cuda_memory_fraction,
    )


def _read_job_result(job: ArmbarJobSpec) -> dict[str, Any]:
    root = job.output_root / job.run_name
    result_path = root / "result.json"
    completion_path = root / "RUN_COMPLETE"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"armbar worker did not complete validly: {job.run_name}") from error
    if completion.get("result_sha256") != _sha256(result_path):
        raise RuntimeError(f"armbar worker result checksum mismatch: {job.run_name}")
    return result


def _guarded_job_runner(campaign: ArmbarCampaignSpec) -> Callable[[ArmbarJobSpec], dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/run_armbar_job.py"

    def run(job: ArmbarJobSpec) -> dict[str, Any]:
        payload = armbar_job_spec_to_dict(job)
        work_item = campaign.output_root / "work-items" / f"{job.run_name}.json"
        if work_item.exists():
            if json.loads(work_item.read_text(encoding="utf-8")) != payload:
                raise RuntimeError(f"armbar work item changed: {job.run_name}")
        else:
            _atomic_json(work_item, payload)
        if (job.output_root / job.run_name / "RUN_COMPLETE").exists():
            return _read_job_result(job)
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
            min_swap_free_bytes=campaign.min_swap_free_bytes,
            min_gpu_free_bytes=(campaign.min_gpu_free_bytes if campaign.device == "cuda" else 0),
            max_gpu_used_fraction=(
                campaign.max_gpu_used_fraction if campaign.device == "cuda" else 1.0
            ),
        )
        last_result = None
        for attempt in range(1, campaign.attempts_per_job + 1):
            last_result = run_guarded(
                [str(campaign.python_executable), str(worker), "--spec", str(work_item)],
                limits=limits,
                log_path=campaign.output_root / "logs" / f"{job.run_name}.attempt-{attempt}.log",
                telemetry_path=(
                    campaign.output_root
                    / "telemetry"
                    / f"{job.run_name}.attempt-{attempt}.jsonl"
                ),
                cwd=project_root,
                env=environment,
                child_memory_max_bytes=campaign.child_memory_max_bytes,
                maximum_runtime_seconds=campaign.maximum_job_runtime_seconds,
                poll_interval_seconds=campaign.resource_poll_interval_seconds,
                terminate_grace_seconds=campaign.terminate_grace_seconds,
            )
            _atomic_json(
                campaign.output_root
                / "guard-results"
                / f"{job.run_name}.attempt-{attempt}.json",
                asdict(last_result),
            )
            if last_result.returncode == 0:
                return _read_job_result(job)
        assert last_result is not None
        raise RuntimeError(
            f"armbar job failed after {campaign.attempts_per_job} isolated attempts: "
            f"{job.run_name}; termination={last_result.termination_reason}; "
            f"violations={[value.resource for value in last_result.violations]}"
        )

    return run


def _select_best(results: Sequence[tuple[Any, dict[str, Any]]]) -> tuple[Any, dict[str, Any]]:
    if not results:
        raise ValueError("armbar model selection requires validation results")
    return max(
        results,
        key=lambda item: float(
            item[1]["training"]["best_validation_metrics"]["macro_actor_iou"]
        ),
    )


def _aggregate_runs(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("armbar aggregation requires at least one run")
    metric_records = [result["evaluation_metrics"] for result in results]
    common = set(metric_records[0])
    for record in metric_records[1:]:
        common &= set(record)
    metrics: dict[str, Any] = {}
    for name in sorted(common):
        values = [float(record[name]) for record in metric_records]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"armbar metric is non-finite: {name}")
        metrics[name] = {
            "mean": fmean(values),
            "population_std": pstdev(values),
            "values": values,
        }
    return {
        "run_count": len(results),
        "run_names": [str(result["run_name"]) for result in results],
        "metrics": metrics,
    }


def _aggregate_substitutions(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = set(results[0]["fixed_substitutions"] or {})
    if not names:
        raise ValueError("final real-action jobs are missing fixed substitutions")
    if any(set(result["fixed_substitutions"] or {}) != names for result in results):
        raise ValueError("fixed-substitution inventories differ between seeds")
    return {
        name: _aggregate_runs(
            [
                {
                    "run_name": result["run_name"],
                    "evaluation_metrics": result["fixed_substitutions"][name],
                }
                for result in results
            ]
        )
        for name in sorted(names)
    }


def _mean(aggregate: Mapping[str, Any], metric: str) -> float:
    return float(aggregate["metrics"][metric]["mean"])


def _exploratory_signal(
    aggregates: Mapping[str, Mapping[str, Any]],
    substitutions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    action = aggregates["action_real"]
    static = aggregates["static"]
    random_control = aggregates["action_random_matched"]
    ordered = substitutions["action_ordered_4fps_off"]
    reversed_control = substitutions["action_reversed_4fps_off"]
    shuffled_control = substitutions["action_shuffled_4fps_off"]
    evidence = {
        "action_over_static_macro_iou": _mean(action, "macro_actor_iou")
        - _mean(static, "macro_actor_iou"),
        "action_over_random_macro_iou": _mean(action, "macro_actor_iou")
        - _mean(random_control, "macro_actor_iou"),
        "action_over_static_contact_margin": _mean(action, "contact_margin")
        - _mean(static, "contact_margin"),
        "ordered_over_best_temporal_null_macro_iou": _mean(ordered, "macro_actor_iou")
        - max(
            _mean(reversed_control, "macro_actor_iou"),
            _mean(shuffled_control, "macro_actor_iou"),
        ),
        "action_background_stability": _mean(action, "background_stability"),
    }
    gates = {
        "action_beats_static": evidence["action_over_static_macro_iou"] >= 0.01,
        "action_beats_random": evidence["action_over_random_macro_iou"] >= 0.01,
        "contact_margin_improves": evidence["action_over_static_contact_margin"] >= 0.05,
        "ordered_beats_temporal_nulls": evidence[
            "ordered_over_best_temporal_null_macro_iou"
        ]
        >= 0.005,
        "background_stable": evidence["action_background_stability"] >= 0.90,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "evidence": evidence,
        "interpretation": (
            "exploratory single-clip signal only; not a multi-clip decoder north-star result"
        ),
    }


def run_armbar_exploratory_campaign(
    spec: ArmbarCampaignSpec,
    *,
    job_runner: Callable[[ArmbarJobSpec], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run validation-only selection, then paired final armbar controls serially."""

    _validate_campaign_spec(spec)
    spec.output_root.mkdir(parents=True, exist_ok=True)
    contract = _campaign_contract(spec)
    contract_path = spec.output_root / "campaign-spec.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("armbar campaign specification changed")
    else:
        _atomic_json(contract_path, contract)
    result_path = spec.output_root / "campaign-result.json"
    completion_path = spec.output_root / "RUN_COMPLETE"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("result_sha256") != _sha256(result_path):
            raise RuntimeError("armbar campaign result checksum mismatch")
        return json.loads(result_path.read_text(encoding="utf-8"))
    run_job = job_runner or _guarded_job_runner(spec)
    selection_seed = spec.seeds[0]

    static_screen = []
    for arm in spec.static_arms:
        job = _job_spec(
            spec,
            run_name=f"screen__static__{arm}__seed-{selection_seed}",
            spatial_arm=arm,
            split="screen",
            seed=selection_seed,
        )
        static_screen.append((arm, run_job(job)))
    selected_static, selected_static_result = _select_best(static_screen)

    layer_screen = []
    for layer in spec.semantic_layers:
        job = _job_spec(
            spec,
            run_name=(
                f"screen__action_relational__{selected_static}__l{layer:02d}__"
                f"seed-{selection_seed}"
            ),
            spatial_arm=selected_static,
            split="screen",
            seed=selection_seed,
            semantic_condition="action_relational",
            language_layer=layer,
        )
        layer_screen.append((layer, run_job(job)))
    selected_layer, selected_layer_result = _select_best(layer_screen)

    final: dict[str, list[dict[str, Any]]] = {
        "static": [],
        "action_real": [],
        "action_random_matched": [],
        "identity_real": [],
        "contact_real": [],
        "action_zero": [],
        "action_mean": [],
    }
    for seed in spec.seeds:
        final["static"].append(
            run_job(
                _job_spec(
                    spec,
                    run_name=f"final__static__{selected_static}__seed-{seed}",
                    spatial_arm=selected_static,
                    split="final",
                    seed=seed,
                )
            )
        )
        for key, control in (
            ("action_real", "real"),
            ("action_random_matched", "random_matched"),
        ):
            final[key].append(
                run_job(
                    _job_spec(
                        spec,
                        run_name=(
                            f"final__action_relational__{control}__{selected_static}__"
                            f"l{selected_layer:02d}__seed-{seed}"
                        ),
                        spatial_arm=selected_static,
                        split="final",
                        seed=seed,
                        semantic_condition="action_relational",
                        language_layer=selected_layer,
                        training_control=control,
                    )
                )
            )

    for key, condition, control in (
        ("identity_real", "identity_only", "real"),
        ("contact_real", "contact_ownership", "real"),
        ("action_zero", "action_relational", "zero"),
        ("action_mean", "action_relational", "mean"),
    ):
        final[key].append(
            run_job(
                _job_spec(
                    spec,
                    run_name=(
                        f"final__{condition}__{control}__{selected_static}__"
                        f"l{selected_layer:02d}__seed-{selection_seed}"
                    ),
                    spatial_arm=selected_static,
                    split="final",
                    seed=selection_seed,
                    semantic_condition=condition,
                    language_layer=selected_layer,
                    training_control=control,
                )
            )
        )

    aggregates = {name: _aggregate_runs(runs) for name, runs in final.items()}
    substitutions = _aggregate_substitutions(final["action_real"])
    result = {
        "format": "armbar-exploratory-campaign-result-v1",
        "campaign_spec_sha256": _sha256(contract_path),
        "supervision_status": ARMBAR_SUPERVISION_STATUS,
        "north_star_eligible": False,
        "selected_static_arm": selected_static,
        "selected_static_screen_run": selected_static_result["run_name"],
        "selected_action_language_layer": selected_layer,
        "selected_action_screen_run": selected_layer_result["run_name"],
        "static_screen": {
            arm: result["training"]["best_validation_metrics"]
            for arm, result in static_screen
        },
        "action_layer_screen": {
            str(layer): result["training"]["best_validation_metrics"]
            for layer, result in layer_screen
        },
        "final_aggregates": aggregates,
        "fixed_substitution_aggregates": substitutions,
        "paired_seed_deltas": {
            "action_minus_static_macro_iou": [
                float(action["evaluation_metrics"]["macro_actor_iou"])
                - float(static["evaluation_metrics"]["macro_actor_iou"])
                for action, static in zip(
                    final["action_real"], final["static"], strict=True
                )
            ],
            "action_minus_random_macro_iou": [
                float(action["evaluation_metrics"]["macro_actor_iou"])
                - float(random_result["evaluation_metrics"]["macro_actor_iou"])
                for action, random_result in zip(
                    final["action_real"],
                    final["action_random_matched"],
                    strict=True,
                )
            ],
        },
        "exploratory_signal": _exploratory_signal(aggregates, substitutions),
        "limitations": [
            "single source video",
            "legacy conservative pseudo-labels rather than fully human-reviewed masks",
            "contact ownership truth exists only in the final held-out contact frame",
            "global cached marker states cannot provide frame-varying temporal memory",
        ],
    }
    if result_path.exists():
        raise RuntimeError("refusing to overwrite armbar campaign result")
    _atomic_json(result_path, result)
    _atomic_json(completion_path, {"result_sha256": _sha256(result_path)})
    return result
