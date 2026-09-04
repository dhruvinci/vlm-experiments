from __future__ import annotations

import hashlib
import json
import os
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .image_agreement import run_image_agreement_campaign
from .mask_geometry import mask_pair_to_geometry_prompts
from .remote_canary import run_sam31_tracker_canary, run_sam3_image_canary
from .remote_preflight import (
    RemoteRuntimeApproval,
    RemoteRuntimeContract,
    RequiredArtifact,
    perform_remote_preflight,
)
from .remote_telemetry import RuntimeTelemetryWriter
from .sam31_tracker_adapter import build_tracker_only_sam31
from .sam3_image_adapter import REQUIRED_SAM3_IMAGE_FILES, build_sam3_image_predictor
from .tracking import (
    TrackingPlan,
    _artifact_paths as _tracking_artifact_paths,
    _load_mask_artifact as _load_tracking_mask_artifact,
    load_tracking_plan_config,
)
from .tracking_campaign import run_tracking_campaign


FINAL_FORMAT = "ownership-remote-mask-campaign-v1"


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
class RemoteMaskCampaignSpec:
    config_paths: tuple[Path, ...]
    input_root: Path
    output_root: Path
    sam_repo_path: Path
    sam_repo_revision: str
    sam31_checkpoint_path: Path
    sam31_checkpoint_sha256: str
    sam3_model_directory: Path
    sam3_model_revision: str
    sam3_model_artifacts: tuple[RequiredArtifact, ...]
    workspace_path: Path
    required_distribution_versions: tuple[tuple[str, str], ...]
    attempt_index: int
    minimum_prompt_area: int = 64
    box_padding_fraction: float = 0.01

    def __post_init__(self) -> None:
        if not self.config_paths:
            raise ValueError("remote mask campaign requires at least one config")
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if not self.sam3_model_revision.strip():
            raise ValueError("sam3_model_revision cannot be empty")
        if self.minimum_prompt_area < 2:
            raise ValueError("minimum_prompt_area must be at least two pixels")
        expected = {
            (self.sam3_model_directory / name).resolve()
            for name in REQUIRED_SAM3_IMAGE_FILES
        }
        observed = {artifact.path.resolve() for artifact in self.sam3_model_artifacts}
        missing = sorted(str(path) for path in expected - observed)
        if missing:
            raise ValueError(f"base SAM3 artifact contract is incomplete: {missing}")

    @property
    def runtime_contract(self) -> RemoteRuntimeContract:
        return RemoteRuntimeContract(
            sam_repo_path=self.sam_repo_path,
            sam_repo_revision=self.sam_repo_revision,
            checkpoint_path=self.sam31_checkpoint_path,
            checkpoint_sha256=self.sam31_checkpoint_sha256,
            workspace_path=self.workspace_path,
            required_distribution_versions=self.required_distribution_versions,
            additional_artifacts=self.sam3_model_artifacts,
        )

    @property
    def tracker_root(self) -> Path:
        return self.output_root / "sam31-tracking"

    @property
    def image_root(self) -> Path:
        return self.output_root / "sam3-image-agreement"

    @property
    def attempt_dir(self) -> Path:
        return self.output_root / "_runtime" / f"attempt_{self.attempt_index:02d}"

    @property
    def tracker_revision(self) -> str:
        return (
            f"repo-{self.sam_repo_revision}+sam31-"
            f"{self.sam31_checkpoint_sha256[:12]}"
        )

    @property
    def image_revision(self) -> str:
        weights = next(
            artifact
            for artifact in self.sam3_model_artifacts
            if artifact.path.name == "model.safetensors"
        )
        return f"hf-{self.sam3_model_revision}+weights-{weights.sha256[:12]}"


def _load_plans(spec: RemoteMaskCampaignSpec) -> tuple[tuple[Path, TrackingPlan], ...]:
    loaded = sorted(
        (
            (
                path,
                load_tracking_plan_config(path, input_root=spec.input_root),
            )
            for path in spec.config_paths
        ),
        key=lambda item: item[1].clip_id,
    )
    ids = [plan.clip_id for _, plan in loaded]
    if len(ids) != len(set(ids)):
        raise ValueError(f"remote mask campaign clip IDs must be unique: {ids}")
    return tuple(loaded)


def _run_sam3_image_canary_from_tracking(
    predictor: Any,
    plan: TrackingPlan,
    *,
    tracker_root: Path,
    tracker_backend: str,
    tracker_revision: str,
    minimum_prompt_area: int,
    box_padding_fraction: float,
) -> dict[str, Any]:
    seed_index = plan.seeds[0].frame_index
    frames = {frame.frame_index: frame for frame in plan.frames}
    frame = frames[seed_index]
    tracker_artifact, tracker_sidecar = _tracking_artifact_paths(
        tracker_root / plan.clip_id,
        seed_index,
    )
    _, values = _load_tracking_mask_artifact(
        tracker_artifact,
        tracker_sidecar,
        expected_plan_sha256=plan.sha256,
        expected_backend=tracker_backend,
        expected_revision=tracker_revision,
        expected_frame=frame,
    )
    prompts = mask_pair_to_geometry_prompts(
        values["A1"],
        values["A2"],
        minimum_area=minimum_prompt_area,
        box_padding_fraction=box_padding_fraction,
    )
    return run_sam3_image_canary(
        predictor,
        image_path=frame.path,
        prompts=prompts,
        expected_shape=plan.frame_shape,
        clip_id=plan.clip_id,
        frame_index=seed_index,
    )


def _preflight_report(
    spec: RemoteMaskCampaignSpec,
    approval: RemoteRuntimeApproval,
    plans: tuple[tuple[Path, TrackingPlan], ...],
) -> dict[str, Any]:
    return {
        "format": "ownership-remote-mask-preflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempt_index": spec.attempt_index,
        "contract": spec.runtime_contract.as_dict(),
        "contract_sha256": approval.contract_sha256,
        "checkpoint_sha256": approval.checkpoint_sha256,
        "hardware": approval.hardware.as_dict(),
        "sam3_model_revision": spec.sam3_model_revision,
        "tracker_revision": spec.tracker_revision,
        "image_revision": spec.image_revision,
        "plans": [
            {
                "clip_id": plan.clip_id,
                "config_path": str(path.resolve()),
                "config_sha256": _sha256(path),
                "plan_sha256": plan.sha256,
                "frame_count": len(plan.frames),
            }
            for path, plan in plans
        ],
    }


def _load_final(spec: RemoteMaskCampaignSpec) -> dict[str, Any]:
    manifest_path = spec.output_root / "campaign-manifest.json"
    completion_path = spec.output_root / "RUN_COMPLETE"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise RuntimeError("remote mask campaign is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("remote mask campaign manifest checksum mismatch")
    expected = {
        "format": FINAL_FORMAT,
        "contract_sha256": spec.runtime_contract.sha256,
        "tracker_revision": spec.tracker_revision,
        "image_revision": spec.image_revision,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"remote mask campaign metadata mismatch: {key}")
    return manifest


def run_remote_mask_campaign(
    spec: RemoteMaskCampaignSpec,
    *,
    preflight_fn: Callable[
        [RemoteRuntimeContract], RemoteRuntimeApproval
    ] = perform_remote_preflight,
    sam31_builder: Callable[..., Any] = build_tracker_only_sam31,
    sam31_canary_fn: Callable[..., dict[str, Any]] = run_sam31_tracker_canary,
    tracking_runner: Callable[..., dict[str, Any]] = run_tracking_campaign,
    sam3_builder: Callable[..., Any] = build_sam3_image_predictor,
    sam3_image_canary_fn: Callable[
        ..., dict[str, Any]
    ] = _run_sam3_image_canary_from_tracking,
    image_runner: Callable[..., dict[str, Any]] = run_image_agreement_campaign,
    telemetry_factory: Callable[..., RuntimeTelemetryWriter] = RuntimeTelemetryWriter,
) -> dict[str, Any]:
    """Run tracker then image agreement sequentially with only one model resident."""

    plans = _load_plans(spec)
    if (spec.output_root / "RUN_COMPLETE").exists():
        return _load_final(spec)
    contract = spec.runtime_contract
    approval = preflight_fn(contract)
    approval.require_contract(contract)

    attempt_dir = spec.attempt_dir
    attempt_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = attempt_dir / "preflight.json"
    if preflight_path.exists():
        raise RuntimeError(f"refusing to overwrite remote attempt metadata: {preflight_path}")
    _atomic_json(preflight_path, _preflight_report(spec, approval, plans))
    telemetry = telemetry_factory(
        attempt_dir / "telemetry.jsonl",
        workspace_path=spec.workspace_path,
    )
    tracker_built = False
    image_built = False
    try:
        telemetry.record(
            {
                "event": "preflight_completed",
                "attempt_index": spec.attempt_index,
                "contract_sha256": approval.contract_sha256,
            }
        )

        def tracker_factory() -> Any:
            nonlocal tracker_built
            if tracker_built:
                raise RuntimeError("SAM3.1 predictor was requested more than once")
            tracker_built = True
            telemetry.record({"event": "sam31_load_started"})
            predictor = sam31_builder(contract, approval)
            telemetry.record({"event": "sam31_loaded"})
            try:
                canary = sam31_canary_fn(predictor, plans[0][1])
                _atomic_json(attempt_dir / "sam31-canary.json", canary)
                telemetry.record(
                    {
                        "event": "sam31_canary_completed",
                        "clip_id": canary.get("clip_id"),
                        "frame_index": canary.get("frame_index"),
                    }
                )
            except Exception:
                close = getattr(predictor, "close", None)
                if callable(close):
                    close()
                raise
            return predictor

        tracking = tracking_runner(
            spec.config_paths,
            input_root=spec.input_root,
            output_root=spec.tracker_root,
            backend="sam3.1-tracker-only",
            revision=spec.tracker_revision,
            predictor_factory=tracker_factory,
            event_callback=telemetry.record,
        )
        telemetry.record(
            {
                "event": "sam31_stage_completed",
                "predictor_loaded": tracker_built,
                "frame_count": tracking.get("frame_count"),
            }
        )

        def image_factory() -> Any:
            nonlocal image_built
            if image_built:
                raise RuntimeError("base SAM3 predictor was requested more than once")
            image_built = True
            telemetry.record({"event": "sam3_image_load_started"})
            predictor = sam3_builder(
                contract,
                approval,
                model_directory=spec.sam3_model_directory,
            )
            telemetry.record({"event": "sam3_image_loaded"})
            try:
                canary = sam3_image_canary_fn(
                    predictor,
                    plans[0][1],
                    tracker_root=spec.tracker_root,
                    tracker_backend="sam3.1-tracker-only",
                    tracker_revision=spec.tracker_revision,
                    minimum_prompt_area=spec.minimum_prompt_area,
                    box_padding_fraction=spec.box_padding_fraction,
                )
                _atomic_json(attempt_dir / "sam3-image-canary.json", canary)
                telemetry.record(
                    {
                        "event": "sam3_image_canary_completed",
                        "clip_id": canary.get("clip_id"),
                        "frame_index": canary.get("frame_index"),
                    }
                )
            except Exception:
                close = getattr(predictor, "close", None)
                if callable(close):
                    close()
                raise
            return predictor

        agreement = image_runner(
            spec.config_paths,
            input_root=spec.input_root,
            tracker_root=spec.tracker_root,
            output_root=spec.image_root,
            tracker_backend="sam3.1-tracker-only",
            tracker_revision=spec.tracker_revision,
            backend="sam3-tracker-image-pvs",
            revision=spec.image_revision,
            predictor_factory=image_factory,
            minimum_prompt_area=spec.minimum_prompt_area,
            box_padding_fraction=spec.box_padding_fraction,
            event_callback=telemetry.record,
        )
        telemetry.record(
            {
                "event": "sam3_image_stage_completed",
                "predictor_loaded": image_built,
                "frame_count": agreement.get("frame_count"),
            }
        )
        manifest = {
            "format": FINAL_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": contract.sha256,
            "tracker_revision": spec.tracker_revision,
            "image_revision": spec.image_revision,
            "clip_count": len(plans),
            "frame_count": sum(len(plan.frames) for _, plan in plans),
            "tracking_manifest_sha256": _canonical_sha256(tracking),
            "image_agreement_manifest_sha256": _canonical_sha256(agreement),
            "localization_dependency": "sam3.1-tracker",
            "scientific_independence": (
                "SAM3 image contours are independently decoded, but localization is "
                "correlated through SAM3.1-derived geometry prompts"
            ),
            "plan_sha256_by_clip": {
                plan.clip_id: plan.sha256 for _, plan in plans
            },
        }
        manifest_path = spec.output_root / "campaign-manifest.json"
        if manifest_path.exists():
            raise RuntimeError(
                f"refusing to overwrite remote campaign manifest: {manifest_path}"
            )
        _atomic_json(manifest_path, manifest)
        _atomic_json(
            spec.output_root / "RUN_COMPLETE",
            {"manifest_sha256": _sha256(manifest_path)},
        )
        _atomic_json(
            attempt_dir / "SUCCESS.json",
            {
                "format": "ownership-remote-mask-attempt-success-v1",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "attempt_index": spec.attempt_index,
                "manifest_sha256": _sha256(manifest_path),
            },
        )
        return _load_final(spec)
    except Exception as error:
        _atomic_json(
            attempt_dir / "FAILURE.json",
            {
                "format": "ownership-remote-mask-attempt-failure-v1",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "attempt_index": spec.attempt_index,
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
