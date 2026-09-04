from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .tracking import (
    TrackingArtifactError,
    TrackingPlan,
    load_completed_tracking_run,
    load_tracking_plan_config,
    run_tracking_plan,
)


CAMPAIGN_FORMAT = "ownership-tracking-campaign-v1"


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


def _load_plans(
    config_paths: Sequence[str | Path],
    *,
    input_root: str | Path,
) -> tuple[tuple[Path, TrackingPlan], ...]:
    if not config_paths:
        raise ValueError("tracking campaign requires at least one config")
    loaded = [
        (Path(path), load_tracking_plan_config(path, input_root=input_root))
        for path in config_paths
    ]
    loaded.sort(key=lambda item: item[1].clip_id)
    clip_ids = [plan.clip_id for _, plan in loaded]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError(f"tracking campaign clip IDs must be unique: {clip_ids}")
    return tuple(loaded)


def _campaign_clip_record(
    output_root: Path,
    config_path: Path,
    plan: TrackingPlan,
    *,
    backend: str,
    revision: str,
) -> dict[str, Any]:
    clip_output = output_root / plan.clip_id
    run = load_completed_tracking_run(
        clip_output,
        expected_plan=plan,
        expected_backend=backend,
        expected_revision=revision,
    )
    run_manifest = clip_output / "run-manifest.json"
    return {
        "clip_id": plan.clip_id,
        "config_name": config_path.name,
        "config_sha256": _sha256(config_path),
        "plan_sha256": plan.sha256,
        "frame_count": len(plan.frames),
        "run_manifest_path": f"{plan.clip_id}/run-manifest.json",
        "run_manifest_sha256": _sha256(run_manifest),
        "run_created_at": run["created_at"],
    }


def _verify_campaign_manifest(
    output_root: Path,
    manifest: dict[str, Any],
    *,
    plans: tuple[tuple[Path, TrackingPlan], ...],
    backend: str,
    revision: str,
) -> None:
    if manifest.get("format") != CAMPAIGN_FORMAT:
        raise TrackingArtifactError("tracking campaign manifest has an unsupported format")
    if manifest.get("backend") != backend or manifest.get("revision") != revision:
        raise TrackingArtifactError("tracking campaign backend or revision mismatch")
    recorded = manifest.get("clips")
    if not isinstance(recorded, list) or len(recorded) != len(plans):
        raise TrackingArtifactError("tracking campaign clip inventory is incomplete")
    expected_records = [
        _campaign_clip_record(
            output_root,
            config_path,
            plan,
            backend=backend,
            revision=revision,
        )
        for config_path, plan in plans
    ]
    if recorded != expected_records:
        raise TrackingArtifactError("tracking campaign inventory does not match clip artifacts")


def load_completed_tracking_campaign(
    output_root: str | Path,
    *,
    config_paths: Sequence[str | Path],
    input_root: str | Path,
    expected_backend: str,
    expected_revision: str,
) -> dict[str, Any]:
    output = Path(output_root)
    plans = _load_plans(config_paths, input_root=input_root)
    manifest_path = output / "campaign-manifest.json"
    completion_path = output / "RUN_COMPLETE"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise TrackingArtifactError(f"tracking campaign is incomplete: {output}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrackingArtifactError(f"tracking campaign metadata is invalid: {output}") from error
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise TrackingArtifactError("tracking campaign manifest checksum mismatch")
    _verify_campaign_manifest(
        output,
        manifest,
        plans=plans,
        backend=expected_backend,
        revision=expected_revision,
    )
    return manifest


def run_tracking_campaign(
    config_paths: Sequence[str | Path],
    *,
    input_root: str | Path,
    output_root: str | Path,
    backend: str,
    revision: str,
    predictor_factory: Callable[[], Any],
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate every clip first, then reuse exactly one predictor for pending work."""

    if not backend.strip() or not revision.strip():
        raise ValueError("backend and revision must be non-empty")
    plans = _load_plans(config_paths, input_root=input_root)
    if event_callback is not None:
        event_callback(
            {
                "event": "campaign_started",
                "clip_count": len(plans),
                "frame_count": sum(len(plan.frames) for _, plan in plans),
            }
        )
    output = Path(output_root)
    completion_path = output / "RUN_COMPLETE"
    if completion_path.exists():
        completed = load_completed_tracking_campaign(
            output,
            config_paths=config_paths,
            input_root=input_root,
            expected_backend=backend,
            expected_revision=revision,
        )
        if event_callback is not None:
            event_callback(
                {
                    "event": "campaign_completed",
                    "clip_count": len(plans),
                    "resumed": True,
                }
            )
        return completed
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "campaign-manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TrackingArtifactError(f"partial campaign manifest is invalid: {manifest_path}") from error
        _verify_campaign_manifest(
            output,
            manifest,
            plans=plans,
            backend=backend,
            revision=revision,
        )
        _atomic_json(completion_path, {"manifest_sha256": _sha256(manifest_path)})
        if event_callback is not None:
            event_callback(
                {
                    "event": "campaign_completed",
                    "clip_count": len(plans),
                    "resumed": True,
                }
            )
        return manifest

    pending: list[TrackingPlan] = []
    for _, plan in plans:
        clip_output = output / plan.clip_id
        if (clip_output / "RUN_COMPLETE").exists():
            load_completed_tracking_run(
                clip_output,
                expected_plan=plan,
                expected_backend=backend,
                expected_revision=revision,
            )
        else:
            pending.append(plan)

    predictor: Any | None = None
    if pending:
        predictor = predictor_factory()
        try:
            for plan in pending:
                if event_callback is not None:
                    event_callback(
                        {
                            "event": "clip_started",
                            "clip_id": plan.clip_id,
                            "frame_count": len(plan.frames),
                        }
                    )
                run_tracking_plan(
                    predictor,
                    plan,
                    output / plan.clip_id,
                    backend=backend,
                    revision=revision,
                    event_callback=event_callback,
                )
                if event_callback is not None:
                    event_callback(
                        {
                            "event": "clip_completed",
                            "clip_id": plan.clip_id,
                            "frame_count": len(plan.frames),
                        }
                    )
        finally:
            close = getattr(predictor, "close", None)
            if callable(close):
                close()

    records = [
        _campaign_clip_record(
            output,
            config_path,
            plan,
            backend=backend,
            revision=revision,
        )
        for config_path, plan in plans
    ]
    manifest = {
        "format": CAMPAIGN_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "revision": revision,
        "clip_count": len(records),
        "frame_count": sum(record["frame_count"] for record in records),
        "predictor_load_count": 1 if pending else 0,
        "clips": records,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(completion_path, {"manifest_sha256": _sha256(manifest_path)})
    completed = load_completed_tracking_campaign(
        output,
        config_paths=config_paths,
        input_root=input_root,
        expected_backend=backend,
        expected_revision=revision,
    )
    if event_callback is not None:
        event_callback(
            {
                "event": "campaign_completed",
                "clip_count": len(plans),
                "resumed": False,
            }
        )
    return completed
