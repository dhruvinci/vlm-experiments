from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .mask_geometry import mask_pair_to_geometry_prompts
from .tracking import (
    TrackingPlan,
    _artifact_paths as _tracking_artifact_paths,
    _load_mask_artifact as _load_tracking_mask_artifact,
    load_completed_tracking_run,
    load_tracking_plan_config,
)


MASK_FORMAT = "ownership-sam3-image-mask-v1"
CLIP_FORMAT = "ownership-sam3-image-clip-v1"
CAMPAIGN_FORMAT = "ownership-sam3-image-campaign-v1"


class ImageAgreementArtifactError(RuntimeError):
    """Raised when a SAM3 image agreement artifact cannot be trusted."""


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


def _append_journal(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _plans(
    config_paths: Sequence[str | Path],
    *,
    input_root: str | Path,
) -> tuple[tuple[Path, TrackingPlan], ...]:
    if not config_paths:
        raise ValueError("image agreement campaign requires at least one config")
    loaded = sorted(
        (
            (Path(path), load_tracking_plan_config(path, input_root=input_root))
            for path in config_paths
        ),
        key=lambda item: item[1].clip_id,
    )
    ids = [plan.clip_id for _, plan in loaded]
    if len(ids) != len(set(ids)):
        raise ValueError(f"image agreement clip IDs must be unique: {ids}")
    return tuple(loaded)


def _prompt_record(prompts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "localization_source": "sam3.1_tracker",
        "independence_note": (
            "base SAM3 predicts contours independently, but localization is correlated "
            "because boxes and points are derived from SAM3.1 masks"
        ),
        "actors": prompts,
    }


def _normalize_prediction(
    prediction: dict[str, Any],
    *,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    required = {
        "logits_A1",
        "logits_A2",
        "raw_A1",
        "raw_A2",
        "score_A1",
        "score_A2",
        "selected_index_A1",
        "selected_index_A2",
    }
    if not isinstance(prediction, dict) or set(prediction) != required:
        raise ValueError("SAM3 image prediction has an invalid schema")
    logits_a1 = np.asarray(prediction["logits_A1"], dtype=np.float32)
    logits_a2 = np.asarray(prediction["logits_A2"], dtype=np.float32)
    raw_a1 = np.asarray(prediction["raw_A1"], dtype=bool)
    raw_a2 = np.asarray(prediction["raw_A2"], dtype=bool)
    arrays = (logits_a1, logits_a2, raw_a1, raw_a2)
    if any(value.shape != expected_shape for value in arrays):
        raise ValueError("SAM3 image prediction shape does not match the source frame")
    if not np.all(np.isfinite(logits_a1)) or not np.all(np.isfinite(logits_a2)):
        raise ValueError("SAM3 image logits must be finite")
    if not np.array_equal(raw_a1, logits_a1 > 0.0) or not np.array_equal(
        raw_a2, logits_a2 > 0.0
    ):
        raise ValueError("SAM3 image raw masks must equal positive selected logits")
    scores = (float(prediction["score_A1"]), float(prediction["score_A2"]))
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("SAM3 image scores must be finite")
    selected = (
        int(prediction["selected_index_A1"]),
        int(prediction["selected_index_A2"]),
    )
    if any(value < 0 for value in selected):
        raise ValueError("SAM3 selected candidate indices must be non-negative")

    overlap = raw_a1 & raw_a2
    a1 = raw_a1 & (~overlap | (logits_a1 > logits_a2))
    a2 = raw_a2 & (~overlap | (logits_a2 > logits_a1))
    ties = overlap & (logits_a1 == logits_a2)
    return {
        "raw_A1": raw_a1,
        "raw_A2": raw_a2,
        "A1": a1,
        "A2": a2,
        "score_A1": scores[0],
        "score_A2": scores[1],
        "selected_index_A1": selected[0],
        "selected_index_A2": selected[1],
        "raw_overlap_pixels": int(np.count_nonzero(overlap)),
        "unresolved_tie_pixels": int(np.count_nonzero(ties)),
    }


def _artifact_paths(output_dir: Path, frame_index: int) -> tuple[Path, Path]:
    artifact = output_dir / "masks" / f"frame_{frame_index:06d}.npz"
    return artifact, artifact.with_suffix(".npz.json")


def _load_artifact(
    output_dir: Path,
    *,
    plan: TrackingPlan,
    frame_index: int,
    backend: str,
    revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, sidecar = _artifact_paths(output_dir, frame_index)
    if not artifact.is_file() or not sidecar.is_file():
        raise ImageAgreementArtifactError(f"image agreement artifact is incomplete: {artifact}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImageAgreementArtifactError(f"image agreement sidecar is invalid: {sidecar}") from error
    frame = plan.frames[frame_index]
    expected = {
        "format": MASK_FORMAT,
        "plan_sha256": plan.sha256,
        "backend": backend,
        "revision": revision,
        "frame_index": frame_index,
        "source_sha256": frame.sha256,
        "shape": [frame.height, frame.width],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ImageAgreementArtifactError(
                f"image agreement metadata mismatch for {artifact}: {key}"
            )
    if metadata.get("size_bytes") != artifact.stat().st_size:
        raise ImageAgreementArtifactError(f"image agreement size mismatch: {artifact}")
    if metadata.get("sha256") != _sha256(artifact):
        raise ImageAgreementArtifactError(f"image agreement checksum mismatch: {artifact}")
    prompt = metadata.get("prompt")
    if not isinstance(prompt, dict) or prompt.get("localization_source") != "sam3.1_tracker":
        raise ImageAgreementArtifactError(f"image agreement prompt provenance is invalid: {artifact}")
    if metadata.get("prompt_sha256") != _canonical_sha256(prompt):
        raise ImageAgreementArtifactError(f"image agreement prompt checksum mismatch: {artifact}")
    try:
        with np.load(artifact, allow_pickle=False) as arrays:
            required = {
                "raw_A1",
                "raw_A2",
                "A1",
                "A2",
                "score_A1",
                "score_A2",
                "selected_index_A1",
                "selected_index_A2",
            }
            if set(arrays.files) != required:
                raise ImageAgreementArtifactError(
                    f"image agreement array keys are invalid: {artifact}"
                )
            values = {name: arrays[name].copy() for name in arrays.files}
    except ImageAgreementArtifactError:
        raise
    except Exception as error:
        raise ImageAgreementArtifactError(f"image agreement payload is invalid: {artifact}") from error
    for name in ("raw_A1", "raw_A2", "A1", "A2"):
        if values[name].shape != plan.frame_shape:
            raise ImageAgreementArtifactError(
                f"image agreement {name} shape is invalid: {artifact}"
            )
    if np.any(values["A1"].astype(bool) & values["A2"].astype(bool)):
        raise ImageAgreementArtifactError(f"image agreement actor masks overlap: {artifact}")
    return metadata, values


def _write_artifact(
    output_dir: Path,
    *,
    plan: TrackingPlan,
    frame_index: int,
    backend: str,
    revision: str,
    tracker_metadata: dict[str, Any],
    prompt: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    artifact, sidecar = _artifact_paths(output_dir, frame_index)
    if artifact.exists() or sidecar.exists():
        raise ImageAgreementArtifactError(
            f"refusing to overwrite image agreement artifact: {artifact}"
        )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=artifact.parent,
            prefix=f".{artifact.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                raw_A1=prediction["raw_A1"],
                raw_A2=prediction["raw_A2"],
                A1=prediction["A1"],
                A2=prediction["A2"],
                score_A1=np.asarray(prediction["score_A1"], dtype=np.float32),
                score_A2=np.asarray(prediction["score_A2"], dtype=np.float32),
                selected_index_A1=np.asarray(
                    prediction["selected_index_A1"], dtype=np.int16
                ),
                selected_index_A2=np.asarray(
                    prediction["selected_index_A2"], dtype=np.int16
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        frame = plan.frames[frame_index]
        metadata = {
            "format": MASK_FORMAT,
            "plan_sha256": plan.sha256,
            "backend": backend,
            "revision": revision,
            "frame_index": frame_index,
            "source_sha256": frame.sha256,
            "shape": [frame.height, frame.width],
            "tracker_mask_sha256": tracker_metadata["sha256"],
            "prompt": prompt,
            "prompt_sha256": _canonical_sha256(prompt),
            "size_bytes": temporary.stat().st_size,
            "sha256": _sha256(temporary),
            "scores": {
                "A1": prediction["score_A1"],
                "A2": prediction["score_A2"],
            },
            "selected_indices": {
                "A1": prediction["selected_index_A1"],
                "A2": prediction["selected_index_A2"],
            },
            "areas": {
                "A1": int(np.count_nonzero(prediction["A1"])),
                "A2": int(np.count_nonzero(prediction["A2"])),
            },
            "raw_overlap_pixels": prediction["raw_overlap_pixels"],
            "unresolved_tie_pixels": prediction["unresolved_tie_pixels"],
        }
        os.replace(temporary, artifact)
        _atomic_json(sidecar, metadata)
        return metadata
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _record(metadata: dict[str, Any]) -> dict[str, Any]:
    frame_index = int(metadata["frame_index"])
    return {
        "frame_index": frame_index,
        "path": f"masks/frame_{frame_index:06d}.npz",
        "sha256": metadata["sha256"],
        "size_bytes": metadata["size_bytes"],
        "tracker_mask_sha256": metadata["tracker_mask_sha256"],
        "prompt_sha256": metadata["prompt_sha256"],
        "areas": metadata["areas"],
        "scores": metadata["scores"],
        "raw_overlap_pixels": metadata["raw_overlap_pixels"],
        "unresolved_tie_pixels": metadata["unresolved_tie_pixels"],
    }


def _verify_clip_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    plan: TrackingPlan,
    backend: str,
    revision: str,
) -> None:
    if manifest.get("format") != CLIP_FORMAT:
        raise ImageAgreementArtifactError("image agreement clip format is invalid")
    if manifest.get("backend") != backend or manifest.get("revision") != revision:
        raise ImageAgreementArtifactError("image agreement clip backend mismatch")
    if manifest.get("plan_sha256") != plan.sha256:
        raise ImageAgreementArtifactError("image agreement clip plan mismatch")
    records = manifest.get("frames")
    if not isinstance(records, list) or len(records) != len(plan.frames):
        raise ImageAgreementArtifactError("image agreement clip inventory is incomplete")
    expected = []
    for frame in plan.frames:
        metadata, _ = _load_artifact(
            output_dir,
            plan=plan,
            frame_index=frame.frame_index,
            backend=backend,
            revision=revision,
        )
        expected.append(_record(metadata))
    if records != expected:
        raise ImageAgreementArtifactError("image agreement clip inventory disagrees with artifacts")


def _load_clip(
    output_dir: Path,
    *,
    plan: TrackingPlan,
    backend: str,
    revision: str,
) -> dict[str, Any]:
    manifest_path = output_dir / "run-manifest.json"
    completion_path = output_dir / "RUN_COMPLETE"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise ImageAgreementArtifactError(f"image agreement clip is incomplete: {output_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImageAgreementArtifactError(f"image agreement clip metadata is invalid: {output_dir}") from error
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise ImageAgreementArtifactError("image agreement clip manifest checksum mismatch")
    _verify_clip_manifest(
        output_dir,
        manifest,
        plan=plan,
        backend=backend,
        revision=revision,
    )
    return manifest


def _run_clip(
    predictor: Any,
    plan: TrackingPlan,
    *,
    tracker_dir: Path,
    output_dir: Path,
    tracker_backend: str,
    tracker_revision: str,
    backend: str,
    revision: str,
    minimum_prompt_area: int,
    box_padding_fraction: float,
    event_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    if (output_dir / "RUN_COMPLETE").exists():
        return _load_clip(
            output_dir,
            plan=plan,
            backend=backend,
            revision=revision,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run-manifest.json"
    policy = {
        "minimum_prompt_area": minimum_prompt_area,
        "box_padding_fraction": box_padding_fraction,
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ImageAgreementArtifactError(
                f"partial image agreement manifest is invalid: {manifest_path}"
            ) from error
        if existing.get("prompt_policy") != policy:
            raise ImageAgreementArtifactError("image agreement prompt policy mismatch")
        _verify_clip_manifest(
            output_dir,
            existing,
            plan=plan,
            backend=backend,
            revision=revision,
        )
        _atomic_json(output_dir / "RUN_COMPLETE", {"manifest_sha256": _sha256(manifest_path)})
        return existing

    tracker_run = load_completed_tracking_run(
        tracker_dir,
        expected_plan=plan,
        expected_backend=tracker_backend,
        expected_revision=tracker_revision,
    )
    records: dict[int, dict[str, Any]] = {}
    for frame in plan.frames:
        artifact, sidecar = _artifact_paths(output_dir, frame.frame_index)
        if artifact.exists() or sidecar.exists():
            metadata, _ = _load_artifact(
                output_dir,
                plan=plan,
                frame_index=frame.frame_index,
                backend=backend,
                revision=revision,
            )
            records[frame.frame_index] = _record(metadata)
            continue
        tracking_artifact, tracking_sidecar = _tracking_artifact_paths(
            tracker_dir, frame.frame_index
        )
        tracker_metadata, tracker_values = _load_tracking_mask_artifact(
            tracking_artifact,
            tracking_sidecar,
            expected_plan_sha256=plan.sha256,
            expected_backend=tracker_backend,
            expected_revision=tracker_revision,
            expected_frame=frame,
        )
        prompts = mask_pair_to_geometry_prompts(
            tracker_values["A1"],
            tracker_values["A2"],
            minimum_area=minimum_prompt_area,
            box_padding_fraction=box_padding_fraction,
        )
        prompt = _prompt_record(prompts)
        if event_callback is not None:
            event_callback(
                {
                    "event": "image_agreement_frame_started",
                    "clip_id": plan.clip_id,
                    "frame_index": frame.frame_index,
                }
            )
        raw_prediction = predictor.segment(
            frame.path,
            prompts,
            expected_shape=plan.frame_shape,
        )
        prediction = _normalize_prediction(
            raw_prediction,
            expected_shape=plan.frame_shape,
        )
        metadata = _write_artifact(
            output_dir,
            plan=plan,
            frame_index=frame.frame_index,
            backend=backend,
            revision=revision,
            tracker_metadata=tracker_metadata,
            prompt=prompt,
            prediction=prediction,
        )
        records[frame.frame_index] = _record(metadata)
        _append_journal(
            output_dir / "journal.jsonl",
            {
                "event": "image_agreement_mask_committed",
                "frame_index": frame.frame_index,
                "sha256": metadata["sha256"],
                "unix_time": datetime.now(timezone.utc).timestamp(),
            },
        )
        if event_callback is not None:
            event_callback(
                {
                    "event": "image_agreement_frame_completed",
                    "clip_id": plan.clip_id,
                    "frame_index": frame.frame_index,
                    "raw_overlap_pixels": metadata["raw_overlap_pixels"],
                    "unresolved_tie_pixels": metadata["unresolved_tie_pixels"],
                }
            )
        raw_prediction = None
        prediction = None
        tracker_values = None

    if set(records) != set(range(len(plan.frames))):
        raise ImageAgreementArtifactError("image agreement did not produce every frame")
    manifest = {
        "format": CLIP_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "revision": revision,
        "clip_id": plan.clip_id,
        "plan_sha256": plan.sha256,
        "tracker_backend": tracker_backend,
        "tracker_revision": tracker_revision,
        "tracker_manifest_sha256": _sha256(tracker_dir / "run-manifest.json"),
        "localization_dependency": "sam3.1-tracker",
        "prompt_policy": policy,
        "frame_count": len(records),
        "frames": [records[index] for index in sorted(records)],
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(output_dir / "RUN_COMPLETE", {"manifest_sha256": _sha256(manifest_path)})
    return _load_clip(
        output_dir,
        plan=plan,
        backend=backend,
        revision=revision,
    )


def _campaign_record(
    output_root: Path,
    config_path: Path,
    plan: TrackingPlan,
    *,
    backend: str,
    revision: str,
) -> dict[str, Any]:
    manifest = _load_clip(
        output_root / plan.clip_id,
        plan=plan,
        backend=backend,
        revision=revision,
    )
    manifest_path = output_root / plan.clip_id / "run-manifest.json"
    return {
        "clip_id": plan.clip_id,
        "config_name": config_path.name,
        "config_sha256": _sha256(config_path),
        "plan_sha256": plan.sha256,
        "frame_count": manifest["frame_count"],
        "run_manifest_path": f"{plan.clip_id}/run-manifest.json",
        "run_manifest_sha256": _sha256(manifest_path),
    }


def load_completed_image_agreement_campaign(
    output_root: str | Path,
    *,
    config_paths: Sequence[str | Path],
    input_root: str | Path,
    tracker_root: str | Path,
    tracker_backend: str,
    tracker_revision: str,
    expected_backend: str,
    expected_revision: str,
) -> dict[str, Any]:
    output = Path(output_root)
    plans = _plans(config_paths, input_root=input_root)
    manifest_path = output / "campaign-manifest.json"
    completion_path = output / "RUN_COMPLETE"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise ImageAgreementArtifactError(f"image agreement campaign is incomplete: {output}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImageAgreementArtifactError("image agreement campaign metadata is invalid") from error
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise ImageAgreementArtifactError("image agreement campaign manifest checksum mismatch")
    if manifest.get("format") != CAMPAIGN_FORMAT:
        raise ImageAgreementArtifactError("image agreement campaign format is invalid")
    if manifest.get("backend") != expected_backend or manifest.get("revision") != expected_revision:
        raise ImageAgreementArtifactError("image agreement campaign backend mismatch")
    if manifest.get("tracker_backend") != tracker_backend or manifest.get(
        "tracker_revision"
    ) != tracker_revision:
        raise ImageAgreementArtifactError("image agreement tracker dependency mismatch")
    tracker_output = Path(tracker_root)
    for _, plan in plans:
        source = load_completed_tracking_run(
            tracker_output / plan.clip_id,
            expected_plan=plan,
            expected_backend=tracker_backend,
            expected_revision=tracker_revision,
        )
        clip_manifest = output / plan.clip_id / "run-manifest.json"
        try:
            recorded_source_hash = json.loads(
                clip_manifest.read_text(encoding="utf-8")
            ).get("tracker_manifest_sha256")
        except (OSError, json.JSONDecodeError) as error:
            raise ImageAgreementArtifactError(
                f"image agreement clip metadata is invalid: {clip_manifest}"
            ) from error
        if recorded_source_hash != _sha256(
            tracker_output / plan.clip_id / "run-manifest.json"
        ):
            raise ImageAgreementArtifactError(
                f"image agreement source lineage changed for {plan.clip_id}"
            )
        del source
    records = [
        _campaign_record(
            output,
            config_path,
            plan,
            backend=expected_backend,
            revision=expected_revision,
        )
        for config_path, plan in plans
    ]
    if manifest.get("clips") != records:
        raise ImageAgreementArtifactError("image agreement campaign inventory mismatch")
    return manifest


def run_image_agreement_campaign(
    config_paths: Sequence[str | Path],
    *,
    input_root: str | Path,
    tracker_root: str | Path,
    output_root: str | Path,
    tracker_backend: str,
    tracker_revision: str,
    backend: str,
    revision: str,
    predictor_factory: Callable[[], Any],
    minimum_prompt_area: int = 64,
    box_padding_fraction: float = 0.01,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run resumable, frame-at-a-time SAM3 contour agreement for every clip."""

    if not backend.strip() or not revision.strip():
        raise ValueError("image agreement backend and revision must be non-empty")
    if minimum_prompt_area < 2:
        raise ValueError("minimum_prompt_area must be at least two pixels")
    plans = _plans(config_paths, input_root=input_root)
    tracker_output = Path(tracker_root)
    for _, plan in plans:
        load_completed_tracking_run(
            tracker_output / plan.clip_id,
            expected_plan=plan,
            expected_backend=tracker_backend,
            expected_revision=tracker_revision,
        )
    output = Path(output_root)
    if (output / "RUN_COMPLETE").exists():
        return load_completed_image_agreement_campaign(
            output,
            config_paths=config_paths,
            input_root=input_root,
            tracker_root=tracker_root,
            tracker_backend=tracker_backend,
            tracker_revision=tracker_revision,
            expected_backend=backend,
            expected_revision=revision,
        )
    output.mkdir(parents=True, exist_ok=True)
    pending = [
        plan for _, plan in plans if not (output / plan.clip_id / "RUN_COMPLETE").exists()
    ]
    predictor: Any | None = None
    if pending:
        predictor = predictor_factory()
        try:
            for plan in pending:
                if event_callback is not None:
                    event_callback(
                        {
                            "event": "image_agreement_clip_started",
                            "clip_id": plan.clip_id,
                            "frame_count": len(plan.frames),
                        }
                    )
                _run_clip(
                    predictor,
                    plan,
                    tracker_dir=tracker_output / plan.clip_id,
                    output_dir=output / plan.clip_id,
                    tracker_backend=tracker_backend,
                    tracker_revision=tracker_revision,
                    backend=backend,
                    revision=revision,
                    minimum_prompt_area=minimum_prompt_area,
                    box_padding_fraction=box_padding_fraction,
                    event_callback=event_callback,
                )
        finally:
            close = getattr(predictor, "close", None)
            if callable(close):
                close()

    records = [
        _campaign_record(
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
        "tracker_backend": tracker_backend,
        "tracker_revision": tracker_revision,
        "localization_dependency": "sam3.1-tracker",
        "clip_count": len(records),
        "frame_count": sum(record["frame_count"] for record in records),
        "predictor_load_count": 1 if pending else 0,
        "clips": records,
    }
    manifest_path = output / "campaign-manifest.json"
    if manifest_path.exists():
        raise ImageAgreementArtifactError(
            f"refusing to overwrite image agreement campaign manifest: {manifest_path}"
        )
    _atomic_json(manifest_path, manifest)
    _atomic_json(output / "RUN_COMPLETE", {"manifest_sha256": _sha256(manifest_path)})
    completed = load_completed_image_agreement_campaign(
        output,
        config_paths=config_paths,
        input_root=input_root,
        tracker_root=tracker_root,
        tracker_backend=tracker_backend,
        tracker_revision=tracker_revision,
        expected_backend=backend,
        expected_revision=revision,
    )
    if event_callback is not None:
        event_callback(
            {
                "event": "image_agreement_campaign_completed",
                "clip_count": len(records),
                "frame_count": completed["frame_count"],
            }
        )
    return completed
