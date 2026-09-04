from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


MASK_FORMAT = "ownership-tracking-mask-v1"
RUN_FORMAT = "ownership-tracking-run-v1"
MAX_PROPAGATION_RESPONSE_FRAMES = 8
_ACTOR_TO_OBJECT_ID = {"A1": 1, "A2": 2}
_CLIP_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class TrackingArtifactError(RuntimeError):
    """Raised when an existing tracking artifact cannot be trusted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: str, *, field: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be hexadecimal") from error


def _validate_point(point: Sequence[float], *, field: str) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{field} must contain normalized x/y coordinates")
    x, y = (float(point[0]), float(point[1]))
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{field} coordinates must be finite")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError(f"{field} coordinates must be in [0, 1]")
    return x, y


@dataclass(frozen=True)
class ActorPrompt:
    """Geometry-only prompt for one actor; no appearance or action text is allowed."""

    actor_id: str
    bbox: tuple[float, float, float, float]
    positive_points: tuple[tuple[float, float], ...]
    negative_points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.actor_id not in _ACTOR_TO_OBJECT_ID:
            raise ValueError("actor_id must be A1 or A2")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain normalized x1, y1, x2, y2")
        x1, y1 = _validate_point(self.bbox[:2], field="bbox minimum")
        x2, y2 = _validate_point(self.bbox[2:], field="bbox maximum")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must have strictly increasing corners")
        if len(self.positive_points) != 2 or len(self.negative_points) != 2:
            raise ValueError("each actor requires exactly two positive and two negative points")
        positives = tuple(
            _validate_point(point, field=f"{self.actor_id} positive point")
            for point in self.positive_points
        )
        negatives = tuple(
            _validate_point(point, field=f"{self.actor_id} negative point")
            for point in self.negative_points
        )
        if len(set(positives)) != len(positives) or len(set(negatives)) != len(negatives):
            raise ValueError("positive and negative points must be distinct within each set")
        for x, y in positives:
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                raise ValueError("positive points must fall inside their actor bbox")

    def predictor_payload(self) -> dict[str, list[Any]]:
        x1, y1, x2, y2 = self.bbox
        return {
            "points": [
                [x1, y1],
                [x2, y2],
                *[list(point) for point in self.positive_points],
                *[list(point) for point in self.negative_points],
            ],
            "point_labels": [2, 3, 1, 1, 0, 0],
        }


@dataclass(frozen=True)
class FrameSpec:
    frame_index: int
    path: Path
    sha256: str
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.height < 1 or self.width < 1:
            raise ValueError("frame dimensions must be positive")
        _validate_sha256(self.sha256, field="frame sha256")


@dataclass(frozen=True)
class SeedPair:
    frame_index: int
    actors: tuple[ActorPrompt, ActorPrompt]

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("seed frame_index must be non-negative")
        if len(self.actors) != 2 or {actor.actor_id for actor in self.actors} != {"A1", "A2"}:
            raise ValueError("each seed must contain exactly one A1 and one A2 prompt")
        by_actor = {actor.actor_id: actor for actor in self.actors}
        if by_actor["A1"].bbox == by_actor["A2"].bbox:
            raise ValueError("A1 and A2 bboxes must be distinct")
        if by_actor["A1"].positive_points == by_actor["A2"].positive_points:
            raise ValueError("A1 and A2 positive points must be distinct")

    def ordered_actors(self) -> tuple[ActorPrompt, ActorPrompt]:
        by_actor = {actor.actor_id: actor for actor in self.actors}
        return by_actor["A1"], by_actor["A2"]


@dataclass(frozen=True)
class PropagationSegment:
    start_frame_idx: int
    max_frame_num_to_track: int
    reverse: bool = False

    def __post_init__(self) -> None:
        if self.start_frame_idx < 0:
            raise ValueError("propagation start_frame_idx must be non-negative")
        if not 1 <= self.max_frame_num_to_track <= MAX_PROPAGATION_RESPONSE_FRAMES:
            raise ValueError(
                "propagation segments must contain between 1 and "
                f"{MAX_PROPAGATION_RESPONSE_FRAMES} frames to bound host memory"
            )

    def frame_indices(self) -> tuple[int, ...]:
        direction = -1 if self.reverse else 1
        return tuple(
            self.start_frame_idx + direction * offset
            for offset in range(self.max_frame_num_to_track)
        )


def forward_propagation_chunks(
    frame_count: int,
    *,
    max_frames_per_response: int = MAX_PROPAGATION_RESPONSE_FRAMES,
) -> tuple[PropagationSegment, ...]:
    """Build contiguous forward chunks without retaining a whole video response."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 1 <= max_frames_per_response <= MAX_PROPAGATION_RESPONSE_FRAMES:
        raise ValueError(
            f"max_frames_per_response must be in [1, {MAX_PROPAGATION_RESPONSE_FRAMES}]"
        )
    return tuple(
        PropagationSegment(
            start_frame_idx=start,
            max_frame_num_to_track=min(max_frames_per_response, frame_count - start),
            reverse=False,
        )
        for start in range(0, frame_count, max_frames_per_response)
    )


@dataclass(frozen=True)
class TrackingPlan:
    clip_id: str
    frames: tuple[FrameSpec, ...]
    seeds: tuple[SeedPair, ...]
    propagations: tuple[PropagationSegment, ...]

    def __post_init__(self) -> None:
        if not _CLIP_ID.fullmatch(self.clip_id):
            raise ValueError("clip_id must contain only lowercase letters, digits, '_' or '-'")
        if not self.frames:
            raise ValueError("tracking plan requires frames")
        expected_indices = tuple(range(len(self.frames)))
        if tuple(frame.frame_index for frame in self.frames) != expected_indices:
            raise ValueError("frames must be ordered and use contiguous zero-based indices")
        parents = {frame.path.parent.resolve() for frame in self.frames}
        if len(parents) != 1:
            raise ValueError("all frames must be located in one directory")
        names = tuple(frame.path.name for frame in self.frames)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("frame filenames must be unique and lexically ordered")
        shapes = {(frame.height, frame.width) for frame in self.frames}
        if len(shapes) != 1:
            raise ValueError("every frame in a clip must have the same dimensions")
        if not self.seeds:
            raise ValueError("tracking plan requires at least one actor-pair seed")
        seed_indices = tuple(seed.frame_index for seed in self.seeds)
        if seed_indices != tuple(sorted(set(seed_indices))):
            raise ValueError("seed frame indices must be unique and ordered")
        if any(index >= len(self.frames) for index in seed_indices):
            raise ValueError("seed frame index is outside the clip")
        if not self.propagations:
            raise ValueError("tracking plan requires propagation segments")
        coverage: set[int] = set()
        for segment in self.propagations:
            indices = segment.frame_indices()
            if min(indices) < 0 or max(indices) >= len(self.frames):
                raise ValueError("propagation segment extends outside the clip")
            coverage.update(indices)
        missing = sorted(set(expected_indices) - coverage)
        extra = sorted(coverage - set(expected_indices))
        if missing or extra:
            raise ValueError(f"propagation coverage is incomplete: missing={missing}, extra={extra}")

    @property
    def frame_directory(self) -> Path:
        return self.frames[0].path.parent

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.frames[0].height, self.frames[0].width

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "clip_id": self.clip_id,
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "path": frame.path.name,
                    "sha256": frame.sha256,
                    "height": frame.height,
                    "width": frame.width,
                }
                for frame in self.frames
            ],
            "seeds": [
                {
                    "frame_index": seed.frame_index,
                    "actors": [asdict(actor) for actor in seed.ordered_actors()],
                }
                for seed in self.seeds
            ],
            "propagations": [asdict(segment) for segment in self.propagations],
            "prompt_contract": (
                "normalized geometry only: bbox corner labels 2/3, exactly two actor "
                "positive labels 1/1, exactly two exclusion labels 0/0"
            ),
            "max_propagation_response_frames": MAX_PROPAGATION_RESPONSE_FRAMES,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def verify_frame_files(self) -> None:
        for frame in self.frames:
            if not frame.path.is_file():
                raise FileNotFoundError(f"missing input frame: {frame.path}")
            actual = _sha256(frame.path)
            if actual != frame.sha256:
                raise TrackingArtifactError(
                    f"input frame checksum mismatch at index {frame.frame_index}: {frame.path}"
                )


def _require_exact_keys(payload: dict[str, Any], expected: set[str], *, context: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    missing = sorted(expected - set(payload))
    unexpected = sorted(set(payload) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{context} keys are invalid: missing={missing}, unexpected={unexpected}"
        )


def _relative_child(root: Path, relative: str, *, context: str) -> Path:
    candidate_value = Path(relative)
    if candidate_value.is_absolute():
        raise ValueError(f"{context} must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate_value).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{context} escapes its input root")
    return resolved


def load_tracking_plan_config(
    config_path: str | Path,
    *,
    input_root: str | Path,
) -> TrackingPlan:
    """Load a strict geometry-only plan bound to a hashed clip manifest.

    Rejecting unknown fields is intentional: appearance descriptions, action labels,
    or other semantic nudges cannot silently enter the segmentation prompt packet.
    """

    path = Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"tracking config is unreadable or invalid JSON: {path}") from error
    _require_exact_keys(
        config,
        {
            "schema_version",
            "clip_id",
            "clip_manifest_path",
            "clip_manifest_sha256",
            "seeds",
            "propagations",
        },
        context="tracking config",
    )
    if config["schema_version"] != "1.0":
        raise ValueError("tracking config schema_version must be 1.0")
    _validate_sha256(config["clip_manifest_sha256"], field="clip manifest sha256")
    manifest_path = _relative_child(
        Path(input_root),
        config["clip_manifest_path"],
        context="clip_manifest_path",
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"clip manifest is missing: {manifest_path}")
    if _sha256(manifest_path) != config["clip_manifest_sha256"]:
        raise TrackingArtifactError(f"clip manifest checksum mismatch: {manifest_path}")
    try:
        clip_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_clip_id = clip_manifest["contract"]["clip_id"]
        manifest_frames = clip_manifest["frames"]
        decoded_count = int(clip_manifest["decode"]["frame_count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"clip manifest schema is invalid: {manifest_path}") from error
    if config["clip_id"] != manifest_clip_id:
        raise ValueError("tracking config clip_id does not match clip manifest")
    if not isinstance(manifest_frames, list) or len(manifest_frames) != decoded_count:
        raise ValueError("clip manifest frame inventory does not match decoded frame count")

    frame_specs = []
    for position, frame in enumerate(manifest_frames):
        required = {"clip_frame_index", "path", "sha256", "height", "width"}
        if not isinstance(frame, dict) or not required.issubset(frame):
            raise ValueError(f"clip manifest frame {position} is incomplete")
        frame_path = _relative_child(
            manifest_path.parent,
            str(frame["path"]),
            context=f"frame {position} path",
        )
        frame_specs.append(
            FrameSpec(
                frame_index=int(frame["clip_frame_index"]),
                path=frame_path,
                sha256=str(frame["sha256"]),
                height=int(frame["height"]),
                width=int(frame["width"]),
            )
        )

    if not isinstance(config["seeds"], list):
        raise ValueError("tracking config seeds must be a list")
    seeds = []
    for seed_position, seed in enumerate(config["seeds"]):
        _require_exact_keys(
            seed,
            {"frame_index", "actors"},
            context=f"seed {seed_position}",
        )
        if not isinstance(seed["actors"], list):
            raise ValueError(f"seed {seed_position} actors must be a list")
        actors = []
        for actor_position, actor in enumerate(seed["actors"]):
            _require_exact_keys(
                actor,
                {"actor_id", "bbox", "positive_points", "negative_points"},
                context=f"seed {seed_position} actor {actor_position}",
            )
            actors.append(
                ActorPrompt(
                    actor_id=str(actor["actor_id"]),
                    bbox=tuple(actor["bbox"]),
                    positive_points=tuple(tuple(point) for point in actor["positive_points"]),
                    negative_points=tuple(tuple(point) for point in actor["negative_points"]),
                )
            )
        seeds.append(SeedPair(frame_index=int(seed["frame_index"]), actors=tuple(actors)))

    if not isinstance(config["propagations"], list):
        raise ValueError("tracking config propagations must be a list")
    propagations = []
    for position, segment in enumerate(config["propagations"]):
        _require_exact_keys(
            segment,
            {"start_frame_idx", "max_frame_num_to_track", "reverse"},
            context=f"propagation {position}",
        )
        if not isinstance(segment["reverse"], bool):
            raise ValueError(f"propagation {position} reverse must be boolean")
        propagations.append(
            PropagationSegment(
                start_frame_idx=int(segment["start_frame_idx"]),
                max_frame_num_to_track=int(segment["max_frame_num_to_track"]),
                reverse=segment["reverse"],
            )
        )
    plan = TrackingPlan(
        clip_id=str(config["clip_id"]),
        frames=tuple(frame_specs),
        seeds=tuple(seeds),
        propagations=tuple(propagations),
    )
    plan.verify_frame_files()
    return plan


@dataclass(frozen=True)
class _NormalizedFrame:
    frame_index: int
    raw_a1: np.ndarray
    raw_a2: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    score_a1: float
    score_a2: float


def _atomic_json(path: Path, value: Any) -> None:
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
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _normalize_predictor_frame(
    payload: dict[str, Any],
    *,
    expected_shape: tuple[int, int],
    frame_count: int,
) -> _NormalizedFrame:
    try:
        frame_index = int(payload["frame_index"])
        outputs = payload["outputs"]
        object_ids = np.asarray(outputs["out_obj_ids"]).astype(np.int64).reshape(-1)
        masks = np.asarray(outputs["out_binary_masks"])
        scores = np.asarray(outputs["out_probs"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("predictor frame has an invalid output schema") from error
    if not 0 <= frame_index < frame_count:
        raise ValueError(f"predictor returned out-of-range frame {frame_index}")
    if object_ids.tolist() and len(set(object_ids.tolist())) != len(object_ids):
        raise ValueError("predictor returned duplicate object IDs")
    if set(object_ids.tolist()) != {1, 2}:
        raise ValueError(f"predictor must return exactly object IDs 1 and 2, got {object_ids.tolist()}")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] != len(object_ids):
        raise ValueError("predictor masks must have shape [objects, height, width]")
    if tuple(masks.shape[1:]) != expected_shape:
        raise ValueError(
            f"predictor mask shape {tuple(masks.shape[1:])} does not match {expected_shape}"
        )
    if np.issubdtype(masks.dtype, np.number) and not np.all(np.isfinite(masks)):
        raise ValueError("predictor masks must be finite")
    if len(scores) != len(object_ids) or not np.all(np.isfinite(scores)):
        raise ValueError("predictor object scores must be finite and match object IDs")
    index_by_id = {int(object_id): index for index, object_id in enumerate(object_ids)}
    raw_a1 = np.asarray(masks[index_by_id[1]] > 0, dtype=bool)
    raw_a2 = np.asarray(masks[index_by_id[2]] > 0, dtype=bool)
    score_a1 = float(scores[index_by_id[1]])
    score_a2 = float(scores[index_by_id[2]])
    overlap = raw_a1 & raw_a2
    a1, a2 = raw_a1.copy(), raw_a2.copy()
    if score_a1 >= score_a2:
        a2 &= ~overlap
    else:
        a1 &= ~overlap
    return _NormalizedFrame(frame_index, raw_a1, raw_a2, a1, a2, score_a1, score_a2)


def _artifact_paths(output_dir: Path, frame_index: int) -> tuple[Path, Path]:
    artifact = output_dir / "masks" / f"frame_{frame_index:06d}.npz"
    return artifact, artifact.with_suffix(artifact.suffix + ".json")


def _load_mask_artifact(
    artifact: Path,
    sidecar: Path,
    *,
    expected_plan_sha256: str,
    expected_backend: str,
    expected_revision: str,
    expected_frame: FrameSpec,
) -> tuple[dict[str, Any], dict[str, np.ndarray | float]]:
    if not artifact.is_file() or not sidecar.is_file():
        raise TrackingArtifactError(f"mask artifact is incomplete: {artifact}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrackingArtifactError(f"mask sidecar is invalid: {sidecar}") from error
    expected_metadata = {
        "format": MASK_FORMAT,
        "plan_sha256": expected_plan_sha256,
        "backend": expected_backend,
        "revision": expected_revision,
        "frame_index": expected_frame.frame_index,
        "source_sha256": expected_frame.sha256,
        "shape": [expected_frame.height, expected_frame.width],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise TrackingArtifactError(
                f"mask metadata mismatch for {artifact}: {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    actual_size = artifact.stat().st_size
    if metadata.get("size_bytes") != actual_size:
        raise TrackingArtifactError(f"mask artifact size mismatch: {artifact}")
    actual_sha256 = _sha256(artifact)
    if metadata.get("sha256") != actual_sha256:
        raise TrackingArtifactError(f"mask artifact checksum mismatch: {artifact}")
    try:
        with np.load(artifact, allow_pickle=False) as arrays:
            required = {"raw_A1", "raw_A2", "A1", "A2", "score_A1", "score_A2"}
            if set(arrays.files) != required:
                raise TrackingArtifactError(f"mask artifact keys are invalid: {artifact}")
            values: dict[str, np.ndarray | float] = {
                "raw_A1": arrays["raw_A1"].astype(bool),
                "raw_A2": arrays["raw_A2"].astype(bool),
                "A1": arrays["A1"].astype(bool),
                "A2": arrays["A2"].astype(bool),
                "score_A1": float(arrays["score_A1"]),
                "score_A2": float(arrays["score_A2"]),
            }
    except TrackingArtifactError:
        raise
    except Exception as error:
        raise TrackingArtifactError(f"mask artifact payload is invalid: {artifact}") from error
    for key in ("raw_A1", "raw_A2", "A1", "A2"):
        if np.asarray(values[key]).shape != (expected_frame.height, expected_frame.width):
            raise TrackingArtifactError(f"mask array shape mismatch for {key}: {artifact}")
    if not math.isfinite(float(values["score_A1"])) or not math.isfinite(float(values["score_A2"])):
        raise TrackingArtifactError(f"mask scores are non-finite: {artifact}")
    if np.any(np.asarray(values["A1"]) & np.asarray(values["A2"])):
        raise TrackingArtifactError(f"exclusive actor masks overlap: {artifact}")
    return metadata, values


def _write_or_verify_mask(
    output_dir: Path,
    frame: _NormalizedFrame,
    *,
    plan: TrackingPlan,
    backend: str,
    revision: str,
) -> tuple[dict[str, Any], bool]:
    artifact, sidecar = _artifact_paths(output_dir, frame.frame_index)
    frame_spec = plan.frames[frame.frame_index]
    exists = artifact.exists() or sidecar.exists()
    if exists:
        metadata, values = _load_mask_artifact(
            artifact,
            sidecar,
            expected_plan_sha256=plan.sha256,
            expected_backend=backend,
            expected_revision=revision,
            expected_frame=frame_spec,
        )
        comparisons = {
            "raw_A1": frame.raw_a1,
            "raw_A2": frame.raw_a2,
            "A1": frame.a1,
            "A2": frame.a2,
        }
        if any(not np.array_equal(values[key], value) for key, value in comparisons.items()):
            raise TrackingArtifactError(f"resume output conflicts with completed artifact: {artifact}")
        if not math.isclose(float(values["score_A1"]), frame.score_a1, rel_tol=0, abs_tol=1e-7):
            raise TrackingArtifactError(f"resume A1 score conflicts with completed artifact: {artifact}")
        if not math.isclose(float(values["score_A2"]), frame.score_a2, rel_tol=0, abs_tol=1e-7):
            raise TrackingArtifactError(f"resume A2 score conflicts with completed artifact: {artifact}")
        return metadata, False

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
                raw_A1=frame.raw_a1,
                raw_A2=frame.raw_a2,
                A1=frame.a1,
                A2=frame.a2,
                score_A1=np.asarray(frame.score_a1, dtype=np.float32),
                score_A2=np.asarray(frame.score_a2, dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        metadata = {
            "format": MASK_FORMAT,
            "plan_sha256": plan.sha256,
            "backend": backend,
            "revision": revision,
            "frame_index": frame.frame_index,
            "source_sha256": frame_spec.sha256,
            "shape": [frame_spec.height, frame_spec.width],
            "size_bytes": temporary.stat().st_size,
            "sha256": _sha256(temporary),
            "scores": {"A1": frame.score_a1, "A2": frame.score_a2},
            "areas": {"A1": int(frame.a1.sum()), "A2": int(frame.a2.sum())},
            "raw_overlap_pixels": int((frame.raw_a1 & frame.raw_a2).sum()),
        }
        os.replace(temporary, artifact)
        _atomic_json(sidecar, metadata)
        return metadata, True
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _verify_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    expected_plan: TrackingPlan | None,
    expected_backend: str | None,
    expected_revision: str | None,
) -> None:
    if manifest.get("format") != RUN_FORMAT:
        raise TrackingArtifactError("tracking run manifest has an unsupported format")
    if expected_plan is not None and manifest.get("plan_sha256") != expected_plan.sha256:
        raise TrackingArtifactError("tracking run plan checksum mismatch")
    if expected_backend is not None and manifest.get("backend") != expected_backend:
        raise TrackingArtifactError("tracking run backend mismatch")
    if expected_revision is not None and manifest.get("revision") != expected_revision:
        raise TrackingArtifactError("tracking run revision mismatch")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or manifest.get("frame_count") != len(frames):
        raise TrackingArtifactError("tracking run frame inventory is invalid")
    if expected_plan is None:
        return
    if len(frames) != len(expected_plan.frames):
        raise TrackingArtifactError("tracking run frame inventory is incomplete")
    for frame_spec, recorded in zip(expected_plan.frames, frames, strict=True):
        if recorded.get("frame_index") != frame_spec.frame_index:
            raise TrackingArtifactError("tracking run frame order is invalid")
        artifact, sidecar = _artifact_paths(output_dir, frame_spec.frame_index)
        metadata, _ = _load_mask_artifact(
            artifact,
            sidecar,
            expected_plan_sha256=expected_plan.sha256,
            expected_backend=manifest["backend"],
            expected_revision=manifest["revision"],
            expected_frame=frame_spec,
        )
        if recorded.get("sha256") != metadata["sha256"] or recorded.get("size_bytes") != metadata["size_bytes"]:
            raise TrackingArtifactError(f"run inventory disagrees with mask sidecar: {artifact}")


def load_completed_tracking_run(
    output_dir: str | Path,
    *,
    expected_plan: TrackingPlan | None = None,
    expected_backend: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "run-manifest.json"
    completion_path = output / "RUN_COMPLETE"
    if not completion_path.is_file() or not manifest_path.is_file():
        raise TrackingArtifactError(f"tracking run is incomplete: {output}")
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrackingArtifactError(f"tracking run metadata is invalid: {output}") from error
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise TrackingArtifactError("tracking run manifest checksum mismatch")
    _verify_manifest(
        output,
        manifest,
        expected_plan=expected_plan,
        expected_backend=expected_backend,
        expected_revision=expected_revision,
    )
    return manifest


def _complete_existing_manifest(
    output_dir: Path,
    *,
    plan: TrackingPlan,
    backend: str,
    revision: str,
) -> dict[str, Any] | None:
    manifest_path = output_dir / "run-manifest.json"
    if not manifest_path.exists():
        return None
    if (output_dir / "RUN_COMPLETE").exists():
        return load_completed_tracking_run(
            output_dir,
            expected_plan=plan,
            expected_backend=backend,
            expected_revision=revision,
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrackingArtifactError(f"partial run manifest is invalid: {manifest_path}") from error
    _verify_manifest(
        output_dir,
        manifest,
        expected_plan=plan,
        expected_backend=backend,
        expected_revision=revision,
    )
    _atomic_json(output_dir / "RUN_COMPLETE", {"manifest_sha256": _sha256(manifest_path)})
    return manifest


def run_tracking_plan(
    predictor: Any,
    plan: TrackingPlan,
    output_dir: str | Path,
    *,
    backend: str,
    revision: str,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one bounded two-actor tracking plan with atomic, resumable outputs.

    The caller owns model construction. This module deliberately accepts an injected
    predictor and never imports SAM, Torch, or model-builder code.
    """

    if not backend.strip() or not revision.strip():
        raise ValueError("backend and revision must be non-empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    existing = _complete_existing_manifest(
        output,
        plan=plan,
        backend=backend,
        revision=revision,
    )
    if existing is not None:
        return existing
    plan.verify_frame_files()
    (output / "masks").mkdir(parents=True, exist_ok=True)
    journal_path = output / "journal.jsonl"
    records: dict[int, dict[str, Any]] = {}
    session_id: str | None = None
    try:
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(plan.frame_directory),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": False,
            }
        )
        session_id = str(response["session_id"])
        if not session_id:
            raise RuntimeError("predictor returned an empty session ID")
        if event_callback is not None:
            event_callback(
                {
                    "event": "session_started",
                    "clip_id": plan.clip_id,
                    "session_id": session_id,
                }
            )
        for seed in plan.seeds:
            for actor in seed.ordered_actors():
                predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": seed.frame_index,
                        "obj_id": _ACTOR_TO_OBJECT_ID[actor.actor_id],
                        **actor.predictor_payload(),
                        "clear_old_points": True,
                        "rel_coordinates": True,
                    }
                )
                if event_callback is not None:
                    event_callback(
                        {
                            "event": "seed_prompted",
                            "clip_id": plan.clip_id,
                            "frame_index": seed.frame_index,
                            "actor_id": actor.actor_id,
                        }
                    )
        for segment_index, segment in enumerate(plan.propagations):
            if event_callback is not None:
                event_callback(
                    {
                        "event": "segment_started",
                        "clip_id": plan.clip_id,
                        "segment_index": segment_index,
                        "start_frame_idx": segment.start_frame_idx,
                        "max_frame_num_to_track": segment.max_frame_num_to_track,
                        "reverse": segment.reverse,
                    }
                )
            response = predictor.handle_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_idx": segment.start_frame_idx,
                    "max_frame_num_to_track": segment.max_frame_num_to_track,
                    "reverse": segment.reverse,
                    "tqdm_disable": True,
                }
            )
            response_frames = response.get("frames") if isinstance(response, dict) else None
            if not isinstance(response_frames, list):
                raise ValueError("predictor propagation response must contain a frames list")
            if len(response_frames) > MAX_PROPAGATION_RESPONSE_FRAMES:
                raise ValueError("predictor exceeded the bounded propagation response size")
            for payload in response_frames:
                normalized = _normalize_predictor_frame(
                    payload,
                    expected_shape=plan.frame_shape,
                    frame_count=len(plan.frames),
                )
                metadata, created = _write_or_verify_mask(
                    output,
                    normalized,
                    plan=plan,
                    backend=backend,
                    revision=revision,
                )
                prior = records.get(normalized.frame_index)
                if prior is not None and (
                    prior["sha256"] != metadata["sha256"]
                    or prior["size_bytes"] != metadata["size_bytes"]
                ):
                    raise TrackingArtifactError(
                        f"multiple propagations disagree at frame {normalized.frame_index}"
                    )
                records[normalized.frame_index] = {
                    "frame_index": normalized.frame_index,
                    "path": f"masks/frame_{normalized.frame_index:06d}.npz",
                    "sha256": metadata["sha256"],
                    "size_bytes": metadata["size_bytes"],
                    "areas": metadata["areas"],
                    "scores": metadata["scores"],
                    "raw_overlap_pixels": metadata["raw_overlap_pixels"],
                }
                if created:
                    _append_journal(
                        journal_path,
                        {
                            "event": "mask_committed",
                            "frame_index": normalized.frame_index,
                            "segment_index": segment_index,
                            "sha256": metadata["sha256"],
                            "unix_time": datetime.now(timezone.utc).timestamp(),
                        },
                    )
            del response
            if event_callback is not None:
                event_callback(
                    {
                        "event": "segment_completed",
                        "clip_id": plan.clip_id,
                        "segment_index": segment_index,
                        "committed_frame_count": len(response_frames),
                    }
                )
    finally:
        if session_id is not None:
            predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": True,
                }
            )
            if event_callback is not None:
                event_callback(
                    {
                        "event": "session_closed",
                        "clip_id": plan.clip_id,
                        "session_id": session_id,
                    }
                )

    missing = sorted(set(range(len(plan.frames))) - set(records))
    if missing:
        raise TrackingArtifactError(f"predictor did not produce every frame: missing={missing}")
    manifest_path = output / "run-manifest.json"
    if manifest_path.exists():
        raise TrackingArtifactError(f"refusing to overwrite run manifest: {manifest_path}")
    manifest = {
        "format": RUN_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "revision": revision,
        "clip_id": plan.clip_id,
        "plan_sha256": plan.sha256,
        "plan": plan.as_dict(),
        "frame_count": len(records),
        "frames": [records[index] for index in sorted(records)],
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(output / "RUN_COMPLETE", {"manifest_sha256": _sha256(manifest_path)})
    completed = load_completed_tracking_run(
        output,
        expected_plan=plan,
        expected_backend=backend,
        expected_revision=revision,
    )
    if event_callback is not None:
        event_callback(
            {
                "event": "run_completed",
                "clip_id": plan.clip_id,
                "frame_count": len(records),
                "manifest_sha256": _sha256(manifest_path),
            }
        )
    return completed
