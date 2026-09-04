from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .labels import (
    LABEL_A1,
    LABEL_A2,
    LABEL_BACKGROUND,
    LABEL_IGNORE,
    apply_reviewed_contact_ownership,
    build_agreement_ownership_labels,
    label_summary,
    validate_label_ready,
)
from .image_agreement import load_completed_image_agreement_campaign
from .tracking import load_tracking_plan_config
from .tracking_campaign import load_completed_tracking_campaign


REVIEW_ATTESTATION = (
    "I visually reviewed actor identity and every marked contact region against "
    "the source frame."
)
_ACTOR_COLORS = {
    LABEL_A1: np.asarray((0, 220, 255), dtype=np.float32),
    LABEL_A2: np.asarray((255, 55, 210), dtype=np.float32),
}


@dataclass(frozen=True)
class CandidateFrameInput:
    clip_id: str
    frame_index: int
    source_path: Path
    source_sha256: str
    tracker_mask_path: Path
    tracker_mask_sha256: str
    image_mask_path: Path
    image_mask_sha256: str
    qwen_spatial_path: Path
    qwen_spatial_sha256: str
    output_hw: tuple[int, int]


def read_qwen_full_grid_hw(path: str | Path) -> tuple[int, int]:
    """Read only the tiny safetensors header and ``grid_thw`` payload.

    This deliberately avoids importing PyTorch or materializing the multi-megabyte
    hidden tensor while assembling a label package.
    """

    artifact = Path(path)
    try:
        file_size = artifact.stat().st_size
        with artifact.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("safetensors header length is truncated")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length < 2 or header_length > 100 * 1024 * 1024:
                raise ValueError("safetensors header length is invalid")
            encoded = handle.read(header_length)
            if len(encoded) != header_length:
                raise ValueError("safetensors header is truncated")
            try:
                header = json.loads(encoded.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("safetensors header JSON is invalid") from error
            grid = header.get("grid_thw")
            hidden = header.get("hidden")
            metadata = header.get("__metadata__")
            if not isinstance(grid, dict) or not isinstance(hidden, dict):
                raise ValueError("Qwen artifact is missing grid_thw or hidden metadata")
            if grid.get("dtype") != "I64" or grid.get("shape") != [1, 3]:
                raise ValueError("Qwen grid_thw contract is invalid")
            grid_offsets = grid.get("data_offsets")
            if (
                not isinstance(grid_offsets, list)
                or len(grid_offsets) != 2
                or int(grid_offsets[1]) - int(grid_offsets[0]) != 24
            ):
                raise ValueError("Qwen grid_thw byte range is invalid")
            data_start = 8 + header_length
            grid_start = data_start + int(grid_offsets[0])
            grid_end = data_start + int(grid_offsets[1])
            if grid_start < data_start or grid_end > file_size:
                raise ValueError("Qwen grid_thw byte range escapes the artifact")
            handle.seek(grid_start)
            raw_grid = handle.read(24)
            if len(raw_grid) != 24:
                raise ValueError("Qwen grid_thw payload is truncated")
    except OSError as error:
        raise ValueError(f"could not read Qwen spatial artifact: {artifact}") from error

    if not isinstance(metadata, dict) or "campaign" not in metadata:
        raise ValueError("Qwen spatial artifact is missing campaign metadata")
    try:
        campaign = json.loads(str(metadata["campaign"]))
    except json.JSONDecodeError as error:
        raise ValueError("Qwen spatial campaign metadata is invalid") from error
    if campaign.get("stage") != "spatial_full":
        raise ValueError("Qwen spatial artifact is not a full-resolution vision layer")
    temporal, height, width = struct.unpack("<qqq", raw_grid)
    if temporal != 1 or height < 1 or width < 1:
        raise ValueError("Qwen spatial grid values are invalid")
    if hidden.get("dtype") != "BF16" or hidden.get("shape") != [height * width, 1152]:
        raise ValueError("Qwen hidden tensor contract disagrees with grid_thw")
    hidden_offsets = hidden.get("data_offsets")
    expected_hidden_bytes = height * width * 1152 * 2
    if (
        not isinstance(hidden_offsets, list)
        or len(hidden_offsets) != 2
        or int(hidden_offsets[1]) - int(hidden_offsets[0]) != expected_hidden_bytes
        or data_start + int(hidden_offsets[1]) > file_size
    ):
        raise ValueError("Qwen hidden tensor byte range is invalid")
    return int(height), int(width)


def collect_breadth_candidate_inputs(
    config_paths: Sequence[str | Path],
    *,
    input_root: str | Path,
    mask_campaign_root: str | Path,
    qwen_breadth_root: str | Path,
    spatial_layer: int = 11,
) -> tuple[CandidateFrameInput, ...]:
    """Join two verified mask campaigns to the matching cached Qwen frame grids."""

    if not config_paths:
        raise ValueError("candidate collection requires at least one tracking config")
    if spatial_layer < 0 or spatial_layer > 26:
        raise ValueError("Qwen spatial layer must be in [0, 26]")
    campaign_root = Path(mask_campaign_root)
    manifest_path = campaign_root / "campaign-manifest.json"
    completion_path = campaign_root / "RUN_COMPLETE"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"remote mask campaign is incomplete: {campaign_root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("remote mask campaign metadata is invalid") from error
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("remote mask campaign manifest SHA-256 mismatch")
    if manifest.get("format") != "ownership-remote-mask-campaign-v1":
        raise ValueError("remote mask campaign format is invalid")
    tracker_revision = str(manifest.get("tracker_revision", ""))
    image_revision = str(manifest.get("image_revision", ""))
    if not tracker_revision or not image_revision:
        raise ValueError("remote mask campaign revisions are missing")

    configs = tuple(Path(value) for value in config_paths)
    plans = tuple(
        sorted(
            (
                load_tracking_plan_config(path, input_root=input_root)
                for path in configs
            ),
            key=lambda plan: plan.clip_id,
        )
    )
    clip_ids = [plan.clip_id for plan in plans]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("candidate collection clip IDs are duplicated")
    expected_frame_count = sum(len(plan.frames) for plan in plans)
    if manifest.get("clip_count") not in (None, len(plans)) or manifest.get(
        "frame_count"
    ) not in (None, expected_frame_count):
        raise ValueError("remote mask campaign inventory does not match tracking plans")

    tracker_root = campaign_root / "sam31-tracking"
    image_root = campaign_root / "sam3-image-agreement"
    load_completed_tracking_campaign(
        tracker_root,
        config_paths=configs,
        input_root=input_root,
        expected_backend="sam3.1-tracker-only",
        expected_revision=tracker_revision,
    )
    load_completed_image_agreement_campaign(
        image_root,
        config_paths=configs,
        input_root=input_root,
        tracker_root=tracker_root,
        tracker_backend="sam3.1-tracker-only",
        tracker_revision=tracker_revision,
        expected_backend="sam3-tracker-image-pvs",
        expected_revision=image_revision,
    )

    qwen_root = Path(qwen_breadth_root)
    values: list[CandidateFrameInput] = []
    for plan in plans:
        for frame in plan.frames:
            filename = f"frame_{frame.frame_index:06d}"
            tracker_mask = tracker_root / plan.clip_id / "masks" / f"{filename}.npz"
            image_mask = image_root / plan.clip_id / "masks" / f"{filename}.npz"
            qwen_spatial = (
                qwen_root
                / plan.clip_id
                / "spatial"
                / "full"
                / f"layer_{spatial_layer:02d}"
                / f"{filename}.safetensors"
            )
            if not tracker_mask.is_file() or not image_mask.is_file():
                raise FileNotFoundError(
                    f"verified campaign is missing a joined mask artifact: {plan.clip_id}/{filename}"
                )
            output_hw = read_qwen_full_grid_hw(qwen_spatial)
            values.append(
                CandidateFrameInput(
                    clip_id=plan.clip_id,
                    frame_index=frame.frame_index,
                    source_path=frame.path,
                    source_sha256=frame.sha256,
                    tracker_mask_path=tracker_mask,
                    tracker_mask_sha256=_sha256(tracker_mask),
                    image_mask_path=image_mask,
                    image_mask_sha256=_sha256(image_mask),
                    qwen_spatial_path=qwen_spatial,
                    qwen_spatial_sha256=_sha256(qwen_spatial),
                    output_hw=output_hw,
                )
            )
    return tuple(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"could not hash artifact: {path}") from error
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, *, role: str) -> None:
    if len(expected) != 64:
        raise ValueError(f"{role} SHA-256 is malformed: {path}")
    try:
        int(expected, 16)
    except ValueError as error:
        raise ValueError(f"{role} SHA-256 is malformed: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"{role} artifact is missing: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{role} SHA-256 mismatch for {path}: expected {expected}, got {observed}"
        )


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


def _atomic_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            image.save(handle, format="PNG", optimize=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_relative(root: Path, value: Any, *, role: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{role} path is unsafe: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{role} path escapes its package: {relative}")
    return resolved


def _safe_package_reference(
    base: Path,
    package_root: Path,
    value: Any,
    *,
    role: str,
) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"{role} path is unsafe: {relative}")
    resolved = (base / relative).resolve()
    if not resolved.is_relative_to(package_root.resolve()):
        raise ValueError(f"{role} path escapes its package: {relative}")
    return resolved


def _load_actor_masks(path: Path, expected_sha256: str, *, role: str) -> tuple[np.ndarray, np.ndarray]:
    _verify_hash(path, expected_sha256, role=role)
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if not {"A1", "A2"}.issubset(artifact.files):
                raise ValueError(f"{role} mask artifact must contain A1 and A2: {path}")
            a1 = np.asarray(artifact["A1"], dtype=bool)
            a2 = np.asarray(artifact["A2"], dtype=bool)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"could not decode {role} mask artifact: {path}") from error
    if a1.ndim != 2 or a2.shape != a1.shape:
        raise ValueError(f"{role} actor masks must have one identical 2D shape: {path}")
    if np.any(a1 & a2):
        raise ValueError(f"{role} actor masks overlap: {path}")
    return a1, a2


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1 or radius % 2 == 0:
        raise ValueError("dilation radius must be a positive odd integer")
    if radius == 1:
        return mask.copy()
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(radius))) > 0


def _resize_fraction(mask: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    height, width = output_hw
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    return np.asarray(
        image.resize((width, height), Image.Resampling.BOX),
        dtype=np.float32,
    )


def _candidate_arrays(
    tracker_a1: np.ndarray,
    tracker_a2: np.ndarray,
    image_a1: np.ndarray,
    image_a2: np.ndarray,
    *,
    output_hw: tuple[int, int],
    dilation_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    if output_hw[0] < 1 or output_hw[1] < 1:
        raise ValueError("candidate output dimensions must be positive")
    arrays = tuple(np.asarray(value, dtype=bool) for value in (
        tracker_a1,
        tracker_a2,
        image_a1,
        image_a2,
    ))
    if arrays[0].ndim != 2 or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("candidate source masks must have one identical 2D shape")
    t_a1, t_a2, i_a1, i_a2 = arrays
    actor_union = t_a1 | t_a2 | i_a1 | i_a2
    possible_foreground = _dilate(actor_union, dilation_radius)
    labels = build_agreement_ownership_labels(
        t_a1,
        t_a2,
        i_a1,
        i_a2,
        possible_foreground=possible_foreground,
        output_hw=output_hw,
        erosion_radius=dilation_radius,
    )

    any_a1 = t_a1 | i_a1
    any_a2 = t_a2 | i_a2
    contact_pixels = (any_a1 & _dilate(any_a2, dilation_radius)) | (
        any_a2 & _dilate(any_a1, dilation_radius)
    )
    contact_cells = _resize_fraction(contact_pixels, output_hw) > 0.0
    proposal = np.zeros(output_hw, dtype=np.uint8)
    proposal[contact_cells] = LABEL_IGNORE
    proposal[contact_cells & (labels == LABEL_A1)] = LABEL_A1
    proposal[contact_cells & (labels == LABEL_A2)] = LABEL_A2
    return labels, proposal


def _mask_overlay(
    source: Image.Image,
    a1: np.ndarray,
    a2: np.ndarray,
    *,
    alpha: float = 0.46,
) -> Image.Image:
    base = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    for label, mask in ((LABEL_A1, a1), (LABEL_A2, a2)):
        if mask.shape != (source.height, source.width):
            resized = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                source.size,
                Image.Resampling.NEAREST,
            )
            selected = np.asarray(resized) > 0
        else:
            selected = mask
        base[selected] = (1.0 - alpha) * base[selected] + alpha * _ACTOR_COLORS[label]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def _labeled_panel(image: Image.Image, title: str, width: int) -> Image.Image:
    scale = width / image.width
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    header = 24
    panel = Image.new("RGB", (width, height + header), "black")
    panel.paste(resized, (0, header))
    ImageDraw.Draw(panel).text((6, 6), title, fill="white", font=ImageFont.load_default())
    return panel


def _render_preview(
    source: Image.Image,
    tracker: tuple[np.ndarray, np.ndarray],
    image_masks: tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    proposal: np.ndarray,
    *,
    width: int,
) -> Image.Image:
    candidate_a1 = labels == LABEL_A1
    candidate_a2 = labels == LABEL_A2
    candidate = _mask_overlay(source, candidate_a1, candidate_a2)
    contact = proposal != LABEL_BACKGROUND
    if np.any(contact):
        contact_full = Image.fromarray(contact.astype(np.uint8) * 255, mode="L").resize(
            source.size,
            Image.Resampling.NEAREST,
        )
        candidate_pixels = np.asarray(candidate, dtype=np.float32).copy()
        selected = np.asarray(contact_full) > 0
        yellow = np.asarray((255, 205, 0), dtype=np.float32)
        candidate_pixels[selected] = 0.70 * candidate_pixels[selected] + 0.30 * yellow
        candidate = Image.fromarray(np.clip(candidate_pixels, 0, 255).astype(np.uint8), mode="RGB")
    panels = [
        _labeled_panel(source.convert("RGB"), "source", width),
        _labeled_panel(_mask_overlay(source, *tracker), "SAM3.1 temporal", width),
        _labeled_panel(_mask_overlay(source, *image_masks), "base SAM3 image", width),
        _labeled_panel(candidate, "agreement + yellow contact-review band", width),
    ]
    canvas = Image.new(
        "RGB",
        (2 * width, panels[0].height + panels[2].height),
        "black",
    )
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (width, 0))
    canvas.paste(panels[2], (0, panels[0].height))
    canvas.paste(panels[3], (width, panels[0].height))
    for panel in panels:
        panel.close()
    candidate.close()
    return canvas


def write_candidate_review_package(
    inputs: Sequence[CandidateFrameInput],
    output_root: str | Path,
    *,
    preview_width: int = 480,
    dilation_radius: int = 31,
) -> dict[str, Any]:
    """Build an immutable, explicitly non-training label-review package."""

    if not inputs:
        raise ValueError("candidate review package requires at least one frame")
    if preview_width < 64:
        raise ValueError("preview width must be at least 64 pixels")
    identities = [(value.clip_id, value.frame_index) for value in inputs]
    if len(set(identities)) != len(identities):
        raise ValueError("candidate frame identities must be unique")
    if any(not value.clip_id.strip() or value.frame_index < 0 for value in inputs):
        raise ValueError("candidate clip IDs and frame indices are invalid")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate review package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    committed = False
    try:
        records: list[dict[str, Any]] = []
        for item in sorted(inputs, key=lambda value: (value.clip_id, value.frame_index)):
            _verify_hash(item.source_path, item.source_sha256, role="source frame")
            _verify_hash(item.qwen_spatial_path, item.qwen_spatial_sha256, role="Qwen spatial")
            tracker = _load_actor_masks(
                item.tracker_mask_path,
                item.tracker_mask_sha256,
                role="SAM3.1 tracker",
            )
            image_masks = _load_actor_masks(
                item.image_mask_path,
                item.image_mask_sha256,
                role="base SAM3 image",
            )
            if tracker[0].shape != image_masks[0].shape:
                raise ValueError("SAM3.1 and base-SAM3 masks must share source dimensions")
            with Image.open(item.source_path) as opened:
                source = opened.convert("RGB")
            if tracker[0].shape != (source.height, source.width):
                source.close()
                raise ValueError("mask dimensions do not match the frozen source frame")
            labels, proposal = _candidate_arrays(
                *tracker,
                *image_masks,
                output_hw=item.output_hw,
                dilation_radius=dilation_radius,
            )
            relative_stem = Path(item.clip_id) / f"frame_{item.frame_index:06d}"
            label_path = temporary / "labels" / relative_stem.with_suffix(".png")
            proposal_path = temporary / "contact-proposals" / relative_stem.with_suffix(".png")
            preview_path = temporary / "previews" / relative_stem.with_suffix(".png")
            _atomic_image(label_path, Image.fromarray(labels, mode="L"))
            _atomic_image(proposal_path, Image.fromarray(proposal, mode="L"))
            preview = _render_preview(
                source,
                tracker,
                image_masks,
                labels,
                proposal,
                width=preview_width,
            )
            _atomic_image(preview_path, preview)
            preview.close()
            source.close()
            contact_candidate = proposal != LABEL_BACKGROUND
            summary = label_summary(
                labels,
                np.zeros_like(labels, dtype=bool),
                contact_reviewed=False,
            )
            records.append(
                {
                    "clip_id": item.clip_id,
                    "frame_index": item.frame_index,
                    "grid_hw": list(item.output_hw),
                    "source_path": str(item.source_path.resolve()),
                    "source_sha256": item.source_sha256,
                    "tracker_mask_path": str(item.tracker_mask_path.resolve()),
                    "tracker_mask_sha256": item.tracker_mask_sha256,
                    "image_mask_path": str(item.image_mask_path.resolve()),
                    "image_mask_sha256": item.image_mask_sha256,
                    "qwen_spatial_path": str(item.qwen_spatial_path.resolve()),
                    "qwen_spatial_sha256": item.qwen_spatial_sha256,
                    "label_path": label_path.relative_to(temporary).as_posix(),
                    "label_sha256": _sha256(label_path),
                    "contact_proposal_path": proposal_path.relative_to(temporary).as_posix(),
                    "contact_proposal_sha256": _sha256(proposal_path),
                    "preview_path": preview_path.relative_to(temporary).as_posix(),
                    "preview_sha256": _sha256(preview_path),
                    "contact_candidate_patch_count": int(np.count_nonzero(contact_candidate)),
                    "unresolved_contact_patch_count": int(
                        np.count_nonzero(contact_candidate & (proposal == LABEL_IGNORE))
                    ),
                    **summary,
                }
            )
        clips = sorted({record["clip_id"] for record in records})
        manifest = {
            "format": "ownership-label-candidates-v1",
            "training_eligible": False,
            "reason": "human contact-region review has not been completed",
            "clip_ids": clips,
            "clip_count": len(clips),
            "frame_count": len(records),
            "construction": {
                "actor_labels": "eroded exclusive cross-backend agreement",
                "possible_foreground": "dilated union of both actors and both backends",
                "contact_review_band": "actor pixels within the dilation radius of the other actor",
                "dilation_radius_source_pixels": dilation_radius,
                "review_required": True,
            },
            "records": records,
        }
        manifest_path = temporary / "candidate-manifest.json"
        _atomic_json(manifest_path, manifest)
        (temporary / "candidate-manifest.json.sha256").write_text(
            _sha256(manifest_path) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        committed = True
        return manifest
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def _load_hashed_manifest(path: Path, *, expected_format: str) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"manifest or checksum sidecar is missing: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != _sha256(path):
        raise ValueError(f"manifest SHA-256 mismatch: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"manifest is invalid: {path}") from error
    if value.get("format") != expected_format:
        raise ValueError(f"manifest format is not {expected_format}: {path}")
    return value


def _load_review_manifest(path: Path, candidate_sha256: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"human review manifest is invalid: {path}") from error
    if value.get("format") != "ownership-contact-review-v1":
        raise ValueError("human review manifest format is invalid")
    if value.get("candidate_manifest_sha256") != candidate_sha256:
        raise ValueError("human review is not bound to this candidate manifest")
    if value.get("attestation") != REVIEW_ATTESTATION:
        raise ValueError("human review attestation is missing or invalid")
    if not str(value.get("reviewer", "")).strip():
        raise ValueError("human review must name a reviewer")
    try:
        reviewed_at = datetime.fromisoformat(str(value["reviewed_at"]))
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("human review timestamp is invalid") from error
    if reviewed_at.tzinfo is None:
        raise ValueError("human review timestamp must include a timezone")
    if not isinstance(value.get("records"), list):
        raise ValueError("human review records must be a list")
    return value


def write_review_template(
    candidate_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Copy contact proposals into an editable package that remains pending."""

    candidate_path = Path(candidate_manifest_path)
    candidate = _load_hashed_manifest(
        candidate_path,
        expected_format="ownership-label-candidates-v1",
    )
    if candidate.get("training_eligible") is not False:
        raise ValueError("candidate package must explicitly prohibit training")
    records = candidate.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("candidate frame inventory is invalid")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite human review template: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    committed = False
    try:
        review_records = []
        for record in records:
            clip_id = str(record["clip_id"])
            frame_index = int(record["frame_index"])
            proposal = _safe_relative(
                candidate_path.parent,
                record.get("contact_proposal_path"),
                role="contact proposal",
            )
            _verify_hash(
                proposal,
                str(record.get("contact_proposal_sha256")),
                role="contact proposal",
            )
            owner = temporary / "contact-owner" / clip_id / f"frame_{frame_index:06d}.png"
            owner.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(proposal, owner)
            review_records.append(
                {
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "decision": "pending",
                    "contact_owner_path": owner.relative_to(temporary).as_posix(),
                    "contact_owner_sha256": None,
                    "notes": "",
                }
            )
        manifest = {
            "format": "ownership-contact-review-v1",
            "candidate_manifest_sha256": _sha256(candidate_path),
            "reviewer": "",
            "reviewed_at": "",
            "attestation": "",
            "instructions": (
                "Inspect each full preview against the source, edit the contact-owner PNG "
                "so it contains only 0=not contact, 1=A1, or 2=A2, and then run the "
                "explicit finalize-review command. Yellow/255 cells are unresolved. "
                "If a frame truly has no contact, set every cell to 0 and change that "
                "record's decision to no_contact."
            ),
            "records": review_records,
        }
        _atomic_json(temporary / "review-manifest.json", manifest)
        os.replace(temporary, output)
        committed = True
        return manifest
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def finalize_review_manifest(
    candidate_manifest_path: str | Path,
    review_manifest_path: str | Path,
    *,
    reviewer: str,
    attested: bool,
) -> dict[str, Any]:
    """Bind edited contact maps only after an explicit human attestation."""

    if not attested:
        raise ValueError("the human reviewer must explicitly attest the completed review")
    if not reviewer.strip():
        raise ValueError("the human reviewer must provide a name")
    candidate_path = Path(candidate_manifest_path)
    candidate = _load_hashed_manifest(
        candidate_path,
        expected_format="ownership-label-candidates-v1",
    )
    candidate_sha = _sha256(candidate_path)
    review_path = Path(review_manifest_path)
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"human review template is invalid: {review_path}") from error
    if review.get("format") != "ownership-contact-review-v1" or review.get(
        "candidate_manifest_sha256"
    ) != candidate_sha:
        raise ValueError("human review template is not bound to this candidate package")
    candidate_records = candidate.get("records")
    review_records = review.get("records")
    if not isinstance(candidate_records, list) or not isinstance(review_records, list):
        raise ValueError("candidate or review frame inventory is invalid")
    candidates = {
        (str(record.get("clip_id")), int(record.get("frame_index"))): record
        for record in candidate_records
    }
    identities = [
        (str(record.get("clip_id")), int(record.get("frame_index")))
        for record in review_records
        if isinstance(record, dict)
    ]
    if len(identities) != len(set(identities)) or set(identities) != set(candidates):
        raise ValueError("human review frame inventory is incomplete or duplicated")
    review_root = review_path.parent
    finalized_records: list[dict[str, Any]] = []
    for identity, record in zip(identities, review_records, strict=True):
        candidate_record = candidates[identity]
        owner_path = _safe_relative(
            review_root,
            record.get("contact_owner_path"),
            role="contact owner",
        )
        if not owner_path.is_file():
            raise FileNotFoundError(f"edited contact owner is missing: {owner_path}")
        proposal_path = _safe_relative(
            candidate_path.parent,
            candidate_record.get("contact_proposal_path"),
            role="contact proposal",
        )
        _verify_hash(
            proposal_path,
            str(candidate_record.get("contact_proposal_sha256")),
            role="contact proposal",
        )
        with Image.open(owner_path) as image:
            owners = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        with Image.open(proposal_path) as image:
            proposal = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        if owners.shape != proposal.shape:
            raise ValueError(f"edited contact owner grid is invalid: {identity}")
        unsupported = set(np.unique(owners).tolist()) - {
            LABEL_BACKGROUND,
            LABEL_A1,
            LABEL_A2,
        }
        if unsupported:
            raise ValueError(
                f"edited contact owner still has unresolved/unsupported values "
                f"{sorted(unsupported)}: {identity}"
            )
        contact = (owners == LABEL_A1) | (owners == LABEL_A2)
        proposed_band = proposal != LABEL_BACKGROUND
        if np.any(contact & ~proposed_band):
            raise ValueError(f"edited frame marks contact outside the review band: {identity}")
        requested_decision = str(record.get("decision", ""))
        if requested_decision not in {"pending", "approved", "no_contact"}:
            raise ValueError(f"human review decision is invalid: {identity}")
        if requested_decision == "no_contact":
            if np.any(contact):
                raise ValueError(
                    f"no-contact decision contains actor-owned contact cells: {identity}"
                )
            final_decision = "approved_no_contact"
        else:
            if not np.any(contact & proposed_band):
                raise ValueError(
                    f"edited frame contains no contact truth in the review band: {identity}"
                )
            final_decision = "approved"
        finalized_records.append(
            {
                **record,
                "decision": final_decision,
                "contact_owner_sha256": _sha256(owner_path),
            }
        )
    finalized = {
        **review,
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "attestation": REVIEW_ATTESTATION,
        "records": finalized_records,
    }
    _atomic_json(review_path, finalized)
    return finalized


def freeze_reviewed_label_package(
    candidate_manifest_path: str | Path,
    review_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Freeze labels only after a complete, hash-bound human contact review."""

    candidate_path = Path(candidate_manifest_path)
    candidate = _load_hashed_manifest(
        candidate_path,
        expected_format="ownership-label-candidates-v1",
    )
    if candidate.get("training_eligible") is not False:
        raise ValueError("candidate package must explicitly prohibit training")
    candidate_sha = _sha256(candidate_path)
    review_path = Path(review_manifest_path)
    review = _load_review_manifest(review_path, candidate_sha)
    candidate_records = candidate.get("records")
    if not isinstance(candidate_records, list) or not candidate_records:
        raise ValueError("candidate frame inventory is invalid")
    expected = {
        (str(record.get("clip_id")), int(record.get("frame_index")))
        for record in candidate_records
    }
    review_records = review["records"]
    observed_list = [
        (str(record.get("clip_id")), int(record.get("frame_index")))
        for record in review_records
        if isinstance(record, dict)
    ]
    if len(observed_list) != len(set(observed_list)) or set(observed_list) != expected:
        raise ValueError("human review frame inventory is incomplete or duplicated")
    review_by_identity = {
        identity: record for identity, record in zip(observed_list, review_records, strict=True)
    }

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reviewed label package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    committed = False
    try:
        provenance = temporary / "provenance"
        shutil.copytree(candidate_path.parent, provenance / "candidates")
        shutil.copytree(review_path.parent, provenance / "review")
        by_clip: dict[str, list[dict[str, Any]]] = {}
        candidate_root = candidate_path.parent
        review_root = review_path.parent
        for candidate_record in candidate_records:
            clip_id = str(candidate_record["clip_id"])
            frame_index = int(candidate_record["frame_index"])
            identity = (clip_id, frame_index)
            reviewed = review_by_identity[identity]
            if reviewed.get("decision") not in {"approved", "approved_no_contact"}:
                raise ValueError(f"frame was not approved by the human reviewer: {identity}")
            candidate_label = _safe_relative(
                candidate_root,
                candidate_record.get("label_path"),
                role="candidate label",
            )
            _verify_hash(
                candidate_label,
                str(candidate_record.get("label_sha256")),
                role="candidate label",
            )
            owner_path = _safe_relative(
                review_root,
                reviewed.get("contact_owner_path"),
                role="contact owner",
            )
            _verify_hash(
                owner_path,
                str(reviewed.get("contact_owner_sha256")),
                role="contact owner",
            )
            with Image.open(candidate_label) as image:
                labels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            with Image.open(owner_path) as image:
                owners = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            if owners.shape != labels.shape:
                raise ValueError(f"contact owner grid does not match candidate label: {identity}")
            unsupported = set(np.unique(owners).tolist()) - {
                LABEL_BACKGROUND,
                LABEL_A1,
                LABEL_A2,
            }
            if unsupported:
                raise ValueError(
                    f"reviewed contact owner contains unsupported values {sorted(unsupported)}: {identity}"
                )
            corrected, contact = apply_reviewed_contact_ownership(
                labels,
                owners,
                reviewed=True,
            )
            if (reviewed.get("decision") == "approved_no_contact") != (not np.any(contact)):
                raise ValueError(f"reviewed frame decision and contact truth disagree: {identity}")
            validate_label_ready(corrected, contact, require_both_actors=True)
            clip_root = temporary / "clips" / clip_id
            label_path = clip_root / "labels" / f"frame_{frame_index:06d}.png"
            contact_path = clip_root / "contact" / f"frame_{frame_index:06d}.png"
            _atomic_image(label_path, Image.fromarray(corrected, mode="L"))
            _atomic_image(
                contact_path,
                Image.fromarray(contact.astype(np.uint8), mode="L"),
            )
            summary = label_summary(corrected, contact, contact_reviewed=True)
            by_clip.setdefault(clip_id, []).append(
                {
                    "frame_index": frame_index,
                    "subset": "leave_one_clip_out",
                    "screen_subset": "leave_one_clip_out",
                    "label_path": label_path.relative_to(clip_root).as_posix(),
                    "label_sha256": _sha256(label_path),
                    "contact_path": contact_path.relative_to(clip_root).as_posix(),
                    "contact_sha256": _sha256(contact_path),
                    "candidate_label_sha256": candidate_record["label_sha256"],
                    "contact_owner_sha256": reviewed["contact_owner_sha256"],
                    **summary,
                }
            )

        clip_records: list[dict[str, Any]] = []
        for clip_id in sorted(by_clip):
            clip_root = temporary / "clips" / clip_id
            records = sorted(by_clip[clip_id], key=lambda value: value["frame_index"])
            if sum(int(record["contact_patch_count"]) for record in records) < 1:
                raise ValueError(
                    f"reviewed clip contains no explicit contact evidence: {clip_id}"
                )
            manifest = {
                "format": "reviewed-ownership-labels-v1",
                "clip_id": clip_id,
                "training_eligible": True,
                "contact_reviewed": True,
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "candidate_manifest_path": "../../provenance/candidates/candidate-manifest.json",
                "candidate_manifest_sha256": candidate_sha,
                "review_manifest_path": "../../provenance/review/review-manifest.json",
                "review_manifest_sha256": _sha256(review_path),
                "frame_count": len(records),
                "records": records,
            }
            manifest_path = clip_root / "label-manifest.json"
            _atomic_json(manifest_path, manifest)
            sidecar = manifest_path.with_suffix(".json.sha256")
            sidecar.write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
            clip_records.append(
                {
                    "clip_id": clip_id,
                    "frame_count": len(records),
                    "label_manifest_path": manifest_path.relative_to(temporary).as_posix(),
                    "label_manifest_sha256": _sha256(manifest_path),
                }
            )
        campaign = {
            "format": "reviewed-ownership-label-campaign-v1",
            "training_eligible": True,
            "candidate_manifest_sha256": candidate_sha,
            "review_manifest_sha256": _sha256(review_path),
            "reviewer": review["reviewer"],
            "reviewed_at": review["reviewed_at"],
            "clip_count": len(clip_records),
            "approved_frame_count": sum(record["frame_count"] for record in clip_records),
            "clips": clip_records,
        }
        campaign_path = temporary / "campaign-manifest.json"
        _atomic_json(campaign_path, campaign)
        (temporary / "campaign-manifest.json.sha256").write_text(
            _sha256(campaign_path) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        committed = True
        return campaign
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def verify_reviewed_label_manifest(
    manifest_path: str | Path,
    *,
    expected_clip_id: str | None = None,
    expected_frame_count: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless every reviewed label and provenance binding is intact."""

    path = Path(manifest_path)
    manifest = _load_hashed_manifest(path, expected_format="reviewed-ownership-labels-v1")
    if manifest.get("training_eligible") is not True or manifest.get("contact_reviewed") is not True:
        raise ValueError("label manifest is not approved for training")
    if expected_clip_id is not None and manifest.get("clip_id") != expected_clip_id:
        raise ValueError("reviewed label clip ID mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("reviewed label record inventory is invalid")
    if expected_frame_count is not None and len(records) != expected_frame_count:
        raise ValueError("reviewed label frame count mismatch")
    if manifest.get("frame_count") != len(records):
        raise ValueError("reviewed label manifest frame count is inconsistent")
    frame_indices = [int(record.get("frame_index")) for record in records]
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError("reviewed label frame indices are duplicated")
    root = path.parent
    if len(path.parents) < 3:
        raise ValueError("reviewed label manifest is not inside a campaign package")
    package_root = path.parents[2]
    for provenance_role in ("candidate", "review"):
        provenance = _safe_package_reference(
            root,
            package_root,
            manifest.get(f"{provenance_role}_manifest_path"),
            role=f"{provenance_role} manifest",
        )
        _verify_hash(
            provenance,
            str(manifest.get(f"{provenance_role}_manifest_sha256")),
            role=f"{provenance_role} manifest",
        )
    for record in records:
        label_path = _safe_relative(root, record.get("label_path"), role="reviewed label")
        contact_path = _safe_relative(root, record.get("contact_path"), role="reviewed contact")
        _verify_hash(label_path, str(record.get("label_sha256")), role="reviewed label")
        _verify_hash(contact_path, str(record.get("contact_sha256")), role="reviewed contact")
        with Image.open(label_path) as image:
            labels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        with Image.open(contact_path) as image:
            contact_values = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        if set(np.unique(contact_values).tolist()) - {0, 1}:
            raise ValueError(f"reviewed contact mask is not binary: {contact_path}")
        contact = contact_values.astype(bool)
        validate_label_ready(labels, contact, require_both_actors=True)
        if record.get("contact_reviewed") is not True:
            raise ValueError("reviewed record lacks explicit contact review")
        observed = label_summary(labels, contact, contact_reviewed=True)
        for key, value in observed.items():
            if record.get(key) != value:
                raise ValueError(f"reviewed label summary mismatch for frame {record.get('frame_index')}")
    if sum(int(record.get("contact_patch_count", 0)) for record in records) < 1:
        raise ValueError("reviewed clip lacks explicit contact evidence")
    return manifest
