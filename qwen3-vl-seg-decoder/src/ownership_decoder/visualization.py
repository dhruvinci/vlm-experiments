from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_CLASS_COLORS = np.array(
    [
        [40, 40, 40],
        [0, 210, 145],
        [235, 72, 110],
        [0, 0, 0],
    ],
    dtype=np.float32,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=0, keepdims=True)
    exponent = np.exp(shifted)
    return (exponent / exponent.sum(axis=0, keepdims=True)).astype(np.float32)


def _contact_edge(contact: np.ndarray) -> np.ndarray:
    inner = contact.copy()
    inner[0, :] = False
    inner[-1, :] = False
    inner[:, 0] = False
    inner[:, -1] = False
    if contact.shape[0] > 2 and contact.shape[1] > 2:
        inner[1:-1, 1:-1] &= contact[:-2, 1:-1]
        inner[1:-1, 1:-1] &= contact[2:, 1:-1]
        inner[1:-1, 1:-1] &= contact[1:-1, :-2]
        inner[1:-1, 1:-1] &= contact[1:-1, 2:]
    return contact & ~inner


def _overlay(base: np.ndarray, classes: np.ndarray, contact: np.ndarray) -> np.ndarray:
    output = base.astype(np.float32).copy()
    actor = (classes == 1) | (classes == 2)
    colors = _CLASS_COLORS[np.where(classes == 255, 3, classes)]
    output[actor] = 0.52 * output[actor] + 0.48 * colors[actor]
    output[classes == 255] *= 0.22
    output[_contact_edge(contact)] = np.array([255, 224, 0], dtype=np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def _margin_image(probabilities: np.ndarray, contact: np.ndarray) -> np.ndarray:
    margin = np.clip(probabilities[2] - probabilities[1], -1.0, 1.0)
    magnitude = np.abs(margin)[..., None]
    white = np.full((*margin.shape, 3), 235.0, dtype=np.float32)
    negative = np.array([35.0, 95.0, 245.0], dtype=np.float32)
    positive = np.array([245.0, 70.0, 45.0], dtype=np.float32)
    endpoint = np.where((margin >= 0)[..., None], positive, negative)
    output = white * (1.0 - magnitude) + endpoint * magnitude
    output[_contact_edge(contact)] = np.array([255, 224, 0], dtype=np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def _resize_panel(values: np.ndarray, size: tuple[int, int], *, nearest: bool) -> Image.Image:
    image = Image.fromarray(values, mode="RGB")
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return image.resize(size, resampling)


def _crop_for_contact(
    values: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = bbox
    height, width = values.shape[:2]
    padding = max(2, math.ceil(max(right - left, bottom - top) * 0.75))
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    return values[top:bottom, left:right]


def _validate_inputs(
    labels: np.ndarray,
    logits: np.ndarray,
    contact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    contact = np.asarray(contact)
    if labels.ndim != 2 or logits.shape != (3, *labels.shape) or contact.shape != labels.shape:
        raise ValueError("labels, logits, and contact shapes are inconsistent")
    if not np.isfinite(logits).all():
        raise ValueError("ownership logits must be finite")
    if not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("ownership labels must contain only 0,1,2,255")
    contact = contact.astype(bool)
    if not contact.any():
        raise ValueError("a diagnostic requires reviewed contact pixels")
    return labels.astype(np.uint8), logits.astype(np.float32), contact


def render_ownership_diagnostic(
    *,
    rgb_path: str | Path,
    labels: np.ndarray,
    logits: np.ndarray,
    contact: np.ndarray,
    output_path: str | Path,
    title: str = "ownership diagnostic",
    panel_size: tuple[int, int] = (480, 320),
) -> dict[str, Any]:
    """Render a deterministic six-panel ownership and contact diagnostic."""

    labels, logits, contact = _validate_inputs(labels, logits, contact)
    panel_width, panel_height = (int(value) for value in panel_size)
    if panel_width < 32 or panel_height < 32:
        raise ValueError("diagnostic panels must be at least 32 pixels per side")
    source_path = Path(rgb_path)
    try:
        with Image.open(source_path) as image:
            source = np.asarray(
                image.convert("RGB").resize(
                    (labels.shape[1], labels.shape[0]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            ).copy()
    except OSError as error:
        raise ValueError(f"could not decode diagnostic RGB frame: {source_path}") from error

    probabilities = _softmax(logits)
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    truth_overlay = _overlay(source, labels, contact)
    prediction_overlay = _overlay(source, prediction, contact)
    margin = _margin_image(probabilities, contact)
    y_values, x_values = np.nonzero(contact)
    bbox = (
        int(x_values.min()),
        int(y_values.min()),
        int(x_values.max()) + 1,
        int(y_values.max()) + 1,
    )
    panels = (
        (source, False, "source"),
        (truth_overlay, True, "reviewed truth"),
        (prediction_overlay, True, "prediction"),
        (margin, False, "A2 - A1 probability"),
        (_crop_for_contact(truth_overlay, bbox), True, "truth contact zoom"),
        (_crop_for_contact(prediction_overlay, bbox), True, "prediction contact zoom"),
    )
    title_height = 22
    legend_height = 16
    canvas = Image.new(
        "RGB",
        (panel_width * 3, title_height + panel_height * 2 + legend_height),
        color=(18, 18, 20),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 5), str(title), fill=(245, 245, 245), font=font)
    for index, (values, nearest, label) in enumerate(panels):
        column = index % 3
        row = index // 3
        panel = _resize_panel(values, (panel_width, panel_height), nearest=nearest)
        x = column * panel_width
        y = title_height + row * panel_height
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + panel_width, y + 14), fill=(0, 0, 0))
        draw.text((x + 4, y + 2), label, fill=(255, 255, 255), font=font)
    legend_y = title_height + 2 * panel_height
    draw.text(
        (6, legend_y + 2),
        "A1=green  A2=pink  contact=yellow  margin: blue=A1 / red=A2",
        fill=(230, 230, 230),
        font=font,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            canvas.save(handle, format="PNG", optimize=False, compress_level=9)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "format": "ownership-diagnostic-v1",
        "output_path": str(destination.resolve()),
        "output_sha256": _sha256(destination),
        "contact_bbox_grid": list(bbox),
        "prediction_classes": sorted(int(value) for value in np.unique(prediction)),
        "a2_minus_a1_min": float((probabilities[2] - probabilities[1]).min()),
        "a2_minus_a1_max": float((probabilities[2] - probabilities[1]).max()),
    }
