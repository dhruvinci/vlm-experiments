from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .tracking import TrackingArtifactError, TrackingPlan


ACTOR_COLORS = {"A1": (0, 220, 255), "A2": (255, 55, 210)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _draw_seed_panel(plan: TrackingPlan, seed_index: int, max_panel_width: int) -> Image.Image:
    seed = plan.seeds[seed_index]
    frame = plan.frames[seed.frame_index]
    with Image.open(frame.path) as source:
        image = source.convert("RGB")
    if image.size != (frame.width, frame.height):
        raise TrackingArtifactError(
            f"review frame dimensions disagree with manifest: {frame.path}"
        )
    scale = min(1.0, max_panel_width / image.width)
    display_width = max(1, round(image.width * scale))
    display_height = max(1, round(image.height * scale))
    image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
    header_height = max(28, display_width // 24)
    panel = Image.new("RGB", (display_width, display_height + header_height), "black")
    panel.paste(image, (0, header_height))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text(
        (8, max(2, (header_height - 12) // 2)),
        f"{plan.clip_id} | seed frame {seed.frame_index} | {frame.path.name}",
        fill="white",
        font=font,
    )
    line_width = max(2, display_width // 300)
    point_radius = max(4, display_width // 100)
    for actor in seed.ordered_actors():
        color = ACTOR_COLORS[actor.actor_id]
        x1, y1, x2, y2 = actor.bbox
        box = (
            round(x1 * display_width),
            header_height + round(y1 * display_height),
            round(x2 * display_width),
            header_height + round(y2 * display_height),
        )
        draw.rectangle(box, outline=color, width=line_width)
        draw.rectangle((box[0], box[1], box[0] + 24, box[1] + 15), fill="black")
        draw.text((box[0] + 2, box[1] + 1), actor.actor_id, fill=color, font=font)
        # Draw exclusion crosses first; mirrored opponent positives then remain legible.
        for x, y in actor.negative_points:
            px = round(x * display_width)
            py = header_height + round(y * display_height)
            draw.line(
                (px - point_radius, py - point_radius, px + point_radius, py + point_radius),
                fill=color,
                width=line_width,
            )
            draw.line(
                (px - point_radius, py + point_radius, px + point_radius, py - point_radius),
                fill=color,
                width=line_width,
            )
        for point_number, (x, y) in enumerate(actor.positive_points, start=1):
            px = round(x * display_width)
            py = header_height + round(y * display_height)
            draw.ellipse(
                (
                    px - point_radius,
                    py - point_radius,
                    px + point_radius,
                    py + point_radius,
                ),
                fill=color,
                outline="black",
                width=1,
            )
            draw.text((px + point_radius + 1, py - point_radius), str(point_number), fill=color, font=font)
    return panel


def render_seed_prompt_review(
    plan: TrackingPlan,
    output_path: str | Path,
    *,
    max_panel_width: int = 600,
) -> dict[str, Any]:
    """Render only the frozen geometry sent to the tracker for visual QA."""

    if max_panel_width < 64:
        raise ValueError("max_panel_width must be at least 64 pixels")
    output = Path(output_path)
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise TrackingArtifactError(f"refusing to overwrite seed review artifact: {output}")
    if output.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("seed review output must be PNG or JPEG")
    output.parent.mkdir(parents=True, exist_ok=True)
    plan.verify_frame_files()

    panels = [
        _draw_seed_panel(plan, seed_index, max_panel_width)
        for seed_index in range(len(plan.seeds))
    ]
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    review = Image.new("RGB", (width, height), "black")
    y = 0
    for panel in panels:
        review.paste(panel, (0, y))
        y += panel.height

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output.parent,
            prefix=f".{output.stem}.",
            suffix=output.suffix,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            image_format = "PNG" if output.suffix.lower() == ".png" else "JPEG"
            review.save(handle, format=image_format, quality=94)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        for panel in panels:
            panel.close()
        review.close()

    metadata = {
        "format": "ownership-seed-prompt-review-v1",
        "plan_sha256": plan.sha256,
        "image_sha256": _sha256(output),
        "clip_id": plan.clip_id,
        "seed_frame_indices": [seed.frame_index for seed in plan.seeds],
        "actor_colors_rgb": {key: list(value) for key, value in ACTOR_COLORS.items()},
        "legend": "solid circles=positive; crosses=opponent exclusion; boxes=actor extent",
    }
    _atomic_json(sidecar, metadata)
    return metadata
