from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class GeometryDerivationError(ValueError):
    """A tracker mask cannot safely define an image-segmentation prompt."""


@dataclass(frozen=True)
class _Run:
    y: int
    x1: int
    x2: int

    @property
    def area(self) -> int:
        return self.x2 - self.x1 + 1


@dataclass(frozen=True)
class _Component:
    area: int
    x1: int
    y1: int
    x2: int
    y2: int
    runs: tuple[_Run, ...]


def _validated_mask(value: Any, *, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2 or min(mask.shape) < 2:
        raise GeometryDerivationError(f"{name} mask shape must be a two-dimensional image")
    if np.issubdtype(mask.dtype, np.number) and not np.all(np.isfinite(mask)):
        raise GeometryDerivationError(f"{name} mask must be finite")
    return mask.astype(bool, copy=False)


def _row_runs(row: np.ndarray, y: int) -> list[_Run]:
    changes = np.flatnonzero(
        np.diff(row.astype(np.int8, copy=False), prepend=np.int8(0), append=np.int8(0))
    )
    return [
        _Run(y=y, x1=int(start), x2=int(end) - 1)
        for start, end in zip(changes[::2], changes[1::2], strict=True)
    ]


def _largest_component(mask: np.ndarray, *, minimum_area: int) -> _Component:
    runs: list[_Run] = []
    parent: list[int] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

    previous: list[int] = []
    for y in range(mask.shape[0]):
        current: list[int] = []
        for run in _row_runs(mask[y], y):
            index = len(runs)
            runs.append(run)
            parent.append(index)
            current.append(index)
            for prior_index in previous:
                prior = runs[prior_index]
                if prior.x2 < run.x1 - 1:
                    continue
                if prior.x1 > run.x2 + 1:
                    break
                union(index, prior_index)
        previous = current

    grouped: dict[int, list[_Run]] = {}
    for index, run in enumerate(runs):
        grouped.setdefault(find(index), []).append(run)
    if not grouped:
        raise GeometryDerivationError("mask has no foreground component")
    components = [
        _Component(
            area=sum(run.area for run in component_runs),
            x1=min(run.x1 for run in component_runs),
            y1=min(run.y for run in component_runs),
            x2=max(run.x2 for run in component_runs),
            y2=max(run.y for run in component_runs),
            runs=tuple(component_runs),
        )
        for component_runs in grouped.values()
    ]
    largest = max(components, key=lambda item: (item.area, -item.y1, -item.x1))
    if largest.area < minimum_area:
        raise GeometryDerivationError(
            f"largest foreground component has area {largest.area}, below {minimum_area}"
        )
    return largest


def _interior_points(component: _Component) -> list[list[float]]:
    candidates: dict[tuple[int, int], float] = {}
    for run in component.runs:
        length = run.area
        positions = {
            (run.x1 + run.x2) // 2,
            run.x1 + length // 4,
            run.x1 + (3 * length) // 4,
        }
        for x in positions:
            horizontal_margin = min(x - run.x1, run.x2 - x) + 1
            vertical_margin = min(run.y - component.y1, component.y2 - run.y) + 1
            candidates[(x, run.y)] = float(min(horizontal_margin, vertical_margin))
    if len(candidates) < 2:
        raise GeometryDerivationError("foreground component cannot supply two distinct points")
    first = max(
        candidates,
        key=lambda point: (candidates[point], -point[1], -point[0]),
    )
    second = max(
        (point for point in candidates if point != first),
        key=lambda point: (
            candidates[point]
            * math.sqrt((point[0] - first[0]) ** 2 + (point[1] - first[1]) ** 2),
            candidates[point],
            -point[1],
            -point[0],
        ),
    )
    return [[float(first[0]), float(first[1])], [float(second[0]), float(second[1])]]


def _box(
    component: _Component,
    *,
    shape: tuple[int, int],
    padding_fraction: float,
) -> list[float]:
    height, width = shape
    pad_x = width * padding_fraction
    pad_y = height * padding_fraction
    return [
        float(max(0.0, component.x1 - pad_x)),
        float(max(0.0, component.y1 - pad_y)),
        float(min(width - 1.0, component.x2 + pad_x)),
        float(min(height - 1.0, component.y2 + pad_y)),
    ]


def mask_pair_to_geometry_prompts(
    mask_a1: Any,
    mask_a2: Any,
    *,
    minimum_area: int = 64,
    box_padding_fraction: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Convert exclusive tracker masks into correlated, geometry-only SAM prompts."""

    if minimum_area < 2:
        raise ValueError("minimum_area must be at least two pixels")
    if not 0.0 <= box_padding_fraction <= 0.25:
        raise ValueError("box_padding_fraction must be in [0, 0.25]")
    a1 = _validated_mask(mask_a1, name="A1")
    a2 = _validated_mask(mask_a2, name="A2")
    if a1.shape != a2.shape:
        raise GeometryDerivationError(
            f"actor mask shape mismatch: A1={a1.shape}, A2={a2.shape}"
        )
    if np.any(a1 & a2):
        raise GeometryDerivationError(
            "actor masks overlap; prompt ownership must be resolved before conversion"
        )
    components = {
        "A1": _largest_component(a1, minimum_area=minimum_area),
        "A2": _largest_component(a2, minimum_area=minimum_area),
    }
    positives = {actor: _interior_points(value) for actor, value in components.items()}
    return {
        actor: {
            "box": _box(
                component,
                shape=a1.shape,
                padding_fraction=box_padding_fraction,
            ),
            "points": [*positives[actor], *positives["A2" if actor == "A1" else "A1"]],
            "labels": [1, 1, 0, 0],
            "derivation": "largest_tracker_component_box_and_interior_points",
            "localization_source": "sam3.1_tracker",
        }
        for actor, component in components.items()
    }
