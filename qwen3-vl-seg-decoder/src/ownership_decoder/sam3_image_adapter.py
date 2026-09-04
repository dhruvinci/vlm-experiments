from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager

import numpy as np

from .remote_preflight import (
    RemoteRuntimeApproval,
    RemoteRuntimeContract,
    RemoteRuntimePreflightError,
)


REQUIRED_SAM3_IMAGE_FILES = (
    "config.json",
    "model.safetensors",
    "processor_config.json",
)


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.cpu()
    return np.asarray(value)


def _cpu_float(value: Any) -> Any:
    """Preserve a Torch tensor for official post-processing, but move it off CUDA."""

    if hasattr(value, "detach"):
        return value.detach().float().cpu()
    return np.asarray(value, dtype=np.float32)


def select_actor_multimasks(
    mask_logits: Any,
    iou_scores: Any,
    *,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    """Select each actor's highest predicted-IoU mask without merging actors."""

    logits = _numpy(mask_logits).astype(np.float32, copy=False)
    scores = _numpy(iou_scores).astype(np.float32, copy=False)
    if logits.ndim == 5 and logits.shape[0] == 1:
        logits = logits[0]
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores[0]
    if logits.ndim != 4 or logits.shape[0] != 2:
        raise ValueError(
            "SAM3 image logits must have shape [2 actors, candidates, height, width]"
        )
    if scores.ndim != 2 or scores.shape != logits.shape[:2]:
        raise ValueError("SAM3 IoU scores must match the actor/candidate dimensions")
    if tuple(logits.shape[-2:]) != tuple(expected_shape):
        raise ValueError(
            f"SAM3 image mask shape {tuple(logits.shape[-2:])} does not match "
            f"{tuple(expected_shape)}"
        )
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(scores)):
        raise ValueError("SAM3 image masks and IoU scores must be finite")

    selected = np.argmax(scores, axis=1)
    actor_logits = [logits[index, selected[index]].copy() for index in range(2)]
    return {
        "logits_A1": actor_logits[0],
        "logits_A2": actor_logits[1],
        "raw_A1": actor_logits[0] > 0.0,
        "raw_A2": actor_logits[1] > 0.0,
        "score_A1": float(scores[0, selected[0]]),
        "score_A2": float(scores[1, selected[1]]),
        "selected_index_A1": int(selected[0]),
        "selected_index_A2": int(selected[1]),
    }


@dataclass(frozen=True)
class Sam3ImageRuntimeBindings:
    load_model: Callable[[Path], Any]
    load_processor: Callable[[Path], Any]
    inference_context_factory: Callable[[], ContextManager[Any]]
    load_rgb_image: Callable[[Path], Any]
    collect: Callable[[], Any]
    release_cuda_cache: Callable[[], Any]


def _load_default_runtime_bindings() -> Sam3ImageRuntimeBindings:
    """Import Transformers, Torch and Pillow only after remote approval."""

    import gc

    import torch
    from PIL import Image
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    def load_processor(path: Path) -> Any:
        return Sam3TrackerProcessor.from_pretrained(
            str(path),
            local_files_only=True,
        )

    def load_model(path: Path) -> Any:
        return Sam3TrackerModel.from_pretrained(
            str(path),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )

    @contextmanager
    def inference_context() -> Any:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            yield

    def load_rgb_image(path: Path) -> Any:
        with Image.open(path) as source:
            return source.convert("RGB")

    def release_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return Sam3ImageRuntimeBindings(
        load_model=load_model,
        load_processor=load_processor,
        inference_context_factory=inference_context,
        load_rgb_image=load_rgb_image,
        collect=gc.collect,
        release_cuda_cache=release_cuda_cache,
    )


def _require_bound_model_directory(
    contract: RemoteRuntimeContract,
    model_directory: Path,
) -> None:
    bound = {artifact.path.resolve() for artifact in contract.additional_artifacts}
    for filename in REQUIRED_SAM3_IMAGE_FILES:
        expected = (model_directory / filename).resolve()
        if expected not in bound:
            raise RemoteRuntimePreflightError(
                f"base SAM3 model file is not contract-bound: {filename}"
            )


def _validate_prompts(prompts: dict[str, dict[str, Any]]) -> None:
    if set(prompts) != {"A1", "A2"}:
        raise ValueError("SAM3 image prompts must contain exactly A1 and A2")
    for actor in ("A1", "A2"):
        prompt = prompts[actor]
        required = {"box", "points", "labels"}
        if not required.issubset(prompt):
            raise ValueError(f"{actor} SAM3 image prompt is incomplete")
        box = np.asarray(prompt["box"], dtype=np.float64)
        points = np.asarray(prompt["points"], dtype=np.float64)
        labels = np.asarray(prompt["labels"])
        if box.shape != (4,) or points.shape != (4, 2) or labels.shape != (4,):
            raise ValueError(f"{actor} SAM3 image prompt has an invalid shape")
        if not np.all(np.isfinite(box)) or not np.all(np.isfinite(points)):
            raise ValueError(f"{actor} SAM3 image prompt must be finite")
        if labels.tolist() != [1, 1, 0, 0]:
            raise ValueError(f"{actor} SAM3 image labels must be [1, 1, 0, 0]")


class Sam3ImagePredictor:
    """Memory-bounded two-actor image segmentation using SAM3 tracker weights."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        bindings: Sam3ImageRuntimeBindings,
    ) -> None:
        self.model = model
        self.processor = processor
        self._bindings = bindings
        self._closed = False

    def segment(
        self,
        image_path: Path,
        prompts: dict[str, dict[str, Any]],
        *,
        expected_shape: tuple[int, int],
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("SAM3 image predictor is closed")
        _validate_prompts(prompts)
        actors = (prompts["A1"], prompts["A2"])
        image: Any | None = None
        inputs: Any | None = None
        outputs: Any | None = None
        try:
            image = self._bindings.load_rgb_image(Path(image_path))
            inputs = self.processor(
                images=image,
                input_points=[[actor["points"] for actor in actors]],
                input_labels=[[actor["labels"] for actor in actors]],
                input_boxes=[[actor["box"] for actor in actors]],
                return_tensors="pt",
            )
            original_sizes = inputs["original_sizes"]
            inputs = inputs.to("cuda")
            with self._bindings.inference_context_factory():
                outputs = self.model(**inputs, multimask_output=True)
            logits = self.processor.post_process_masks(
                _cpu_float(outputs.pred_masks),
                original_sizes,
                mask_threshold=0.0,
                binarize=False,
                max_hole_area=0.0,
                max_sprinkle_area=0.0,
                apply_non_overlapping_constraints=False,
            )[0]
            return select_actor_multimasks(
                logits,
                outputs.iou_scores,
                expected_shape=expected_shape,
            )
        finally:
            outputs = None
            inputs = None
            close = getattr(image, "close", None)
            if callable(close):
                close()
            image = None
            self._bindings.collect()
            self._bindings.release_cuda_cache()

    def close(self) -> None:
        if self._closed:
            return
        self.model = None
        self.processor = None
        self._closed = True
        self._bindings.collect()
        self._bindings.release_cuda_cache()


def build_sam3_image_predictor(
    contract: RemoteRuntimeContract,
    approval: RemoteRuntimeApproval,
    *,
    model_directory: Path,
    runtime_bindings_factory: Callable[
        [], Sam3ImageRuntimeBindings
    ] = _load_default_runtime_bindings,
) -> Sam3ImagePredictor:
    """Build base SAM3 image PVS only after all weight files are approved."""

    approval.require_contract(contract)
    _require_bound_model_directory(contract, Path(model_directory))
    bindings = runtime_bindings_factory()
    model: Any | None = None
    processor: Any | None = None
    try:
        processor = bindings.load_processor(Path(model_directory))
        model = bindings.load_model(Path(model_directory))
        model.to(device="cuda").eval()
        bindings.collect()
        bindings.release_cuda_cache()
        return Sam3ImagePredictor(model, processor, bindings)
    except Exception:
        model = None
        processor = None
        bindings.collect()
        bindings.release_cuda_cache()
        raise
