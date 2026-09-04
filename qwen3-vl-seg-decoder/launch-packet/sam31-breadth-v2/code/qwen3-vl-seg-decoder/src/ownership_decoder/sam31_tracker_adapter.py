from __future__ import annotations

import inspect
import sys
import uuid
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


def remap_tracker_only_checkpoint(
    checkpoint: dict[str, Any], expected_keys: set[str]
) -> dict[str, Any]:
    """Map only tracker weights plus the detector's shared visual backbone."""

    remapped: dict[str, Any] = {}
    for target in expected_keys:
        tracker_source = f"tracker.model.{target}"
        detector_source = f"detector.{target}"
        if tracker_source in checkpoint:
            remapped[target] = checkpoint[tracker_source]
        elif target.startswith("backbone.") and detector_source in checkpoint:
            remapped[target] = checkpoint[detector_source]
    return remapped


def initialize_dimension_tracker_state(
    model: Any,
    resource_path: Path,
    *,
    frame_loader: Callable[..., tuple[Any, int, int]],
) -> dict[str, Any]:
    """Preload official normalized frames for the dimension-only tracker class."""

    images, video_height, video_width = frame_loader(
        video_path=str(resource_path),
        image_size=model.image_size,
        offload_video_to_cpu=True,
        async_loading_frames=False,
    )
    state = model.init_state(
        video_height=video_height,
        video_width=video_width,
        num_frames=len(images),
        offload_video_to_cpu=True,
        # The official multiplex demo concatenates a CUDA mask with this state.
        offload_state_to_cpu=False,
    )
    state["images"] = images
    return state


def disable_unused_sam3_branch(model: Any) -> None:
    """Prevent the open-vocabulary detector branch from running in tracker-only mode."""

    original_forward_image = model.forward_image

    def tracker_forward_image(
        image: Any,
        *,
        need_sam3_out: bool = False,
        need_interactive_out: bool = False,
        need_propagation_out: bool = False,
    ) -> Any:
        return original_forward_image(
            image,
            need_sam3_out=False,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )

    model.forward_image = tracker_forward_image


def preserve_images_in_interaction_state(model: Any) -> None:
    """Repair the official singleton extractor's omission of preloaded video tensors."""

    original = getattr(model, "_extract_object_for_interaction", None)
    if original is None or getattr(model, "_ownership_preserves_interaction_images", False):
        return

    def extract_with_images(inference_state: Any, obj_id: int, frame_idx: int) -> Any:
        singleton_state, original_obj_idx = original(inference_state, obj_id, frame_idx)
        if "images" in inference_state:
            singleton_state["images"] = inference_state["images"]
        return singleton_state, original_obj_idx

    model._extract_object_for_interaction = extract_with_images
    model._ownership_preserves_interaction_images = True


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.cpu()
    return np.asarray(value)


@dataclass(frozen=True)
class Sam31RuntimeBindings:
    build_model: Callable[..., Any]
    load_checkpoint: Callable[[Path], dict[str, Any]]
    inference_context_factory: Callable[[], ContextManager[Any]]
    point_tensor_factory: Callable[[Any, Any], tuple[Any, Any]]
    collect: Callable[[], Any]
    release_cuda_cache: Callable[[], Any]


def _load_default_runtime_bindings() -> Sam31RuntimeBindings:
    """Import all heavyweight dependencies only after remote preflight approval."""

    import gc

    import torch
    from sam3.model_builder import build_sam3_multiplex_video_model

    @contextmanager
    def inference_context() -> Any:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield

    def point_tensors(points: Any, labels: Any) -> tuple[Any, Any]:
        return (
            torch.as_tensor(points, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.int32),
        )

    def load_checkpoint(path: Path) -> dict[str, Any]:
        return torch.load(str(path), map_location="cpu", weights_only=True)

    def release_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return Sam31RuntimeBindings(
        build_model=build_sam3_multiplex_video_model,
        load_checkpoint=load_checkpoint,
        inference_context_factory=inference_context,
        point_tensor_factory=point_tensors,
        collect=gc.collect,
        release_cuda_cache=release_cuda_cache,
    )


class TrackerOnlySam31Predictor:
    """Request-compatible adapter around Meta's official tracker-only API."""

    def __init__(
        self,
        model: Any,
        *,
        video_state_loader: Callable[[Any, Path], dict[str, Any]] | None = None,
        inference_context_factory: Callable[[], ContextManager[Any]],
        point_tensor_factory: Callable[[Any, Any], tuple[Any, Any]],
        session_cleanup: Callable[[], Any] | None = None,
        final_cleanup: Callable[[], Any] | None = None,
    ) -> None:
        self.model = model
        preserve_images_in_interaction_state(model)
        self._video_state_loader = video_state_loader
        self._inference_context_factory = inference_context_factory
        self._point_tensor_factory = point_tensor_factory
        self._session_cleanup = session_cleanup or (lambda: None)
        self._final_cleanup = final_cleanup or (lambda: None)
        self._states: dict[str, Any] = {}
        self._object_states: dict[str, dict[int, Any]] = {}
        self._resource_paths: dict[str, Path] = {}
        self._video_images: dict[str, Any] = {}
        self._shared_feature_cache: dict[str, Any] = {}
        self._prompt_masks: dict[str, dict[int, dict[int, np.ndarray]]] = {}
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SAM3.1 tracker predictor is closed")

    def _start_video_state(self, resource_path: Path) -> Any:
        if self._video_state_loader is not None:
            return self._video_state_loader(self.model, resource_path)
        parameters = inspect.signature(self.model.init_state).parameters
        if "video_path" in parameters:
            return self.model.init_state(
                video_path=str(resource_path),
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
        from sam3.model.io_utils import load_video_frames

        return initialize_dimension_tracker_state(
            self.model,
            resource_path,
            frame_loader=load_video_frames,
        )

    def _new_object_state(self, session_id: str) -> Any:
        template = self._states[session_id]
        if not self._object_states[session_id]:
            return template
        parameters = inspect.signature(self.model.init_state).parameters
        if "video_height" in parameters:
            state = self.model.init_state(
                video_height=template["video_height"],
                video_width=template["video_width"],
                num_frames=template["num_frames"],
                cached_features=self._shared_feature_cache.get(session_id),
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
            )
            if self._video_images.get(session_id) is not None:
                state["images"] = self._video_images[session_id]
            return state
        return self._start_video_state(self._resource_paths[session_id])

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require_open()
        request_type = request.get("type")
        if request_type == "start_session":
            resource_path = Path(request["resource_path"])
            session_id = str(uuid.uuid4())
            state = self._start_video_state(resource_path)
            self._states[session_id] = state
            self._object_states[session_id] = {}
            self._resource_paths[session_id] = resource_path
            self._video_images[session_id] = state.get("images")
            self._shared_feature_cache[session_id] = state.get("cached_features", {})
            self._prompt_masks[session_id] = {}
            return {"session_id": session_id}
        if request_type == "close_session":
            self._close_session(str(request["session_id"]))
            if request.get("run_gc_collect", False):
                self._session_cleanup()
            return {"is_success": True}
        if request_type == "propagate_in_video":
            return self._propagate(request)
        if request_type != "add_prompt":
            raise ValueError(f"unsupported tracker-only request: {request_type!r}")
        return self._add_prompt(request)

    def _add_prompt(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request["session_id"])
        if session_id not in self._states:
            raise KeyError(f"unknown tracker session: {session_id}")
        frame_index = int(request["frame_index"])
        object_id = int(request["obj_id"])
        if object_id not in (1, 2):
            raise ValueError("tracker object ID must be 1 or 2")
        points, labels = self._point_tensor_factory(
            request["points"], request["point_labels"]
        )
        object_states = self._object_states[session_id]
        if object_id not in object_states:
            object_states[object_id] = self._new_object_state(session_id)
        object_state = object_states[object_id]
        self._restore_shared_state(session_id, object_state)
        with self._inference_context_factory():
            result = self.model.add_new_points(
                inference_state=object_state,
                frame_idx=frame_index,
                obj_id=object_id,
                points=points,
                labels=labels,
                clear_old_points=bool(request.get("clear_old_points", True)),
                rel_coordinates=bool(request.get("rel_coordinates", True)),
            )
        self._publish_shared_features(session_id, object_state)
        returned_frame, object_ids, _, video_masks = result[:4]
        masks = _numpy(video_masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        current = self._prompt_masks[session_id].setdefault(frame_index, {})
        for index, returned_id in enumerate(_numpy(object_ids).astype(int).tolist()):
            current[returned_id] = masks[index] > 0.0
        ordered_ids = sorted(current)
        return {
            "frame_index": int(returned_frame),
            "outputs": {
                "out_obj_ids": np.asarray(ordered_ids, dtype=np.int64),
                "out_probs": np.ones(len(ordered_ids), dtype=np.float32),
                "out_binary_masks": np.stack([current[value] for value in ordered_ids]),
            },
        }

    def _restore_shared_state(self, session_id: str, state: Any) -> None:
        if "images" not in state and self._video_images.get(session_id) is not None:
            state["images"] = self._video_images[session_id]
        if "cached_features" in state:
            state["cached_features"] = self._shared_feature_cache[session_id]

    def _publish_shared_features(self, session_id: str, state: Any) -> None:
        if "cached_features" not in state:
            return
        shared = state["cached_features"]
        self._shared_feature_cache[session_id] = shared
        for candidate in self._object_states[session_id].values():
            if "cached_features" in candidate:
                candidate["cached_features"] = shared

    def _propagate(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request["session_id"])
        if session_id not in self._states:
            raise KeyError(f"unknown tracker session: {session_id}")
        object_states = self._object_states[session_id]
        if set(object_states) != {1, 2}:
            raise RuntimeError("both actor objects must be prompted before propagation")
        start = int(request["start_frame_idx"])
        maximum = int(request["max_frame_num_to_track"])
        if not 1 <= maximum <= 8:
            raise ValueError("tracker propagation responses are capped at eight frames")
        reverse = bool(request.get("reverse", False))
        direction = -1 if reverse else 1
        allowed_indices = {start + direction * offset for offset in range(maximum)}
        call_masks: dict[int, dict[int, np.ndarray]] = {}
        call_probs: dict[int, dict[int, float]] = {}

        for object_id in sorted(object_states):
            state = object_states[object_id]
            self._restore_shared_state(session_id, state)
            with self._inference_context_factory():
                generator = self.model.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=start,
                    max_frame_num_to_track=maximum,
                    reverse=reverse,
                    tqdm_disable=bool(request.get("tqdm_disable", True)),
                )
                for result in generator:
                    frame_index, object_ids, _, video_masks = result[:4]
                    frame_index = int(frame_index)
                    if frame_index not in allowed_indices:
                        raise RuntimeError(
                            f"SAM3.1 returned frame {frame_index} outside the requested chunk"
                        )
                    returned_ids = _numpy(object_ids).astype(int).reshape(-1).tolist()
                    masks = _numpy(video_masks)
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0]
                    scores = (
                        _numpy(result[4]).reshape(-1)
                        if len(result) > 4
                        else np.ones(len(returned_ids), dtype=np.float32)
                    )
                    current_masks = call_masks.setdefault(frame_index, {})
                    current_probs = call_probs.setdefault(frame_index, {})
                    for position, returned_id in enumerate(returned_ids):
                        current_masks[returned_id] = masks[position] > 0.0
                        current_probs[returned_id] = float(
                            scores[min(position, len(scores) - 1)]
                        )
            self._publish_shared_features(session_id, state)

        frames: list[dict[str, Any]] = []
        for frame_index in sorted(call_masks):
            current = call_masks[frame_index]
            if set(current) != {1, 2}:
                continue
            frames.append(
                {
                    "frame_index": frame_index,
                    "outputs": {
                        "out_obj_ids": np.asarray([1, 2], dtype=np.int64),
                        "out_probs": np.asarray(
                            [call_probs[frame_index][1], call_probs[frame_index][2]],
                            dtype=np.float32,
                        ),
                        "out_binary_masks": np.stack([current[1], current[2]]),
                    },
                }
            )
        return {"frames": frames}

    def _close_session(self, session_id: str) -> None:
        if session_id not in self._states:
            raise KeyError(f"unknown tracker session: {session_id}")
        base_state = self._states.pop(session_id)
        object_states = self._object_states.pop(session_id)
        self._resource_paths.pop(session_id, None)
        self._video_images.pop(session_id, None)
        self._shared_feature_cache.pop(session_id, None)
        self._prompt_masks.pop(session_id, None)
        reset_state = getattr(self.model, "reset_state", None)
        if callable(reset_state):
            unique = {id(base_state): base_state}
            unique.update({id(value): value for value in object_states.values()})
            for state in unique.values():
                reset_state(state)

    def close(self) -> None:
        if self._closed:
            return
        if self._states:
            raise RuntimeError(
                f"cannot close SAM3.1 tracker with active sessions: {sorted(self._states)}"
            )
        self.model = None
        self._closed = True
        self._final_cleanup()


def build_tracker_only_sam31(
    contract: RemoteRuntimeContract,
    approval: RemoteRuntimeApproval,
    *,
    runtime_bindings_factory: Callable[[], Sam31RuntimeBindings] = _load_default_runtime_bindings,
) -> TrackerOnlySam31Predictor:
    """Build the validated tracker-only model after a contract-bound remote preflight."""

    approval.require_contract(contract)
    if approval.checkpoint_sha256 != contract.checkpoint_sha256:
        raise RemoteRuntimePreflightError(
            "runtime approval checkpoint does not match the requested contract"
        )
    repo_path = str(contract.sam_repo_path.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    bindings = runtime_bindings_factory()
    model: Any | None = None
    try:
        model = bindings.build_model(
            checkpoint_path=None,
            load_from_HF=False,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=False,
            strict_state_dict_loading=False,
            device="cpu",
            compile=False,
        )
        checkpoint = bindings.load_checkpoint(contract.checkpoint_path)
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        expected_state = model.state_dict()
        remapped = remap_tracker_only_checkpoint(checkpoint, set(expected_state))
        missing = sorted(set(expected_state) - set(remapped))
        shape_mismatches = sorted(
            key
            for key, value in remapped.items()
            if tuple(value.shape) != tuple(expected_state[key].shape)
        )
        if missing or shape_mismatches:
            raise RuntimeError(
                "tracker-only checkpoint mapping failed: "
                f"{len(missing)} missing, {len(shape_mismatches)} shape mismatches"
            )
        model.load_state_dict(remapped, strict=True)
        checkpoint = None
        remapped = None
        expected_state = None
        bindings.collect()
        disable_unused_sam3_branch(model)
        model.to(device="cuda").eval()

        def final_cleanup() -> None:
            bindings.collect()
            bindings.release_cuda_cache()

        return TrackerOnlySam31Predictor(
            model,
            inference_context_factory=bindings.inference_context_factory,
            point_tensor_factory=bindings.point_tensor_factory,
            session_cleanup=final_cleanup,
            final_cleanup=final_cleanup,
        )
    except Exception:
        model = None
        bindings.collect()
        bindings.release_cuda_cache()
        raise
