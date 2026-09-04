from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.remote_mask_campaign import (
    RemoteMaskCampaignSpec,
    run_remote_mask_campaign,
)
from ownership_decoder.remote_preflight import (
    RemoteHardwareSnapshot,
    RemoteRuntimeApproval,
    RequiredArtifact,
)


GIB = 1024**3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(root: Path) -> Path:
    clip = root / "inputs" / "clip_a"
    frames = clip / "frames"
    frames.mkdir(parents=True)
    frame = frames / "frame_000001.jpg"
    frame.write_bytes(b"frame")
    manifest = {
        "contract": {"clip_id": "clip_a"},
        "decode": {"frame_count": 1},
        "frames": [
            {
                "clip_frame_index": 0,
                "path": "frames/frame_000001.jpg",
                "sha256": _sha(frame),
                "height": 4,
                "width": 5,
            }
        ],
    }
    manifest_path = clip / "clip-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    payload = {
        "schema_version": "1.0",
        "clip_id": "clip_a",
        "clip_manifest_path": "clip_a/clip-manifest.json",
        "clip_manifest_sha256": _sha(manifest_path),
        "seeds": [
            {
                "frame_index": 0,
                "actors": [
                    {
                        "actor_id": "A1",
                        "bbox": [0.0, 0.0, 0.45, 1.0],
                        "positive_points": [[0.1, 0.2], [0.3, 0.8]],
                        "negative_points": [[0.7, 0.2], [0.9, 0.8]],
                    },
                    {
                        "actor_id": "A2",
                        "bbox": [0.55, 0.0, 1.0, 1.0],
                        "positive_points": [[0.7, 0.2], [0.9, 0.8]],
                        "negative_points": [[0.1, 0.2], [0.3, 0.8]],
                    },
                ],
            }
        ],
        "propagations": [
            {"start_frame_idx": 0, "max_frame_num_to_track": 1, "reverse": False}
        ],
    }
    config = root / "clip_a.json"
    config.write_text(json.dumps(payload))
    return config


def _spec(root: Path, config: Path) -> RemoteMaskCampaignSpec:
    repo = root / "repo"
    repo.mkdir()
    checkpoint = root / "sam31.pt"
    checkpoint.write_bytes(b"sam31")
    model = root / "sam3"
    model.mkdir()
    extras = []
    for name in ("config.json", "model.safetensors", "processor_config.json"):
        path = model / name
        path.write_bytes(name.encode())
        extras.append(RequiredArtifact(path, _sha(path), path.stat().st_size))
    return RemoteMaskCampaignSpec(
        config_paths=(config,),
        input_root=root / "inputs",
        output_root=root / "outputs",
        sam_repo_path=repo,
        sam_repo_revision="repo-rev",
        sam31_checkpoint_path=checkpoint,
        sam31_checkpoint_sha256="a" * 64,
        sam3_model_directory=model,
        sam3_model_revision="hf-rev",
        sam3_model_artifacts=tuple(extras),
        workspace_path=root,
        required_distribution_versions=(("torch", "2.12.1+cu130"),),
        attempt_index=0,
    )


def _approval(spec: RemoteMaskCampaignSpec) -> RemoteRuntimeApproval:
    contract = spec.runtime_contract
    return RemoteRuntimeApproval(
        contract_sha256=contract.sha256,
        checkpoint_sha256=contract.checkpoint_sha256,
        hardware=RemoteHardwareSnapshot(
            gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            gpu_total_bytes=96 * GIB,
            gpu_free_bytes=90 * GIB,
            compute_capability=(12, 0),
            driver_version=(580, 65, 6),
            host_available_bytes=64 * GIB,
            workspace_free_bytes=200 * GIB,
        ),
        additional_artifact_sha256=tuple(
            (str(item.path.resolve()), item.sha256)
            for item in contract.additional_artifacts
        ),
    )


class Telemetry:
    def __init__(self, path: Path, **kwargs) -> None:
        self.events = []

    def record(self, value: dict) -> None:
        self.events.append(value)


class Predictor:
    def __init__(self, name: str, order: list[str]) -> None:
        self.name = name
        self.order = order
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.order.append(f"{self.name}_closed")


class RemoteMaskCampaignTests(unittest.TestCase):
    def test_preflights_once_then_never_overlaps_tracker_and_image_models(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = _config(root)
            spec = _spec(root, config)
            order = []
            preflights = []

            def preflight(contract):
                preflights.append(contract)
                return _approval(spec)

            def tracker_builder(contract, approval):
                self.assertTrue((spec.attempt_dir / "preflight.json").is_file())
                order.append("tracker_built")
                return Predictor("tracker", order)

            def tracker_canary(predictor, plan):
                self.assertEqual(plan.clip_id, "clip_a")
                order.append("tracker_canary")
                return {"format": "tracker-canary", "clip_id": plan.clip_id}

            def tracking_runner(configs, **kwargs):
                predictor = kwargs["predictor_factory"]()
                predictor.close()
                return {"format": "tracking", "clip_count": 1, "frame_count": 1}

            def image_builder(contract, approval, *, model_directory):
                self.assertEqual(order[-1], "tracker_closed")
                self.assertEqual(model_directory, spec.sam3_model_directory)
                order.append("image_built")
                return Predictor("image", order)

            def image_canary(
                predictor,
                plan,
                *,
                tracker_root,
                tracker_backend,
                tracker_revision,
                minimum_prompt_area,
                box_padding_fraction,
            ):
                self.assertEqual(order[-1], "image_built")
                self.assertEqual(plan.clip_id, "clip_a")
                self.assertEqual(tracker_root, spec.tracker_root)
                order.append("image_canary")
                return {"format": "image-canary", "clip_id": plan.clip_id}

            def image_runner(configs, **kwargs):
                predictor = kwargs["predictor_factory"]()
                predictor.close()
                return {"format": "agreement", "clip_count": 1, "frame_count": 1}

            manifest = run_remote_mask_campaign(
                spec,
                preflight_fn=preflight,
                sam31_builder=tracker_builder,
                sam31_canary_fn=tracker_canary,
                tracking_runner=tracking_runner,
                sam3_builder=image_builder,
                sam3_image_canary_fn=image_canary,
                image_runner=image_runner,
                telemetry_factory=Telemetry,
            )

            self.assertEqual(len(preflights), 1)
            self.assertEqual(
                order,
                [
                    "tracker_built",
                    "tracker_canary",
                    "tracker_closed",
                    "image_built",
                    "image_canary",
                    "image_closed",
                ],
            )
            self.assertEqual(manifest["frame_count"], 1)
            self.assertTrue((spec.output_root / "RUN_COMPLETE").is_file())
            self.assertTrue((spec.attempt_dir / "SUCCESS.json").is_file())
            report = json.loads((spec.attempt_dir / "preflight.json").read_text())
            self.assertEqual(len(report["contract"]["additional_artifacts"]), 3)
            self.assertEqual(
                json.loads((spec.attempt_dir / "sam31-canary.json").read_text())["format"],
                "tracker-canary",
            )
            self.assertEqual(
                json.loads((spec.attempt_dir / "sam3-image-canary.json").read_text())["format"],
                "image-canary",
            )

    def test_invalid_config_fails_before_preflight_or_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = _config(root)
            spec = _spec(root, config)
            config.write_text("{}")
            called = False

            def preflight(contract):
                nonlocal called
                called = True
                raise AssertionError("preflight called")

            with self.assertRaises(ValueError):
                run_remote_mask_campaign(spec, preflight_fn=preflight)

            self.assertFalse(called)

    def test_tracker_failure_is_recorded_and_prevents_image_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = _config(root)
            spec = _spec(root, config)
            image_loaded = False

            def tracking_runner(configs, **kwargs):
                raise RuntimeError("tracker failed")

            def image_builder(*args, **kwargs):
                nonlocal image_loaded
                image_loaded = True
                raise AssertionError("image loaded")

            with self.assertRaisesRegex(RuntimeError, "tracker failed"):
                run_remote_mask_campaign(
                    spec,
                    preflight_fn=lambda contract: _approval(spec),
                    tracking_runner=tracking_runner,
                    sam3_builder=image_builder,
                    telemetry_factory=Telemetry,
                )

            self.assertFalse(image_loaded)
            failure = json.loads((spec.attempt_dir / "FAILURE.json").read_text())
            self.assertEqual(failure["exception_type"], "RuntimeError")
            self.assertFalse((spec.output_root / "RUN_COMPLETE").exists())

    def test_tracker_canary_failure_closes_model_before_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = _config(root)
            spec = _spec(root, config)
            order = []

            def tracker_builder(contract, approval):
                order.append("tracker_built")
                return Predictor("tracker", order)

            def tracker_canary(predictor, plan):
                order.append("tracker_canary_failed")
                raise RuntimeError("canary failed")

            with self.assertRaisesRegex(RuntimeError, "canary failed"):
                run_remote_mask_campaign(
                    spec,
                    preflight_fn=lambda contract: _approval(spec),
                    sam31_builder=tracker_builder,
                    sam31_canary_fn=tracker_canary,
                    telemetry_factory=Telemetry,
                )

            self.assertEqual(
                order,
                ["tracker_built", "tracker_canary_failed", "tracker_closed"],
            )
            self.assertTrue((spec.attempt_dir / "FAILURE.json").is_file())
            self.assertFalse((spec.attempt_dir / "sam31-canary.json").exists())


if __name__ == "__main__":
    unittest.main()
