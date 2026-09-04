from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ownership_decoder.cloud_launch import (
    FULL_GPU_TYPES,
    MIG_GPU_TYPE,
    MaskCloudLaunchContract,
    launch_mask_pod_once,
    monitor_mask_pod,
)


def _fixture(root: Path) -> MaskCloudLaunchContract:
    packet = root / "manifest.json"
    packet.write_text('{"format":"packet"}\n', encoding="utf-8")
    digest = hashlib.sha256(packet.read_bytes()).hexdigest()
    return MaskCloudLaunchContract(
        run_id="sam31-breadth-20260904t120000z",
        packet_manifest_path=packet,
        packet_sha256=digest,
        approved_by="user",
        approved_at=datetime(2026, 9, 4, 12, tzinfo=timezone.utc),
        network_volume_id="0vnqaqwt1r",
        datacenter="US-NC-2",
        container_image=(
            "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:"
            + "8" * 64
        ),
        spend_before_usd=6.30,
        frozen_hourly_rate_usd=2.09,
        terminate_at_usd=10.30,
        max_runtime_seconds=7200,
    )


class FakeRunPod:
    def __init__(self, created: dict | None = None) -> None:
        self.created = created or {
            "id": "pod-123",
            "gpu": {"displayName": FULL_GPU_TYPES[0]},
            "costPerHr": 2.09,
        }
        self.calls: list[tuple[str, str, dict | None]] = []

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        self.calls.append((method, path, payload))
        if method == "POST":
            return self.created
        if method == "DELETE":
            return {}
        raise AssertionError((method, path, payload))


class CloudLaunchTests(unittest.TestCase):
    def test_payload_is_exactly_scoped_to_the_approved_gpu_volume_and_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract = _fixture(Path(raw))
            payload = contract.pod_payload(api_key="secret")

            self.assertEqual(payload["cloudType"], "SECURE")
            self.assertEqual(payload["gpuCount"], 1)
            self.assertEqual(payload["gpuTypeIds"], list(FULL_GPU_TYPES))
            self.assertEqual(payload["networkVolumeId"], "0vnqaqwt1r")
            self.assertEqual(payload["volumeMountPath"], "/workspace")
            self.assertEqual(payload["allowedCudaVersions"], ["13.0"])
            self.assertEqual(
                payload["dockerStartCmd"],
                ["/bin/bash", "/workspace/qwen38-campaign/mask-campaign/v2/packet/launch.sh"],
            )
            self.assertEqual(payload["env"]["MASK_CAMPAIGN_RUN_ID"], contract.run_id)
            self.assertEqual(payload["env"]["RUNPOD_API_KEY"], "secret")
            self.assertEqual(payload["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertFalse(payload["interruptible"])

    def test_mig_payload_cannot_fall_back_to_a_full_or_other_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = _fixture(Path(raw))
            contract = MaskCloudLaunchContract(
                **{**base.__dict__, "purpose": "mig"}
            )

            payload = contract.pod_payload(api_key="secret")

            self.assertEqual(payload["gpuTypeIds"], [MIG_GPU_TYPE])
            self.assertEqual(
                payload["dockerStartCmd"],
                ["/bin/bash", "/workspace/qwen38-campaign/mask-campaign/v2/packet/preflight.sh"],
            )

    def test_launch_writes_control_and_local_state_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _fixture(root)
            client = FakeRunPod()
            controls = []
            state_path = root / "launch-state.json"
            started = datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc)

            result = launch_mask_pod_once(
                contract,
                api_key="secret",
                request_fn=client.request,
                put_control_fn=lambda key, value: controls.append((key, value)),
                state_path=state_path,
                now_fn=lambda: started,
            )

            self.assertEqual(result["pod_id"], "pod-123")
            self.assertEqual(len(controls), 1)
            key, control = controls[0]
            self.assertEqual(key, f"qwen38-campaign/mask-campaign/v2/control/{contract.run_id}.json")
            self.assertEqual(control["pod_id"], "pod-123")
            self.assertEqual(control["packet_sha256"], contract.packet_sha256)
            self.assertEqual(control["billing_started_at"], started.isoformat())
            self.assertNotIn("api_key", json.loads(state_path.read_text()))

            with self.assertRaisesRegex(FileExistsError, "replacement"):
                launch_mask_pod_once(
                    contract,
                    api_key="secret",
                    request_fn=client.request,
                    put_control_fn=lambda *_args: None,
                    state_path=state_path,
                    now_fn=lambda: started,
                )
            self.assertEqual(
                len([call for call in client.calls if call[0] == "POST"]), 1
            )

    def test_control_upload_failure_terminates_new_pod(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _fixture(root)
            client = FakeRunPod()

            with self.assertRaisesRegex(OSError, "S3 failed"):
                launch_mask_pod_once(
                    contract,
                    api_key="secret",
                    request_fn=client.request,
                    put_control_fn=lambda *_args: (_ for _ in ()).throw(
                        OSError("S3 failed")
                    ),
                    state_path=root / "state.json",
                    now_fn=lambda: datetime(2026, 9, 4, 12, tzinfo=timezone.utc),
                )

            self.assertIn(("DELETE", "/pods/pod-123", None), client.calls)

    def test_wrong_returned_hardware_is_terminated_before_control(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _fixture(root)
            client = FakeRunPod(
                {
                    "id": "pod-bad",
                    "gpu": {
                        "displayName": (
                            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
                        )
                    },
                    "costPerHr": 1.0,
                }
            )
            controls = []

            with self.assertRaisesRegex(RuntimeError, "unapproved hardware"):
                launch_mask_pod_once(
                    contract,
                    api_key="secret",
                    request_fn=client.request,
                    put_control_fn=lambda *args: controls.append(args),
                    state_path=root / "state.json",
                    now_fn=lambda: datetime(2026, 9, 4, 12, tzinfo=timezone.utc),
                )

            self.assertEqual(controls, [])
            self.assertIn(("DELETE", "/pods/pod-bad", None), client.calls)

    def test_packet_mutation_is_rejected_before_cloud_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _fixture(root)
            contract.packet_manifest_path.write_text("mutated", encoding="utf-8")
            client = FakeRunPod()

            with self.assertRaisesRegex(RuntimeError, "packet SHA-256"):
                launch_mask_pod_once(
                    contract,
                    api_key="secret",
                    request_fn=client.request,
                    put_control_fn=lambda *_args: None,
                    state_path=root / "state.json",
                )

            self.assertEqual(client.calls, [])

    def test_monitor_terminates_at_local_budget_guard(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _fixture(root)
            started = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)
            terminated = []

            result = monitor_mask_pod(
                contract,
                pod_id="pod-123",
                billing_started_at=started,
                get_pod_fn=lambda _pod: {"desiredStatus": "RUNNING", "costPerHr": 4.0},
                terminate_fn=terminated.append,
                read_remote_json_fn=lambda _key: {"decision": "continue"},
                remote_exists_fn=lambda _key: False,
                local_log=root / "monitor.jsonl",
                now_fn=lambda: started + timedelta(hours=1),
                sleep_fn=lambda _seconds: self.fail("budget decision must be immediate"),
            )

            self.assertEqual(result, "terminate_budget")
            self.assertEqual(terminated, ["pod-123"])

    def test_monitor_terminates_when_no_guard_heartbeat_arrives(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _fixture(root)
            started = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)
            terminated = []

            result = monitor_mask_pod(
                contract,
                pod_id="pod-123",
                billing_started_at=started,
                get_pod_fn=lambda _pod: {"desiredStatus": "RUNNING", "costPerHr": 2.09},
                terminate_fn=terminated.append,
                read_remote_json_fn=lambda _key: (_ for _ in ()).throw(
                    FileNotFoundError("not yet")
                ),
                remote_exists_fn=lambda _key: False,
                local_log=root / "monitor.jsonl",
                now_fn=lambda: started + timedelta(seconds=601),
                sleep_fn=lambda _seconds: self.fail("heartbeat timeout must be immediate"),
                startup_heartbeat_grace_seconds=600,
            )

            self.assertEqual(result, "terminate_missing_guard")
            self.assertEqual(terminated, ["pod-123"])

    def test_mig_monitor_uses_preflight_run_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = _fixture(root)
            contract = MaskCloudLaunchContract(**{**base.__dict__, "purpose": "mig"})
            started = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)
            observed_keys = []
            terminated = []

            def exists(key: str) -> bool:
                observed_keys.append(key)
                return key.endswith("/RUN_COMPLETE")

            result = monitor_mask_pod(
                contract,
                pod_id="pod-mig",
                billing_started_at=started,
                get_pod_fn=lambda _pod: {"desiredStatus": "RUNNING", "costPerHr": 0.59},
                terminate_fn=terminated.append,
                read_remote_json_fn=lambda key: {"key": key},
                remote_exists_fn=exists,
                local_log=root / "monitor.jsonl",
                now_fn=lambda: started + timedelta(seconds=30),
                sleep_fn=lambda _seconds: self.fail("completion must terminate immediately"),
            )

            expected_prefix = (
                "qwen38-campaign/mask-campaign/preflight-runs/"
                f"{contract.run_id}"
            )
            self.assertEqual(result, "terminate_success")
            self.assertTrue(all(key.startswith(expected_prefix) for key in observed_keys))
            self.assertEqual(terminated, ["pod-mig"])

    def test_monitor_terminates_a_stale_guard_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _fixture(root)
            started = datetime(2026, 9, 4, 12, tzinfo=timezone.utc)
            terminated = []

            result = monitor_mask_pod(
                contract,
                pod_id="pod-123",
                billing_started_at=started,
                get_pod_fn=lambda _pod: {"desiredStatus": "RUNNING", "costPerHr": 2.09},
                terminate_fn=terminated.append,
                read_remote_json_fn=lambda key: (
                    {"at": (started + timedelta(seconds=300)).isoformat()}
                    if key.endswith("pod-guard-heartbeat.json")
                    else {"updated_at": (started + timedelta(seconds=599)).isoformat()}
                ),
                remote_exists_fn=lambda _key: False,
                local_log=root / "monitor.jsonl",
                now_fn=lambda: started + timedelta(seconds=601),
                sleep_fn=lambda _seconds: self.fail("stale guard must terminate immediately"),
                heartbeat_max_age_seconds=180,
            )

            self.assertEqual(result, "terminate_stale_guard")
            self.assertEqual(terminated, ["pod-123"])


if __name__ == "__main__":
    unittest.main()
