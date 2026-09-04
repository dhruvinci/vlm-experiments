from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .cloud_launch import (
    MaskCloudLaunchContract,
    launch_mask_pod_once,
    monitor_mask_pod,
)


REST_BASE = "https://rest.runpod.io/v1"
STAGING_COMPLETE_KEY = "qwen38-campaign/mask-campaign/v2/STAGING_COMPLETE.json"
MIG_COMPLETE_KEY = "qwen38-campaign/mask-campaign/v2/MIG_PREFLIGHT_COMPLETE.json"


def load_secret_environment(
    path: str | Path,
    *,
    required: Sequence[str],
) -> dict[str, str]:
    """Read only explicitly allowed secrets from KEY=value or KEY: value files."""

    requested = set(required)
    if not requested or len(requested) != len(tuple(required)):
        raise ValueError("required secret names must be non-empty and unique")
    values: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        separators = [position for token in ("=", ":") if (position := line.find(token)) >= 0]
        if not separators:
            continue
        index = min(separators)
        key = line[:index].strip()
        if key not in requested:
            continue
        value = line[index + 1 :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value:
            raise EnvironmentError(f"required secret is empty: {key}")
        if key in values:
            raise ValueError(f"duplicate secret key: {key}")
        values[key] = value
    missing = sorted(requested - set(values))
    if missing:
        raise EnvironmentError(f"required secrets are missing: {missing}")
    return values


def require_remote_staging(
    value: Any,
    *,
    packet_sha256: str,
    expected_artifact_count: int | None = None,
    expected_total_bytes: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("format") != "ownership-mask-staging-complete-v1":
        raise RuntimeError("remote staging sentinel format is invalid")
    if value.get("packet_sha256") != packet_sha256:
        raise RuntimeError("remote staging is bound to a different packet")
    artifact_count = int(value.get("artifact_count", 0))
    total_bytes = int(value.get("total_bytes", 0))
    if artifact_count < 1 or total_bytes < 1:
        raise RuntimeError("remote staging inventory is empty")
    if expected_artifact_count is not None and artifact_count != expected_artifact_count:
        raise RuntimeError("remote staging artifact count mismatch")
    if expected_total_bytes is not None and total_bytes != expected_total_bytes:
        raise RuntimeError("remote staging byte total mismatch")
    inventory_sha = str(value.get("inventory_sha256", ""))
    if len(inventory_sha) != 64:
        raise RuntimeError("remote staging inventory SHA-256 is invalid")
    return value


class RunPodRestApi:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("RunPod API key is required")
        self._api_key = api_key

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            REST_BASE + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            if method == "DELETE" and error.code == 404:
                return {}
            safe = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(
                f"RunPod API {method} {path} failed with HTTP {error.code}: {safe}"
            ) from error
        return json.loads(body) if body else {}

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(pod_id, safe="")
        value = self.request("GET", f"/pods/{encoded}")
        if not isinstance(value, dict):
            raise RuntimeError("RunPod returned an invalid Pod response")
        return value

    def terminate(self, pod_id: str) -> None:
        encoded = urllib.parse.quote(pod_id, safe="")
        self.request("DELETE", f"/pods/{encoded}")


class S3ControlStore:
    def __init__(
        self,
        *,
        volume_id: str,
        datacenter: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        import boto3
        from botocore.config import Config

        self.volume_id = volume_id
        self.client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=datacenter,
            endpoint_url=f"https://s3api-{datacenter.lower()}.runpod.io/",
            config=Config(
                retries={"max_attempts": 10, "mode": "standard"},
                s3={"addressing_style": "path"},
            ),
        )

    def put_json(self, key: str, value: dict[str, Any]) -> None:
        self.client.put_object(
            Bucket=self.volume_id,
            Key=key,
            Body=(json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    def read_json(self, key: str) -> dict[str, Any]:
        response = self.client.get_object(Bucket=self.volume_id, Key=key)
        body = response["Body"]
        try:
            value = json.loads(body.read())
        finally:
            body.close()
        if not isinstance(value, dict):
            raise RuntimeError(f"remote JSON object is invalid: {key}")
        return value

    def exists(self, key: str) -> bool:
        response = self.client.list_objects_v2(
            Bucket=self.volume_id,
            Prefix=key,
            MaxKeys=1,
        )
        return any(item.get("Key") == key for item in response.get("Contents", []))


def _terminate_with_retries(
    api: RunPodRestApi,
    pod_id: str,
    *,
    attempts: int = 120,
    interval_seconds: float = 5.0,
) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            api.terminate(pod_id)
            return
        except Exception as error:
            last_error = error
            time.sleep(interval_seconds)
    raise RuntimeError("emergency Pod termination did not succeed") from last_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch and independently monitor exactly one mask-campaign Pod."
    )
    parser.add_argument("--purpose", choices=("mig", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--network-volume-id", default="0vnqaqwt1r")
    parser.add_argument("--datacenter", default="US-NC-2")
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--spend-before-usd", type=float, required=True)
    parser.add_argument("--hourly-rate-usd", type=float, required=True)
    parser.add_argument("--terminate-at-usd", type=float, required=True)
    parser.add_argument("--max-runtime-seconds", type=float, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--monitor-log", type=Path, required=True)
    parser.add_argument("--expected-staging-artifacts", type=int, default=717)
    parser.add_argument("--expected-staging-bytes", type=int, default=7_100_837_883)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    secrets = load_secret_environment(
        args.env_file,
        required=(
            "RUNPOD_API_KEY",
            "RUNPOD_S3_ACCESS_KEY",
            "RUNPOD_S3_SECRET_KEY",
        ),
    )
    store = S3ControlStore(
        volume_id=args.network_volume_id,
        datacenter=args.datacenter,
        access_key=secrets["RUNPOD_S3_ACCESS_KEY"],
        secret_key=secrets["RUNPOD_S3_SECRET_KEY"],
    )
    require_remote_staging(
        store.read_json(STAGING_COMPLETE_KEY),
        packet_sha256=args.packet_sha256,
        expected_artifact_count=args.expected_staging_artifacts,
        expected_total_bytes=args.expected_staging_bytes,
    )
    if args.purpose == "full":
        preflight = store.read_json(MIG_COMPLETE_KEY)
        if (
            preflight.get("format") != "ownership-mask-mig-preflight-v1"
            or preflight.get("packet_sha256") != args.packet_sha256
        ):
            raise RuntimeError("full Pod requires a packet-bound MIG preflight")

    approved_at = datetime.fromisoformat(args.approved_at)
    contract = MaskCloudLaunchContract(
        run_id=args.run_id,
        packet_manifest_path=args.packet_manifest,
        packet_sha256=args.packet_sha256,
        approved_by=args.approved_by,
        approved_at=approved_at,
        network_volume_id=args.network_volume_id,
        datacenter=args.datacenter,
        container_image=args.container_image,
        spend_before_usd=args.spend_before_usd,
        frozen_hourly_rate_usd=args.hourly_rate_usd,
        terminate_at_usd=args.terminate_at_usd,
        max_runtime_seconds=args.max_runtime_seconds,
        purpose=args.purpose,
    )
    if args.dry_run:
        payload = contract.pod_payload(api_key="<redacted>")
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "remote_staging_verified": True,
                    "packet_sha256": args.packet_sha256,
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    api = RunPodRestApi(secrets["RUNPOD_API_KEY"])
    launch = launch_mask_pod_once(
        contract,
        api_key=secrets["RUNPOD_API_KEY"],
        request_fn=api.request,
        put_control_fn=store.put_json,
        state_path=args.state_path,
    )
    pod_id = str(launch["pod_id"])
    started = datetime.fromisoformat(str(launch["billing_started_at"]))
    try:
        decision = monitor_mask_pod(
            contract,
            pod_id=pod_id,
            billing_started_at=started,
            get_pod_fn=api.get_pod,
            terminate_fn=api.terminate,
            read_remote_json_fn=store.read_json,
            remote_exists_fn=store.exists,
            local_log=args.monitor_log,
        )
    except BaseException:
        _terminate_with_retries(api, pod_id)
        raise
    print(
        json.dumps(
            {
                "status": decision,
                "purpose": args.purpose,
                "pod_id": pod_id,
                "run_id": args.run_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
