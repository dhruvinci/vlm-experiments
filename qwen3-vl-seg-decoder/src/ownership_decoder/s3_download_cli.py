from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .cloud_controller import load_secret_environment
from .s3_download import download_run_prefix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a completed mask campaign from a RunPod network volume."
    )
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--network-volume-id", default="0vnqaqwt1r")
    parser.add_argument("--datacenter", default="US-NC-2")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_id or "/" in args.run_id or ".." in args.run_id:
        raise ValueError("run ID must be one safe path component")
    secrets = load_secret_environment(
        args.env_file,
        required=("RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY"),
    )
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        aws_access_key_id=secrets["RUNPOD_S3_ACCESS_KEY"],
        aws_secret_access_key=secrets["RUNPOD_S3_SECRET_KEY"],
        region_name=args.datacenter,
        endpoint_url=f"https://s3api-{args.datacenter.lower()}.runpod.io/",
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )
    result = download_run_prefix(
        client,
        bucket=args.network_volume_id,
        prefix=f"qwen38-campaign/mask-campaign/runs/{args.run_id}/",
        output_root=args.output_root,
    )
    print(json.dumps({"status": "complete", "object_count": result["object_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
