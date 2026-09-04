from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .remote_campaign import RemoteSam31CampaignSpec, run_remote_sam31_campaign
from .remote_preflight import RemoteRuntimePreflightError
from .remote_telemetry import RuntimeFatalSafetyError, RuntimeWorkerRestartRequired


DEFAULT_DISTRIBUTION_PINS = (
    "torch==2.12.1+cu130",
    "torchvision==0.27.1+cu130",
    "numpy==1.26.4",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed tracker-only SAM3.1 breadth campaign."
    )
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sam-repo", required=True, type=Path)
    parser.add_argument("--sam-revision", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument(
        "--require-distribution",
        action="append",
        default=list(DEFAULT_DISTRIBUTION_PINS),
        metavar="NAME==VERSION",
        help="Exact installed distribution pin; repeat for additional dependencies.",
    )
    return parser


def _parse_distribution_pins(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    parsed: dict[str, str] = {}
    for value in values:
        if value.count("==") != 1:
            raise ValueError(
                f"distribution pin must use exactly name==version syntax: {value!r}"
            )
        name, version = (piece.strip() for piece in value.split("==", 1))
        if not name or not version:
            raise ValueError(
                f"distribution pin must use non-empty name==version syntax: {value!r}"
            )
        if name in parsed:
            raise ValueError(f"duplicate distribution pin: {name}")
        parsed[name] = version
    return tuple(sorted(parsed.items()))


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[RemoteSam31CampaignSpec], dict[str, Any]] = run_remote_sam31_campaign,
) -> int:
    args = build_parser().parse_args(argv)
    spec = RemoteSam31CampaignSpec(
        config_paths=tuple(args.config),
        input_root=args.input_root,
        output_root=args.output_root,
        sam_repo_path=args.sam_repo,
        sam_repo_revision=args.sam_revision,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        workspace_path=args.workspace,
        required_distribution_versions=_parse_distribution_pins(
            args.require_distribution
        ),
        attempt_index=args.attempt_index,
    )
    manifest = runner(spec)
    print(json.dumps({"status": "complete", "manifest_format": manifest.get("format")}))
    return 0


def _is_recoverable_oom(error: BaseException) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        isinstance(error, MemoryError)
        or "outofmemory" in name
        or "cuda out of memory" in message
        or "cuda error: out of memory" in message
    )


def entrypoint(
    argv: Sequence[str] | None = None,
    *,
    main_fn: Callable[[Sequence[str] | None], int] = main,
) -> int:
    try:
        return main_fn(argv)
    except RuntimeWorkerRestartRequired as error:
        print(f"recoverable worker restart requested: {error}", file=sys.stderr)
        return 75
    except (RemoteRuntimePreflightError, RuntimeFatalSafetyError) as error:
        print(f"fatal remote safety failure: {error}", file=sys.stderr)
        return 70
    except Exception as error:
        if _is_recoverable_oom(error):
            print(f"recoverable worker OOM: {error}", file=sys.stderr)
            return 75
        raise
