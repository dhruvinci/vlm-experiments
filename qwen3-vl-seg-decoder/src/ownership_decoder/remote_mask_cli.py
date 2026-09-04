from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .remote_mask_campaign import RemoteMaskCampaignSpec, run_remote_mask_campaign
from .remote_preflight import RemoteRuntimePreflightError, RequiredArtifact
from .remote_telemetry import RuntimeFatalSafetyError, RuntimeWorkerRestartRequired


DEFAULT_DISTRIBUTION_PINS = (
    "torch==2.12.1+cu130",
    "torchvision==0.27.1+cu130",
    "annotated-doc==0.0.5",
    "anyio==4.14.2",
    "certifi==2026.7.22",
    "click==8.5.0",
    "einops==0.8.2",
    "filelock==3.32.4",
    "fsspec==2026.7.0",
    "ftfy==6.1.1",
    "h11==0.16.0",
    "hf-xet==1.6.0",
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "huggingface-hub==1.28.0",
    "idna==3.19",
    "iopath==0.1.10",
    "markdown-it-py==4.2.0",
    "mdurl==0.1.2",
    "numpy==1.26.4",
    "packaging==26.3",
    "Pillow==12.3.0",
    "portalocker==4.3.0",
    "psutil==7.2.2",
    "Pygments==2.21.0",
    "PyYAML==6.0.3",
    "regex==2026.7.19",
    "rich==15.0.0",
    "sam3==0.1.0",
    "safetensors==0.8.0",
    "shellingham==1.5.4",
    "timm==1.0.28",
    "tokenizers==0.23.1",
    "tqdm==4.70.0",
    "transformers==5.16.1",
    "typer==0.27.1",
    "typing-extensions==4.16.0",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run sequential tracker-only SAM3.1 and base-SAM3 image agreement "
            "on a preflight-approved 96 GB Blackwell GPU."
        )
    )
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sam-repo", required=True, type=Path)
    parser.add_argument("--sam-revision", required=True)
    parser.add_argument("--sam31-checkpoint", required=True, type=Path)
    parser.add_argument("--sam31-checkpoint-sha256", required=True)
    parser.add_argument("--sam3-model-directory", required=True, type=Path)
    parser.add_argument("--sam3-artifact-manifest", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--minimum-prompt-area", type=int, default=64)
    parser.add_argument("--box-padding-fraction", type=float, default=0.01)
    parser.add_argument(
        "--require-distribution",
        action="append",
        default=list(DEFAULT_DISTRIBUTION_PINS),
        metavar="NAME==VERSION",
    )
    return parser


def _parse_distribution_pins(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    parsed: dict[str, str] = {}
    for value in values:
        if value.count("==") != 1:
            raise ValueError(f"distribution pin must use name==version syntax: {value!r}")
        name, version = (piece.strip() for piece in value.split("==", 1))
        normalized = name.lower().replace("_", "-")
        if not name or not version:
            raise ValueError(f"distribution pin must use name==version syntax: {value!r}")
        if normalized in parsed:
            raise ValueError(f"duplicate distribution pin: {name}")
        parsed[normalized] = version
    return tuple(sorted(parsed.items()))


def _exact_keys(value: Any, expected: set[str], *, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{context} keys are invalid: missing={missing}, unexpected={unexpected}"
        )


def _load_artifact_manifest(
    path: Path,
    *,
    model_directory: Path,
) -> tuple[str, tuple[RequiredArtifact, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"SAM3 artifact manifest is unreadable: {path}") from error
    _exact_keys(
        payload,
        {"format", "model_revision", "artifacts"},
        context="SAM3 artifact manifest",
    )
    if payload["format"] != "sam3-model-artifacts-v1":
        raise ValueError("SAM3 artifact manifest format is unsupported")
    revision = str(payload["model_revision"])
    if not revision.strip():
        raise ValueError("SAM3 artifact manifest model_revision cannot be empty")
    if not isinstance(payload["artifacts"], list) or not payload["artifacts"]:
        raise ValueError("SAM3 artifact manifest artifacts must be a non-empty list")
    root = model_directory.resolve()
    records: list[RequiredArtifact] = []
    seen: set[Path] = set()
    for index, value in enumerate(payload["artifacts"]):
        _exact_keys(
            value,
            {"relative_path", "sha256", "size_bytes"},
            context=f"SAM3 artifact {index}",
        )
        relative = Path(str(value["relative_path"]))
        if relative.is_absolute():
            raise ValueError("SAM3 artifact relative_path cannot be absolute")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("SAM3 artifact relative_path cannot escape the model directory")
        if resolved in seen:
            raise ValueError(f"duplicate SAM3 artifact path: {relative}")
        seen.add(resolved)
        records.append(
            RequiredArtifact(
                path=resolved,
                sha256=str(value["sha256"]),
                size_bytes=int(value["size_bytes"]),
            )
        )
    records.sort(key=lambda item: str(item.path))
    return revision, tuple(records)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[RemoteMaskCampaignSpec], dict[str, Any]] = run_remote_mask_campaign,
) -> int:
    args = build_parser().parse_args(argv)
    model_revision, model_artifacts = _load_artifact_manifest(
        args.sam3_artifact_manifest,
        model_directory=args.sam3_model_directory,
    )
    spec = RemoteMaskCampaignSpec(
        config_paths=tuple(args.config),
        input_root=args.input_root,
        output_root=args.output_root,
        sam_repo_path=args.sam_repo,
        sam_repo_revision=args.sam_revision,
        sam31_checkpoint_path=args.sam31_checkpoint,
        sam31_checkpoint_sha256=args.sam31_checkpoint_sha256,
        sam3_model_directory=args.sam3_model_directory,
        sam3_model_revision=model_revision,
        sam3_model_artifacts=model_artifacts,
        workspace_path=args.workspace,
        required_distribution_versions=_parse_distribution_pins(
            args.require_distribution
        ),
        attempt_index=args.attempt_index,
        minimum_prompt_area=args.minimum_prompt_area,
        box_padding_fraction=args.box_padding_fraction,
    )
    manifest = runner(spec)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest_format": manifest.get("format"),
                "frame_count": manifest.get("frame_count"),
            }
        )
    )
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
