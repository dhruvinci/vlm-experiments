from __future__ import annotations

import hashlib
import json
import os
import tempfile
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .remote_preflight import (
    RemoteRuntimeApproval,
    RemoteRuntimeContract,
    perform_remote_preflight,
)
from .remote_telemetry import RuntimeTelemetryWriter
from .sam31_tracker_adapter import build_tracker_only_sam31
from .tracking import TrackingPlan, load_tracking_plan_config
from .tracking_campaign import run_tracking_campaign


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


@dataclass(frozen=True)
class RemoteSam31CampaignSpec:
    config_paths: tuple[Path, ...]
    input_root: Path
    output_root: Path
    sam_repo_path: Path
    sam_repo_revision: str
    checkpoint_path: Path
    checkpoint_sha256: str
    workspace_path: Path
    required_distribution_versions: tuple[tuple[str, str], ...]
    attempt_index: int

    def __post_init__(self) -> None:
        if not self.config_paths:
            raise ValueError("remote SAM3.1 campaign requires at least one config")
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")

    @property
    def runtime_contract(self) -> RemoteRuntimeContract:
        return RemoteRuntimeContract(
            sam_repo_path=self.sam_repo_path,
            sam_repo_revision=self.sam_repo_revision,
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
            workspace_path=self.workspace_path,
            required_distribution_versions=self.required_distribution_versions,
        )

    @property
    def backend_revision(self) -> str:
        return f"{self.sam_repo_revision}+checkpoint-{self.checkpoint_sha256[:12]}"

    @property
    def attempt_dir(self) -> Path:
        return self.output_root / "_runtime" / f"attempt_{self.attempt_index:02d}"


def _load_all_plans(spec: RemoteSam31CampaignSpec) -> tuple[tuple[Path, TrackingPlan], ...]:
    loaded = tuple(
        (path, load_tracking_plan_config(path, input_root=spec.input_root))
        for path in spec.config_paths
    )
    clip_ids = [plan.clip_id for _, plan in loaded]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError(f"remote campaign clip IDs must be unique: {clip_ids}")
    return loaded


def _preflight_report(
    spec: RemoteSam31CampaignSpec,
    approval: RemoteRuntimeApproval,
    plans: Sequence[tuple[Path, TrackingPlan]],
) -> dict[str, Any]:
    plan_records = [
        {
            "clip_id": plan.clip_id,
            "config_path": str(path.resolve()),
            "config_sha256": _sha256(path),
            "plan_sha256": plan.sha256,
            "frame_count": len(plan.frames),
        }
        for path, plan in plans
    ]
    return {
        "format": "sam31-remote-preflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempt_index": spec.attempt_index,
        "contract": spec.runtime_contract.as_dict(),
        "contract_sha256": approval.contract_sha256,
        "checkpoint_sha256": approval.checkpoint_sha256,
        "hardware": approval.hardware.as_dict(),
        "plans": plan_records,
        "plan_sha256_by_clip": {
            record["clip_id"]: record["plan_sha256"] for record in plan_records
        },
    }


def run_remote_sam31_campaign(
    spec: RemoteSam31CampaignSpec,
    *,
    preflight_fn: Callable[[RemoteRuntimeContract], RemoteRuntimeApproval] = perform_remote_preflight,
    predictor_builder: Callable[
        [RemoteRuntimeContract, RemoteRuntimeApproval], Any
    ] = build_tracker_only_sam31,
    campaign_runner: Callable[..., dict[str, Any]] = run_tracking_campaign,
    telemetry_factory: Callable[..., RuntimeTelemetryWriter] = RuntimeTelemetryWriter,
) -> dict[str, Any]:
    """Run one resumable worker attempt after immutable data and hardware validation."""

    # All data and geometry-only prompt hashes are checked before the multi-GB hash or
    # any model import occurs.
    plans = _load_all_plans(spec)
    contract = spec.runtime_contract
    approval = preflight_fn(contract)
    approval.require_contract(contract)

    attempt_dir = spec.attempt_dir
    attempt_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = attempt_dir / "preflight.json"
    if preflight_path.exists():
        raise RuntimeError(f"refusing to overwrite remote attempt metadata: {preflight_path}")
    _atomic_json(preflight_path, _preflight_report(spec, approval, plans))
    telemetry = telemetry_factory(
        attempt_dir / "telemetry.jsonl",
        workspace_path=spec.workspace_path,
    )
    built = False

    try:
        telemetry.record(
            {
                "event": "preflight_completed",
                "attempt_index": spec.attempt_index,
                "contract_sha256": approval.contract_sha256,
            }
        )

        def predictor_factory() -> Any:
            nonlocal built
            if built:
                raise RuntimeError("remote campaign attempted to build the predictor twice")
            built = True
            telemetry.record(
                {"event": "predictor_load_started", "attempt_index": spec.attempt_index}
            )
            predictor = predictor_builder(contract, approval)
            telemetry.record(
                {"event": "predictor_loaded", "attempt_index": spec.attempt_index}
            )
            return predictor

        manifest = campaign_runner(
            spec.config_paths,
            input_root=spec.input_root,
            output_root=spec.output_root,
            backend="sam3.1-tracker-only",
            revision=spec.backend_revision,
            predictor_factory=predictor_factory,
            event_callback=telemetry.record,
        )
        _atomic_json(
            attempt_dir / "SUCCESS.json",
            {
                "format": "sam31-remote-attempt-success-v1",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "attempt_index": spec.attempt_index,
                "campaign_format": manifest.get("format"),
                "campaign_clip_count": manifest.get("clip_count"),
            },
        )
        return manifest
    except Exception as error:
        _atomic_json(
            attempt_dir / "FAILURE.json",
            {
                "format": "sam31-remote-attempt-failure-v1",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "attempt_index": spec.attempt_index,
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
