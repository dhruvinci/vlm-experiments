from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .resource_guard import ResourceLimits, run_guarded


GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class DiagnosticCampaignSpec:
    decoder_output_root: Path
    output_root: Path
    python_executable: Path = Path(sys.executable)
    child_memory_max_bytes: int = 4 * GIB
    min_host_available_bytes: int = 2 * GIB
    min_gpu_free_bytes: int = 512 * MIB
    max_gpu_used_fraction: float = 0.92
    attempts_per_fold: int = 2
    panel_width: int = 480
    panel_height: int = 320


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


def _read_decoder_campaign(root: Path) -> tuple[dict[str, Any], str]:
    result_path = root / "campaign-result.json"
    completion_path = root / "RUN_COMPLETE"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("decoder campaign is incomplete or invalid") from error
    result_sha = _sha256(result_path)
    if completion.get("result_sha256") != result_sha:
        raise RuntimeError("decoder campaign result checksum mismatch")
    if result.get("format") != "ownership-local-decoder-campaign-result-v1":
        raise RuntimeError("decoder campaign result format is unsupported")
    return result, result_sha


def _read_fold_diagnostic(
    destination: Path,
    *,
    run_name: str,
    heldout_clip: str,
) -> tuple[dict[str, Any], str]:
    manifest_path = destination / "diagnostic-manifest.json"
    completion_path = destination / "RUN_COMPLETE"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"fold diagnostic is incomplete: {run_name}") from error
    manifest_sha = _sha256(manifest_path)
    if completion.get("manifest_sha256") != manifest_sha:
        raise RuntimeError(f"fold diagnostic checksum mismatch: {run_name}")
    if (
        manifest.get("format") != "ownership-fold-diagnostics-v1"
        or manifest.get("run_name") != run_name
        or manifest.get("heldout_clip") != heldout_clip
        or int(manifest.get("frame_count", 0)) < 1
        or len(manifest.get("records", [])) != int(manifest.get("frame_count", 0))
    ):
        raise RuntimeError(f"fold diagnostic inventory is invalid: {run_name}")
    return manifest, manifest_sha


def _guarded_renderer(spec: DiagnosticCampaignSpec) -> Callable[[Path, Path], dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "scripts/render_decoder_fold.py"

    def render(work_item: Path, destination: Path) -> dict[str, Any]:
        payload = json.loads(work_item.read_text(encoding="utf-8"))
        device = str(payload.get("device", ""))
        if device not in {"cpu", "cuda"}:
            raise ValueError("diagnostic work item device is invalid")
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(project_root / "src"),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        limits = ResourceLimits(
            min_host_available_bytes=spec.min_host_available_bytes,
            min_gpu_free_bytes=(spec.min_gpu_free_bytes if device == "cuda" else 0),
            max_gpu_used_fraction=(
                spec.max_gpu_used_fraction if device == "cuda" else 1.0
            ),
        )
        last_result = None
        for attempt in range(1, spec.attempts_per_fold + 1):
            last_result = run_guarded(
                [
                    str(spec.python_executable),
                    str(worker),
                    "--spec",
                    str(work_item),
                    "--output-root",
                    str(destination),
                    "--panel-width",
                    str(spec.panel_width),
                    "--panel-height",
                    str(spec.panel_height),
                ],
                limits=limits,
                log_path=(
                    spec.output_root
                    / "logs"
                    / f"{payload['run_name']}.attempt-{attempt}.log"
                ),
                telemetry_path=(
                    spec.output_root
                    / "telemetry"
                    / f"{payload['run_name']}.attempt-{attempt}.jsonl"
                ),
                cwd=project_root,
                env=environment,
                child_memory_max_bytes=spec.child_memory_max_bytes,
            )
            if last_result.returncode == 0:
                manifest, _ = _read_fold_diagnostic(
                    destination,
                    run_name=str(payload["run_name"]),
                    heldout_clip=str(payload["heldout_clip"]),
                )
                return manifest
        assert last_result is not None
        raise RuntimeError(
            f"fold diagnostic failed after {spec.attempts_per_fold} attempts: "
            f"{payload['run_name']}"
        )

    return render


def run_diagnostic_campaign(
    spec: DiagnosticCampaignSpec,
    *,
    job_runner: Callable[[Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render selected static/action folds serially in restart-safe workers."""

    if not spec.python_executable.is_file():
        raise FileNotFoundError("diagnostic Python executable is missing")
    if spec.child_memory_max_bytes < GIB or spec.attempts_per_fold not in {1, 2}:
        raise ValueError("diagnostic resource limits are invalid")
    if min(spec.panel_width, spec.panel_height) < 32:
        raise ValueError("diagnostic panel dimensions are too small")
    campaign, campaign_sha = _read_decoder_campaign(spec.decoder_output_root)
    result_path = spec.output_root / "diagnostic-campaign-result.json"
    completion_path = spec.output_root / "RUN_COMPLETE"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("result_sha256") != _sha256(result_path):
            raise RuntimeError("diagnostic campaign result checksum mismatch")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("decoder_campaign_sha256") != campaign_sha:
            raise RuntimeError("diagnostic campaign is bound to a different decoder campaign")
        return result

    clips = tuple(str(value) for value in campaign.get("clip_ids", []))
    static = campaign.get("selected_static_run_by_heldout")
    action = campaign.get("selected_action_run_by_heldout")
    if (
        len(clips) < 4
        or not isinstance(static, dict)
        or not isinstance(action, dict)
        or set(static) != set(clips)
        or set(action) != set(clips)
    ):
        raise RuntimeError("decoder campaign does not contain complete selected-run maps")
    render = job_runner or _guarded_renderer(spec)
    output_records = []
    for mode, selected in (("static", static), ("action_relational", action)):
        for clip_id in sorted(clips):
            run_name = str(selected[clip_id])
            work_item = spec.decoder_output_root / "work-items" / f"{run_name}.json"
            try:
                payload = json.loads(work_item.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"diagnostic work item is missing: {run_name}") from error
            expected_condition = None if mode == "static" else "action_relational"
            if (
                payload.get("run_name") != run_name
                or payload.get("heldout_clip") != clip_id
                or payload.get("semantic_condition") != expected_condition
            ):
                raise RuntimeError(f"selected diagnostic work item is inconsistent: {run_name}")
            destination = spec.output_root / mode / clip_id
            if (destination / "RUN_COMPLETE").exists():
                _, manifest_sha = _read_fold_diagnostic(
                    destination,
                    run_name=run_name,
                    heldout_clip=clip_id,
                )
            else:
                render(work_item, destination)
                _, manifest_sha = _read_fold_diagnostic(
                    destination,
                    run_name=run_name,
                    heldout_clip=clip_id,
                )
            output_records.append(
                {
                    "mode": mode,
                    "clip_id": clip_id,
                    "run_name": run_name,
                    "manifest_path": str(
                        (destination / "diagnostic-manifest.json").resolve()
                    ),
                    "manifest_sha256": manifest_sha,
                }
            )
    result = {
        "format": "ownership-diagnostic-campaign-result-v1",
        "decoder_campaign_sha256": campaign_sha,
        "diagnostic_count": len(output_records),
        "diagnostics": output_records,
    }
    if result_path.exists():
        raise RuntimeError("refusing to overwrite diagnostic campaign result")
    _atomic_json(result_path, result)
    _atomic_json(completion_path, {"result_sha256": _sha256(result_path)})
    return result
