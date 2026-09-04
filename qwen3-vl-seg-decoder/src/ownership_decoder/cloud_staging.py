from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


REMOTE_V2_PREFIX = "qwen38-campaign/mask-campaign/v2"
REMOTE_INPUT_PREFIX = "qwen38-campaign/launch-packet/inputs/breadth"


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StagingArtifact:
    local_path: Path
    remote_key: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.local_path)
        key = PurePosixPath(self.remote_key)
        if not path.is_file():
            raise FileNotFoundError(f"staging source is missing: {path}")
        if key.is_absolute() or ".." in key.parts or not self.remote_key.strip("/"):
            raise ValueError(f"unsafe staging key: {self.remote_key}")
        if path.stat().st_size != self.size_bytes or self.size_bytes < 1:
            raise ValueError(f"staging source size mismatch: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("staging SHA-256 is invalid")

    @classmethod
    def from_file(cls, local_path: str | Path, remote_key: str) -> "StagingArtifact":
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"staging source is missing: {path}")
        return cls(
            local_path=path,
            remote_key=remote_key,
            size_bytes=path.stat().st_size,
            sha256=_sha256_stream(path),
        )

    def record(self) -> dict[str, Any]:
        return {
            "remote_key": self.remote_key,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _tree_artifacts(local_root: Path, remote_prefix: str) -> list[StagingArtifact]:
    if not local_root.is_dir():
        raise FileNotFoundError(f"staging tree is missing: {local_root}")
    records = []
    for path in sorted(local_root.rglob("*")):
        relative_path = path.relative_to(local_root)
        if "__pycache__" in relative_path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"staging trees cannot contain symlinks: {path}")
        if path.is_file():
            relative = relative_path.as_posix()
            records.append(
                StagingArtifact.from_file(path, f"{remote_prefix}/{relative}")
            )
    if not records:
        raise ValueError(f"staging tree contains no files: {local_root}")
    return records


def build_mask_staging_inventory(
    *,
    packet_root: str | Path,
    input_root: str | Path,
    sam_repo_root: str | Path,
    sam31_checkpoint: str | Path,
    sam3_model_root: str | Path,
    sam3_artifact_names: Sequence[str],
) -> list[StagingArtifact]:
    """Build the exact local-to-volume inventory used by both Pod stages."""

    packet = Path(packet_root)
    inputs = Path(input_root)
    repository = Path(sam_repo_root)
    checkpoint = Path(sam31_checkpoint)
    base_model = Path(sam3_model_root)
    if not sam3_artifact_names or len(set(sam3_artifact_names)) != len(
        sam3_artifact_names
    ):
        raise ValueError("base SAM3 artifact names must be non-empty and unique")
    inventory = [
        *_tree_artifacts(packet, f"{REMOTE_V2_PREFIX}/packet"),
        *_tree_artifacts(inputs, REMOTE_INPUT_PREFIX),
        *_tree_artifacts(repository, f"{REMOTE_V2_PREFIX}/repos/sam3-official"),
        StagingArtifact.from_file(
            checkpoint,
            f"{REMOTE_V2_PREFIX}/models/sam3.1/sam3.1_multiplex.pt",
        ),
    ]
    for name in sorted(sam3_artifact_names):
        relative = PurePosixPath(name)
        if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
            raise ValueError(f"invalid base SAM3 artifact name: {name}")
        inventory.append(
            StagingArtifact.from_file(
                base_model / name,
                f"{REMOTE_V2_PREFIX}/models/sam3/{name}",
            )
        )
    _require_unique_inventory(inventory)
    return sorted(inventory, key=lambda artifact: artifact.remote_key)


def _require_unique_inventory(artifacts: Sequence[StagingArtifact]) -> None:
    if not artifacts:
        raise ValueError("staging inventory cannot be empty")
    keys = [artifact.remote_key for artifact in artifacts]
    if len(keys) != len(set(keys)):
        raise ValueError("staging inventory contains duplicate remote keys")


def _missing_object(error: Exception) -> bool:
    response = getattr(error, "response", {})
    code = str(response.get("Error", {}).get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


def _remote_matches(client: Any, bucket: str, artifact: StagingArtifact) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=artifact.remote_key)
    except Exception as error:
        if _missing_object(error):
            return False
        raise
    if int(head.get("ContentLength", -1)) != artifact.size_bytes:
        return False
    metadata = head.get("Metadata") or {}
    if metadata.get("sha256") == artifact.sha256:
        return True
    # RunPod's S3-compatible gateway may discard user metadata. In that case,
    # content hashing is the only safe resume signal; matching size alone is
    # never accepted.
    response = client.get_object(Bucket=bucket, Key=artifact.remote_key)
    body = response["Body"]
    digest = hashlib.sha256()
    observed_size = 0
    try:
        while True:
            chunk = body.read(8 * 1024 * 1024)
            if not chunk:
                break
            observed_size += len(chunk)
            digest.update(chunk)
    finally:
        body.close()
    return observed_size == artifact.size_bytes and digest.hexdigest() == artifact.sha256


def stage_artifacts(
    client: Any,
    bucket: str,
    artifacts: Sequence[StagingArtifact],
) -> dict[str, int]:
    """Upload one bounded artifact at a time and safely resume exact matches."""

    _require_unique_inventory(artifacts)
    uploaded = 0
    resumed = 0
    bytes_uploaded = 0
    for artifact in artifacts:
        if _remote_matches(client, bucket, artifact):
            resumed += 1
            continue
        client.upload_file(
            str(artifact.local_path),
            bucket,
            artifact.remote_key,
            ExtraArgs={
                "Metadata": {
                    "sha256": artifact.sha256,
                    "source-size-bytes": str(artifact.size_bytes),
                }
            },
        )
        if not _remote_matches(client, bucket, artifact):
            raise RuntimeError(f"uploaded object metadata mismatch: {artifact.remote_key}")
        uploaded += 1
        bytes_uploaded += artifact.size_bytes
    return {
        "uploaded": uploaded,
        "resumed": resumed,
        "bytes_uploaded": bytes_uploaded,
    }


def verify_remote_artifacts(
    client: Any,
    bucket: str,
    artifacts: Sequence[StagingArtifact],
) -> dict[str, int]:
    """Stream every remote byte; metadata alone is never treated as proof."""

    _require_unique_inventory(artifacts)
    verified_bytes = 0
    for artifact in artifacts:
        response = client.get_object(Bucket=bucket, Key=artifact.remote_key)
        body = response["Body"]
        digest = hashlib.sha256()
        observed_size = 0
        try:
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                observed_size += len(chunk)
                digest.update(chunk)
        finally:
            body.close()
        if observed_size != artifact.size_bytes:
            raise RuntimeError(f"remote size mismatch: {artifact.remote_key}")
        if digest.hexdigest() != artifact.sha256:
            raise RuntimeError(f"remote SHA-256 mismatch: {artifact.remote_key}")
        verified_bytes += observed_size
    return {"verified": len(artifacts), "verified_bytes": verified_bytes}


def publish_staging_completion(
    client: Any,
    bucket: str,
    completion_key: str,
    artifacts: Sequence[StagingArtifact],
    *,
    verified_count: int,
    packet_sha256: str,
) -> dict[str, Any]:
    _require_unique_inventory(artifacts)
    if verified_count != len(artifacts):
        raise ValueError("verified count does not match the staging inventory")
    if not re.fullmatch(r"[0-9a-f]{64}", packet_sha256):
        raise ValueError("packet SHA-256 is invalid")
    records = [artifact.record() for artifact in artifacts]
    inventory_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "format": "ownership-mask-staging-complete-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "packet_sha256": packet_sha256,
        "artifact_count": len(records),
        "total_bytes": sum(artifact.size_bytes for artifact in artifacts),
        "inventory_sha256": inventory_sha256,
        "artifacts": records,
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=completion_key,
        Body=encoded,
        Metadata={"sha256": hashlib.sha256(encoded).hexdigest()},
    )
    return result
