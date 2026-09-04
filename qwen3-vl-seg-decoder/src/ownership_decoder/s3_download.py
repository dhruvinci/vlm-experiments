from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


_RESERVED_LOCAL_NAMES = {
    ".download-state.json",
    "download-manifest.json",
    "DOWNLOAD_COMPLETE",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _relative_key(key: str, prefix: str) -> str:
    if not key.startswith(prefix):
        raise ValueError(f"S3 object key is outside the requested prefix: {key}")
    relative = key[len(prefix) :]
    pure = PurePosixPath(relative)
    if (
        not relative
        or relative.endswith("/")
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or pure.parts[0] in _RESERVED_LOCAL_NAMES
    ):
        raise ValueError(f"unsafe S3 object key: {key}")
    return pure.as_posix()


def _list_remote(client: Any, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    records = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token is not None:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = str(item.get("Key", ""))
            if key.endswith("/"):
                continue
            relative = _relative_key(key, prefix)
            size = int(item.get("Size", -1))
            if size < 0:
                raise RuntimeError(f"S3 object size is invalid: {key}")
            records.append(
                {
                    "key": key,
                    "relative_path": relative,
                    "size_bytes": size,
                    "etag": str(item.get("ETag", "")),
                }
            )
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not token:
            raise RuntimeError("S3 listing was truncated without a continuation token")
    records.sort(key=lambda value: value["key"])
    paths = [value["relative_path"] for value in records]
    if not records or len(paths) != len(set(paths)):
        raise RuntimeError("S3 run inventory is empty or contains duplicate paths")
    return records


def _validate_local_manifest(
    output_root: Path,
    manifest: dict[str, Any],
    *,
    bucket: str,
    prefix: str,
    remote_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        manifest.get("format") != "ownership-s3-run-download-v1"
        or manifest.get("bucket") != bucket
        or manifest.get("prefix") != prefix
        or manifest.get("remote_inventory_sha256")
        != _canonical_sha256(remote_inventory)
    ):
        raise RuntimeError("completed S3 download provenance changed")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(remote_inventory):
        raise RuntimeError("completed S3 download record inventory is invalid")
    for record in records:
        path = output_root / str(record["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"downloaded S3 artifact checksum mismatch: {path}")
    return manifest


def download_run_prefix(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Download a completed network-volume run atomically with exact resume."""

    if not bucket.strip() or not prefix.strip() or not prefix.endswith("/"):
        raise ValueError("S3 bucket and slash-terminated prefix are required")
    remote_inventory = _list_remote(client, bucket=bucket, prefix=prefix)
    if "RUN_COMPLETE" not in {record["relative_path"] for record in remote_inventory}:
        raise RuntimeError("remote run has no RUN_COMPLETE sentinel")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download-manifest.json"
    completion_path = root / "DOWNLOAD_COMPLETE"
    if completion_path.exists():
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("completed S3 download metadata is invalid") from error
        if completion.get("manifest_sha256") != _sha256(manifest_path):
            raise RuntimeError("completed S3 download manifest checksum mismatch")
        return _validate_local_manifest(
            root,
            manifest,
            bucket=bucket,
            prefix=prefix,
            remote_inventory=remote_inventory,
        )

    state_path = root / ".download-state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("partial S3 download state is invalid") from error
        if state.get("format") != "ownership-s3-download-state-v1":
            raise RuntimeError("partial S3 download state format is invalid")
        completed = {record["key"]: record for record in state.get("records", [])}
    else:
        completed = {}
    final_records = []
    for remote in remote_inventory:
        key = remote["key"]
        destination = (root / remote["relative_path"]).resolve()
        if not destination.is_relative_to(root.resolve()):
            raise ValueError(f"unsafe local destination for S3 key: {key}")
        existing = completed.get(key)
        if existing is not None:
            remote_fields = {field: existing.get(field) for field in remote}
            if remote_fields != remote:
                raise RuntimeError(f"remote S3 object changed during resume: {key}")
            if (
                not destination.is_file()
                or destination.stat().st_size != remote["size_bytes"]
                or _sha256(destination) != existing.get("sha256")
            ):
                raise RuntimeError(f"partial S3 download artifact is corrupt: {key}")
            final_records.append(existing)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            if int(response.get("ContentLength", remote["size_bytes"])) != remote["size_bytes"]:
                raise RuntimeError(f"S3 download content length changed: {key}")
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                digest = hashlib.sha256()
                count = 0
                while True:
                    block = body.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
                    count += len(block)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        try:
            if count != remote["size_bytes"]:
                raise RuntimeError(f"S3 download size mismatch: {key}")
            local_sha = digest.hexdigest()
            if destination.exists():
                if destination.stat().st_size != count or _sha256(destination) != local_sha:
                    raise RuntimeError(f"uncommitted local S3 artifact differs: {key}")
                temporary.unlink()
                temporary = None
            else:
                os.replace(temporary, destination)
                temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        record = {**remote, "sha256": local_sha}
        completed[key] = record
        final_records.append(record)
        _atomic_json(
            state_path,
            {
                "format": "ownership-s3-download-state-v1",
                "bucket": bucket,
                "prefix": prefix,
                "records": [completed[name] for name in sorted(completed)],
            },
        )

    observed_after = _list_remote(client, bucket=bucket, prefix=prefix)
    if observed_after != remote_inventory:
        raise RuntimeError("remote S3 inventory changed during download")
    manifest = {
        "format": "ownership-s3-run-download-v1",
        "bucket": bucket,
        "prefix": prefix,
        "remote_inventory_sha256": _canonical_sha256(remote_inventory),
        "object_count": len(final_records),
        "total_bytes": sum(int(record["size_bytes"]) for record in final_records),
        "records": final_records,
    }
    if manifest_path.exists():
        raise RuntimeError("refusing to overwrite S3 download manifest")
    _atomic_json(manifest_path, manifest)
    _atomic_json(completion_path, {"manifest_sha256": _sha256(manifest_path)})
    return _validate_local_manifest(
        root,
        manifest,
        bucket=bucket,
        prefix=prefix,
        remote_inventory=remote_inventory,
    )
