from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ownership_decoder.cloud_staging import (
    StagingArtifact,
    build_mask_staging_inventory,
    publish_staging_completion,
    stage_artifacts,
    verify_remote_artifacts,
)


class FakeBody(io.BytesIO):
    def close(self) -> None:
        super().close()


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.uploaded: list[str] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        try:
            payload, metadata = self.objects[(Bucket, Key)]
        except KeyError as error:
            missing = RuntimeError("missing")
            missing.response = {"Error": {"Code": "404"}}  # type: ignore[attr-defined]
            raise missing from error
        return {"ContentLength": len(payload), "Metadata": metadata}

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs=None) -> None:
        metadata = dict((ExtraArgs or {}).get("Metadata", {}))
        self.objects[(bucket, key)] = (Path(filename).read_bytes(), metadata)
        self.uploaded.append(key)

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        payload, _ = self.objects[(Bucket, Key)]
        return {"Body": FakeBody(payload)}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata=None) -> None:
        self.objects[(Bucket, Key)] = (bytes(Body), dict(Metadata or {}))


class CloudStagingTests(unittest.TestCase):
    def artifact(self, path: Path, key: str) -> StagingArtifact:
        payload = path.read_bytes()
        return StagingArtifact(
            local_path=path,
            remote_key=key,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def test_stage_resumes_only_with_matching_size_and_sha_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"correct")
            artifact = self.artifact(source, "prefix/source.bin")
            client = FakeS3()
            client.objects[("volume", artifact.remote_key)] = (
                b"correct",
                {"sha256": artifact.sha256},
            )
            result = stage_artifacts(client, "volume", [artifact])
            self.assertEqual(result, {"uploaded": 0, "resumed": 1, "bytes_uploaded": 0})
            self.assertEqual(client.uploaded, [])

            client.objects[("volume", artifact.remote_key)] = (
                b"wrong!!",
                {"sha256": "0" * 64},
            )
            result = stage_artifacts(client, "volume", [artifact])
            self.assertEqual(result["uploaded"], 1)
            self.assertEqual(client.uploaded, [artifact.remote_key])
            self.assertEqual(
                client.objects[("volume", artifact.remote_key)][1]["sha256"],
                artifact.sha256,
            )

    def test_stage_resumes_from_stream_hash_when_gateway_strips_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"correct")
            artifact = self.artifact(source, "prefix/source.bin")
            client = FakeS3()
            client.objects[("volume", artifact.remote_key)] = (b"correct", {})
            result = stage_artifacts(client, "volume", [artifact])
            self.assertEqual(result["resumed"], 1)
            self.assertEqual(client.uploaded, [])

    def test_inventory_maps_only_declared_model_artifacts_to_remote_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = root / "packet"
            inputs = root / "inputs"
            repo = root / "repo"
            model = root / "model"
            for directory in (packet, inputs, repo, model):
                directory.mkdir()
            (packet / "manifest.json").write_text("packet", encoding="utf-8")
            packet_cache = packet / "code/__pycache__"
            packet_cache.mkdir(parents=True)
            (packet_cache / "module.cpython-311.pyc").write_bytes(b"derived")
            (inputs / "frame.jpg").write_bytes(b"frame")
            (repo / "source.py").write_text("source", encoding="utf-8")
            repo_cache = repo / "package/__pycache__"
            repo_cache.mkdir(parents=True)
            (repo_cache / "source.cpython-312.pyc").write_bytes(b"derived")
            checkpoint = root / "sam31.pt"
            checkpoint.write_bytes(b"checkpoint")
            (model / "declared.bin").write_bytes(b"declared")
            (model / "undeclared.bin").write_bytes(b"do not upload")

            artifacts = build_mask_staging_inventory(
                packet_root=packet,
                input_root=inputs,
                sam_repo_root=repo,
                sam31_checkpoint=checkpoint,
                sam3_model_root=model,
                sam3_artifact_names=("declared.bin",),
            )
            self.assertEqual(
                {item.remote_key for item in artifacts},
                {
                    "qwen38-campaign/mask-campaign/v2/packet/manifest.json",
                    "qwen38-campaign/launch-packet/inputs/breadth/frame.jpg",
                    "qwen38-campaign/mask-campaign/v2/repos/sam3-official/source.py",
                    "qwen38-campaign/mask-campaign/v2/models/sam3.1/sam3.1_multiplex.pt",
                    "qwen38-campaign/mask-campaign/v2/models/sam3/declared.bin",
                },
            )

    def test_verify_streams_content_and_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"payload")
            artifact = self.artifact(source, "prefix/source.bin")
            client = FakeS3()
            client.objects[("volume", artifact.remote_key)] = (
                b"payload",
                {"sha256": artifact.sha256},
            )
            self.assertEqual(
                verify_remote_artifacts(client, "volume", [artifact]),
                {"verified": 1, "verified_bytes": 7},
            )
            client.objects[("volume", artifact.remote_key)] = (
                b"corrupt",
                {"sha256": artifact.sha256},
            )
            with self.assertRaisesRegex(RuntimeError, "remote SHA-256 mismatch"):
                verify_remote_artifacts(client, "volume", [artifact])

    def test_completion_is_published_only_for_exact_verified_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"payload")
            artifact = self.artifact(source, "prefix/source.bin")
            client = FakeS3()
            with self.assertRaisesRegex(ValueError, "verified count"):
                publish_staging_completion(
                    client,
                    "volume",
                    "prefix/STAGING_COMPLETE.json",
                    [artifact],
                    verified_count=0,
                    packet_sha256="a" * 64,
                )
            self.assertNotIn(("volume", "prefix/STAGING_COMPLETE.json"), client.objects)

            result = publish_staging_completion(
                client,
                "volume",
                "prefix/STAGING_COMPLETE.json",
                [artifact],
                verified_count=1,
                packet_sha256="a" * 64,
            )
            payload = json.loads(
                client.objects[("volume", "prefix/STAGING_COMPLETE.json")][0]
            )
            self.assertEqual(payload["artifact_count"], 1)
            self.assertEqual(payload["total_bytes"], 7)
            self.assertEqual(result["packet_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
