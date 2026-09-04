from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.s3_download import download_run_prefix


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)
        self.downloads = []

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        self.bucket = Bucket
        records = [
            {
                "Key": key,
                "Size": len(value),
                "ETag": f'"{_sha256_bytes(value)[:32]}"',
            }
            for key, value in sorted(self.objects.items())
            if key.startswith(Prefix)
        ]
        return {"Contents": records, "IsTruncated": False}

    def get_object(self, *, Bucket, Key):
        self.downloads.append(Key)
        value = self.objects[Key]
        return {"Body": io.BytesIO(value), "ContentLength": len(value)}


class S3DownloadTests(unittest.TestCase):
    def test_download_freezes_inventory_hashes_every_file_and_resumes(self) -> None:
        prefix = "qwen38-campaign/mask-campaign/runs/run-1/"
        objects = {
            prefix + "RUN_COMPLETE": b'{"ok":true}\n',
            prefix + "campaign-manifest.json": b'{"format":"test"}\n',
            prefix + "sam31-tracking/a/frame_000000.npz": b"mask-data",
        }
        client = FakeS3(objects)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = download_run_prefix(
                client,
                bucket="volume",
                prefix=prefix,
                output_root=root,
            )
            resumed = download_run_prefix(
                client,
                bucket="volume",
                prefix=prefix,
                output_root=root,
            )

            self.assertEqual(first, resumed)
            self.assertEqual(first["object_count"], 3)
            self.assertEqual(first["total_bytes"], sum(map(len, objects.values())))
            self.assertEqual(len(client.downloads), 3)
            self.assertTrue((root / "sam31-tracking/a/frame_000000.npz").is_file())
            self.assertTrue((root / "DOWNLOAD_COMPLETE").is_file())

    def test_download_requires_remote_completion_and_rejects_unsafe_keys(self) -> None:
        prefix = "runs/run-1/"
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(RuntimeError, "RUN_COMPLETE"):
                download_run_prefix(
                    FakeS3({prefix + "partial.bin": b"x"}),
                    bucket="volume",
                    prefix=prefix,
                    output_root=Path(raw),
                )
            with self.assertRaisesRegex(ValueError, "unsafe"):
                download_run_prefix(
                    FakeS3(
                        {
                            prefix + "RUN_COMPLETE": b"ok",
                            prefix + "../escape": b"bad",
                        }
                    ),
                    bucket="volume",
                    prefix=prefix,
                    output_root=Path(raw),
                )

    def test_remote_inventory_change_during_download_fails_closed(self) -> None:
        prefix = "runs/run-1/"

        class MutatingS3(FakeS3):
            def list_objects_v2(self, **kwargs):
                result = super().list_objects_v2(**kwargs)
                if getattr(self, "listed", False):
                    self.objects[prefix + "late.bin"] = b"late"
                    result = super().list_objects_v2(**kwargs)
                self.listed = True
                return result

        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(RuntimeError, "changed"):
                download_run_prefix(
                    MutatingS3({prefix + "RUN_COMPLETE": b"ok"}),
                    bucket="volume",
                    prefix=prefix,
                    output_root=Path(raw),
                )


if __name__ == "__main__":
    unittest.main()
