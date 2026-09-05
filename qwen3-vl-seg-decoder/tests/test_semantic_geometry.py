from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class BreadthSemanticGeometryTests(unittest.TestCase):
    def test_cli_requires_explicit_unique_top_actor_mapping(self) -> None:
        from ownership_decoder import semantic_geometry_cli

        returned = {
            "selected_layers": {"action_delta": 12, "contact_delta": 45},
            "artifact_count": 24,
        }
        with patch.object(
            semantic_geometry_cli,
            "audit_breadth_condition_deltas",
            return_value=returned,
        ) as audit:
            status = semantic_geometry_cli.main(
                [
                    "--cache-root",
                    "/cache/breadth",
                    "--output",
                    "/output/audit.json",
                    "--top-actor",
                    "back_seatbelt=A1",
                    "--top-actor",
                    "guard_scramble=A1",
                    "--top-actor",
                    "half_guard=A2",
                    "--top-actor",
                    "mount=A1",
                ]
            )

        self.assertEqual(status, 0)
        _, kwargs = audit.call_args
        self.assertEqual(
            kwargs["top_actor_by_clip"],
            {
                "back_seatbelt": "A1",
                "guard_scramble": "A1",
                "half_guard": "A2",
                "mount": "A1",
            },
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            semantic_geometry_cli.parse_top_actor_mapping(
                ["mount=A1", "mount=A2"]
            )

    def test_audit_selects_and_hash_binds_slot_cancelled_role_layers(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("ownership_decoder.semantic_geometry"))
        from ownership_decoder.semantic_geometry import audit_breadth_condition_deltas

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / "breadth"
            roles = {
                "back_seatbelt": "A1",
                "guard_scramble": "A1",
                "half_guard": "A2",
                "mount": "A1",
            }
            for clip_index, (clip, top_actor) in enumerate(roles.items()):
                identity_a1 = torch.zeros((64, 5120), dtype=torch.bfloat16)
                identity_a2 = torch.zeros_like(identity_a1)
                identity_a1[:, 0] = 1.0 + clip_index / 16
                identity_a2[:, 0] = -1.0 - clip_index / 16
                role_sign = 1.0 if top_actor == "A1" else -1.0
                action_a1 = identity_a1.clone()
                action_a2 = identity_a2.clone()
                action_a1[12, 1] += 0.5 * role_sign
                action_a2[12, 1] -= 0.5 * role_sign
                contact_a1 = identity_a1.clone()
                contact_a2 = identity_a2.clone()
                contact_a1[45, 2] += 0.5 * role_sign
                contact_a2[45, 2] -= 0.5 * role_sign
                for condition, pairs in (
                    ("identity_only", (identity_a1, identity_a2)),
                    ("action_relational", (action_a1, action_a2)),
                    ("contact_ownership", (contact_a1, contact_a2)),
                ):
                    for actor, states in zip(("A1", "A2"), pairs, strict=True):
                        path = (
                            cache
                            / clip
                            / "semantic/video/4fps"
                            / condition
                            / "off"
                            / f"{actor}.safetensors"
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        save_file(
                            {"marker_states": states},
                            path,
                            metadata={
                                "campaign": json.dumps(
                                    {
                                        "actor": actor,
                                        "stage": "semantic_video",
                                        "condition": condition,
                                        "context": "4fps",
                                        "thinking_mode": "off",
                                    }
                                )
                            },
                        )

            output = root / "geometry-audit.json"
            audit = audit_breadth_condition_deltas(
                cache,
                output,
                top_actor_by_clip=roles,
            )

            self.assertEqual(audit["selected_layers"], {"action_delta": 12, "contact_delta": 45})
            self.assertEqual(audit["artifact_count"], 24)
            self.assertEqual(
                set(audit["implementation_sha256"]),
                {"cache.py", "semantic_geometry.py"},
            )
            for digest in audit["implementation_sha256"].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(len(audit["conditions"]["action_delta"]["layers"]), 64)
            self.assertEqual(
                audit["conditions"]["action_delta"]["layers"][12][
                    "leave_one_clip_out_accuracy"
                ],
                1.0,
            )
            self.assertEqual(
                (output.with_suffix(".json.sha256")).read_text().strip(),
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                audit_breadth_condition_deltas(
                    cache,
                    output,
                    top_actor_by_clip=roles,
                ),
                audit,
            )


if __name__ == "__main__":
    unittest.main()
