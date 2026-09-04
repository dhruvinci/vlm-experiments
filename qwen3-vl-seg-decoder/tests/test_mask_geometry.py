from __future__ import annotations

import unittest

import numpy as np

from ownership_decoder.mask_geometry import (
    GeometryDerivationError,
    mask_pair_to_geometry_prompts,
)


class MaskGeometryTests(unittest.TestCase):
    def test_pair_produces_boxes_two_interior_positives_and_opponent_negatives(self) -> None:
        a1 = np.zeros((20, 30), dtype=bool)
        a2 = np.zeros((20, 30), dtype=bool)
        a1[3:17, 2:12] = True
        a2[4:18, 18:28] = True

        prompts = mask_pair_to_geometry_prompts(a1, a2, minimum_area=20)

        self.assertEqual(set(prompts), {"A1", "A2"})
        for actor, own, opponent in (("A1", a1, a2), ("A2", a2, a1)):
            prompt = prompts[actor]
            self.assertEqual(prompt["labels"], [1, 1, 0, 0])
            self.assertEqual(len(prompt["points"]), 4)
            self.assertEqual(len({tuple(point) for point in prompt["points"][:2]}), 2)
            for x, y in prompt["points"][:2]:
                self.assertTrue(own[round(y), round(x)])
            for x, y in prompt["points"][2:]:
                self.assertTrue(opponent[round(y), round(x)])
            x1, y1, x2, y2 = prompt["box"]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

    def test_distant_speck_is_excluded_by_largest_component_geometry(self) -> None:
        a1 = np.zeros((40, 50), dtype=bool)
        a2 = np.zeros((40, 50), dtype=bool)
        a1[10:30, 5:20] = True
        a1[0, 49] = True
        a2[8:32, 30:45] = True

        prompts = mask_pair_to_geometry_prompts(a1, a2, minimum_area=20, box_padding_fraction=0)

        self.assertEqual(prompts["A1"]["box"], [5.0, 10.0, 19.0, 29.0])

    def test_overlap_is_rejected_instead_of_silently_assigning_prompt_ownership(self) -> None:
        a1 = np.zeros((10, 10), dtype=bool)
        a2 = np.zeros((10, 10), dtype=bool)
        a1[1:8, 1:6] = True
        a2[3:9, 4:9] = True

        with self.assertRaisesRegex(GeometryDerivationError, "overlap"):
            mask_pair_to_geometry_prompts(a1, a2, minimum_area=5)

    def test_tiny_or_malformed_masks_are_rejected(self) -> None:
        valid = np.zeros((10, 10), dtype=bool)
        valid[1:5, 1:5] = True
        tiny = np.zeros((10, 10), dtype=bool)
        tiny[8, 8] = True
        with self.assertRaisesRegex(GeometryDerivationError, "foreground"):
            mask_pair_to_geometry_prompts(tiny, valid, minimum_area=4)
        with self.assertRaisesRegex(GeometryDerivationError, "shape"):
            mask_pair_to_geometry_prompts(valid, valid[:8], minimum_area=4)


if __name__ == "__main__":
    unittest.main()
