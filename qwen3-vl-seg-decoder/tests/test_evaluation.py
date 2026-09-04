from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ownership_decoder.evaluation import (
    NorthStarThresholds,
    aggregate_clip_metrics,
    evaluate_north_star,
    summarize_paired_clip_improvement,
)


def metrics(
    *,
    iou: float,
    contact: float,
    positive_regions: float,
    background: float,
    contact_pixels: float = 10,
    contact_regions: float = 1,
) -> dict[str, float]:
    return {
        "macro_actor_iou": iou,
        "contact_accuracy": contact,
        "positive_contact_region_fraction": positive_regions,
        "background_stability": background,
        "contact_pixel_count": contact_pixels,
        "contact_region_count": contact_regions,
    }


class DecoderEvaluationTests(unittest.TestCase):
    def test_paired_clip_uncertainty_is_deterministic_and_clip_balanced(self) -> None:
        baseline = {
            clip: metrics(
                iou=iou,
                contact=contact,
                positive_regions=0.8,
                background=0.9,
            )
            for clip, iou, contact in (
                ("a", 0.50, 0.50),
                ("b", 0.60, 0.60),
                ("c", 0.70, 0.70),
                ("d", 0.80, 0.80),
            )
        }
        candidate = {
            clip: metrics(
                iou=value["macro_actor_iou"] + 0.05,
                contact=value["contact_accuracy"] + 0.10,
                positive_regions=0.8,
                background=0.9,
            )
            for clip, value in baseline.items()
        }

        first = summarize_paired_clip_improvement(candidate, baseline)
        second = summarize_paired_clip_improvement(candidate, baseline)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["macro_actor_iou"]["mean_delta"], 0.05)
        self.assertAlmostEqual(first["contact_accuracy"]["mean_delta"], 0.10)
        self.assertAlmostEqual(first["macro_actor_iou"]["population_std"], 0.0)
        self.assertEqual(first["bootstrap_resample_count"], 256)

    def test_invalid_or_nonfinite_metrics_fail_closed(self) -> None:
        invalid_cases = (
            metrics(
                iou=math.nan,
                contact=0.8,
                positive_regions=0.8,
                background=0.9,
            ),
            metrics(
                iou=1.01,
                contact=0.8,
                positive_regions=0.8,
                background=0.9,
            ),
            metrics(
                iou=0.7,
                contact=0.8,
                positive_regions=0.8,
                background=0.9,
                contact_pixels=-1,
            ),
        )
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    aggregate_clip_metrics({"bad": invalid})

    def test_clip_aggregation_rejects_missing_contact_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "contact"):
            aggregate_clip_metrics(
                {
                    "bad": metrics(
                        iou=0.7,
                        contact=0.0,
                        positive_regions=0.0,
                        background=0.9,
                        contact_pixels=0,
                        contact_regions=0,
                    )
                }
            )

    def test_north_star_rejects_invalid_swap_metrics(self) -> None:
        valid = metrics(
            iou=0.7,
            contact=0.8,
            positive_regions=0.8,
            background=0.9,
        )
        for invalid_swap in (
            {
                "actor_prediction_flip_fraction": math.nan,
                "background_probability_delta": 0.0,
            },
            {
                "actor_prediction_flip_fraction": 1.1,
                "background_probability_delta": 0.0,
            },
            {
                "actor_prediction_flip_fraction": 0.8,
                "background_probability_delta": -0.1,
            },
        ):
            with self.subTest(invalid_swap=invalid_swap):
                with self.assertRaises(ValueError):
                    evaluate_north_star(
                        candidate=valid,
                        strongest_baseline=valid,
                        semantic_controls={
                            "real": valid,
                            "random_matched": valid,
                            "zero": valid,
                            "mean": valid,
                        },
                        swap_metrics=invalid_swap,
                    )

    def test_clip_aggregation_is_macro_for_iou_and_weighted_for_contact(self) -> None:
        aggregate = aggregate_clip_metrics(
            {
                "a": metrics(
                    iou=0.5,
                    contact=1.0,
                    positive_regions=1.0,
                    background=0.9,
                    contact_pixels=2,
                    contact_regions=1,
                ),
                "b": metrics(
                    iou=0.7,
                    contact=0.0,
                    positive_regions=0.0,
                    background=1.0,
                    contact_pixels=6,
                    contact_regions=3,
                ),
            }
        )

        self.assertAlmostEqual(aggregate["macro_actor_iou"], 0.6)
        self.assertAlmostEqual(aggregate["background_stability"], 0.95)
        self.assertAlmostEqual(aggregate["contact_accuracy"], 0.25)
        self.assertAlmostEqual(aggregate["positive_contact_region_fraction"], 0.25)

    def test_north_star_passes_with_contact_gain_even_when_iou_gain_is_small(self) -> None:
        result = evaluate_north_star(
            candidate=metrics(iou=0.65, contact=0.75, positive_regions=0.80, background=0.95),
            strongest_baseline=metrics(
                iou=0.63,
                contact=0.60,
                positive_regions=0.60,
                background=0.96,
            ),
            semantic_controls={
                "real": metrics(iou=0.65, contact=0.75, positive_regions=0.80, background=0.95),
                "random_matched": metrics(iou=0.60, contact=0.60, positive_regions=0.60, background=0.95),
                "zero": metrics(iou=0.55, contact=0.50, positive_regions=0.50, background=0.95),
                "mean": metrics(iou=0.54, contact=0.50, positive_regions=0.50, background=0.95),
            },
            swap_metrics={
                "actor_prediction_flip_fraction": 0.90,
                "background_probability_delta": 0.005,
            },
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.gates["baseline_improvement"])
        self.assertEqual(result.evidence["winning_improvement_metric"], "contact_accuracy")

    def test_north_star_exposes_semantic_random_code_failure(self) -> None:
        candidate = metrics(iou=0.66, contact=0.75, positive_regions=0.80, background=0.95)
        result = evaluate_north_star(
            candidate=candidate,
            strongest_baseline=metrics(
                iou=0.60,
                contact=0.60,
                positive_regions=0.60,
                background=0.95,
            ),
            semantic_controls={
                "real": candidate,
                "random_matched": metrics(
                    iou=0.67,
                    contact=0.78,
                    positive_regions=0.80,
                    background=0.95,
                ),
                "zero": metrics(iou=0.50, contact=0.40, positive_regions=0.40, background=0.95),
                "mean": metrics(iou=0.50, contact=0.40, positive_regions=0.40, background=0.95),
            },
            swap_metrics={
                "actor_prediction_flip_fraction": 0.90,
                "background_probability_delta": 0.005,
            },
            thresholds=NorthStarThresholds(),
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.gates["real_beats_random_matched"])
