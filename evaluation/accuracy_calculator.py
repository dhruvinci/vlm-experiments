#!/usr/bin/env python3
"""
Accuracy Calculator for BJJ Analysis Experiments

Compares experiment outputs against ground truth labels and calculates metrics.
"""

import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path


class AccuracyCalculator:
    """Calculate accuracy metrics for BJJ analysis experiments."""

    def __init__(self, ground_truth_path: str):
        """
        Initialize with ground truth data.

        Args:
            ground_truth_path: Path to ground_truth.json file
        """
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)

    def calculate_all_metrics(self, experiment_output_path: str) -> Dict[str, Any]:
        """
        Calculate all accuracy metrics for an experiment.

        Args:
            experiment_output_path: Path to experiment's analysis.json

        Returns:
            Dictionary of all metrics
        """
        with open(experiment_output_path, 'r') as f:
            experiment = json.load(f)

        metrics = {
            "experiment_path": experiment_output_path,
            "timestamp": datetime.now().isoformat(),
            "position_metrics": self._calculate_position_metrics(experiment),
            "transition_metrics": self._calculate_transition_metrics(experiment),
            "scoring_metrics": self._calculate_scoring_metrics(experiment),
            "temporal_metrics": self._calculate_temporal_metrics(experiment),
            "overall_score": 0.0  # Calculated at end
        }

        # Calculate weighted overall score
        metrics["overall_score"] = self._calculate_overall_score(metrics)

        return metrics

    def _calculate_position_metrics(self, experiment: Dict[str, Any]) -> Dict[str, float]:
        """Calculate position detection accuracy metrics."""
        gt_positions = self.ground_truth.get("ground_truth_positions", [])
        exp_positions = experiment.get("position_timeline", [])

        if not gt_positions:
            return {"error": "No ground truth positions"}

        # Convert to comparable format
        gt_set = self._positions_to_set(gt_positions)
        exp_set = self._positions_to_set(exp_positions)

        # Calculate metrics
        true_positives = len(gt_set & exp_set)
        false_positives = len(exp_set - gt_set)
        false_negatives = len(gt_set - exp_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / len(gt_set) if len(gt_set) > 0 else 0

        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_ground_truth": len(gt_set),
            "total_detected": len(exp_set)
        }

    def _calculate_transition_metrics(self, experiment: Dict[str, Any]) -> Dict[str, float]:
        """Calculate transition detection metrics."""
        gt_transitions = self.ground_truth.get("ground_truth_transitions", [])
        exp_transitions = experiment.get("transitions", [])

        if not gt_transitions:
            return {"info": "No ground truth transitions"}

        # Convert to comparable format
        gt_set = self._transitions_to_set(gt_transitions)
        exp_set = self._transitions_to_set(exp_transitions)

        # Calculate metrics
        true_positives = len(gt_set & exp_set)
        false_positives = len(exp_set - gt_set)
        false_negatives = len(gt_set - exp_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_ground_truth": len(gt_set),
            "total_detected": len(exp_set)
        }

    def _calculate_scoring_metrics(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ADCC scoring accuracy."""
        gt_scoring = self.ground_truth.get("ground_truth_scoring_adcc", {})
        exp_scoring = experiment.get("scoring_adcc", {}) or experiment.get("scoring", {})

        if not gt_scoring:
            return {"info": "No ground truth scoring"}

        metrics = {}

        # For each athlete
        for athlete_key in gt_scoring.keys():
            if athlete_key in ["ruleset", "winner", "winning_margin"]:
                continue

            gt_athlete = gt_scoring[athlete_key]
            exp_athlete = exp_scoring.get(athlete_key, {}) or exp_scoring.get(f"athlete_{athlete_key[-1]}", {})

            if isinstance(gt_athlete, dict) and isinstance(exp_athlete, dict):
                gt_total = gt_athlete.get("total_points", 0)
                exp_total = exp_athlete.get("total_points", 0)

                # Calculate MAE and exact match
                mae = abs(gt_total - exp_total)
                exact_match = (gt_total == exp_total)

                metrics[f"{athlete_key}_mae"] = mae
                metrics[f"{athlete_key}_exact_match"] = exact_match
                metrics[f"{athlete_key}_gt_points"] = gt_total
                metrics[f"{athlete_key}_exp_points"] = exp_total

        # Overall scoring metrics
        all_maes = [v for k, v in metrics.items() if k.endswith("_mae")]
        all_matches = [v for k, v in metrics.items() if k.endswith("_exact_match")]

        if all_maes:
            metrics["average_mae"] = round(sum(all_maes) / len(all_maes), 2)
            metrics["exact_match_rate"] = round(sum(all_matches) / len(all_matches), 2)

        return metrics

    def _calculate_temporal_metrics(self, experiment: Dict[str, Any]) -> Dict[str, float]:
        """Calculate temporal overlap metrics (IoU)."""
        gt_positions = self.ground_truth.get("ground_truth_positions", [])
        exp_positions = experiment.get("position_timeline", [])

        if not gt_positions or not exp_positions:
            return {"info": "Insufficient data for temporal metrics"}

        # Calculate average IoU across all positions
        ious = []

        for gt_pos in gt_positions:
            gt_start = self._time_to_seconds(gt_pos.get("start_time", "00:00"))
            gt_end = self._time_to_seconds(gt_pos.get("end_time", "00:00"))

            best_iou = 0.0

            for exp_pos in exp_positions:
                exp_start = self._time_to_seconds(exp_pos.get("start_time", "00:00"))
                exp_end = self._time_to_seconds(exp_pos.get("end_time", "00:00"))

                # Calculate IoU
                intersection_start = max(gt_start, exp_start)
                intersection_end = min(gt_end, exp_end)
                intersection = max(0, intersection_end - intersection_start)

                union_start = min(gt_start, exp_start)
                union_end = max(gt_end, exp_end)
                union = union_end - union_start

                iou = intersection / union if union > 0 else 0
                best_iou = max(best_iou, iou)

            ious.append(best_iou)

        average_iou = sum(ious) / len(ious) if ious else 0

        return {
            "average_iou": round(average_iou, 3),
            "min_iou": round(min(ious), 3) if ious else 0,
            "max_iou": round(max(ious), 3) if ious else 0
        }

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall score."""
        weights = {
            "position_accuracy": 0.35,
            "transition_recall": 0.25,
            "scoring_exact_match": 0.25,
            "temporal_iou": 0.15
        }

        score = 0.0

        # Position accuracy
        pos_metrics = metrics.get("position_metrics", {})
        if "accuracy" in pos_metrics:
            score += weights["position_accuracy"] * pos_metrics["accuracy"]

        # Transition recall
        trans_metrics = metrics.get("transition_metrics", {})
        if "recall" in trans_metrics:
            score += weights["transition_recall"] * trans_metrics["recall"]

        # Scoring exact match
        scoring_metrics = metrics.get("scoring_metrics", {})
        if "exact_match_rate" in scoring_metrics:
            score += weights["scoring_exact_match"] * scoring_metrics["exact_match_rate"]

        # Temporal IoU
        temp_metrics = metrics.get("temporal_metrics", {})
        if "average_iou" in temp_metrics:
            score += weights["temporal_iou"] * temp_metrics["average_iou"]

        return round(score, 3)

    def _positions_to_set(self, positions: List[Dict[str, Any]]) -> set:
        """Convert position list to set of tuples for comparison."""
        position_set = set()

        for pos in positions:
            # Use position name and approximate time range
            position = pos.get("position", "unknown")
            start_time = pos.get("start_time", "00:00")
            end_time = pos.get("end_time", "00:00")

            # Round to 5-second buckets for fuzzy matching
            start_bucket = self._time_to_bucket(start_time, bucket_size=5)
            end_bucket = self._time_to_bucket(end_time, bucket_size=5)

            position_set.add((position, start_bucket, end_bucket))

        return position_set

    def _transitions_to_set(self, transitions: List[Dict[str, Any]]) -> set:
        """Convert transition list to set of tuples for comparison."""
        transition_set = set()

        for trans in transitions:
            timestamp = trans.get("timestamp", "00:00")
            from_pos = trans.get("from_position", "unknown")
            to_pos = trans.get("to_position", "unknown")

            # Round to 5-second buckets
            time_bucket = self._time_to_bucket(timestamp, bucket_size=5)

            transition_set.add((time_bucket, from_pos, to_pos))

        return transition_set

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS or HH:MM:SS to seconds."""
        try:
            parts = time_str.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return 0.0
        except:
            return 0.0

    def _time_to_bucket(self, time_str: str, bucket_size: int = 5) -> int:
        """Convert time to bucket number (for fuzzy matching)."""
        seconds = self._time_to_seconds(time_str)
        return int(seconds // bucket_size)


def compare_experiments(
    ground_truth_path: str,
    experiment_paths: List[str]
) -> Dict[str, Any]:
    """
    Compare multiple experiments against ground truth.

    Args:
        ground_truth_path: Path to ground_truth.json
        experiment_paths: List of paths to experiment analysis.json files

    Returns:
        Comparison results with rankings
    """
    calculator = AccuracyCalculator(ground_truth_path)

    results = {}

    for exp_path in experiment_paths:
        exp_name = Path(exp_path).parent.name
        try:
            metrics = calculator.calculate_all_metrics(exp_path)
            results[exp_name] = metrics
        except Exception as e:
            results[exp_name] = {"error": str(e)}

    # Rank experiments by overall score
    ranked = sorted(
        [(name, data.get("overall_score", 0)) for name, data in results.items()],
        key=lambda x: x[1],
        reverse=True
    )

    comparison = {
        "ground_truth": ground_truth_path,
        "experiments": results,
        "ranking": [
            {
                "rank": i + 1,
                "experiment": name,
                "overall_score": score
            }
            for i, (name, score) in enumerate(ranked)
        ],
        "timestamp": datetime.now().isoformat()
    }

    return comparison


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python accuracy_calculator.py <ground_truth.json> <experiment1/analysis.json> [experiment2/analysis.json ...]")
        sys.exit(1)

    ground_truth = sys.argv[1]
    experiments = sys.argv[2:]

    comparison = compare_experiments(ground_truth, experiments)

    print(json.dumps(comparison, indent=2))
