#!/usr/bin/env python3
"""
Evaluate Experiment 2 (Multi-Pass Analysis) against Ground Truth
"""

import json
from pathlib import Path
from datetime import datetime


def time_to_seconds(time_str):
    """Convert MM:SS or M:SS format to seconds."""
    time_str = str(time_str).strip()

    # Handle formats: "0:06", "00:06", "10:15", etc.
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds

    # Try parsing as just seconds
    try:
        return int(float(time_str))
    except:
        return 0


def positions_overlap(gt_pos, exp_pos, tolerance=5):
    """
    Check if two positions overlap within tolerance (seconds).

    Returns tuple: (overlaps, iou)
    """
    gt_start = time_to_seconds(gt_pos['start_time'])
    gt_end = time_to_seconds(gt_pos['end_time'])
    exp_start = time_to_seconds(exp_pos['start_time'])
    exp_end = time_to_seconds(exp_pos['end_time'])

    # Calculate overlap
    overlap_start = max(gt_start, exp_start)
    overlap_end = min(gt_end, exp_end)
    overlap = max(0, overlap_end - overlap_start)

    # Calculate IoU (Intersection over Union)
    gt_duration = gt_end - gt_start
    exp_duration = exp_end - exp_start
    union = gt_duration + exp_duration - overlap
    iou = overlap / union if union > 0 else 0

    # Check if positions match within tolerance
    start_diff = abs(gt_start - exp_start)
    end_diff = abs(gt_end - exp_end)

    overlaps = (start_diff <= tolerance and end_diff <= tolerance) or iou > 0.5

    return overlaps, iou


def normalize_position(pos_str):
    """Normalize position names for comparison."""
    pos_str = str(pos_str).lower().strip()

    # Map variations
    mappings = {
        'side_control': 'side_control',
        'side control': 'side_control',
        'mount': 'mount',
        'back_control': 'back_control',
        'back control': 'back_control',
        'guard': 'guard',
        'closed_guard': 'guard',
        'open_guard': 'guard',
        'de_la_riva_guard': 'guard',
        'half_guard': 'guard',
        'standing': 'standing',
        'takedown': 'takedown',
        'passing_guard': 'passing_guard',
        'sweep': 'sweep',
        'escape': 'escape',
    }

    return mappings.get(pos_str, pos_str)


def calculate_position_accuracy(ground_truth_path, experiment_path):
    """Calculate position detection accuracy."""

    # Load data
    with open(ground_truth_path, 'r') as f:
        gt = json.load(f)

    with open(experiment_path, 'r') as f:
        exp = json.load(f)

    # Extract positions
    gt_positions = gt.get('ground_truth_positions', [])
    exp_positions = exp.get('pass3_synthesis', {}).get('unified_timeline', [])

    print(f"\n=== EXPERIMENT 2 EVALUATION ===\n")
    print(f"Ground Truth Positions: {len(gt_positions)}")
    print(f"Detected Positions: {len(exp_positions)}")

    # Match positions
    matched = 0
    position_correct = 0
    total_iou = 0.0

    matches = []
    unmatched_gt = []

    for gt_pos in gt_positions:
        best_match = None
        best_iou = 0

        for exp_pos in exp_positions:
            overlaps, iou = positions_overlap(gt_pos, exp_pos, tolerance=10)
            if overlaps and iou > best_iou:
                best_iou = iou
                best_match = exp_pos

        if best_match:
            matched += 1
            total_iou += best_iou

            # Check if position labels match
            gt_label = normalize_position(gt_pos.get('position', ''))
            exp_label = normalize_position(best_match.get('position', ''))

            if gt_label == exp_label:
                position_correct += 1

            matches.append({
                'gt': gt_pos,
                'exp': best_match,
                'iou': best_iou,
                'label_match': gt_label == exp_label
            })
        else:
            unmatched_gt.append(gt_pos)

    # Calculate metrics
    recall = matched / len(gt_positions) if len(gt_positions) > 0 else 0
    precision = matched / len(exp_positions) if len(exp_positions) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    position_accuracy = position_correct / matched if matched > 0 else 0
    avg_iou = total_iou / matched if matched > 0 else 0

    # Calculate overall score (0-100 scale)
    # Weights: Position Accuracy (40%), Temporal IoU (30%), Recall (20%), Precision (10%)
    overall_score = (
        position_accuracy * 40 +
        avg_iou * 30 +
        recall * 20 +
        precision * 10
    )

    # Count sub-segments and labels
    total_sub_segments = 0
    total_labels = 0
    for exp_pos in exp_positions:
        sub_segs = exp_pos.get('sub_segments', [])
        total_sub_segments += len(sub_segs)
        for seg in sub_segs:
            total_labels += len(seg.get('labels', []))

    # Calculate detail quality
    avg_sub_segments = total_sub_segments / len(exp_positions) if len(exp_positions) > 0 else 0
    avg_labels_per_position = total_labels / len(exp_positions) if len(exp_positions) > 0 else 0

    # Detail quality bonus (up to 20 points)
    # Target: 3-5 sub-segments per position, 5-10 labels per position
    detail_score = min(20, (
        min(20, avg_sub_segments * 4) * 0.5 +  # 5 sub-segments = 20 points
        min(20, avg_labels_per_position * 2) * 0.5  # 10 labels = 20 points
    ))

    final_score = overall_score + detail_score

    # Print results
    print(f"\n--- Temporal Alignment ---")
    print(f"  Matched Positions: {matched}/{len(gt_positions)} ({recall*100:.1f}% recall)")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  F1 Score: {f1*100:.1f}%")
    print(f"  Average IoU: {avg_iou:.3f}")

    print(f"\n--- Position Classification ---")
    print(f"  Correct Labels: {position_correct}/{matched} ({position_accuracy*100:.1f}%)")

    print(f"\n--- Detail Quality ---")
    print(f"  Total Sub-Segments: {total_sub_segments}")
    print(f"  Total Labels: {total_labels}")
    print(f"  Avg Sub-Segments/Position: {avg_sub_segments:.1f}")
    print(f"  Avg Labels/Position: {avg_labels_per_position:.1f}")
    print(f"  Detail Score: {detail_score:.1f}/20")

    print(f"\n--- OVERALL SCORE ---")
    print(f"  Base Score: {overall_score:.1f}/100")
    print(f"  Detail Bonus: +{detail_score:.1f}")
    print(f"  FINAL SCORE: {final_score:.1f}/120")

    print(f"\n--- Unmatched Ground Truth Positions ({len(unmatched_gt)}) ---")
    for pos in unmatched_gt[:5]:  # Show first 5
        print(f"  {pos['start_time']}-{pos['end_time']}: {pos['position']}")
    if len(unmatched_gt) > 5:
        print(f"  ... and {len(unmatched_gt)-5} more")

    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp2_multipass',
        'ground_truth_positions': len(gt_positions),
        'detected_positions': len(exp_positions),
        'metrics': {
            'matched': matched,
            'recall': round(recall, 3),
            'precision': round(precision, 3),
            'f1_score': round(f1, 3),
            'position_accuracy': round(position_accuracy, 3),
            'average_iou': round(avg_iou, 3),
        },
        'detail_quality': {
            'total_sub_segments': total_sub_segments,
            'total_labels': total_labels,
            'avg_sub_segments_per_position': round(avg_sub_segments, 2),
            'avg_labels_per_position': round(avg_labels_per_position, 2),
            'detail_score': round(detail_score, 1),
        },
        'scores': {
            'base_score': round(overall_score, 1),
            'detail_bonus': round(detail_score, 1),
            'final_score': round(final_score, 1),
            'max_score': 120,
        }
    }

    report_path = Path(experiment_path).parent / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Detailed report saved to: {report_path}")

    return report


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python evaluate_exp2.py <ground_truth.json> <analysis.json>")
        sys.exit(1)

    gt_path = sys.argv[1]
    exp_path = sys.argv[2]

    calculate_position_accuracy(gt_path, exp_path)
