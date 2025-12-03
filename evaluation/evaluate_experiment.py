#!/usr/bin/env python3
"""
Compare AI experiment output against human ground truth annotations.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

def time_to_seconds(time_str: str) -> float:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Calculate Intersection over Union for two time segments."""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    return intersection / union if union > 0 else 0

def match_segments(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> Tuple[List, List, List]:
    """
    Match predicted segments to ground truth segments using IoU.
    Returns: (matched_pairs, unmatched_predictions, unmatched_ground_truths)
    """
    matched = []
    unmatched_preds = list(range(len(predictions)))
    unmatched_gts = list(range(len(ground_truths)))

    # For each ground truth, find best matching prediction
    for gt_idx, gt in enumerate(ground_truths):
        gt_start = time_to_seconds(gt['start_time'])
        gt_end = time_to_seconds(gt['end_time'])

        best_iou = 0
        best_pred_idx = None

        for pred_idx in unmatched_preds:
            pred = predictions[pred_idx]
            pred_start = time_to_seconds(pred['start_time'])
            pred_end = time_to_seconds(pred['end_time'])

            iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)

            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_iou >= iou_threshold and best_pred_idx is not None:
            matched.append({
                'pred_idx': best_pred_idx,
                'gt_idx': gt_idx,
                'pred': predictions[best_pred_idx],
                'gt': gt,
                'iou': best_iou
            })
            unmatched_preds.remove(best_pred_idx)
            unmatched_gts.remove(gt_idx)

    return matched, unmatched_preds, unmatched_gts

def evaluate_experiment(experiment_path: str, ground_truth_path: str) -> Dict:
    """
    Evaluate experiment against ground truth.
    """
    # Load data
    with open(experiment_path) as f:
        experiment = json.load(f)

    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    predictions = experiment.get('position_timeline', [])
    ground_truths = ground_truth.get('ground_truth_positions', [])

    # Match segments
    matched, unmatched_preds, unmatched_gts = match_segments(predictions, ground_truths)

    # Calculate metrics
    total_gt = len(ground_truths)
    total_pred = len(predictions)
    total_matched = len(matched)

    # Precision/Recall/F1
    precision = total_matched / total_pred if total_pred > 0 else 0
    recall = total_matched / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Position accuracy (for matched segments)
    position_correct = 0
    sub_position_correct = 0

    for match in matched:
        pred = match['pred']
        gt = match['gt']

        if pred.get('position', '').lower() == gt.get('position', '').lower():
            position_correct += 1

        if pred.get('sub_position', '').lower() == gt.get('sub_position', '').lower():
            sub_position_correct += 1

    position_accuracy = position_correct / total_matched if total_matched > 0 else 0
    sub_position_accuracy = sub_position_correct / total_matched if total_matched > 0 else 0

    # Average IoU for matched segments
    avg_iou = sum(m['iou'] for m in matched) / len(matched) if matched else 0

    # Detailed comparison
    detailed_results = []
    for match in matched:
        pred = match['pred']
        gt = match['gt']

        detailed_results.append({
            'time_range': f"{gt['start_time']} - {gt['end_time']}",
            'predicted_position': pred.get('position'),
            'ground_truth_position': gt.get('position'),
            'predicted_sub_position': pred.get('sub_position'),
            'ground_truth_sub_position': gt.get('sub_position'),
            'position_match': pred.get('position', '').lower() == gt.get('position', '').lower(),
            'sub_position_match': pred.get('sub_position', '').lower() == gt.get('sub_position', '').lower(),
            'iou': match['iou'],
            'user_rating': gt.get('rating')
        })

    return {
        'summary': {
            'total_ground_truth_segments': total_gt,
            'total_predicted_segments': total_pred,
            'matched_segments': total_matched,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'position_accuracy': position_accuracy,
            'sub_position_accuracy': sub_position_accuracy,
            'average_iou': avg_iou,
            'overall_score': (f1 + position_accuracy + sub_position_accuracy + avg_iou) / 4
        },
        'detailed_results': detailed_results,
        'unmatched_predictions': [predictions[i] for i in unmatched_preds],
        'missed_ground_truths': [ground_truths[i] for i in unmatched_gts]
    }

def print_evaluation_report(results: Dict):
    """Print a formatted evaluation report."""
    summary = results['summary']

    print("\n" + "="*80)
    print("BJJ VIDEO ANALYSIS EVALUATION REPORT")
    print("="*80)

    print("\n📊 OVERALL METRICS:")
    print(f"  Overall Score: {summary['overall_score']:.2%}")
    print(f"  F1 Score: {summary['f1_score']:.2%}")
    print(f"  Precision: {summary['precision']:.2%}")
    print(f"  Recall: {summary['recall']:.2%}")

    print("\n🎯 ACCURACY METRICS:")
    print(f"  Position Accuracy: {summary['position_accuracy']:.2%}")
    print(f"  Sub-Position Accuracy: {summary['sub_position_accuracy']:.2%}")
    print(f"  Average IoU (Temporal): {summary['average_iou']:.2%}")

    print("\n📈 COVERAGE:")
    print(f"  Ground Truth Segments: {summary['total_ground_truth_segments']}")
    print(f"  Predicted Segments: {summary['total_predicted_segments']}")
    print(f"  Matched Segments: {summary['matched_segments']}")
    print(f"  Missed Segments: {len(results['missed_ground_truths'])}")
    print(f"  False Positives: {len(results['unmatched_predictions'])}")

    print("\n📝 DETAILED BREAKDOWN:")
    print(f"  Total comparisons: {len(results['detailed_results'])}")

    # Group by user rating
    rating_groups = {}
    for detail in results['detailed_results']:
        rating = detail.get('user_rating')
        if rating not in rating_groups:
            rating_groups[rating] = []
        rating_groups[rating].append(detail)

    for rating in sorted(rating_groups.keys(), key=lambda x: x if x is not None else -1):
        segments = rating_groups[rating]
        pos_correct = sum(1 for s in segments if s['position_match'])
        sub_pos_correct = sum(1 for s in segments if s['sub_position_match'])

        print(f"\n  Rating {rating if rating is not None else 'N/A'} segments ({len(segments)} total):")
        print(f"    Position accuracy: {pos_correct}/{len(segments)} ({pos_correct/len(segments):.1%})")
        print(f"    Sub-position accuracy: {sub_pos_correct}/{len(segments)} ({sub_pos_correct/len(segments):.1%})")

    # Show some examples of errors
    print("\n❌ SAMPLE ERRORS (first 5):")
    errors = [d for d in results['detailed_results'] if not d['position_match']][:5]
    for i, error in enumerate(errors, 1):
        print(f"\n  {i}. Time: {error['time_range']}")
        print(f"     Predicted: {error['predicted_position']} / {error['predicted_sub_position']}")
        print(f"     Ground Truth: {error['ground_truth_position']} / {error['ground_truth_sub_position']}")
        print(f"     IoU: {error['iou']:.2%}, User Rating: {error.get('user_rating')}")

    if results['missed_ground_truths']:
        print(f"\n⚠️  MISSED SEGMENTS ({len(results['missed_ground_truths'])} total):")
        for missed in results['missed_ground_truths'][:5]:
            print(f"  {missed['start_time']}-{missed['end_time']}: {missed['position']} / {missed.get('sub_position')}")

    print("\n" + "="*80)

if __name__ == "__main__":
    experiment_path = "results/gemini/20251007_163528_youtube_SMRbZEbxepA/analysis.json"
    ground_truth_path = "evaluation/ground_truth.json"

    results = evaluate_experiment(experiment_path, ground_truth_path)
    print_evaluation_report(results)

    # Save detailed results
    output_path = f"evaluation/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_path}")
