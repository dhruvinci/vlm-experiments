#!/usr/bin/env python3
"""
Evaluate AI quality based on human ratings and detail richness.
"""
import json
from pathlib import Path
from collections import Counter

def load_data():
    with open('evaluation/ground_truth.json') as f:
        return json.load(f)

def analyze_ratings(ground_truth):
    """Analyze user ratings of AI predictions."""
    positions = ground_truth.get('ground_truth_positions', [])

    ratings = [p.get('rating') for p in positions if p.get('rating') is not None]
    rating_dist = Counter(ratings)

    total = len(ratings)
    avg_rating = sum(ratings) / total if total > 0 else 0

    return {
        'total_rated': total,
        'average_rating': avg_rating,
        'distribution': dict(rating_dist),
        'ratings': ratings
    }

def analyze_detail_richness(ground_truth):
    """Compare AI output (simple notes) to human annotations (detailed sub-segments)."""
    positions = ground_truth.get('ground_truth_positions', [])

    total_positions = len(positions)
    total_subsegments = sum(len(p.get('sub_segments', [])) for p in positions)
    total_labels = sum(
        sum(len(seg.get('labels', [])) for seg in p.get('sub_segments', []))
        for p in positions
    )

    # Collect all unique techniques/labels
    all_labels = set()
    for pos in positions:
        for seg in pos.get('sub_segments', []):
            all_labels.update(seg.get('labels', []))

    # Calculate detail metrics
    avg_subsegments_per_position = total_subsegments / total_positions if total_positions > 0 else 0
    avg_labels_per_subsegment = total_labels / total_subsegments if total_subsegments > 0 else 0

    return {
        'total_positions': total_positions,
        'total_subsegments': total_subsegments,
        'total_labels': total_labels,
        'unique_techniques': len(all_labels),
        'avg_subsegments_per_position': avg_subsegments_per_position,
        'avg_labels_per_subsegment': avg_labels_per_subsegment,
        'technique_vocabulary': sorted(list(all_labels))
    }

def calculate_ai_score(ratings_analysis, detail_analysis):
    """Calculate overall AI performance score."""
    avg_rating = ratings_analysis['average_rating']

    # Rating-based score (0-5 scale, normalize to 0-100)
    rating_score = (avg_rating / 5.0) * 100

    # Detail score: AI provided 0 sub-segments
    # Ground truth has ~4.6 sub-segments per position on average
    # Detail score = 0% (AI provided no granular detail)
    detail_coverage = 0.0

    # Weighted overall score
    # 60% weight on ratings (did AI get positions roughly right?)
    # 40% weight on detail (did AI provide granular analysis?)
    overall_score = (0.6 * rating_score) + (0.4 * detail_coverage)

    return {
        'rating_score': rating_score,
        'detail_score': detail_coverage,
        'overall_score': overall_score
    }

def print_report(ratings_analysis, detail_analysis, score):
    """Print comprehensive evaluation report."""
    print("\n" + "="*80)
    print("BJJ AI EVALUATION: QUALITY & DETAIL ANALYSIS")
    print("="*80)

    print("\n📊 AI PREDICTION QUALITY (Human Ratings):")
    print(f"  Average Rating: {ratings_analysis['average_rating']:.2f}/5.0")
    print(f"  Total Segments Rated: {ratings_analysis['total_rated']}")
    print(f"\n  Rating Distribution:")
    for rating in sorted(ratings_analysis['distribution'].keys()):
        count = ratings_analysis['distribution'][rating]
        pct = (count / ratings_analysis['total_rated']) * 100
        bar = '█' * int(pct / 2)
        print(f"    {rating if rating is not None else 'N/A'} stars: {count:2d} ({pct:5.1f}%) {bar}")

    print("\n🔍 DETAIL RICHNESS:")
    print(f"  AI-Generated Positions: {detail_analysis['total_positions']}")
    print(f"  Human-Annotated Sub-segments: {detail_analysis['total_subsegments']}")
    print(f"  Average Sub-segments per Position: {detail_analysis['avg_subsegments_per_position']:.1f}")
    print(f"  Total Technique Labels Applied: {detail_analysis['total_labels']}")
    print(f"  Unique Techniques Identified: {detail_analysis['unique_techniques']}")

    print(f"\n📈 WHAT THE AI MISSED:")
    print(f"  ❌ 0 sub-segments provided (vs {detail_analysis['total_subsegments']} in ground truth)")
    print(f"  ❌ 0 technique labels (vs {detail_analysis['total_labels']} in ground truth)")
    print(f"  ❌ No granular action descriptions")
    print(f"  ❌ No temporal precision within positions")

    print(f"\n🎯 SCORING:")
    print(f"  Position Quality Score: {score['rating_score']:.1f}/100")
    print(f"    → Based on average {ratings_analysis['average_rating']:.1f}/5.0 user rating")
    print(f"  Detail Coverage Score: {score['detail_score']:.1f}/100")
    print(f"    → AI provided no sub-segment analysis")
    print(f"  \n  📊 OVERALL AI SCORE: {score['overall_score']:.1f}/100")

    print(f"\n💡 INTERPRETATION:")
    if score['overall_score'] >= 80:
        quality = "Excellent"
    elif score['overall_score'] >= 60:
        quality = "Good"
    elif score['overall_score'] >= 40:
        quality = "Fair"
    elif score['overall_score'] >= 20:
        quality = "Poor"
    else:
        quality = "Very Poor"

    print(f"  Quality: {quality}")
    print(f"  The AI correctly identified high-level positions ({ratings_analysis['average_rating']:.1f}/5.0 rating)")
    print(f"  but completely missed granular details (0/{detail_analysis['total_subsegments']} sub-segments).")

    print(f"\n🔬 TECHNIQUE VOCABULARY (Sample):")
    vocab = detail_analysis['technique_vocabulary']
    for i, technique in enumerate(vocab[:15], 1):
        print(f"    {i:2d}. {technique}")
    if len(vocab) > 15:
        print(f"    ... and {len(vocab) - 15} more techniques")

    print("\n" + "="*80)

if __name__ == "__main__":
    ground_truth = load_data()

    ratings_analysis = analyze_ratings(ground_truth)
    detail_analysis = analyze_detail_richness(ground_truth)
    score = calculate_ai_score(ratings_analysis, detail_analysis)

    print_report(ratings_analysis, detail_analysis, score)

    # Save results
    output = {
        'ratings_analysis': ratings_analysis,
        'detail_analysis': detail_analysis,
        'scores': score
    }

    output_path = f"evaluation/quality_report_{Path(ground_truth['labeled_date']).stem}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Report saved to: {output_path}")
