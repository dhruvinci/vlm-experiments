#!/usr/bin/env python3
"""
Run Stage 2 only: Lazy Detection and Re-analysis
Loads Stage 1 output and runs targeted re-analysis
"""
import json
import sys
from pathlib import Path

# Add experiment directory to path
sys.path.insert(0, str(Path(__file__).parent / 'experiment'))

from experiment3_stage2 import GeminiStage2Refinement

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Configuration
cv_cache_path = "outputs/experiment3/youtube_SMRbZEbxepA_cv_cache.json"
stage1_output_path = "outputs/experiment3/stage1_timeline_v3.3-pass1.json"
video_path = "data/videos/youtube_SMRbZEbxepA.mp4"
output_dir = "outputs/experiment3"
output_suffix = "_v3.3"

print("="*80)
print("STAGE 2: Lazy Detection and Re-analysis")
print("="*80)
print(f"Stage 1 input: {stage1_output_path}")
print(f"CV cache: {cv_cache_path}")
print(f"Video: {video_path}")
print(f"Output suffix: {output_suffix}")
print("="*80)

# Load Stage 1 results
print("\nLoading Stage 1 results...")
with open(stage1_output_path, 'r') as f:
    stage1_results = json.load(f)
print(f"✓ Loaded {len(stage1_results.get('segments', []))} segments from Stage 1")

# Load CV cache
print("\nLoading CV cache...")
with open(cv_cache_path, 'r') as f:
    cv_data = json.load(f)
print(f"✓ Loaded {len(cv_data.get('per_second_metrics', []))} seconds of CV data")

# Run Stage 2
stage2 = GeminiStage2Refinement(
    stage1_results=stage1_results,
    cv_data=cv_data,
    video_path=video_path,
    output_dir=output_dir,
    output_suffix=output_suffix
)

result = stage2.analyze()

print("\n" + "="*80)
print("STAGE 2 SUMMARY")
print("="*80)
print(f"Flagged segments: {result['detection']['flagged_count']}")
print(f"Windows re-analyzed: {len(result['windows'])}")
print(f"Total cost: ${result['total_cost']:.4f}")
print(f"Total time: {result['total_time']:.1f}s")
print("="*80)
