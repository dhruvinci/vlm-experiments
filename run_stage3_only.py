#!/usr/bin/env python3
"""
Run Stage 3 only: Merge Pass 1 and Pass 2
Loads Stage 1 and Stage 2 outputs and creates final timeline
"""
import json
import sys
from pathlib import Path

# Add experiment directory to path
sys.path.insert(0, str(Path(__file__).parent / 'experiment'))

from experiment3_stage2 import GeminiStage3Synthesis

# Configuration
stage1_output_path = "outputs/experiment3/stage1_timeline_v3.3-pass1.json"
stage2_detection_path = "outputs/experiment3/stage2_lazy_detection_v3.3.json"
output_dir = "outputs/experiment3"
output_suffix = "_v3.3"

print("="*80)
print("STAGE 3: Synthesis and Merge")
print("="*80)
print(f"Stage 1 input: {stage1_output_path}")
print(f"Stage 2 detection: {stage2_detection_path}")
print(f"Output suffix: {output_suffix}")
print("="*80)

# Load Stage 1 results
print("\nLoading Stage 1 results...")
with open(stage1_output_path, 'r') as f:
    stage1_results = json.load(f)
print(f"✓ Loaded {len(stage1_results.get('segments', []))} segments from Stage 1")

# Load Stage 2 detection
print("\nLoading Stage 2 detection...")
with open(stage2_detection_path, 'r') as f:
    stage2_detection = json.load(f)

# Load Stage 2 window results
stage2_windows = []
window_num = 1
while True:
    window_path = Path(output_dir) / f"stage2_window{window_num}_timeline{output_suffix}.json"
    if not window_path.exists():
        break
    with open(window_path, 'r') as f:
        stage2_windows.append(json.load(f))
    window_num += 1

print(f"✓ Loaded {len(stage2_windows)} window results from Stage 2")

# Construct Stage 2 results
stage2_results = {
    'detection': stage2_detection,
    'windows': stage2_windows
}

# Run Stage 3
stage3 = GeminiStage3Synthesis(
    stage1_results=stage1_results,
    stage2_results=stage2_results,
    output_dir=output_dir,
    output_suffix=output_suffix
)

result = stage3.analyze()

print("\n" + "="*80)
print("STAGE 3 SUMMARY")
print("="*80)
print(f"Stage 1 segments: {result['merge_info']['stage1_segments']}")
print(f"Segments replaced: {result['merge_info']['segments_replaced']}")
print(f"Segments added: {result['merge_info']['segments_added']}")
print(f"Final segments: {result['merge_info']['final_segments']}")
print(f"Continuous: {result['validation']['is_continuous']}")
print("="*80)
