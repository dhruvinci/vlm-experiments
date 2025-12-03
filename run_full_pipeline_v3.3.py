#!/usr/bin/env python3
"""
Run Full Experiment 3.3 Pipeline: Stage 1 → Stage 2 → Stage 3
"""
import json
import sys
import subprocess
from pathlib import Path

# Configuration
video_path = "data/videos/youtube_SMRbZEbxepA.mp4"
cv_cache_path = "outputs/experiment3/youtube_SMRbZEbxepA_cv_cache.json"
output_dir = "outputs/experiment3"
output_suffix = "_v3.3"

print("="*80)
print("EXPERIMENT 3.3: Full Pipeline")
print("="*80)
print(f"Video: {video_path}")
print(f"CV cache: {cv_cache_path}")
print(f"Output directory: {output_dir}")
print(f"Output suffix: {output_suffix}")
print("="*80)

# Stage 1: Full Video Analysis
print("\n" + "="*80)
print("RUNNING STAGE 1: Full Video Analysis")
print("="*80)

result = subprocess.run([
    sys.executable,
    "run_stage1_only.py",
    f"{output_suffix}-pass1"
], capture_output=False)

if result.returncode != 0:
    print("❌ Stage 1 failed!")
    sys.exit(1)

print("\n✓ Stage 1 complete")

# Stage 2: Lazy Detection and Re-analysis
print("\n" + "="*80)
print("RUNNING STAGE 2: Lazy Detection and Re-analysis")
print("="*80)

result = subprocess.run([
    sys.executable,
    "run_stage2_only.py"
], capture_output=False)

if result.returncode != 0:
    print("❌ Stage 2 failed!")
    sys.exit(1)

print("\n✓ Stage 2 complete")

# Stage 3: Synthesis and Merge
print("\n" + "="*80)
print("RUNNING STAGE 3: Synthesis and Merge")
print("="*80)

result = subprocess.run([
    sys.executable,
    "run_stage3_only.py"
], capture_output=False)

if result.returncode != 0:
    print("❌ Stage 3 failed!")
    sys.exit(1)

print("\n✓ Stage 3 complete")

# Final Summary
print("\n" + "="*80)
print("EXPERIMENT 3.3 COMPLETE!")
print("="*80)
print("\nOutput files generated:")
print(f"  Stage 1: stage1_timeline{output_suffix}-pass1.json")
print(f"  Stage 2: stage2_lazy_detection{output_suffix}.json")
print(f"           stage2_window*_timeline{output_suffix}.json")
print(f"  Stage 3: stage3_timeline{output_suffix}.json (FINAL)")
print("\nCompare in HITL tool:")
print(f"  - Pass 1 (baseline): stage1_timeline{output_suffix}-pass1.json")
print(f"  - Fixed (final): stage3_timeline{output_suffix}.json")
print("="*80)
