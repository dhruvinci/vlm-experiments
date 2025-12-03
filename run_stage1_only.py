#!/usr/bin/env python3
"""
Run Stage 1 only for testing
"""
import json
import os
from pathlib import Path
from experiment.experiment3_stages123 import GeminiStage1Timeline

# Load .env file if it exists
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Load existing Stage 0 cache
cv_cache_path = "outputs/experiment3/youtube_SMRbZEbxepA_cv_cache.json"
video_path = "data/videos/youtube_SMRbZEbxepA.mp4"
output_dir = "outputs/experiment3"

print("Loading Stage 0 cache...")
with open(cv_cache_path, 'r') as f:
    cv_data = json.load(f)

print(f"✓ Loaded CV data: {len(cv_data['per_second_metrics'])} frames")
print(f"  Duration: {cv_data['duration']}s")
print(f"  Both detected: {cv_data['summary']['athlete_detection']['both_rate']:.1%}")

print("\n" + "="*80)
print("STAGE 1: Position Timeline")
print("="*80)

# Run Stage 1 with dynamic output suffix
import sys
output_suffix = sys.argv[1] if len(sys.argv) > 1 else "_test"

stage1 = GeminiStage1Timeline(
    video_path=video_path,
    cv_data=cv_data,
    output_dir=output_dir,
    output_suffix=output_suffix
)

timeline_data = stage1.analyze()

# Save output (filename determined by output_suffix)
output_filename = f"stage1_timeline{output_suffix}.json"
output_path = Path(output_dir) / output_filename
with open(output_path, 'w') as f:
    json.dump(timeline_data, f, indent=2)

print("\n" + "="*80)
print("STAGE 1 COMPLETE")
print("="*80)
print(f"Output saved: {output_path}")
print(f"\nPositions found: {len(timeline_data.get('positions', []))}")
print(f"Athletes: {len(timeline_data.get('athlete_profiles', {}))}")
print(f"Cost: ${timeline_data.get('cost', {}).get('total', 0):.4f}")
print(f"Time: {timeline_data.get('time_seconds', 0):.1f}s")

# Print position summary
print("\n" + "="*80)
print("POSITION SUMMARY")
print("="*80)
for i, pos in enumerate(timeline_data.get('positions', [])[:10], 1):
    print(f"{i}. {pos['start']}-{pos['end']}: {pos['position']} ({pos.get('sub_position', 'N/A')})")
    
if len(timeline_data.get('positions', [])) > 10:
    print(f"... and {len(timeline_data['positions']) - 10} more positions")
