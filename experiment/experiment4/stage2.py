#!/usr/bin/env python3
"""
Experiment 4 - Stage 2: Flesh Pass
Provides detailed analysis of high-action segments from Stage 1.
Uses cached video from Stage 1 to add HOW/WHY detail.

Run:
  python -m experiment.experiment4.stage2 --stage1_json stage1_skeleton_competition_run1.json --cache_name cachedContents/xxx
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add Google AI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai not installed")
    print("Install with: pip install google-genai")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "experiment4.0"


# ============================================================================
# CONFIGURATION
# ============================================================================

# Filter: Only analyze segments with action >= threshold
ACTION_THRESHOLD = 0.5

# Batch size: Number of segments per API call
MAX_SEGMENTS_PER_BATCH = 10

# Expected output per segment: ~400-500 tokens
EXPECTED_OUTPUT_PER_SEGMENT = 450


# ============================================================================
# PROMPTS
# ============================================================================

COMPETITION_BATCH_PROMPT_TEMPLATE = """You are a world-class BJJ coach providing detailed technical and tactical analysis.

MATCH CONTEXT:
- Athletes: {athletes}
- Duration: {duration}

You previously created a position timeline (Stage 1). Now provide deeper analysis for high-action segments.

ANALYZE THESE {num_segments} SEGMENTS:

{segments_text}

For each segment, provide:
1. **Setup** (100-150 chars): How was this created? What grips/positioning/tactics enabled it?
2. **Execution** (250-350 chars): Technical breakdown - grips, angles, pressure, timing. Why it worked/failed. Key details.
3. **Outcome** (100-150 chars): Result and impact on the match.
4. **Coaching** (150-200 chars): What was done well that can be learned, or what could be improved.

OUTPUT FORMAT (pipe-separated, ONE line per segment):
**DETAIL** | [time] | setup | execution | outcome | coaching

CRITICAL: Use **DETAIL** format, NOT **COACHING** format. Output exactly {num_segments} lines.

Be specific, technical, and concise. Focus on HOW and WHY, not just WHAT (Stage 1 already has that)."""

TRAINING_BATCH_PROMPT_TEMPLATE = """You are a BJJ coach providing constructive feedback on training footage.

TRAINING CONTEXT:
- Duration: {duration}

You previously created a position timeline (Stage 1). Now provide coaching feedback for high-action segments.

ANALYZE THESE SEGMENTS:

{segments_text}

For each segment, provide:
1. **Observation** (100-150 chars): What they attempted and what happened.
2. **Feedback** (250-350 chars): What went well, key errors, why it matters. Be constructive.
3. **Improvement** (150-200 chars): Specific corrections to practice. Actionable advice.

OUTPUT FORMAT (pipe-separated, ONE line per segment):
**COACHING** | [time] | observation | feedback | improvement

Be constructive, specific, and focus on learning. Adapt feedback to skill level shown in video."""


def format_competition_segment(seg: Dict, idx: int) -> str:
    """Format a competition segment for the batch prompt."""
    return f"""---
SEGMENT {idx}: [{seg.get('time', 'unknown')}]
Position: {seg.get('position', 'unknown')} | Top: {seg.get('top', '-')} | Control: {seg.get('control', 'N/A')}
Score: {seg.get('score', '0-0')} | Action: {seg.get('action', 'N/A')}
What happened: {seg.get('reasons', 'unknown')}
Focus areas: {seg.get('focus', 'general_analysis')}
Brief notes: {seg.get('notes', 'No notes')}"""


def format_training_segment(seg: Dict, idx: int) -> str:
    """Format a training segment for the batch prompt."""
    return f"""---
SEGMENT {idx}: [{seg.get('time', 'unknown')}]
Position: {seg.get('position', 'unknown')} | Top: {seg.get('top', '-')} | Control: {seg.get('control', 'N/A')}
Action: {seg.get('action', 'N/A')}
What happened: {seg.get('reasons', 'unknown')}
Focus areas: {seg.get('focus', 'technique_feedback')}
Brief notes: {seg.get('notes', 'No notes')}"""


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_stage1_output(json_path: Path) -> Dict:
    """Load Stage 1 JSON output."""
    with open(json_path, 'r') as f:
        return json.load(f)


def filter_high_action_segments(segments: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """Filter segments with action score above threshold."""
    high_action = []
    for seg in segments:
        try:
            action = float(seg.get('action', 0.0))
            if action >= threshold:
                high_action.append(seg)
        except (ValueError, TypeError):
            continue
    return high_action


def create_batches(segments: List[Dict]) -> List[List[Dict]]:
    """Create batches of segments (max MAX_SEGMENTS_PER_BATCH per batch)."""
    batches = []
    for i in range(0, len(segments), MAX_SEGMENTS_PER_BATCH):
        batch = segments[i:i + MAX_SEGMENTS_PER_BATCH]
        batches.append(batch)
    return batches


def process_batch(batch: List[Dict], mode: str, meta: Dict, cache_name: str, client) -> List[str]:
    """Process a batch of segments using cached video and return analysis lines."""
    print(f"  Processing batch of {len(batch)} segments...")
    
    # Format segments for prompt
    if mode == "competition":
        segments_text = "\n".join([format_competition_segment(seg, i+1) for i, seg in enumerate(batch)])
        prompt = COMPETITION_BATCH_PROMPT_TEMPLATE.format(
            athletes=meta.get('athletes', 'Unknown'),
            duration=meta.get('duration', 'Unknown'),
            segments_text=segments_text,
            num_segments=len(batch)
        )
    else:  # training
        segments_text = "\n".join([format_training_segment(seg, i+1) for i, seg in enumerate(batch)])
        prompt = TRAINING_BATCH_PROMPT_TEMPLATE.format(
            duration=meta.get('duration', 'Unknown'),
            segments_text=segments_text,
            num_segments=len(batch)
        )
    
    # Use gemini-2.5-flash (same as Stage 1)
    model = 'models/gemini-2.5-flash'
    
    # Generate response using cached content
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            max_output_tokens=EXPECTED_OUTPUT_PER_SEGMENT * len(batch),
            temperature=0.4
        )
    )
    
    if not response.text:
        return []
    
    # Parse response lines
    lines = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
    analysis_lines = [l for l in lines if l.startswith('**DETAIL**') or l.startswith('**COACHING**')]
    
    return analysis_lines


def run_stage2_analysis(stage1_data: Dict, cache_name: str, client) -> List[str]:
    """Run Stage 2 analysis on high-action segments using cached video."""
    # Determine mode: check if context contains 'training', otherwise assume competition
    context = stage1_data['meta'].get('context', 'competition').lower()
    mode = 'training' if 'training' in context else 'competition'
    segments = stage1_data.get('segments', [])
    
    print(f"\n[Stage 2] Mode: {mode} (context: '{stage1_data['meta'].get('context', 'unknown')}')")
    print(f"[Stage 2] Filtering high-action segments...")
    high_action = filter_high_action_segments(segments, threshold=ACTION_THRESHOLD)
    print(f"  Found {len(high_action)} high-action segments (out of {len(segments)})")
    print(f"  Threshold: action >= {ACTION_THRESHOLD}")
    
    if not high_action:
        print("  No high-action segments to analyze")
        return []
    
    print(f"\n[Stage 2] Creating batches...")
    batches = create_batches(high_action)
    print(f"  Created {len(batches)} batches ({MAX_SEGMENTS_PER_BATCH} segments per batch)")
    
    print(f"\n[Stage 2] Processing batches...")
    all_analysis = []
    
    for i, batch in enumerate(batches, 1):
        print(f"\n  Batch {i}/{len(batches)}:")
        try:
            analysis_lines = process_batch(batch, mode, stage1_data['meta'], cache_name, client)
            all_analysis.extend(analysis_lines)
            print(f"    ✓ Got {len(analysis_lines)} analysis lines")
            
            # Small delay between batches
            if i < len(batches):
                time.sleep(1)
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_analysis


def save_stage2_output(analysis_lines: List[str], mode: str, stage1_data: Dict, run: int = None) -> None:
    """Save Stage 2 output to files (non-redundant, just new detail)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate filename suffix
    run_suffix = f"_run{run}" if run else ""
    
    # Save raw MD
    md_content = "\n".join(analysis_lines)
    md_path = OUT_DIR / f"stage2_detail_{mode}{run_suffix}.md"
    md_path.write_text(md_content, encoding='utf-8')
    
    # Parse and save JSON
    parsed_analyses = []
    for line in analysis_lines:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 5:
            if parts[0].strip().replace('**', '') == 'DETAIL':
                parsed_analyses.append({
                    'type': 'detail',
                    'time': parts[1].strip('[]'),
                    'setup': parts[2],
                    'execution': parts[3],
                    'outcome': parts[4],
                    'coaching': parts[5] if len(parts) > 5 else '',
                    'raw': line
                })
            elif parts[0].strip().replace('**', '') == 'COACHING':
                parsed_analyses.append({
                    'type': 'coaching',
                    'time': parts[1].strip('[]'),
                    'observation': parts[2],
                    'feedback': parts[3],
                    'improvement': parts[4] if len(parts) > 4 else '',
                    'raw': line
                })
    
    json_data = {
        'meta': {
            'mode': mode,
            'processed_at': datetime.now().isoformat(),
            'stage1_meta': stage1_data.get('meta', {}),
            'total_analyses': len(parsed_analyses),
            'action_threshold': ACTION_THRESHOLD
        },
        'analyses': parsed_analyses,
        'raw_output': md_content
    }
    
    json_path = OUT_DIR / f"stage2_detail_{mode}{run_suffix}.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding='utf-8')
    
    print(f"\n[Stage 2] Output saved:")
    print(f"  MD: {md_path}")
    print(f"  JSON: {json_path}")
    print(f"  Analyses: {len(parsed_analyses)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 4 Stage 2 (Flesh)")
    parser.add_argument("--stage1_json", required=True, help="Path to Stage 1 JSON output")
    parser.add_argument("--cache_name", required=True, help="Cache name from Stage 1 (e.g., cachedContents/xxx)")
    parser.add_argument("--run", type=int, default=None, help="Run number (e.g., 1, 2, 3)")
    parser.add_argument("--api_key", help="Gemini API key (or set GOOGLE_API_KEY env var)")
    args = parser.parse_args()
    
    # Configure API
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: No API key provided")
        print("Set GOOGLE_API_KEY environment variable or use --api_key")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    
    # Load Stage 1 output
    stage1_path = OUT_DIR / args.stage1_json if not Path(args.stage1_json).is_absolute() else Path(args.stage1_json)
    if not stage1_path.exists():
        print(f"Error: Stage 1 JSON not found: {stage1_path}")
        sys.exit(1)
    
    stage1_data = load_stage1_output(stage1_path)
    mode = stage1_data['meta'].get('context', 'competition')
    
    print("=" * 80)
    print("EXPERIMENT 4 - STAGE 2 (FLESH PASS)")
    print("=" * 80)
    print(f"Stage 1 JSON: {stage1_path.name}")
    print(f"Cache: {args.cache_name}")
    print(f"Mode: {mode}")
    print(f"Stage 1 segments: {len(stage1_data.get('segments', []))}")
    
    try:
        # Run analysis using cached video
        analysis_lines = run_stage2_analysis(stage1_data, args.cache_name, client)
        
        # Save results
        if analysis_lines:
            save_stage2_output(analysis_lines, mode, stage1_data, args.run)
        else:
            print("\n[Stage 2] No analyses generated")
        
        print("\n" + "=" * 80)
        print("✓ STAGE 2 COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def combine_with_stage1(stage1_json_path: str, stage2_json_path: str, output_path: str = None) -> Dict:
    """
    Combine Stage 1 and Stage 2 outputs into a single unified JSON.
    
    Args:
        stage1_json_path: Path to Stage 1 JSON
        stage2_json_path: Path to Stage 2 JSON  
        output_path: Optional output path (default: auto-generated)
    
    Returns:
        Combined data dictionary
    """
    # Load both stages
    stage1_path = OUT_DIR / stage1_json_path if not Path(stage1_json_path).is_absolute() else Path(stage1_json_path)
    stage2_path = OUT_DIR / stage2_json_path if not Path(stage2_json_path).is_absolute() else Path(stage2_json_path)
    
    print(f"\n[Combining] Loading Stage 1: {stage1_path.name}")
    stage1_data = load_stage1_output(stage1_path)
    
    print(f"[Combining] Loading Stage 2: {stage2_path.name}")
    with open(stage2_path, 'r') as f:
        stage2_data = json.load(f)
    
    # Create lookup for Stage 2 analyses by time
    print(f"[Combining] Merging {len(stage1_data['segments'])} segments...")
    stage2_by_time = {}
    for analysis in stage2_data.get('analyses', []):
        time = analysis.get('time', '')
        stage2_by_time[time] = analysis
    
    # Merge segments with their detailed analyses
    combined_segments = []
    for seg in stage1_data.get('segments', []):
        time = seg.get('time', '')
        
        # Start with Stage 1 data
        combined_seg = {
            'time': time,
            'position': seg.get('position', ''),
            'top': seg.get('top', ''),
            'control': seg.get('control', ''),
            'score': seg.get('score', ''),
            'action': seg.get('action', ''),
            'reasons': seg.get('reasons', ''),
            'focus': seg.get('focus', ''),
            'notes': seg.get('notes', ''),
            'stage1_raw': seg.get('raw', '')
        }
        
        # Add Stage 2 detail if available
        if time in stage2_by_time:
            detail = stage2_by_time[time]
            combined_seg['has_detail'] = True
            combined_seg['detail_type'] = detail.get('type', '')
            
            if detail.get('type') == 'detail':
                combined_seg['detail'] = {
                    'setup': detail.get('setup', ''),
                    'execution': detail.get('execution', ''),
                    'outcome': detail.get('outcome', ''),
                    'coaching': detail.get('coaching', '')
                }
            elif detail.get('type') == 'coaching':
                combined_seg['detail'] = {
                    'observation': detail.get('observation', ''),
                    'feedback': detail.get('feedback', ''),
                    'improvement': detail.get('improvement', '')
                }
            
            combined_seg['stage2_raw'] = detail.get('raw', '')
        else:
            combined_seg['has_detail'] = False
        
        combined_segments.append(combined_seg)
    
    # Build combined output
    combined = {
        'meta': {
            'combined_at': datetime.now().isoformat(),
            'stage1_meta': stage1_data.get('meta', {}),
            'stage2_meta': stage2_data.get('meta', {}),
            'total_segments': len(combined_segments),
            'segments_with_detail': sum(1 for s in combined_segments if s.get('has_detail')),
            'segments_skeleton_only': sum(1 for s in combined_segments if not s.get('has_detail'))
        },
        'segments': combined_segments
    }
    
    # Save output
    if output_path is None:
        mode = stage1_data['meta'].get('context', 'unknown')
        run = stage2_data['meta'].get('stage1_meta', {}).get('run', '')
        run_suffix = f"_run{run}" if run else ""
        output_path = OUT_DIR / f"combined_{mode}{run_suffix}.json"
    else:
        output_path = OUT_DIR / output_path if not Path(output_path).is_absolute() else Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n[Combining] ✓ Combined output saved:")
    print(f"  File: {output_path}")
    print(f"  Total segments: {combined['meta']['total_segments']}")
    print(f"  With detail: {combined['meta']['segments_with_detail']}")
    print(f"  Skeleton only: {combined['meta']['segments_skeleton_only']}")
    
    return combined


if __name__ == "__main__":
    main()
