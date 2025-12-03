#!/usr/bin/env python3
"""
Experiment 4 - Stage 1: Skeleton Pass
Creates position timeline with interest routing for Stage 2.
Supports both competition and training analysis modes.

Run:
  python -m experiment.experiment4.stage1 --video_path video.mp4 --mode competition
  python -m experiment.experiment4.stage1 --video_path video.mp4 --mode training
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add Google AI SDK
try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed")
    print("Install with: pip install google-generativeai")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "experiment4.0"


# ============================================================================
# PROMPTS
# ============================================================================

COMPETITION_PROMPT = """You are a world class No-Gi Jiu-Jitsu coach (beyond black belt) analyzing a COMPETITIVE BJJ match.

Create a position timeline in PIPE-SEPARATED MARKDOWN format.

STEP 1: IDENTIFY ATHLETES
Analyze the first 30 seconds and identify:
- Athlete 1 (A1): Visual characteristics (gi/rash guard color, body type, distinctive features)
- Athlete 2 (A2): Visual characteristics (gi/rash guard color, body type, distinctive features)
Also try and identify their names and use those instead of A1, A2 across your analysis.

Use these athlete IDs (names or A1, A2) consistently throughout the entire analysis.
If you know their names from the video, include them in the META line also.

STEP 2: CREATE POSITION TIMELINE WITH MICRO-ANALYSIS

Analyze the complete video and create a position timeline with integrated micro-analysis.

OUTPUT STRUCTURE:

Line 1: **META** | context:competition | duration:MM:SS | athletes:Name1,Name2
Line 2: **FORMAT** | [time] | pos | top | ctrl | score | action | conf | reasons (1-10 tags) | focus (1-10 tags) | notes | strategy | setup | execution | outcome | coaching
Lines 3+: **SEG** | [data...]

FIELD DEFINITIONS:

- Time format: [MM:SS-MM:SS]
- Position: standing, guard, mount, side_control, back_control, half_guard, turtle, scramble, etc
- Top: athlete name, A1, A2, or - (for neutral positions)
- Control quality: 0.0-1.0 (how dominant the position is)
- Score: format "X-Y" (current match score if visible)
- Action: 0.0-1.0 (how much action is going on in this segment)
- Confidence: 0.0-1.0 (how confident you are about this segment's accuracy)
  * 1.0 = Clearly saw this happen at this exact timestamp
  * 0.7 = Fairly confident, minor uncertainty about timing or details
  * 0.5 = Moderate confidence, some guessing involved
  * 0.3 = Low confidence, significant uncertainty
- Reasons: Brief tags describing what happened (e.g., "sweep_to_mount", "armbar_attempt", "pass_sequence")
- Focus: Brief tags of a few things in this segment that could be analysed in more depth (e.g., "guard_retention_strategy", "guard_passing_technique", "takedown_setup", "arm_triangle_setup", "mount_pressure")
- Notes: Brief description (ONLY this field has 100 char limit)
- Strategy (100-150 chars): What was the thinking, planning, and game strategy insights for each fighter within this segment?
- Setup (100-150 chars): How was this position/technique created? What grips, positioning, leverages, etc enabled it?
- Execution (250-350 chars): Technical breakdown - specific grips, angles, pressure points, timing. Why it worked or failed. Key micro-details.
- Outcome (100-150 chars): What was the result? Impact on match momentum and positioning.
- Coaching (150-200 chars): What was done well that can be learned? What could be improved? Actionable insights.

SEGMENTATION PHILOSOPHY - How to assign Action scores and determine segment length:

LOW ACTION (Action < 0.30):
- Duration: 30-60 seconds (prefer longer segments for static positions)
- Characteristics: Static position, minimal movement, positional maintenance
- Position changes: Very few (0-1)
- Technique attempts: 0-2 minor attempts (grip adjustments, posture breaks)
- Example: Athlete maintaining closed guard with grip fighting, no significant attempts

MEDIUM ACTION (Action 0.30-0.70):
- Duration: 15-45 seconds (one cohesive exchange per segment)
- Characteristics: One major position change OR sustained technical exchange
- Position changes: 1-2 major transitions (e.g., takedown, guard pass, sweep)
- Technique attempts: 3-6 intermediate actions (pass attempts, escape attempts, positional improvements)
- Example: Guard passing sequence with multiple attempts, or takedown attempt with scramble to guard

HIGH ACTION (Action > 0.70):
- Duration: 5-20 seconds (rapid action only)
- Characteristics: Rapid scrambles, multiple position changes, submission attempts
- Position changes: 3+ rapid transitions
- Technique attempts: 6+ techniques in quick succession
- Example: Explosive scramble with mount escape → sweep → back take → submission defense
- Note: If multiple high-action bursts are chained with only 2-5 second breaks, keep them as ONE segment

RULES:
- Do NOT create micro-segments (under 5 seconds) unless it's a submission finish or scoring sequence
- For static positions, use 30-60 second segments (preferably nor longer)
- Use abbreviations where clear (names, A1, A2, sub, pos, etc.)
- Maintain chronological order
- No gaps in timeline
- Use athlete IDs (names or A1, A2) consistently

Example output:
**META** | context:competition | duration:8:45 | athletes:Gordon Ryan,Felipe Pena
**FORMAT** | [time] | pos | top | ctrl | score | action | conf | reasons | focus | notes | strategy | setup | execution | outcome | coaching
**SEG** | [0:00-0:52] | standing | - | 0.3 | 0-0 | 0.2 | 0.9 | collar_ties,circling,distance_management | - | Both athletes feeling each other out, collar ties, no serious takedown attempts | Ryan assessing Pena's stance and movement, Pena looking for distance and angle | Both athletes in neutral stance, circling to create angles and openings | Light grip fighting with collar ties. Ryan uses shoulder posts to manage distance. Pena circles away, looking for underhook opportunities. Both athletes are cautious, testing each other's reactions without committing to major techniques | Engagement remains neutral with no clear advantage gained by either athlete | Patient opening is smart at this level. Both showed good distance management. Could have been more aggressive with grip fighting to establish dominance early
**SEG** | [0:52-1:28] | guard | Ryan | 0.5 | 0-0 | 0.4 | 0.8 | guard_passing_pressure,leg_drag_attempts | passing_pressure,leg_positioning | Ryan pressuring to pass, Pena using butterfly hooks and frames to maintain guard | Ryan hunting for guard pass, Pena defending with active guard and off-balancing | Ryan achieved top position inside Pena's open guard, looking to consolidate and pass | Ryan applies heavy pressure with his weight, attempting leg drag passes. Pena counters with De La Riva hooks on Ryan's leg, using frames on Ryan's hip to off-balance. Multiple grip adjustments from both athletes as Ryan tries to clear the legs | Ryan maintains top position but fails to complete pass. Pena successfully retains guard and sets up for back take | Ryan's pressure was relentless but lacked commitment to finish. Pena's guard is exceptionally difficult to pass due to constant movement and control of angles. Key learning: sometimes sustained pressure is better than rushing the pass
**SEG** | [1:28-1:42] | half_guard | Ryan | 0.7 | 0-0 | 0.7 | 0.9 | pass_to_half,underhook_battle | half_guard_passing,pressure_application | Ryan secures half guard, working underhook, Pena defending with lockdown | Ryan successfully transitioned from open guard to half guard position | Ryan achieved half guard with underhook control, looking to improve position | Ryan uses underhook to control Pena's hip. Pena locks a tight half guard lockdown with his legs, preventing Ryan from passing. Ryan applies heavy shoulder pressure and works to break the lockdown. Pena uses frames and hip movement to maintain the lockdown | Ryan maintains half guard position with underhook control but cannot break through Pena's lockdown | Excellent half guard lockdown defense from Pena. Ryan's underhook control was good but he needed to attack the lockdown more aggressively. Key learning: control the lockdown leg first, then work the pass
**SEG** | [1:42-1:49] | mount | Ryan | 0.9 | 3-0 | 0.9 | 1.0 | guard_pass_completion,mount_transition,scoring | passing_mechanics,mount_entry,scoring_sequence | Ryan completes pass to mount, 3 points scored | Ryan successfully passed Pena's guard and transitioned to mount position | Ryan cleared Pena's legs and established high mount position | Ryan used hip pressure to clear the lockdown, then drove his hips forward to establish mount. He immediately controlled Pena's upper body with crossface and arm control. Pena attempted to bridge but Ryan's weight distribution prevented the escape | Ryan successfully passed to mount and scored 3 points, establishing dominant position | Excellent pass execution and scoring. Ryan's hip pressure and timing were perfect. The key was committing fully to the pass once the opportunity opened

CRITICAL: 
- You MUST AIM for {segment_range} segments for the full match (quality over quantity).
- You MUST add as many relevant position tags as possible within each segment (limited to 10)
- Analyze the COMPLETE video from 0:00 to {duration}. Video MUST BE COMPLETELY ANALYSED. Verify your last segment reaches the end.

Now provide the complete timeline with all segments and fields **SEG** line."""




TRAINING_PROMPT = """You are analyzing BJJ TRAINING footage.

Create a position timeline in PIPE-SEPARATED MARKDOWN format.

OUTPUT STRUCTURE:

Line 1: **META** | context:training | duration:MM:SS | skill_estimate:beginner/intermediate/advanced

Line 2: **FORMAT** | [time] | pos | working | activity | attempt | quality | interest | reasons | focus | notes(max100char)

Lines 3+: **SEG** | [data...]

SEGMENT GUIDELINES:
- ~30 seconds each, split on position changes or technique attempts
- Time format: [MM:SS-MM:SS]
- Position: standing, guard, mount, side_control, back_control, half_guard, turtle
- Working: A1, A2, both, or - (who's actively working)
- Activity: 0.0-1.0 (how much movement/action)
- Attempt: technique being attempted or "maintain"
- Quality: 0.0-1.0 (execution quality vs fundamentals)
- Interest: 0.0-1.0 (how much coaching feedback needed)
- Reasons: comma-separated tags (no spaces)
- Focus: directive for Stage 2, or - if interest < 0.7
- Notes: MAX 100 characters, concise observation

INTEREST REASON TAGS:
technique_attempt, fundamental_error, good_execution, teaching_moment, 
safety_concern, progression_opportunity, decision_making

ANALYSIS FOCUS OPTIONS:
technique_feedback, fundamental_correction, positive_reinforcement, 
safety_guidance, progression_path, decision_analysis

SKILL ESTIMATION:
- beginner: basic movements, frequent errors, limited technique variety
- intermediate: solid fundamentals, some refinement needed, broader repertoire
- advanced: clean technique, tactical awareness, efficient movement

RULES:
- Be concise (notes max 100 chars)
- Use abbreviations where clear
- Interest > 0.7 requires focus directive
- Interest < 0.7 uses - for focus
- Maintain chronological order
- No gaps in timeline

Example output:
**META** | context:training | duration:16:10 | skill_estimate:intermediate
**FORMAT** | [time] | pos | working | activity | attempt | quality | interest | reasons | focus | notes(max100char)
**SEG** | [0:00-0:30] | standing | both | 0.5 | grips | 0.6 | 0.3 | - | - | Basic collar grips
**SEG** | [0:30-1:15] | guard | A1 | 0.7 | armbar | 0.4 | 0.9 | technique_attempt,fundamental_error | technique_feedback | Armbar, elbow not controlled

Now analyze the video and provide the complete timeline:"""


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def upload_video_to_gemini(video_path: str, client) -> tuple:
    """Upload video to Gemini File API and return file object and name."""
    print(f"\n[Stage 1] Uploading video: {video_path}")
    
    video_file = client.files.upload(file=video_path)
    print(f"  Uploaded as: {video_file.name}")
    
    # Wait for processing
    print("  Waiting for video processing...")
    while video_file.state.name == 'PROCESSING':
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    
    if video_file.state.name == 'FAILED':
        raise ValueError(f"Video processing failed")
    
    print(f"  ✓ Video processing complete: {video_file.uri}")
    
    # Verify video is ACTIVE
    video_file = client.files.get(name=video_file.name)
    if video_file.state.name != 'ACTIVE':
        raise ValueError(f"Video state changed unexpectedly: {video_file.state.name}")
    
    print(f"  ✓ Video ready for caching")
    return video_file, video_file.name


def load_cv_checkpoints(cv_checkpoint_path: str = "outputs/experiment4.0/stage0_cv_checkpoints_5s.json") -> str:
    """Load and format CV checkpoints for the prompt."""
    try:
        with open(cv_checkpoint_path, 'r') as f:
            cv_data = json.load(f)
        
        checkpoints = cv_data['checkpoints']
        
        # Include ALL checkpoints for maximum temporal anchoring
        # This provides a checkpoint every 5 seconds to prevent temporal drift
        formatted = [cp['compressed'] for cp in checkpoints]
        
        print(f"  Loaded {len(formatted)} CV checkpoints (every 5s)")
        
        return "\n".join(formatted)
    
    except FileNotFoundError:
        print(f"  Warning: CV checkpoints not found at {cv_checkpoint_path}")
        return "CV checkpoints not available - analyze video without CV anchors."


def run_stage1_analysis(video_file, mode: str, client, existing_cache_name: str = None, previous_output: str = None, model_choice: str = "gemini-2.5-pro") -> str:
    """Run Stage 1 skeleton analysis with context caching."""
    # Get video duration and calculate segment range
    try:
        if video_file.video_metadata and video_file.video_metadata.video_duration:
            video_duration = video_file.video_metadata.video_duration
            duration_sec = video_duration.total_seconds() if hasattr(video_duration, 'total_seconds') else 0.0
            mm = int(duration_sec // 60)
            ss = int(duration_sec % 60)
            duration_str = f"{mm}:{ss:02d}"
            
            # Calculate segment range based on video duration
            if duration_sec < 240:  # 0-4 minutes
                segment_range = "10-30"
            elif duration_sec < 480:  # 4-8 minutes
                segment_range = "20-40"
            else:  # 8-20 minutes (and beyond)
                segment_range = "30-70"
        else:
            duration_str = "END"
            segment_range = "30-70"  # Default
    except:
        duration_str = "END"
        segment_range = "30-70"  # Default
    
    print(f"[Stage 1] Running {mode} analysis...")
    print(f"  Model: {model_choice}")
    print(f"  Video duration: {duration_str}")
    print(f"  Segment range target: {segment_range}")
    
    # Build continuation prompt if we have previous output
    continuation_context = ""
    if previous_output:
        # Extract last segment to continue from
        lines = previous_output.strip().split('\n')
        last_seg = lines[-1] if lines else ""
        continuation_context = f"\n\nYou previously stopped at:\n{last_seg}\n\nCONTINUE from where you left off. Do NOT repeat previous segments. Pick up immediately after the last segment and continue to the END of the video.\n\n"
    
    # Select prompt based on mode
    if mode == "competition":
        prompt = COMPETITION_PROMPT.format(duration=duration_str, segment_range=segment_range) + continuation_context
    else:
        prompt = TRAINING_PROMPT + continuation_context
    
    from google.genai import types
    
    # Use selected model
    model = f'models/{model_choice}'
    
    # Use existing cache or create new one
    if existing_cache_name:
        cache_name_to_use = existing_cache_name
        print(f"  Using existing cache: {cache_name_to_use}")
    else:
        # Create cache with video (saves 94% on cached tokens)
        print("  Creating context cache...")
        import hashlib
        cache_version = hashlib.md5(f"{mode}_v1".encode()).hexdigest()[:8]
        
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name=f'bjj_{mode}_v{cache_version}',
                system_instruction=f"You are a world-class BJJ coach analyzing {mode} footage.",
                contents=[video_file],  # Pass file directly, not wrapped
                ttl='3600s',  # 1 hour
            )
        )
        cache_name_to_use = cache.name
        print(f"  ✓ Cache created: {cache_name_to_use} (TTL: 1 hour)")
    
    # Generate response using cache
    print("  Analyzing video (this may take 2-3 minutes)...")
    start_time = time.time()
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            cached_content=cache_name_to_use,
            max_output_tokens=25000,
            temperature=0.5
        )
    )
    
    elapsed = time.time() - start_time
    print(f"  ✓ Analysis complete ({elapsed:.1f}s)")
    
    # Print token usage
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        print(f"\n[Token Usage]")
        
        prompt_tokens = getattr(usage, 'prompt_token_count', None)
        cached_tokens = getattr(usage, 'cached_content_token_count', None)
        output_tokens = getattr(usage, 'candidates_token_count', None)
        total_tokens = getattr(usage, 'total_token_count', None)
        
        print(f"  Prompt tokens: {prompt_tokens:,}" if prompt_tokens else "  Prompt tokens: N/A")
        print(f"  Cached tokens: {cached_tokens:,}" if cached_tokens else "  Cached tokens: N/A")
        print(f"  Output tokens: {output_tokens:,}" if output_tokens else "  Output tokens: N/A")
        print(f"  Total tokens: {total_tokens:,}" if total_tokens else "  Total tokens: N/A")
        
        # Calculate cost
        INPUT_PRICE = 0.30 / 1_000_000
        OUTPUT_PRICE = 1.20 / 1_000_000
        CACHED_PRICE = 0.01875 / 1_000_000
        
        prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
        cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
        output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
        
        uncached_input = prompt_tokens - cached_tokens
        input_cost = (uncached_input * INPUT_PRICE) + (cached_tokens * CACHED_PRICE)
        output_cost = output_tokens * OUTPUT_PRICE
        total_cost = input_cost + output_cost
        
        print(f"\n[Cost Analysis]")
        print(f"  Input cost: ${input_cost:.4f} (${uncached_input * INPUT_PRICE:.4f} uncached + ${cached_tokens * CACHED_PRICE:.4f} cached)")
        print(f"  Output cost: ${output_cost:.4f}")
        print(f"  Total cost: ${total_cost:.4f}")
    
    # Extract text
    if not response.text:
        raise ValueError("No response from model")
    
    return response.text


def parse_stage1_output(output_text: str) -> Dict:
    """Parse pipe-separated MD output into structured data (skeleton + micro-analysis combined)."""
    lines = [l.strip() for l in output_text.strip().split('\n') if l.strip()]
    
    meta = {}
    segments = []
    format_line = None
    
    for line in lines:
        if line.startswith('**META**'):
            # Parse meta line
            parts = line.split('|')[1:]
            for part in parts:
                if ':' in part:
                    key, val = part.strip().split(':', 1)
                    meta[key] = val.strip()
        
        elif line.startswith('**FORMAT**'):
            format_line = line
        
        elif line.startswith('**SEG**'):
            # Parse segment line with skeleton + micro-analysis combined
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 10:
                continue
            
            seg = {
                'raw': line,
                'time': parts[1].strip('[]'),
                'position': parts[2],
                'top': parts[3],
                'control': parts[4],
                'score': parts[5],
                'action': parts[6],
                'confidence': parts[7],
                'reasons': parts[8],
                'focus': parts[9],
                'notes': parts[10] if len(parts) > 10 else '',
                'strategy': parts[11] if len(parts) > 11 else '',
                'setup': parts[12] if len(parts) > 12 else '',
                'execution': parts[13] if len(parts) > 13 else '',
                'outcome': parts[14] if len(parts) > 14 else '',
                'coaching': parts[15] if len(parts) > 15 else ''
            }
            
            segments.append(seg)
    
    return {
        'meta': meta,
        'format': format_line,
        'segments': segments,
        'raw_output': output_text
    }


def save_stage1_output(parsed_data: Dict, mode: str, video_path: str, run: int = None) -> None:
    """Save Stage 1v3 output to files (skeleton + details)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine filenames with optional run suffix
    suffix = f"_run{run}" if run is not None else ""
    
    # Save raw MD (using stage1v3 prefix to distinguish from stage1/stage1v2)
    md_path = OUT_DIR / f"stage1v3_skeleton_{mode}{suffix}.md"
    md_path.write_text(parsed_data['raw_output'], encoding='utf-8')
    
    # Save parsed JSON with combined skeleton + micro-analysis
    total_segments = len(parsed_data['segments'])
    segments_with_micro = sum(1 for seg in parsed_data['segments'] if seg.get('strategy'))
    
    json_data = {
        'meta': {
            'video_path': video_path,
            'mode': mode,
            'run': run,
            'processed_at': datetime.now().isoformat(),
            **parsed_data['meta']
        },
        'segments': parsed_data['segments']
    }
    
    json_path = OUT_DIR / f"stage1v3_skeleton_{mode}{suffix}.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding='utf-8')
    
    print(f"\n[Stage 1v3] Output saved:")
    print(f"  MD: {md_path}")
    print(f"  JSON: {json_path}")
    print(f"  Total segments: {total_segments}")
    print(f"  Segments with micro-analysis: {segments_with_micro}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 4 Stage 1 (Skeleton)")
    parser.add_argument("--video_path", required=True, help="Path to video file")
    parser.add_argument("--mode", required=True, choices=["competition", "training"], 
                       help="Analysis mode")
    parser.add_argument("--run", type=int, default=None, 
                       help="Run number (e.g., 1, 2, 3). If not specified, no suffix.")
    parser.add_argument("--cache_name", help="Existing cache name to reuse (skips video upload)")
    parser.add_argument("--continue_from", help="Previous run JSON to continue from (e.g., stage1_skeleton_competition_run1.json)")
    parser.add_argument("--model", default="gemini-2.5-pro", 
                       choices=["gemini-2.5-flash", "gemini-2.5-pro"],
                       help="Model to use (default: gemini-2.5-pro)")
    parser.add_argument("--api_key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()
    
    # Configure API
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: No API key provided")
        print("Set GOOGLE_API_KEY environment variable or use --api_key")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    
    # Create client for caching API
    from google import genai as genai_client
    client = genai_client.Client(api_key=api_key)
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("EXPERIMENT 4 - STAGE 1 (SKELETON PASS)")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Mode: {args.mode}")
    
    try:
        # Handle continuation or fresh run
        if args.cache_name:
            print(f"\n[Reusing cache]: {args.cache_name}")
            video_file = None  # Not needed when using existing cache
            cache_name = args.cache_name
        else:
            # Upload video
            video_file, video_file_name = upload_video_to_gemini(str(video_path), client)
            cache_name = None
        
        # Load previous output if continuing
        previous_output = None
        if args.continue_from:
            continue_path = OUT_DIR / args.continue_from
            if continue_path.exists():
                with open(continue_path, 'r') as f:
                    prev_data = json.load(f)
                    previous_output = prev_data.get('raw_output', '')
                    print(f"\n[Continuing from]: {args.continue_from}")
                    print(f"  Previous segments: {len(prev_data.get('segments', []))}")
            else:
                print(f"Warning: Continue file not found: {continue_path}")
        
        # Run analysis with caching
        output_text = run_stage1_analysis(video_file, args.mode, client, cache_name, previous_output, args.model)
        
        # Parse output
        parsed_data = parse_stage1_output(output_text)
        
        # Save results
        save_stage1_output(parsed_data, args.mode, str(video_path), args.run)
        
        print("\n" + "=" * 80)
        print("✓ STAGE 1 COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
