"""
Gemini Stages for Experiment 3
Stages 1-3: Position Timeline, Adaptive Detail, Synthesis
"""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not available")


# Gemini pricing (as of 2024)
PRICING = {
    'gemini-2.5-flash': {
        'input': 0.30 / 1_000_000,  # $0.30 per 1M tokens
        'output': 1.20 / 1_000_000,  # $1.20 per 1M tokens
        'cached_input': 0.01875 / 1_000_000  # $0.01875 per 1M cached tokens (94% cheaper)
    },
    'models/gemini-2.5-flash': {
        'input': 0.30 / 1_000_000,  # $0.30 per 1M tokens
        'output': 1.20 / 1_000_000,  # $1.20 per 1M tokens
        'cached_input': 0.01875 / 1_000_000  # $0.01875 per 1M cached tokens (94% cheaper)
    },
    'gemini-2.0-flash-exp': {
        'input': 0.00 / 1_000_000,  # Free during preview
        'output': 0.00 / 1_000_000,  # Free during preview
        'cached_input': 0.00 / 1_000_000  # Free during preview
    },
    'models/gemini-2.0-flash-exp': {
        'input': 0.00 / 1_000_000,  # Free during preview
        'output': 0.00 / 1_000_000,  # Free during preview
        'cached_input': 0.00 / 1_000_000  # Free during preview
    }
}


class GeminiStage1Timeline:
    """Stage 1: Position Timeline with Athlete Identification."""
    
    def __init__(self, video_path: str, cv_data: Dict, output_dir: Path, output_suffix: str = ""):
        self.video_path = video_path
        self.cv_data = cv_data
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.output_suffix = output_suffix  # For versioning outputs (e.g., "_v3.3")
        self.model_name = 'models/gemini-2.5-flash'  # Gemini 2.5 Flash with caching support
        self.ontology_cache = None  # Global ontology cache
        
        # Initialize Gemini client
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            self.client = genai.Client(api_key=api_key)
    
    def get_ontology(self) -> str:
        """Get the comprehensive BJJ ontology (for global caching)."""
        return """COMPREHENSIVE BJJ POSITION ONTOLOGY:

STANDING POSITIONS:
- standing, collar_ties, body_lock, front_headlock
- single_leg_attempt, double_leg_attempt, throw_attempt
- guillotine_standing

GUARD POSITIONS (Bottom):
Closed Guard: closed_guard, high_guard, broken_closed_guard
Open Guard: open_guard, spider_guard, de_la_riva, reverse_de_la_riva, x_guard, single_leg_x, butterfly_guard, lasso_guard
Half Guard: half_guard, knee_shield, deep_half, lockdown, 93_guard
Seated/Inverted: seated_guard, shin_to_shin, k_guard, inverted_guard

TOP POSITIONS:
Side Control: side_control, knee_on_belly, north_south, scarf_hold, reverse_scarf_hold
Mount: mount, high_mount, s_mount, technical_mount
Back Control: back_control (specify: hooks_in, body_triangle, one_hook)
Turtle: turtle_top
Passing: guard_pass_attempt, knee_slice, toreando_pass, stack_pass, leg_drag

SUBMISSION POSITIONS:
Chokes: rear_naked_choke, guillotine, arm_triangle, triangle_choke, darce, anaconda, ezekiel, bow_and_arrow, loop_choke
Joint Locks: armbar, kimura, americana, omoplata, straight_ankle_lock, heel_hook, toe_hold, kneebar, wrist_lock

TRANSITIONS:
- scramble, sweep_attempt, sweep_completed, takedown_in_progress, submission_escape, position_recovery, guard_retention, referee_standup, referee_reset

ACTION TYPES (for key_actions):
Offensive: submission_attempt, submission_locked_in, sweep_attempt, sweep_completed, pass_attempt, pass_completed, takedown_attempt, takedown_completed, back_take, mount_achieved
Defensive: submission_defense, submission_escape, sweep_defense, pass_defense, guard_recovery, position_escape
Positional: grip_fighting, position_improvement, position_transition, base_broken, scramble_initiated"""
    
    def create_system_instruction(self) -> str:
        """Create video-specific system instruction (CV constraints only)."""
        cv_constraints = self._generate_cv_constraints()
        
        instruction = f"""You are an expert BJJ match analyzer with deep knowledge of grappling techniques and positions.

CV CONSTRAINTS (GUIDELINES):
{cv_constraints}

Your task is to:
1. FIRST: Identify the two athletes based on visual characteristics (gi/rash guard color, body type, position in frame)
2. THEN: Analyze the full video to identify ALL position changes
3. Output a position timeline with natural durations (no arbitrary limits)

IMPORTANT: Use CV constraints as a GUIDELINE, not absolute rules. The CV data helps you understand the match flow, but:
- If you see something different in the video than what CV suggests, TRUST YOUR VISUAL ANALYSIS
- CV may miss nuances (submissions, transitions, referee interventions)
- Your video analysis is the primary source of truth; CV is supplementary context"""
        
        return instruction
    
    def _generate_cv_constraints(self) -> str:
        """Generate athlete-specific CV constraints from interaction metrics."""
        constraints = []
        metrics = self.cv_data.get('per_second_metrics', [])
        
        # Group by 30-second windows for readability
        window_size = 30
        for i in range(0, len(metrics), window_size):
            window = metrics[i:i+window_size]
            if not window:
                continue
            
            start_time = window[0]['timestamp']
            end_time = window[-1]['timestamp']
            
            # Calculate averages per athlete
            a1_standing = sum(m['athlete_1']['standing_probability'] for m in window if m['athlete_1']['detected']) / max(sum(1 for m in window if m['athlete_1']['detected']), 1)
            a2_standing = sum(m['athlete_2']['standing_probability'] for m in window if m['athlete_2']['detected']) / max(sum(1 for m in window if m['athlete_2']['detected']), 1)
            
            avg_entanglement = sum(m['interaction']['limb_entanglement'] for m in window) / len(window)
            avg_contact = sum(m['interaction']['contact_intensity'] for m in window) / len(window)
            avg_distance = sum(m['interaction']['centroid_distance'] for m in window) / len(window)
            
            # Determine dominant athlete
            dominant_counts = [m['interaction']['dominant_athlete'] for m in window]
            a1_dominant = dominant_counts.count(1)
            a2_dominant = dominant_counts.count(2)
            
            if a1_dominant > a2_dominant:
                dominant = "Athlete1 DOMINANT"
            elif a2_dominant > a1_dominant:
                dominant = "Athlete2 DOMINANT"
            else:
                dominant = "NEUTRAL"
            
            # Build constraint
            constraint = f"{int(start_time)}s-{int(end_time)}s:\n"
            constraint += f"  Athlete1: {'STANDING' if a1_standing > 0.7 else 'GROUND' if a1_standing < 0.3 else 'MIXED'} (prob={a1_standing:.2f})\n"
            constraint += f"  Athlete2: {'STANDING' if a2_standing > 0.7 else 'GROUND' if a2_standing < 0.3 else 'MIXED'} (prob={a2_standing:.2f})\n"
            constraint += f"  Contact: {'HIGH' if avg_contact > 0.6 else 'MEDIUM' if avg_contact > 0.3 else 'LOW'} ({avg_contact:.2f})\n"
            constraint += f"  Entanglement: {avg_entanglement:.1f} limbs, Distance: {avg_distance:.2f}, {dominant}"
            
            constraints.append(constraint)
        
        # Return ALL constraints - the full match needs CV guidance
        return "\n\n".join(constraints)
    
    def _format_suggested_segments(self, segments: List[Dict]) -> str:
        """Format suggested segments for the prompt."""
        if not segments:
            return "No suggested segments available - analyze the full match freely."
        
        # Format first 10 and last 5 segments as examples
        lines = []
        display_segments = segments[:10] + (['...'] if len(segments) > 15 else []) + segments[-5:]
        
        for seg in display_segments:
            if seg == '...':
                lines.append("... ({} more segments) ...".format(len(segments) - 15))
                continue
            
            lines.append(
                f"  {seg['start_time']}-{seg['end_time']} ({seg['duration_sec']}s, "
                f"CV action={seg['cv_avg_action']:.3f}, "
                f"standing={seg['cv_avg_standing']:.2f})"
            )
        
        return "\n".join(lines)
    
    def create_prompt(self) -> str:
        """Create analysis prompt with suggested segments."""
        duration = self.cv_data.get('duration', 0)
        avg_action = self.cv_data['summary']['avg_action']
        
        # Get suggested segments from CV data
        suggested_segments = self.cv_data.get('suggested_segments', [])
        
        prompt = f"""Analyze this COMPLETE {duration/60:.1f}-minute BJJ match from START (0:00) to END ({int(duration//60)}:{int(duration%60):02d}).

STEP 1: ATHLETE IDENTIFICATION
Analyze the first 30 seconds and identify:
- Athlete 1: Visual characteristics (gi/rash guard color, body type, distinctive features)
- Athlete 2: Visual characteristics

After identification, assess their fighting styles:
- Athlete 1 Style: (guard_player/top_pressure/wrestler/submission_hunter)
- Athlete 1 Strengths: (list 2-3 key strengths observed)
- Athlete 2 Style: (guard_player/top_pressure/wrestler/submission_hunter)
- Athlete 2 Strengths: (list 2-3 key strengths observed)

STEP 2: MATCH SEGMENTATION & ANALYSIS
Computer vision has identified {len(suggested_segments)} suggested segments based on action levels and transitions.
These are HINTS to guide your analysis - you should use your video analysis as the primary source.

SUGGESTED SEGMENTS (from CV):
{self._format_suggested_segments(suggested_segments)}

YOUR TASK:
Analyze each suggested segment and decide how to represent it:
- KEEP as one segment if it's cohesive (e.g., "static closed guard 2:00-2:45")
- SPLIT if you see multiple distinct phases (e.g., "guard pass attempt → sweep → back take")
- MERGE with adjacent segments if they're really the same thing

TARGET: 30-100 final segments for the full match

SEGMENTATION PHILOSOPHY - How to assign Action scores and determine segment length:

LOW ACTION (Action < 0.30):
- Duration: 30-60 seconds
- Characteristics: Static position, minimal movement, positional maintenance
- Position changes: Very few (0-1)
- Technique attempts: 0-2 minor attempts (grip adjustments, posture breaks)
- Example: Athlete maintaining closed guard with grip fighting, no significant attempts

MEDIUM ACTION (Action 0.30-0.50):
- Duration: 15-45 seconds
- Characteristics: One major position change OR sustained technical exchange
- Position changes: 1-2 major transitions (e.g., takedown, guard pass, sweep)
- Technique attempts: 3-6 intermediate actions (pass attempts, escape attempts, positional improvements)
- Example: Guard passing sequence with multiple attempts, or takedown attempt with scramble to guard

HIGH ACTION (Action > 0.50):
- Duration: 5-20 seconds
- Characteristics: Rapid scrambles, multiple position changes, submission attempts
- Position changes: 3+ rapid transitions
- Technique attempts: 6+ techniques in quick succession
- Example: Explosive scramble with mount escape → sweep → back take → submission defense
- Note: If multiple high-action bursts are chained with only 2-5 second breaks, you may keep them as one segment OR split them if there's a clear pause/reset

Your Action score should reflect what you SEE in the video, not just the CV suggestion.

OUTPUT FORMAT - COMPRESSED (one line per segment):

**SEG [Start-End]** Pos:position | Sub:sub_position | Top:name | Bot:name | Conf:0.XX | Act:0.XX | Trans:type | KM:ts1,ts2 | KA:action1,action2 | Narr:brief description

EXAMPLES:

**SEG [0:00-0:45]** Pos:standing | Sub:collar_ties | Top:Athlete1 | Bot:Athlete2 | Conf:0.92 | Act:0.15 | Trans:match_start | KM:0:00,0:15,0:32 | KA:takedown_attempt | Narr:Collar ties, grip fighting. A1 single leg attempt, A2 sprawls.

**SEG [3:20-3:50]** Pos:closed_guard | Sub:guard_passing | Top:Athlete1 | Bot:Athlete2 | Conf:0.88 | Act:0.42 | Trans:guard_pass_attempt | KM:3:20,3:28,3:35,3:42 | KA:guard_pass_attempt,guard_retention | Narr:A1 stands to pass. A2 transitions de la riva. Toreando defended, recovers closed guard.

**SEG [5:42-5:55]** Pos:scramble | Sub:mount_escape_to_back | Top:Athlete1 | Bot:Athlete2 | Conf:0.85 | Act:0.68 | Trans:back_take | KM:5:42,5:46,5:50 | KA:mount_escape,scramble,back_take | Narr:Explosive scramble from mount escape. A1 takes back control.

KEY ACTIONS - Include actions that would score points in a no-gi BJJ match (takedowns, sweeps, passes, back takes, mounts, submissions, etc.) for video indexing. Only include actions that actually occurred. Use comma-separated list. Leave blank if no significant actions.

⚠️ CRITICAL: HARD CHARACTER LIMITS PER NARRATIVE (based on Action score):
- Action < 0.30 (LOW): 150 characters MAXIMUM (~40 tokens)
- Action 0.30-0.50 (MEDIUM): 250 characters MAXIMUM (~65 tokens)
- Action > 0.50 (HIGH): 350 characters MAXIMUM (~90 tokens)

Count every character including spaces and punctuation. EXCEED THESE = FAILURE.
These limits are MANDATORY to stay within the 25,000 token output limit.

CRITICAL REQUIREMENTS:
- Analyze the COMPLETE match from 0:00 to {int(duration//60)}:{int(duration%60):02d}
- Do NOT stop early - continue until the end of the video
- TARGET: 30-100 segments total (not 400+!)
- Each segment should represent a cohesive phase of the match
- Use athlete identifiers consistently (Athlete1, Athlete2 or their names if known)
- Submissions do NOT always end the match - continue analyzing after submission attempts

⚠️ OUTPUT LIMITS:
- MAXIMUM: 25,000 tokens OR 87,500 characters
- If approaching limit, reduce narrative detail but maintain full match coverage

⚠️ ANTI-LAZINESS CHECK:
Before finishing, verify you analyzed the ENTIRE duration from 0:00 to {int(duration//60)}:{int(duration%60):02d}.
If your last segment ends early, you missed content. Complete the full match."""
        
        return prompt
    
    def analyze(self) -> Dict:
        """Run Stage 1 analysis."""
        print("\n[Stage 1.1] Uploading video and creating cache...")
        
        if not GEMINI_AVAILABLE:
            return self._mock_output()
        
        start_time = time.time()
        
        # Upload video file
        video_file = self.client.files.upload(file=self.video_path)
        print(f"  Video uploaded: {video_file.name}")
        
        # Wait for processing
        while video_file.state.name == 'PROCESSING':
            print('  Waiting for video to be processed...')
            time.sleep(2)
            video_file = self.client.files.get(name=video_file.name)
        
        print(f"  Video processing complete: {video_file.uri}")
        
        # Create combined system instruction (ontology + CV constraints)
        # Ontology will be cached with the video
        ontology = self.get_ontology()
        cv_instruction = self.create_system_instruction()
        system_instruction = f"{ontology}\n\n{cv_instruction}"
        
        # Create cache with video + ontology (both cached together)
        # Use timestamp in display name to avoid using stale cache
        import hashlib
        # Add version suffix to force new cache for new format
        cache_version = hashlib.md5((system_instruction + "_v2_segment_format").encode()).hexdigest()[:8]
        
        cache = self.client.caches.create(
            model=self.model_name,
            config=types.CreateCachedContentConfig(
                display_name=f'bjj_match_v{cache_version}',
                system_instruction=system_instruction,
                contents=[video_file],
                ttl="3600s",  # 1 hour
            )
        )
        print(f"  Cache created: {cache.name} (v{cache_version}, TTL: 1 hour, includes ontology)")
        
        # Generate analysis using cache
        print("\n[Stage 1.2] Analyzing video...")
        prompt = self.create_prompt()
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                cached_content=cache.name,
                max_output_tokens=25000,  # Increased for enhanced narratives (Option 3)
                temperature=0.4,  # Lower = more consistent (range: 0.0-2.0, default: 1.0)
                top_p=0.95,  # Nucleus sampling (range: 0.0-1.0, default: 0.95)
                top_k=40,  # Top-k sampling (default: 40)
            )
        )
        
        # Check for truncation
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        print(f"\n[DEBUG] Finish reason: {finish_reason}")
        print(f"[DEBUG] Response text length: {len(response.text) if response.text else 0} chars")
        print(f"[DEBUG] Response text word count: {len(response.text.split()) if response.text else 0} words")
        
        if finish_reason in ['MAX_TOKENS', 'LENGTH']:
            print(f"\n⚠️  WARNING: Response was truncated due to {finish_reason}")
            print("  Consider increasing max_output_tokens or breaking into multiple passes")
        
        # Parse response
        print("\n[Stage 1.3] Parsing response...")
        result = self._parse_response(response.text)
        
        # Calculate costs
        usage = response.usage_metadata
        cost = self._calculate_cost(usage)
        
        elapsed = time.time() - start_time
        
        result['cost'] = cost
        result['time_seconds'] = round(elapsed, 2)
        result['cache_name'] = cache.name
        result['raw_markdown'] = response.text
        
        # Save raw markdown
        markdown_filename = f"stage1_markdown{self.output_suffix}.md"
        markdown_path = self.output_dir / markdown_filename
        with open(markdown_path, 'w') as f:
            f.write(response.text)
        
        print(f"\n[Token Usage Details]")
        print(f"  Input tokens: {usage.prompt_token_count}")
        print(f"  Cached tokens: {usage.cached_content_token_count}")
        print(f"  Output tokens (API reported): {usage.candidates_token_count}")
        print(f"  Output chars: {len(response.text) if response.text else 0}")
        print(f"  Output words: {len(response.text.split()) if response.text else 0}")
        print(f"  Estimated tokens (chars/4): {len(response.text) // 4 if response.text else 0}")
        print(f"  Estimated tokens (words): {len(response.text.split()) if response.text else 0}")
        print(f"  Cost: ${cost['total']:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        
        return result
    
    def _parse_response(self, text: str) -> Dict:
        """Parse Gemini response into structured data (new segment-based format)."""
        if text is None:
            print("ERROR: Response text is None (likely truncated)")
            return {
                'athlete_profiles': {},
                'segments': [],
                'raw_text': '',
                'error': 'Response was None - likely truncated'
            }
        
        result = {
            'athlete_profiles': {},
            'segments': [],  # New: segments instead of positions
            'raw_text': text
        }
        
        lines = text.split('\n')
        
        # Parse athlete identification section (keep existing logic)
        in_athlete_section = False
        current_athlete = None
        
        # Parse segments - support both compressed and verbose formats
        import re
        
        # Compressed format: **SEG [Start-End]** Pos:position | Sub:sub_position | ...
        compressed_pattern = r'\*\*SEG\s+\[(\d+:\d+)-(\d+:\d+)\]\*\*\s+(.+)'
        
        # Verbose format: **SEGMENT [Start-End]**
        verbose_pattern = r'\*\*SEGMENT\s+\[(\d+:\d+)-(\d+:\d+)\]\*\*'
        
        current_segment = None
        current_field = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect athlete identification section
            if 'athlete 1' in line_lower or 'athlete1' in line_lower:
                in_athlete_section = True
                current_athlete = 'athlete_1'
                result['athlete_profiles']['athlete_1'] = {
                    'visual_id': '', 
                    'name': 'Athlete 1',
                    'style': '',
                    'strengths': []
                }
            elif 'athlete 2' in line_lower or 'athlete2' in line_lower:
                current_athlete = 'athlete_2'
                result['athlete_profiles']['athlete_2'] = {
                    'visual_id': '', 
                    'name': 'Athlete 2',
                    'style': '',
                    'strengths': []
                }
            elif in_athlete_section and current_athlete and line.strip() and not line.strip().startswith('#'):
                # Parse style and strengths
                if 'style:' in line_lower:
                    result['athlete_profiles'][current_athlete]['style'] = line.split(':', 1)[1].strip()
                elif 'strength' in line_lower:
                    strengths_text = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                    result['athlete_profiles'][current_athlete]['strengths'] = [s.strip() for s in strengths_text.split(',')]
                # Accumulate athlete description
                elif line.strip().startswith('-'):
                    result['athlete_profiles'][current_athlete]['visual_id'] += line.strip()[1:].strip() + ' '
                elif ':' in line and 'style' not in line_lower and 'strength' not in line_lower:
                    result['athlete_profiles'][current_athlete]['visual_id'] += line.strip() + ' '
            
            # Stop athlete section when we hit segments
            if line.strip().startswith('**SEGMENT') or line.strip().startswith('**SEG'):
                in_athlete_section = False
            
            # Try compressed format first: **SEG [Start-End]** Pos:position | Sub:sub_position | ...
            compressed_match = re.match(compressed_pattern, line.strip())
            if compressed_match:
                # Save previous segment if exists
                if current_segment:
                    result['segments'].append(current_segment)
                
                # Parse compressed format
                start_time = compressed_match.group(1)
                end_time = compressed_match.group(2)
                fields_text = compressed_match.group(3)
                
                # Initialize segment
                current_segment = {
                    'start': start_time,
                    'end': end_time,
                    'position': '',
                    'sub_position': '',
                    'top_athlete': '',
                    'bottom_athlete': '',
                    'confidence': 0.9,
                    'avg_action': 0.0,
                    'transition': '',
                    'key_moments': [],
                    'key_actions': [],
                    'narrative': ''
                }
                
                # Split by pipes and parse each field
                fields = fields_text.split('|')
                for field in fields:
                    field = field.strip()
                    if ':' not in field:
                        continue
                    
                    field_name, field_value = field.split(':', 1)
                    field_name = field_name.strip().lower()
                    field_value = field_value.strip()
                    
                    if field_name == 'pos':
                        current_segment['position'] = field_value
                    elif field_name == 'sub':
                        current_segment['sub_position'] = field_value
                    elif field_name == 'top':
                        current_segment['top_athlete'] = field_value
                    elif field_name == 'bot':
                        current_segment['bottom_athlete'] = field_value
                    elif field_name == 'conf':
                        try:
                            current_segment['confidence'] = float(field_value)
                        except:
                            pass
                    elif field_name == 'act':
                        try:
                            current_segment['avg_action'] = float(field_value)
                        except:
                            pass
                    elif field_name == 'trans':
                        current_segment['transition'] = field_value
                    elif field_name == 'km':
                        # Key moments: comma-separated timestamps
                        if field_value:
                            current_segment['key_moments'] = [ts.strip() for ts in field_value.split(',')]
                    elif field_name == 'ka':
                        # Key actions: comma-separated actions
                        if field_value:
                            current_segment['key_actions'] = [a.strip() for a in field_value.split(',')]
                    elif field_name == 'narr':
                        current_segment['narrative'] = field_value
                
                current_field = None
                continue
            
            # Fallback to verbose format: **SEGMENT [Start-End]**
            verbose_match = re.match(verbose_pattern, line)
            if verbose_match:
                # Save previous segment if exists
                if current_segment:
                    result['segments'].append(current_segment)
                
                # Start new segment
                current_segment = {
                    'start': verbose_match.group(1),
                    'end': verbose_match.group(2),
                    'position': '',
                    'sub_position': '',
                    'top_athlete': '',
                    'bottom_athlete': '',
                    'confidence': 0.9,
                    'avg_action': 0.0,
                    'transition': '',
                    'key_moments': [],
                    'key_actions': [],
                    'narrative': ''
                }
                current_field = None
                continue
            
            # Parse segment fields
            if current_segment and ':' in line and not line.strip().startswith('-'):
                field_name = line.split(':', 1)[0].strip().lower()
                field_value = line.split(':', 1)[1].strip() if len(line.split(':', 1)) > 1 else ''
                
                if field_name == 'position':
                    current_segment['position'] = field_value
                elif field_name == 'sub-position':
                    current_segment['sub_position'] = field_value
                elif field_name == 'top':
                    current_segment['top_athlete'] = field_value
                elif field_name == 'bottom':
                    current_segment['bottom_athlete'] = field_value
                elif field_name == 'confidence':
                    try:
                        current_segment['confidence'] = float(field_value)
                    except:
                        pass
                elif field_name == 'action':
                    try:
                        current_segment['avg_action'] = float(field_value)
                    except:
                        pass
                elif field_name == 'transition':
                    current_segment['transition'] = field_value
                elif field_name == 'key moments':
                    current_field = 'key_moments'
                elif field_name == 'key actions':
                    # Parse comma-separated actions
                    if field_value:
                        current_segment['key_actions'] = [a.strip() for a in field_value.split(',')]
                elif field_name == 'narrative':
                    current_segment['narrative'] = field_value
                    current_field = 'narrative'
                continue
            
            # Parse key moments (lines starting with "- ")
            if current_segment and current_field == 'key_moments' and line.strip().startswith('- '):
                moment_text = line.strip()[2:]  # Remove "- "
                current_segment['key_moments'].append(moment_text)
                continue
            
            # Continue narrative on next lines
            if current_segment and current_field == 'narrative' and line.strip() and not line.strip().startswith('**'):
                current_segment['narrative'] += ' ' + line.strip()
                continue
        
        # Add final segment
        if current_segment:
            result['segments'].append(current_segment)
        
        # FALLBACK: If no segments found, try parsing old table format
        if not result['segments']:
            in_table = False
            for line in lines:
                if '|' in line and 'start' in line.lower():
                    in_table = True
                    continue
                if '|' in line and '---' in line:
                    continue
                if in_table and '|' in line and line.strip().startswith('|'):
                    # Parse table row (old format fallback)
                    parts = [p.strip() for p in line.split('|')[1:-1]]
                    if len(parts) >= 6 and parts[0] and (parts[0][0].isdigit() or ':' in parts[0]):
                        try:
                            position = {
                                'start': parts[0],
                                'end': parts[1],
                                'position': parts[2],
                                'sub_position': parts[3] if len(parts) > 3 else '',
                                'top_athlete': parts[4] if len(parts) > 4 else '',
                                'bottom_athlete': parts[5] if len(parts) > 5 else '',
                                'confidence': float(parts[6]) if len(parts) > 6 and parts[6] else 0.9,
                                'avg_action': float(parts[7]) if len(parts) > 7 and parts[7] else 0.5,
                                'transition': parts[8] if len(parts) > 8 else '',
                                'narrative': parts[9] if len(parts) > 9 else '',
                                'key_actions': []
                            }
                            result['segments'].append(position)  # Store as segment for compatibility
                        except (ValueError, IndexError):
                            continue  # Skip malformed rows
        
        # Clean up athlete profiles
        for athlete_id in result['athlete_profiles']:
            result['athlete_profiles'][athlete_id]['visual_id'] = \
                result['athlete_profiles'][athlete_id]['visual_id'].strip()
        
        return result
    
    def _calculate_cost(self, usage) -> Dict:
        """Calculate API cost based on token usage."""
        pricing = PRICING.get(self.model_name, PRICING['models/gemini-2.5-flash'])
        
        input_tokens = usage.prompt_token_count - usage.cached_content_token_count
        cached_tokens = usage.cached_content_token_count
        output_tokens = usage.candidates_token_count if usage.candidates_token_count is not None else 0
        
        input_cost = input_tokens * pricing['input']
        cached_cost = cached_tokens * pricing['cached_input']
        output_cost = output_tokens * pricing['output']
        
        return {
            'input_tokens': usage.prompt_token_count,
            'cached_tokens': usage.cached_content_token_count,
            'non_cached_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': round(input_cost, 6),
            'cached_cost': round(cached_cost, 6),
            'output_cost': round(output_cost, 6),
            'total': round(input_cost + cached_cost + output_cost, 6)
        }
    
    def _mock_output(self) -> Dict:
        """Mock output for testing without Gemini."""
        return {
            'athlete_profiles': {
                'athlete_1': {'visual_id': 'Mock athlete 1'},
                'athlete_2': {'visual_id': 'Mock athlete 2'}
            },
            'positions': [
                {'start': '0:00', 'end': '0:30', 'position': 'standing', 'sub_position': 'collar_ties',
                 'top_athlete': 'Athlete1', 'bottom_athlete': 'Athlete2', 'confidence': 0.9, 'avg_action': 0.2}
            ],
            'cost': {'total': 0.0},
            'time_seconds': 0.0,
            'raw_text': 'Mock output'
        }




if __name__ == '__main__':
    """Run Stage 1 in isolation for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Stage 1: Position Timeline Analysis')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--cv-cache', required=True, help='Path to CV cache JSON')
    parser.add_argument('--output', default='outputs/experiment3', help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STAGE 1: Position Timeline Analysis (Enhanced)")
    print("="*80)
    print(f"Video: {args.video}")
    print(f"CV Cache: {args.cv_cache}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")
    
    # Load CV cache
    print("Loading CV cache...")
    with open(args.cv_cache, 'r') as f:
        cv_data = json.load(f)
    print(f"✓ Loaded {len(cv_data.get('per_second_metrics', []))} seconds of CV data\n")
    
    # Run Stage 1
    stage1 = GeminiStage1Timeline(
        video_path=args.video,
        cv_data=cv_data,
        output_dir=args.output
    )
    
    result = stage1.analyze()
    
    # Save result
    output_suffix = getattr(stage1, 'output_suffix', '')
    output_filename = f"stage1_timeline{output_suffix}.json"
    output_path = Path(args.output) / output_filename
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "="*80)
    print("STAGE 1 COMPLETE!")
    print("="*80)
    print(f"Segments: {len(result.get('segments', []))}")
    print(f"Athletes: {len(result.get('athlete_profiles', {}))}")
    print(f"Cost: ${result.get('cost', {}).get('total', 0):.4f}")
    print(f"Time: {result.get('time_seconds', 0):.1f}s")
    print(f"Output: {output_path}")
    print("="*80 + "\n")
