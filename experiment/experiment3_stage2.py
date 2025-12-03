"""
Stage 2: Lazy Detection and Targeted Re-analysis
Part of Experiment 3.3 - Two-Pass Architecture
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import pricing from stage 123
from experiment3_stages123 import PRICING, GeminiStage1Timeline


class GeminiStage2Refinement:
    """Stage 2: Lazy Detection and Targeted Re-analysis.
    
    Detects segments where Stage 1 was lazy/repetitive and re-analyzes them
    with fresh context in smaller windows.
    """
    
    def __init__(self, stage1_results: Dict, cv_data: Dict, video_path: str, 
                 output_dir: Path, output_suffix: str = ""):
        self.stage1_results = stage1_results
        self.cv_data = cv_data
        self.video_path = video_path
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.output_suffix = output_suffix
        self.model_name = 'models/gemini-2.5-flash'
        
        # Initialize Gemini client
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            self.client = genai.Client(api_key=api_key)
    
    def detect_lazy_segments(self) -> Dict:
        """Detect segments where Stage 1 was lazy using 5 signals."""
        segments = self.stage1_results.get('segments', [])
        if not segments:
            return {'flagged_segments': [], 'signals': {}, 'attention_cliff': None}
        
        flagged = []
        signals_per_segment = {}
        
        for i, seg in enumerate(segments):
            signals = []
            
            # Signal 1: Repetition
            if i >= 2:
                prev_positions = [segments[j]['position'] for j in range(max(0, i-2), i)]
                if all(p == seg['position'] for p in prev_positions):
                    signals.append('repetition')
            
            # Signal 2: CV Mismatch (standing probability)
            seg_start_sec = self._time_to_seconds(seg['start'])
            seg_end_sec = self._time_to_seconds(seg['end'])
            cv_metrics = [m for m in self.cv_data.get('per_second_metrics', [])
                         if seg_start_sec <= m['timestamp'] <= seg_end_sec]
            
            if cv_metrics:
                valid_metrics = [m for m in cv_metrics if m['athlete_1']['detected'] and m['athlete_2']['detected']]
                if valid_metrics:
                    avg_standing = sum(
                        (m['athlete_1']['standing_probability'] + m['athlete_2']['standing_probability']) / 2
                        for m in valid_metrics
                    ) / len(valid_metrics)
                    
                    model_is_standing = seg['position'] in ['standing', 'collar_ties', 'body_lock', 'front_headlock']
                    if model_is_standing and avg_standing < 0.3:
                        signals.append('cv_standing_mismatch')
                    elif not model_is_standing and avg_standing > 0.7:
                        signals.append('cv_standing_mismatch')
            
            # Signal 3: Action Score Mismatch
            if cv_metrics:
                cv_avg_action = sum(m.get('action_score', 0) for m in cv_metrics) / len(cv_metrics)
                model_action = seg.get('avg_action', 0)
                if abs(cv_avg_action - model_action) > 0.3:
                    signals.append('action_mismatch')
            
            # Signal 4: Duration Anomaly
            duration = seg_end_sec - seg_start_sec
            if duration > 60 and cv_metrics:
                action_scores = [m.get('action_score', 0) for m in cv_metrics]
                if len(action_scores) > 1:
                    mean_action = sum(action_scores) / len(action_scores)
                    variance = sum((x - mean_action)**2 for x in action_scores) / len(action_scores)
                    if variance > 0.05:
                        signals.append('duration_anomaly')
            
            # Signal 5: Narrative Similarity
            if i > 0:
                prev_narrative = segments[i-1].get('narrative', '')
                curr_narrative = seg.get('narrative', '')
                if prev_narrative and curr_narrative:
                    similarity = self._calculate_similarity(prev_narrative, curr_narrative)
                    if similarity > 0.8:
                        signals.append('narrative_similarity')
            
            # Flag if 2+ signals
            if len(signals) >= 2:
                flagged.append(i)
                signals_per_segment[i] = signals
        
        attention_cliff = self._detect_attention_cliff(flagged, segments)
        
        return {
            'flagged_segments': flagged,
            'signals': signals_per_segment,
            'attention_cliff': attention_cliff,
            'total_segments': len(segments),
            'flagged_count': len(flagged)
        }
    
    def _detect_attention_cliff(self, flagged_segments: List[int], segments: List[Dict]) -> Dict:
        """Detect where model quality drops."""
        if not flagged_segments:
            return None
        
        first_lazy = min(flagged_segments)
        
        if first_lazy == 0:
            return {
                'segment_index': 0,
                'timestamp': '0:00',
                'success_duration_seconds': 0
            }
        
        cliff_segment = segments[first_lazy]
        cliff_time = self._time_to_seconds(cliff_segment['start'])
        
        return {
            'segment_index': first_lazy,
            'timestamp': cliff_segment['start'],
            'success_duration_seconds': cliff_time
        }
    
    def _create_contextual_windows(self, detection_results: Dict) -> List[Dict]:
        """Create contextual windows for re-analysis."""
        flagged = detection_results['flagged_segments']
        if not flagged:
            return []
        
        segments = self.stage1_results.get('segments', [])
        attention_cliff = detection_results.get('attention_cliff')
        
        # Determine window size
        if attention_cliff and attention_cliff['success_duration_seconds'] > 0:
            window_duration = min(attention_cliff['success_duration_seconds'], 360)
        else:
            window_duration = 360
        
        # Group consecutive flagged segments
        flagged_ranges = []
        current_range = [flagged[0]]
        
        for i in range(1, len(flagged)):
            if flagged[i] == flagged[i-1] + 1:
                current_range.append(flagged[i])
            else:
                flagged_ranges.append(current_range)
                current_range = [flagged[i]]
        flagged_ranges.append(current_range)
        
        # Create windows
        windows = []
        for range_indices in flagged_ranges:
            start_idx = range_indices[0]
            end_idx = range_indices[-1]
            
            range_start_time = self._time_to_seconds(segments[start_idx]['start'])
            range_end_time = self._time_to_seconds(segments[end_idx]['end'])
            range_duration = range_end_time - range_start_time
            
            if range_duration <= window_duration:
                buffer_start_idx = max(0, start_idx - 1)
                windows.append({
                    'start_segment_idx': buffer_start_idx,
                    'end_segment_idx': end_idx,
                    'start_time': segments[buffer_start_idx]['start'],
                    'end_time': segments[end_idx]['end'],
                    'flagged_segments': range_indices
                })
            else:
                # Split into multiple windows
                current_start_idx = start_idx
                while current_start_idx <= end_idx:
                    window_start_time = self._time_to_seconds(segments[current_start_idx]['start'])
                    target_end_time = window_start_time + window_duration
                    
                    current_end_idx = current_start_idx
                    for idx in range(current_start_idx, end_idx + 1):
                        seg_end_time = self._time_to_seconds(segments[idx]['end'])
                        if seg_end_time <= target_end_time:
                            current_end_idx = idx
                        else:
                            break
                    
                    buffer_start_idx = max(0, current_start_idx - 1)
                    
                    windows.append({
                        'start_segment_idx': buffer_start_idx,
                        'end_segment_idx': current_end_idx,
                        'start_time': segments[buffer_start_idx]['start'],
                        'end_time': segments[current_end_idx]['end'],
                        'flagged_segments': [i for i in range_indices if current_start_idx <= i <= current_end_idx]
                    })
                    
                    current_start_idx = current_end_idx + 1
        
        return windows
    
    def _compress_cv_data(self, start_time: str, end_time: str) -> str:
        """Compress CV data to abbreviated format."""
        start_sec = self._time_to_seconds(start_time)
        end_sec = self._time_to_seconds(end_time)
        
        metrics = [m for m in self.cv_data.get('per_second_metrics', [])
                  if start_sec <= m['timestamp'] <= end_sec]
        
        if not metrics:
            return "No CV data available."
        
        compressed = []
        window_size = 30
        
        for i in range(0, len(metrics), window_size):
            window = metrics[i:i+window_size]
            if not window:
                continue
            
            w_start = int(window[0]['timestamp'])
            w_end = int(window[-1]['timestamp'])
            
            valid = [m for m in window if m['athlete_1']['detected'] and m['athlete_2']['detected']]
            if not valid:
                continue
            
            a1_standing = sum(m['athlete_1']['standing_probability'] for m in valid) / len(valid)
            a2_standing = sum(m['athlete_2']['standing_probability'] for m in valid) / len(valid)
            avg_contact = sum(m['interaction']['contact_intensity'] for m in window) / len(window)
            avg_entanglement = sum(m['interaction']['limb_entanglement'] for m in window) / len(window)
            avg_distance = sum(m['interaction']['centroid_distance'] for m in window) / len(window)
            
            dominant_counts = [m['interaction']['dominant_athlete'] for m in window]
            a1_dom = dominant_counts.count(1)
            a2_dom = dominant_counts.count(2)
            dom = "A1>" if a1_dom > a2_dom else "A2>" if a2_dom > a1_dom else "="
            
            a1_state = "S" if a1_standing > 0.7 else "G" if a1_standing < 0.3 else "M"
            a2_state = "S" if a2_standing > 0.7 else "G" if a2_standing < 0.3 else "M"
            
            line = f"{w_start}-{w_end}s: A1:{a1_state}({a1_standing:.2f}) A2:{a2_state}({a2_standing:.2f}) C:{avg_contact:.2f} E:{avg_entanglement:.1f} D:{avg_distance:.2f} {dom}"
            compressed.append(line)
        
        legend = "Legend: A1/A2=Athletes, S/G/M=Standing/Ground/Mixed, C=Contact, E=Entanglement, D=Distance, >=Dominant"
        return legend + "\n\n" + "\n".join(compressed)
    
    def _create_window_prompt(self, window: Dict, segments: List[Dict]) -> str:
        """Create prompt for re-analyzing a window."""
        prev_context = ""
        if window['start_segment_idx'] > 0:
            prev_seg = segments[window['start_segment_idx'] - 1]
            prev_context = f"Before: {prev_seg['start']}-{prev_seg['end']} was {prev_seg['position']}"
        
        next_context = ""
        if window['end_segment_idx'] < len(segments) - 1:
            next_seg = segments[window['end_segment_idx'] + 1]
            next_context = f"After: {next_seg['start']}-{next_seg['end']} is {next_seg['position']}"
        
        cv_data = self._compress_cv_data(window['start_time'], window['end_time'])
        duration_sec = self._time_to_seconds(window['end_time']) - self._time_to_seconds(window['start_time'])
        
        prompt = f"""⚠️ PASS 2 RE-ANALYSIS: Initial analysis was lazy/repetitive here.

Analyze {window['start_time']}-{window['end_time']} ({duration_sec/60:.1f} min) with FRESH ATTENTION.

CONTEXT:
{prev_context}
{next_context}

CV DATA:
{cv_data}

Re-analyze thoroughly. Use SAME format as Pass 1:

**SEGMENT [Start-End]**
Position: primary_position
Sub-position: sub_position
Top: Athlete name
Bottom: Athlete name
Confidence: 0.XX
Action: 0.XX
Transition: transition_type
Key Moments:
- timestamp: description
Key Actions: action1, action2
Narrative: 150-350 chars

Analyze COMPLETE window. Look for position changes and transitions."""
        
        return prompt
    
    def reanalyze_window(self, window: Dict, window_num: int) -> Dict:
        """Re-analyze a single window."""
        print(f"\n[Stage 2.{window_num}] Window {window_num}: {window['start_time']}-{window['end_time']}")
        
        if not GEMINI_AVAILABLE:
            return self._mock_output(window)
        
        start_time = time.time()
        
        # Upload video
        video_file = self.client.files.upload(file=self.video_path)
        print(f"  Video: {video_file.name}")
        
        while video_file.state.name == 'PROCESSING':
            time.sleep(2)
            video_file = self.client.files.get(name=video_file.name)
        
        # Create cache
        import hashlib
        cache_version = hashlib.md5(f"window{window_num}".encode()).hexdigest()[:8]
        
        cache = self.client.caches.create(
            model=self.model_name,
            config=types.CreateCachedContentConfig(
                display_name=f'bjj_w{window_num}_v{cache_version}',
                system_instruction="You are an expert BJJ analyzer.",
                contents=[video_file],
                ttl="3600s",
            )
        )
        
        # Generate
        segments = self.stage1_results.get('segments', [])
        prompt = self._create_window_prompt(window, segments)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                cached_content=cache.name,
                max_output_tokens=25000,
                temperature=0.4,
                top_p=0.95,
                top_k=40,
            )
        )
        
        # Parse
        stage1 = GeminiStage1Timeline(self.video_path, self.cv_data, self.output_dir, self.output_suffix)
        parsed = stage1._parse_response(response.text)
        
        usage = response.usage_metadata
        cost = self._calculate_cost(usage)
        elapsed = time.time() - start_time
        
        result = {
            'window': window,
            'segments': parsed.get('segments', []),
            'athlete_profiles': parsed.get('athlete_profiles', {}),
            'raw_markdown': response.text,
            'cost': cost,
            'time_seconds': round(elapsed, 2)
        }
        
        # Save
        timeline_path = self.output_dir / f"stage2_window{window_num}_timeline{self.output_suffix}.json"
        with open(timeline_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        markdown_path = self.output_dir / f"stage2_window{window_num}_markdown{self.output_suffix}.md"
        with open(markdown_path, 'w') as f:
            f.write(response.text)
        
        print(f"  Segments: {len(result['segments'])}, Cost: ${cost['total']:.4f}, Time: {elapsed:.1f}s")
        
        return result
    
    def analyze(self) -> Dict:
        """Main Stage 2 orchestrator."""
        print("\n" + "="*80)
        print("STAGE 2: Lazy Detection and Re-analysis")
        print("="*80)
        
        # Detect
        print("\n[Stage 2.1] Detecting lazy segments...")
        detection = self.detect_lazy_segments()
        
        print(f"  Total: {detection['total_segments']}, Flagged: {detection['flagged_count']}")
        
        if detection['attention_cliff']:
            cliff = detection['attention_cliff']
            print(f"  Attention cliff: {cliff['timestamp']} ({cliff['success_duration_seconds']/60:.1f} min success)")
        
        # Save detection
        detection_path = self.output_dir / f"stage2_lazy_detection{self.output_suffix}.json"
        with open(detection_path, 'w') as f:
            json.dump(detection, f, indent=2)
        
        if not detection['flagged_segments']:
            print("\n✓ No lazy segments!")
            return {'detection': detection, 'windows': [], 'total_cost': 0, 'total_time': 0}
        
        # Create windows
        print("\n[Stage 2.2] Creating windows...")
        windows = self._create_contextual_windows(detection)
        print(f"  Windows: {len(windows)}")
        for i, w in enumerate(windows, 1):
            print(f"    {i}: {w['start_time']}-{w['end_time']}")
        
        # Re-analyze
        results = []
        total_cost = 0
        total_time = 0
        
        for i, window in enumerate(windows, 1):
            result = self.reanalyze_window(window, i)
            results.append(result)
            total_cost += result['cost']['total']
            total_time += result['time_seconds']
        
        print("\n" + "="*80)
        print("STAGE 2 COMPLETE!")
        print(f"Windows: {len(windows)}, Cost: ${total_cost:.4f}, Time: {total_time:.1f}s")
        print("="*80)
        
        return {
            'detection': detection,
            'windows': results,
            'total_cost': total_cost,
            'total_time': total_time
        }
    
    def _calculate_cost(self, usage) -> Dict:
        """Calculate cost."""
        pricing = PRICING.get(self.model_name, PRICING['models/gemini-2.5-flash'])
        
        input_tokens = usage.prompt_token_count - usage.cached_content_token_count
        cached_tokens = usage.cached_content_token_count
        output_tokens = usage.candidates_token_count
        
        input_cost = input_tokens * pricing['input']
        cached_cost = cached_tokens * pricing['cached_input']
        output_cost = output_tokens * pricing['output']
        
        return {
            'input_tokens': input_tokens,
            'cached_tokens': cached_tokens,
            'output_tokens': output_tokens,
            'input_cost': round(input_cost, 6),
            'cached_cost': round(cached_cost, 6),
            'output_cost': round(output_cost, 6),
            'total': round(input_cost + cached_cost + output_cost, 6)
        }
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS to seconds."""
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def _mock_output(self, window: Dict) -> Dict:
        """Mock output."""
        return {
            'window': window,
            'segments': [],
            'athlete_profiles': {},
            'raw_markdown': 'Mock',
            'cost': {'total': 0.0},
            'time_seconds': 0.0
        }


class GeminiStage3Synthesis:
    """Stage 3: Merge Pass 1 and Pass 2."""
    
    def __init__(self, stage1_results: Dict, stage2_results: Dict, 
                 output_dir: Path, output_suffix: str = ""):
        self.stage1_results = stage1_results
        self.stage2_results = stage2_results
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.output_suffix = output_suffix
    
    def merge_timelines(self) -> Dict:
        """Merge Pass 1 and Pass 2."""
        print("\n[Stage 3.1] Merging...")
        
        stage1_segments = self.stage1_results.get('segments', [])
        detection = self.stage2_results.get('detection', {})
        windows = self.stage2_results.get('windows', [])
        
        if not windows:
            print("  No windows - using Stage 1")
            return {
                'segments': stage1_segments,
                'athlete_profiles': self.stage1_results.get('athlete_profiles', {}),
                'merge_info': {'segments_replaced': 0, 'segments_added': 0}
            }
        
        flagged_indices = set(detection.get('flagged_segments', []))
        
        # Keep non-flagged
        final_segments = [seg for i, seg in enumerate(stage1_segments) if i not in flagged_indices]
        replaced_count = len(flagged_indices)
        added_count = 0
        
        # Add window segments
        for window_result in windows:
            window_segments = window_result.get('segments', [])
            for ws in window_segments:
                final_segments.append(ws)
                added_count += 1
        
        # Sort by time
        final_segments.sort(key=lambda s: self._time_to_seconds(s['start']))
        
        # Renumber
        for i, seg in enumerate(final_segments):
            seg['segment_number'] = i + 1
        
        print(f"  Stage 1: {len(stage1_segments)}, Replaced: {replaced_count}, Added: {added_count}, Final: {len(final_segments)}")
        
        return {
            'segments': final_segments,
            'athlete_profiles': self.stage1_results.get('athlete_profiles', {}),
            'merge_info': {
                'stage1_segments': len(stage1_segments),
                'segments_replaced': replaced_count,
                'segments_added': added_count,
                'final_segments': len(final_segments)
            }
        }
    
    def _validate_continuity(self, segments: List[Dict]) -> Dict:
        """Check gaps/overlaps."""
        print("\n[Stage 3.2] Validating...")
        
        gaps = []
        overlaps = []
        
        for i in range(len(segments) - 1):
            curr_end = self._time_to_seconds(segments[i]['end'])
            next_start = self._time_to_seconds(segments[i+1]['start'])
            
            if next_start > curr_end + 1:
                gaps.append({
                    'after_segment': i,
                    'gap_duration': next_start - curr_end
                })
            elif next_start < curr_end - 1:
                overlaps.append({
                    'segment1': i,
                    'segment2': i+1,
                    'overlap_duration': curr_end - next_start
                })
        
        if gaps:
            print(f"  ⚠️ {len(gaps)} gaps")
        if overlaps:
            print(f"  ⚠️ {len(overlaps)} overlaps")
        if not gaps and not overlaps:
            print("  ✓ Continuous")
        
        return {
            'gaps': gaps,
            'overlaps': overlaps,
            'is_continuous': len(gaps) == 0 and len(overlaps) == 0
        }
    
    def analyze(self) -> Dict:
        """Main Stage 3."""
        print("\n" + "="*80)
        print("STAGE 3: Synthesis")
        print("="*80)
        
        merged = self.merge_timelines()
        validation = self._validate_continuity(merged['segments'])
        
        result = {
            **merged,
            'validation': validation
        }
        
        # Save
        timeline_path = self.output_dir / f"stage3_timeline{self.output_suffix}.json"
        with open(timeline_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print("\n" + "="*80)
        print("STAGE 3 COMPLETE!")
        print(f"Final segments: {len(result['segments'])}")
        print(f"Output: {timeline_path}")
        print("="*80)
        
        return result
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS to seconds."""
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
