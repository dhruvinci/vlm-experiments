#!/usr/bin/env python3
"""
BJJ Video Analyzer - Experiment 2: Multi-Pass Adaptive Analysis

4-pass architecture:
- Pass 0: Action detection (OpenCV optical flow)
- Pass 1: Holistic context (Gemini 2.0 Flash)
- Pass 2: Adaptive detail (Gemini 2.0 Flash)
- Pass 3: Synthesis (Gemini 1.5 Pro)

Usage:
    python bjj_video_analyzer_exp2.py --video path/to/video.mp4
"""

import os
import sys
import time
import json
import argparse
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed")
    print("Install with: pip install opencv-python")
    sys.exit(1)

try:
    import google.generativeai as genai
    from google.generativeai import caching
except ImportError:
    print("Error: google-generativeai not installed")
    print("Install with: pip install google-generativeai")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple progress bar fallback
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n=1):
            self.n += n
            if self.total:
                pct = (self.n / self.total) * 100
                print(f"\r{self.desc}: {self.n}/{self.total} ({pct:.1f}%)", end='', flush=True)

        def close(self):
            print()  # New line

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()


class ActionDetector:
    """Pass 0: Detect action vs stalling periods using optical flow."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video to detect action and stalling segments.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with action_segments and stalling_segments
        """
        if self.verbose:
            print("\n=== PASS 0: ACTION DETECTION ===")
            print(f"Analyzing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps

        if self.verbose:
            print(f"  FPS: {fps:.2f}")
            print(f"  Total frames: {total_frames}")
            print(f"  Duration: {duration_seconds:.1f}s")

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Motion intensity per second
        motion_per_second = []
        frame_idx = 0
        current_second = 0
        motion_sum = 0
        frame_count_in_second = 0

        # Progress bar
        with tqdm(total=total_frames, desc="  Optical flow", unit="frame", disable=not self.verbose) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    flow=None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )

                # Calculate magnitude of motion
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                mean_motion = np.mean(magnitude)

                # Accumulate for current second
                motion_sum += mean_motion
                frame_count_in_second += 1

                # Check if we've completed a second
                second = int(frame_idx / fps)
                if second > current_second:
                    # Store average motion for this second
                    avg_motion = motion_sum / frame_count_in_second if frame_count_in_second > 0 else 0
                    motion_per_second.append(avg_motion)

                    # Reset for next second
                    current_second = second
                    motion_sum = 0
                    frame_count_in_second = 0

                prev_gray = gray
                pbar.update(1)

        # Store last incomplete second if any
        if frame_count_in_second > 0:
            avg_motion = motion_sum / frame_count_in_second
            motion_per_second.append(avg_motion)

        cap.release()

        # Classify segments based on motion intensity
        motion_array = np.array(motion_per_second)
        threshold = np.percentile(motion_array, 60)  # Top 40% = action

        if self.verbose:
            print(f"  Motion threshold: {threshold:.2f}")
            print(f"  Mean motion: {np.mean(motion_array):.2f}")
            print(f"  Max motion: {np.max(motion_array):.2f}")

        # Create segments
        action_segments = []
        stalling_segments = []

        current_type = None
        current_start = 0

        for second, motion in enumerate(motion_per_second):
            is_action = motion >= threshold
            segment_type = "action" if is_action else "stalling"

            if segment_type != current_type:
                # Save previous segment
                if current_type is not None:
                    segment = {
                        "start_time": self._seconds_to_mmss(current_start),
                        "end_time": self._seconds_to_mmss(second),
                        "duration": second - current_start
                    }

                    if current_type == "action":
                        segment["intensity"] = float(np.mean(motion_array[current_start:second]))
                        action_segments.append(segment)
                    else:
                        stalling_segments.append(segment)

                # Start new segment
                current_type = segment_type
                current_start = second

        # Save final segment
        if current_type is not None:
            segment = {
                "start_time": self._seconds_to_mmss(current_start),
                "end_time": self._seconds_to_mmss(len(motion_per_second)),
                "duration": len(motion_per_second) - current_start
            }

            if current_type == "action":
                segment["intensity"] = float(np.mean(motion_array[current_start:]))
                action_segments.append(segment)
            else:
                stalling_segments.append(segment)

        result = {
            "action_segments": action_segments,
            "stalling_segments": stalling_segments,
            "motion_per_second": [float(m) for m in motion_per_second],
            "threshold": float(threshold),
            "video_metadata": {
                "fps": fps,
                "total_frames": int(total_frames),
                "duration_seconds": float(duration_seconds)
            }
        }

        if self.verbose:
            print(f"  ✓ Detected {len(action_segments)} action segments")
            print(f"  ✓ Detected {len(stalling_segments)} stalling segments")
            total_action = sum(s['duration'] for s in action_segments)
            total_stalling = sum(s['duration'] for s in stalling_segments)
            print(f"  Action: {total_action}s ({total_action/duration_seconds*100:.1f}%)")
            print(f"  Stalling: {total_stalling}s ({total_stalling/duration_seconds*100:.1f}%)")

        return result

    def _seconds_to_mmss(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


class GeminiMultiPassAnalyzer:
    """Passes 1-3: Multi-pass Gemini analysis."""

    def __init__(self, api_key: str, verbose: bool = True):
        self.api_key = api_key
        self.verbose = verbose
        genai.configure(api_key=api_key)

        # Load ontologies (macro and micro) and prompts
        self.ontology_macro = self._load_ontology("prompts/bjj_ontology_macro.md")
        self.ontology_micro = self._load_ontology("prompts/bjj_ontology_micro.md")
        self.prompts = self._load_prompts()

    def _load_ontology(self, path: str) -> str:
        """Load BJJ ontology."""
        ontology_path = Path(path)
        with open(ontology_path, 'r') as f:
            return f.read()

    def _load_prompts(self) -> Dict[str, str]:
        """Load Exp2-specific prompts."""
        prompts_path = Path("prompts/exp2_prompts_optimized.md")
        with open(prompts_path, 'r') as f:
            content = f.read()

        # Parse prompts from markdown
        prompts = {}
        current_key = None
        current_content = []

        for line in content.split('\n'):
            if line.startswith('## Pass '):
                if current_key:
                    prompts[current_key] = '\n'.join(current_content).strip()
                current_key = line.replace('##', '').strip().replace(':', '')
                current_content = []
            elif current_key:
                current_content.append(line)

        if current_key:
            prompts[current_key] = '\n'.join(current_content).strip()

        return prompts

    def _upload_video(self, video_path: str):
        """Upload video to Gemini."""
        if self.verbose:
            print(f"  📤 Uploading video: {video_path}")

        start = time.time()
        video_file = genai.upload_file(path=str(video_path))

        if self.verbose:
            elapsed = time.time() - start
            print(f"  ✓ Uploaded ({elapsed:.1f}s): {video_file.name}")
            print(f"  ⏳ Gemini processing video...", end='', flush=True)

        # Wait for processing with progress dots
        processing_start = time.time()
        dots = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            if self.verbose:
                dots += 1
                print('.', end='', flush=True)
                if dots % 15 == 0:  # New line every 30 seconds
                    elapsed = time.time() - processing_start
                    print(f" ({elapsed:.0f}s)", end='', flush=True)

        if video_file.state.name == "FAILED":
            raise Exception(f"Video processing failed: {video_file.state}")

        if self.verbose:
            total_elapsed = time.time() - start
            print(f" ✓ Ready ({total_elapsed:.1f}s total)")

        return video_file

    def _parse_pass1_markdown(self, text: str) -> Dict[str, Any]:
        """Parse Pass 1 markdown output into structured data."""
        import re

        # Extract athlete IDs
        athlete_pattern = r'\*\*Athlete IDs?:\*\*\s*(.*?)(?=\||\Z)'
        athlete_match = re.search(athlete_pattern, text, re.DOTALL | re.IGNORECASE)

        athlete_1_desc = ""
        athlete_2_desc = ""

        if athlete_match:
            athlete_text = athlete_match.group(1)
            # Parse athlete descriptions
            athlete_1_match = re.search(r'-\s*Athlete 1:\s*(.+?)(?=-\s*Athlete 2|$)', athlete_text, re.DOTALL)
            athlete_2_match = re.search(r'-\s*Athlete 2:\s*(.+?)(?=$)', athlete_text, re.DOTALL)

            if athlete_1_match:
                athlete_1_desc = athlete_1_match.group(1).strip()
            if athlete_2_match:
                athlete_2_desc = athlete_2_match.group(1).strip()

        # Parse markdown table
        positions = []
        table_lines = []
        in_table = False

        for line in text.split('\n'):
            if re.match(r'^\|.*\|$', line.strip()):
                if '---' in line:
                    in_table = True
                    continue
                if in_table:
                    table_lines.append(line.strip())

        # Parse table rows
        for line in table_lines:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
            if len(cells) >= 8:
                time_range = cells[0]
                # Parse time range (M:SS-M:SS)
                time_match = re.match(r'(\d+:\d+)-(\d+:\d+)', time_range)
                if time_match:
                    positions.append({
                        'start_time': time_match.group(1),
                        'end_time': time_match.group(2),
                        'position': cells[1],
                        'sub_position': cells[2] if cells[2] not in ['-', ''] else None,
                        'top_athlete': cells[3],
                        'bottom_athlete': cells[4],
                        'control_quality': int(cells[5]) if cells[5].isdigit() else 3,
                        'confidence': float(cells[6]) if cells[6].replace('.', '').isdigit() else 1.0,
                        'notes': cells[7]
                    })

        return {
            'raw_text': text,
            'athlete_1_description': athlete_1_desc,
            'athlete_2_description': athlete_2_desc,
            'position_timeline': positions
        }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from response text."""
        # Try to find JSON in code blocks
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Return text as-is
                return {"raw_text": text}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  ⚠ JSON parse error: {e}")
            return {"raw_text": text, "parse_error": str(e)}

    def _analyze_window(self, cached_model, window_start: str, window_end: str, window_num: int, total_windows: int) -> str:
        """Analyze a single time window and return raw markdown text."""
        prompt_template = self.prompts.get("Pass 1 Holistic Context (Markdown Output)", "")

        # Add time window context to prompt
        windowed_prompt = f"""Analyze ONLY the time window from {window_start} to {window_end}.

{prompt_template}"""

        if self.verbose:
            print(f"  ⏳ Window {window_num}/{total_windows} ({window_start}-{window_end})... (target: 43K tokens)")

        response = cached_model.generate_content(
            windowed_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=43253,  # 66% of 65536 (43K tokens)
            )
        )

        # Check response validity
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            try:
                response_text = response.text
                tokens_est = len(response_text) * 0.75
                if self.verbose:
                    print(f"  ✓ Window {window_num}/{total_windows} complete (~{tokens_est:.0f} tokens)")
                return response_text
            except ValueError:
                return ""
        return ""

    def _deduplicate_positions(self, all_positions: list, overlap_seconds: int = 15) -> list:
        """Deduplicate positions in overlap regions."""
        if not all_positions:
            return []

        # Sort by start time
        sorted_pos = sorted(all_positions, key=lambda p: self._time_to_seconds(p['start_time']))

        deduplicated = []
        skip_indices = set()

        for i, pos in enumerate(sorted_pos):
            if i in skip_indices:
                continue

            # Check if next position overlaps with this one
            if i + 1 < len(sorted_pos):
                next_pos = sorted_pos[i + 1]

                pos_start = self._time_to_seconds(pos['start_time'])
                pos_end = self._time_to_seconds(pos['end_time'])
                next_start = self._time_to_seconds(next_pos['start_time'])
                next_end = self._time_to_seconds(next_pos['end_time'])

                # Check for overlap (within 5 second tolerance)
                if abs(pos_start - next_start) <= 5 and abs(pos_end - next_end) <= 5:
                    # Duplicate detected - keep the one with higher confidence
                    if pos.get('confidence', 0) >= next_pos.get('confidence', 0):
                        deduplicated.append(pos)
                    else:
                        deduplicated.append(next_pos)
                    skip_indices.add(i + 1)
                else:
                    deduplicated.append(pos)
            else:
                deduplicated.append(pos)

        return deduplicated

    def _time_to_seconds(self, time_str: str) -> int:
        """Convert M:SS or MM:SS to seconds."""
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0

    def pass_1_holistic(self, video_path: str, video_duration_seconds: int = 960) -> Dict[str, Any]:
        """
        Pass 1: Multi-window holistic context analysis.

        Breaks video into 2:30 windows with 15s overlap.
        Uses caching to save tokens across windows.
        """
        if self.verbose:
            print("\n=== PASS 1: HOLISTIC CONTEXT (MULTI-WINDOW) ===")
            print(f"  Model: gemini-2.5-flash (with caching)")

        start_time = time.time()

        # Upload video
        video_file = self._upload_video(video_path)

        # Create cache with video + macro ontology
        cache = caching.CachedContent.create(
            model='models/gemini-2.5-flash',
            system_instruction=self.ontology_macro,
            contents=[video_file],
            ttl=datetime.timedelta(hours=1)
        )

        # Define 6:00 windows with 15s overlap
        window_duration = 360  # 6:00 in seconds
        overlap = 15  # 15 seconds
        windows = []

        current_start = 0
        while current_start < video_duration_seconds:
            window_end = min(current_start + window_duration, video_duration_seconds)
            start_str = f"{current_start // 60}:{current_start % 60:02d}"
            end_str = f"{window_end // 60}:{window_end % 60:02d}"
            windows.append((start_str, end_str))

            # Move to next window with overlap
            current_start += window_duration - overlap
            if current_start >= video_duration_seconds:
                break

        if self.verbose:
            print(f"  📊 Processing {len(windows)} windows (6:00 each, 15s overlap)")

        # Analyze each window
        cached_model = genai.GenerativeModel.from_cached_content(cache)
        all_positions = []
        all_raw_texts = []
        athlete_1_desc = ""
        athlete_2_desc = ""

        for idx, (window_start, window_end) in enumerate(windows, 1):
            # Show progress bar
            progress = "=" * idx + ">" + "." * (len(windows) - idx)
            if self.verbose:
                print(f"\n  [{progress}] Window {idx}/{len(windows)}")

            response_text = self._analyze_window(cached_model, window_start, window_end, idx, len(windows))

            if response_text:
                all_raw_texts.append(f"=== WINDOW {idx}: {window_start}-{window_end} ===\n{response_text}\n")

                # Parse this window's positions
                window_data = self._parse_pass1_markdown(response_text)
                all_positions.extend(window_data.get('position_timeline', []))

                # Extract athlete IDs from first window
                if idx == 1:
                    athlete_1_desc = window_data.get('athlete_1_description', '')
                    athlete_2_desc = window_data.get('athlete_2_description', '')

        # Deduplicate overlapping positions
        if self.verbose:
            print(f"  🔄 Deduplicating {len(all_positions)} positions from overlaps...")

        deduplicated_positions = self._deduplicate_positions(all_positions)

        if self.verbose:
            print(f"  ✓ Final timeline: {len(deduplicated_positions)} unique positions")

        # Delete cache
        cache.delete()

        elapsed = time.time() - start_time
        tokens_est = sum(len(text) * 0.75 for text in all_raw_texts)

        if self.verbose:
            print(f"  ✓ All windows complete ({elapsed:.1f}s, ~{tokens_est:.0f} total tokens)")

        result = {
            'raw_text': '\n'.join(all_raw_texts),
            'athlete_1_description': athlete_1_desc,
            'athlete_2_description': athlete_2_desc,
            'position_timeline': deduplicated_positions,
            '_metadata': {
                'model': 'gemini-2.5-flash',
                'processing_time': elapsed,
                'windows_processed': len(windows),
                'positions_before_dedup': len(all_positions),
                'positions_after_dedup': len(deduplicated_positions)
            }
        }

        return result

    def pass_2_adaptive_detail(
        self,
        video_path: str,
        pass1_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pass 2: Detailed analysis of EVERY position from Pass 1.

        Analyzes all positions with 3-5 sub-segments each.
        """
        positions = pass1_output.get('position_timeline', [])

        if self.verbose:
            print("\n=== PASS 2: DETAILED POSITION ANALYSIS ===")
            print(f"  Model: gemini-2.5-flash (with context caching)")
            print(f"  Positions to analyze: {len(positions)}")

        # Upload video once
        video_file = self._upload_video(video_path)

        # Create context cache with video + ontology
        # This caches ~85K tokens (video + system instructions)
        # Reduces per-request tokens from ~100K to ~15K (85% reduction!)
        if self.verbose:
            print(f"  🗂️  Creating context cache (video + ontology)...")

        cache = caching.CachedContent.create(
            model='models/gemini-2.5-flash',
            system_instruction=self.ontology_micro,  # BJJ micro ontology cached
            contents=[video_file],  # Video cached
            ttl=datetime.timedelta(hours=1)
        )

        if self.verbose:
            print(f"  ✓ Cache created (TTL: 1 hour)")

        # Use cached model for all positions
        model = genai.GenerativeModel.from_cached_content(cache)

        # Analyze each position
        all_sub_segments = []
        segment_analyses = []

        # Process ALL positions from Pass 1
        positions_to_process = positions

        # Process positions with progress bar
        with tqdm(positions_to_process, desc="  Analyzing positions", disable=not self.verbose) as pbar:
            for position in pbar:
                position_label = position.get('position', 'unknown')
                start_time_str = position.get('start_time', '0:00')
                end_time_str = position.get('end_time', '0:00')

                pbar.set_postfix_str(f"{position_label} {start_time_str}-{end_time_str}")

                # Calculate duration (simple MM:SS parsing)
                def time_to_seconds(t):
                    parts = str(t).split(':')
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + int(parts[1])
                    return 0

                duration = time_to_seconds(end_time_str) - time_to_seconds(start_time_str)

                # Build prompt with context
                prompt_template = self.prompts.get("Pass 2 Detailed Position Analysis (Plain Text)", "")

                # Create simplified Pass 1 summary for context
                pass1_summary = {
                    "match_summary": pass1_output.get('match_summary', {}),
                    "athlete_profiles": pass1_output.get('athlete_profiles', {})
                }

                # Use replace instead of format to avoid conflicts with any remaining braces
                full_prompt = (prompt_template
                    .replace("{position_start}", start_time_str)
                    .replace("{position_end}", end_time_str)
                    .replace("{pass1_position}", position_label)
                    .replace("{pass1_notes}", position.get('notes', 'No notes from Pass 1'))
                    .replace("{athlete_1_desc}", pass1_output.get('athlete_1_description', 'Unknown'))
                    .replace("{athlete_2_desc}", pass1_output.get('athlete_2_description', 'Unknown'))
                )

                # NOTE: Video and ontology are in the cache, so only send position-specific prompt
                # This reduces tokens from ~100K to ~15K per request!

                try:
                    api_start_time = time.time()
                    response = model.generate_content(
                        full_prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=4096,
                        )
                    )
                    elapsed = time.time() - api_start_time

                    result = self._extract_json(response.text)
                    result["_position_metadata"] = {
                        "position": position_label,
                        "processing_time": elapsed
                    }
                    segment_analyses.append(result)

                    # Extract sub-segments
                    if "sub_segments" in result:
                        all_sub_segments.extend(result["sub_segments"])

                except Exception as e:
                    if self.verbose:
                        pbar.write(f"      ✗ Error on {position_label}: {e}")

        return {
            "position_analyses": segment_analyses,
            "all_sub_segments": all_sub_segments,
            "positions_processed": len(positions_to_process)
        }

    def pass_3_synthesis(
        self,
        pass1_output: Dict[str, Any],
        pass2_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pass 3: Synthesis and meta-analysis with Gemini 1.5 Pro.

        Resolves conflicts, creates unified timeline, deep profiling.
        """
        if self.verbose:
            print("\n=== PASS 3: SYNTHESIS ===")
            print(f"  Model: gemini-2.5-flash")

        start_time = time.time()

        # Build prompt with all context
        prompt_template = self.prompts.get("Pass 3 Synthesis & JSON Formatting", "")

        pass1_str = json.dumps(pass1_output, indent=2)
        pass2_str = json.dumps(pass2_output, indent=2)

        # Use replace instead of format to avoid conflicts with JSON braces
        full_prompt = prompt_template.replace("{pass1_output}", pass1_str).replace("{pass2_outputs}", pass2_str)

        full_prompt = f"""{self.ontology_macro}

---

{full_prompt}
"""

        # Use Gemini 2.5 Flash for synthesis
        model = genai.GenerativeModel("gemini-2.5-flash")

        if self.verbose:
            print(f"  ⏳ Synthesizing...")

        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,  # Slightly higher for creative synthesis
                max_output_tokens=65536,  # Maximum for gemini-2.5-flash
            )
        )

        elapsed = time.time() - start_time
        tokens_est = len(response.text) * 0.75 if response.text else 0

        if self.verbose:
            print(f"  ✓ Complete ({elapsed:.1f}s, ~{tokens_est:.0f} tokens)")

        result = self._extract_json(response.text)
        result["_metadata"] = {
            "model": "gemini-2.5-flash",
            "processing_time": elapsed
        }

        return result


def main():
    parser = argparse.ArgumentParser(description="BJJ Video Analyzer - Experiment 2")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="results/exp2", help="Output directory")
    parser.add_argument("--api-key", help="Google AI API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--skip-pass0", action="store_true", help="Skip action detection")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Google AI API key required")
        print("Set GOOGLE_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(args.video).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"BJJ Video Analyzer - Experiment 2")
    print(f"{'='*60}")
    print(f"Video: {args.video}")
    print(f"Output: {output_dir}")

    start_time = time.time()

    # Pass 0: Action Detection (with checkpoint)
    pass0_checkpoint = output_dir / "pass0_action_detection.json"
    if pass0_checkpoint.exists():
        print("\n=== PASS 0: LOADING FROM CHECKPOINT ===")
        with open(pass0_checkpoint, 'r') as f:
            pass0_output = json.load(f)
        print(f"  ✓ Loaded {len(pass0_output.get('action_segments', []))} action segments")
        print(f"  ✓ Loaded {len(pass0_output.get('stalling_segments', []))} stalling segments")
    elif not args.skip_pass0:
        detector = ActionDetector(verbose=True)
        pass0_output = detector.analyze_video(args.video)

        # Save Pass 0 results
        with open(pass0_checkpoint, 'w') as f:
            json.dump(pass0_output, f, indent=2)
    else:
        print("\n=== PASS 0: SKIPPED ===")
        pass0_output = {"action_segments": [], "stalling_segments": []}

    # Pass 1: Holistic Context (with checkpoint)
    pass1_checkpoint = output_dir / "pass1_holistic.json"
    if pass1_checkpoint.exists():
        print("\n=== PASS 1: LOADING FROM CHECKPOINT ===")
        with open(pass1_checkpoint, 'r') as f:
            pass1_output = json.load(f)
        print(f"  ✓ Loaded holistic context")
    else:
        analyzer = GeminiMultiPassAnalyzer(api_key=api_key, verbose=True)
        pass1_output = analyzer.pass_1_holistic(args.video)

        with open(pass1_checkpoint, 'w') as f:
            json.dump(pass1_output, f, indent=2)

        # Save raw markdown for inspection
        pass1_markdown_path = output_dir / "pass1_markdown.md"
        with open(pass1_markdown_path, 'w') as f:
            f.write(pass1_output.get('raw_text', ''))

    # Pass 2: Detailed Position Analysis (with checkpoint)
    pass2_checkpoint = output_dir / "pass2_detail.json"
    if pass2_checkpoint.exists():
        print("\n=== PASS 2: LOADING FROM CHECKPOINT ===")
        with open(pass2_checkpoint, 'r') as f:
            pass2_output = json.load(f)
        print(f"  ✓ Loaded detailed analysis")
    else:
        if 'analyzer' not in locals():
            analyzer = GeminiMultiPassAnalyzer(api_key=api_key, verbose=True)
        pass2_output = analyzer.pass_2_adaptive_detail(
            args.video,
            pass1_output
        )

        with open(pass2_checkpoint, 'w') as f:
            json.dump(pass2_output, f, indent=2)

    # Pass 3: Synthesis (always run - combines all previous passes)
    if 'analyzer' not in locals():
        analyzer = GeminiMultiPassAnalyzer(api_key=api_key, verbose=True)
    pass3_output = analyzer.pass_3_synthesis(pass1_output, pass2_output)

    with open(output_dir / "pass3_synthesis.json", 'w') as f:
        json.dump(pass3_output, f, indent=2)

    # Save final unified output
    final_output = {
        "experiment": "exp2_multipass",
        "video_path": args.video,
        "timestamp": timestamp,
        "pass0_action_detection": pass0_output,
        "pass1_holistic": pass1_output,
        "pass2_detail": pass2_output,
        "pass3_synthesis": pass3_output
    }

    with open(output_dir / "analysis.json", 'w') as f:
        json.dump(final_output, f, indent=2)

    # Save metadata
    elapsed = time.time() - start_time
    metadata = {
        "experiment_name": "Experiment 2 - Multi-Pass Adaptive",
        "video_path": args.video,
        "video_name": video_name,
        "timestamp": timestamp,
        "processing_time_seconds": elapsed,
        "passes_completed": ["pass0", "pass1", "pass2", "pass3"],
        "success": True
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
