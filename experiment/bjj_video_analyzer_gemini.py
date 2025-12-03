#!/usr/bin/env python3
"""
BJJ Video Analyzer - Gemini Version

Analyzes BJJ videos using Google's Gemini 2.0 Flash or Gemini 1.5 Pro models.
Leverages native video understanding for full temporal analysis.

Usage:
    python bjj_video_analyzer_gemini.py --video path/to/video.mp4
    python bjj_video_analyzer_gemini.py --video path/to/video.mp4 --model gemini-1.5-pro
"""

import os
import sys
import time
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed")
    print("Install with: pip install google-generativeai")
    sys.exit(1)


class GeminiBJJAnalyzer:
    """Analyzes BJJ videos using Gemini AI models."""

    MODELS = {
        "flash": "gemini-2.0-flash-exp",          # Recommended: Fast, cheap, good quality
        "pro": "gemini-1.5-pro",                   # Higher quality, more expensive
        "flash-legacy": "gemini-1.5-flash"         # Being phased out
    }

    def __init__(
        self,
        api_key: str,
        model: str = "flash",
        output_dir: str = "results/gemini",
        verbose: bool = True
    ):
        """
        Initialize the Gemini BJJ Analyzer.

        Args:
            api_key: Google AI API key
            model: Model to use ("flash", "pro", or full model name)
            output_dir: Directory to save results
            verbose: Print progress messages
        """
        self.api_key = api_key
        self.verbose = verbose

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Set model name
        if model in self.MODELS:
            self.model_name = self.MODELS[model]
        else:
            self.model_name = model  # Allow custom model names

        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load ontology and prompts
        self.ontology = self._load_ontology()
        self.prompts = self._load_prompts()

        if self.verbose:
            print(f"✓ Initialized Gemini BJJ Analyzer")
            print(f"  Model: {self.model_name}")
            print(f"  Output: {self.output_dir}")

    def _load_ontology(self) -> str:
        """Load BJJ ontology from markdown file."""
        ontology_path = Path("prompts/bjj_ontology.md")
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology not found: {ontology_path}")

        with open(ontology_path, 'r') as f:
            content = f.read()

        if self.verbose:
            print(f"✓ Loaded BJJ ontology ({len(content)} chars)")

        return content

    def _load_prompts(self) -> Dict[str, str]:
        """Load Gemini-specific prompts from markdown file."""
        prompts_path = Path("prompts/gemini_prompts.md")
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts not found: {prompts_path}")

        with open(prompts_path, 'r') as f:
            content = f.read()

        # Extract prompts from markdown code blocks
        prompts = {}
        current_section = None
        in_code_block = False
        current_prompt = []

        for line in content.split('\n'):
            if line.startswith('## ') and not line.startswith('## Example'):
                current_section = line[3:].strip()
            elif line.startswith('```') and current_section:
                if in_code_block:
                    # End of code block
                    prompts[current_section] = '\n'.join(current_prompt).strip()
                    current_prompt = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_prompt.append(line)

        if self.verbose:
            print(f"✓ Loaded {len(prompts)} prompt templates")

        return prompts

    def _upload_video(self, video_path: str) -> Any:
        """
        Upload video to Gemini.

        Args:
            video_path: Path to video file

        Returns:
            Uploaded file object
        """
        video_path = Path(video_path)

        # Check if file exists, try adding .mp4 extension if not
        if not video_path.exists():
            video_path_with_ext = Path(str(video_path) + ".mp4")
            if video_path_with_ext.exists():
                video_path = video_path_with_ext
            else:
                raise FileNotFoundError(f"Video not found: {video_path}")

        if self.verbose:
            print(f"⏳ Uploading video: {video_path.name}")
            print(f"   Size: {video_path.stat().st_size / (1024*1024):.1f} MB")

        # Upload file
        video_file = genai.upload_file(path=str(video_path))

        if self.verbose:
            print(f"✓ Video uploaded: {video_file.name}")

        # Wait for processing
        if self.verbose:
            print(f"⏳ Processing video (this may take 30-60 seconds)...")

        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise Exception(f"Video processing failed: {video_file.state}")

        if self.verbose:
            print(f"✓ Video ready for analysis")

        return video_file

    def _build_prompt(self, prompt_type: str = "Long Video Prompt (10+ minutes)") -> str:
        """
        Build complete prompt including ontology and analysis instructions.

        Args:
            prompt_type: Type of prompt to use

        Returns:
            Complete prompt string
        """
        # Get base prompt
        analysis_prompt = self.prompts.get(prompt_type, self.prompts["Main Analysis Prompt"])

        # Combine with ontology (provide context)
        full_prompt = f"""You are analyzing a Brazilian Jiu-Jitsu video.

I'm providing you with a complete BJJ knowledge ontology that defines positions, transitions, and scoring rules.

{self.ontology}

---

Now, using this ontology as your reference:

{analysis_prompt}
"""

        return full_prompt

    def analyze_video(
        self,
        video_path: str,
        prompt_type: str = "Long Video Prompt (10+ minutes)",
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze a BJJ video with Gemini.

        Args:
            video_path: Path to video file
            prompt_type: Type of analysis to perform
            max_retries: Number of retries on failure

        Returns:
            Analysis results dict
        """
        video_path = Path(video_path)
        video_name = video_path.stem

        # Create run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"{timestamp}_{video_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Analyzing: {video_path.name}")
            print(f"Output: {run_dir}")
            print(f"{'='*60}\n")

        # Upload video
        video_file = self._upload_video(video_path)

        # Build prompt
        prompt = self._build_prompt(prompt_type)

        # Save prompt for reproducibility
        with open(run_dir / "prompt_used.txt", 'w') as f:
            f.write(prompt)

        # Analyze with retries
        model = genai.GenerativeModel(self.model_name)

        for attempt in range(max_retries):
            try:
                if self.verbose:
                    print(f"⏳ Analyzing video (attempt {attempt + 1}/{max_retries})...")
                    print(f"   Model: {self.model_name}")
                    start_time = time.time()

                # Generate analysis
                response = model.generate_content(
                    [video_file, prompt],
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,  # Low temperature for consistent analysis
                        max_output_tokens=8192,  # Allow long outputs
                    )
                )

                elapsed = time.time() - start_time

                if self.verbose:
                    print(f"✓ Analysis complete ({elapsed:.1f} seconds)")

                # Parse response
                response_text = response.text

                # Save raw response
                with open(run_dir / "raw_response.txt", 'w') as f:
                    f.write(response_text)

                # Try to extract JSON
                analysis = self._extract_json(response_text)

                # Save structured analysis
                with open(run_dir / "analysis.json", 'w') as f:
                    json.dump(analysis, f, indent=2)

                # Generate markdown report
                self._generate_markdown_report(analysis, run_dir, video_name)

                # Save metadata
                metadata = {
                    "video_path": str(video_path),
                    "video_name": video_name,
                    "model": self.model_name,
                    "prompt_type": prompt_type,
                    "timestamp": timestamp,
                    "processing_time_seconds": elapsed,
                    "success": True
                }

                with open(run_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                if self.verbose:
                    print(f"\n✓ Results saved to: {run_dir}")
                    print(f"  - analysis.json (structured data)")
                    print(f"  - analysis_report.md (markdown report)")
                    print(f"  - raw_response.txt (full model output)")

                return analysis

            except Exception as e:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"⚠ Error: {e}")
                        print(f"  Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    # Final attempt failed
                    if self.verbose:
                        print(f"✗ Analysis failed after {max_retries} attempts")
                        print(f"  Error: {e}")

                    # Save error info
                    with open(run_dir / "error.txt", 'w') as f:
                        f.write(f"Error: {e}\n")
                        f.write(f"Response: {response_text if 'response_text' in locals() else 'N/A'}\n")

                    raise

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model response (may be wrapped in markdown code blocks)."""
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Parse JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return structured error
            return {
                "error": "Failed to parse JSON from model response",
                "error_details": str(e),
                "raw_text": text
            }

    def _generate_markdown_report(self, analysis: Dict[str, Any], run_dir: Path, video_name: str):
        """Generate human-readable markdown report from analysis."""
        report_path = run_dir / "analysis_report.md"

        with open(report_path, 'w') as f:
            f.write(f"# BJJ Video Analysis Report\n\n")
            f.write(f"**Video**: {video_name}\n")
            f.write(f"**Model**: {self.model_name}\n")
            f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")

            # Video metadata
            if "video_metadata" in analysis:
                f.write(f"## Video Metadata\n\n")
                meta = analysis["video_metadata"]
                f.write(f"- **Duration**: {meta.get('duration', 'Unknown')}\n")
                f.write(f"- **Athletes**: {meta.get('detected_athletes', 'Unknown')}\n")
                if "athlete_descriptions" in meta:
                    f.write(f"- **Athlete 1**: {meta['athlete_descriptions'].get('athlete_1', 'N/A')}\n")
                    f.write(f"- **Athlete 2**: {meta['athlete_descriptions'].get('athlete_2', 'N/A')}\n")
                f.write(f"\n")

            # Match summary
            if "match_summary" in analysis:
                f.write(f"## Match Summary\n\n")
                summary = analysis["match_summary"]
                f.write(f"{summary.get('overall_assessment', 'N/A')}\n\n")
                f.write(f"- **Dominant Athlete**: {summary.get('dominant_athlete', 'N/A')}\n")
                f.write(f"- **Match Pace**: {summary.get('match_pace', 'N/A')}\n\n")

            # Scoring
            if "scoring_adcc" in analysis:
                f.write(f"## ADCC Scoring\n\n")
                scoring = analysis["scoring_adcc"]

                f.write(f"### Athlete 1: {scoring['athlete_1']['total_points']} points\n")
                self._write_score_breakdown(f, scoring['athlete_1'])

                f.write(f"\n### Athlete 2: {scoring['athlete_2']['total_points']} points\n")
                self._write_score_breakdown(f, scoring['athlete_2'])

                f.write(f"\n**Winner**: {scoring.get('winner', 'Unknown')} ")
                f.write(f"(+{scoring.get('winning_margin', 0)} points)\n\n")

            # Position timeline
            if "position_timeline" in analysis:
                f.write(f"## Position Timeline\n\n")
                for pos in analysis["position_timeline"]:
                    f.write(f"- **{pos['start_time']} - {pos['end_time']}**: {pos['position']}")
                    if pos.get('sub_position'):
                        f.write(f" ({pos['sub_position']})")
                    f.write(f" - Top: {pos['top_athlete']}, Control: {pos.get('control_quality', 'N/A')}/5\n")
                f.write(f"\n")

            # Key moments
            if "key_moments" in analysis:
                f.write(f"## Key Moments\n\n")
                for moment in analysis["key_moments"]:
                    f.write(f"### {moment['timestamp']}\n")
                    f.write(f"{moment['description']}\n\n")
                    f.write(f"*Significance*: {moment['significance']}\n\n")

            # Coaching insights
            if "coaching_insights" in analysis:
                f.write(f"## Coaching Insights\n\n")
                insights = analysis["coaching_insights"]

                if "athlete_1" in insights and insights["athlete_1"]:
                    f.write(f"### Athlete 1\n\n")
                    for insight in insights["athlete_1"]:
                        f.write(f"- {insight}\n")
                    f.write(f"\n")

                if "athlete_2" in insights and insights["athlete_2"]:
                    f.write(f"### Athlete 2\n\n")
                    for insight in insights["athlete_2"]:
                        f.write(f"- {insight}\n")
                    f.write(f"\n")

                if "general" in insights and insights["general"]:
                    f.write(f"### General Observations\n\n")
                    for insight in insights["general"]:
                        f.write(f"- {insight}\n")
                    f.write(f"\n")

    def _write_score_breakdown(self, f, athlete_scoring: Dict[str, Any]):
        """Write score breakdown for one athlete."""
        categories = [
            ("Takedowns", "takedowns", 2),
            ("Guard Passes", "guard_passes", 3),
            ("Knee on Belly", "knee_on_belly", 2),
            ("Mounts", "mounts", 4),
            ("Back Controls", "back_controls", 4)
        ]

        for label, key, points_each in categories:
            count = athlete_scoring.get(key, 0)
            if count > 0:
                f.write(f"- {label}: {count} × {points_each} pts = {count * points_each} pts\n")


def main():
    """Main entry point for Gemini BJJ video analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze BJJ videos using Google Gemini AI"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--model",
        default="flash",
        choices=["flash", "pro", "flash-legacy"],
        help="Gemini model to use (default: flash = gemini-2.0-flash-exp)"
    )
    parser.add_argument(
        "--prompt-type",
        default="Long Video Prompt (10+ minutes)",
        help="Type of analysis prompt (default: Long Video Prompt)"
    )
    parser.add_argument(
        "--output",
        default="results/gemini",
        help="Output directory (default: results/gemini)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google AI API key (or set GOOGLE_API_KEY env var)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: API key required")
        print("  Set GOOGLE_API_KEY environment variable, or")
        print("  Pass --api-key argument")
        sys.exit(1)

    # Create analyzer
    analyzer = GeminiBJJAnalyzer(
        api_key=api_key,
        model=args.model,
        output_dir=args.output,
        verbose=not args.quiet
    )

    # Analyze video
    try:
        analysis = analyzer.analyze_video(
            video_path=args.video,
            prompt_type=args.prompt_type
        )

        print("\n✓ Analysis complete!")

    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
