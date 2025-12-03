#!/usr/bin/env python3
"""
Experiment 3: Adaptive CV + Cached Gemini Architecture
BJJ Video Analysis Pipeline

Usage:
    python3 experiment/bjj_analyzer_exp3.py --video data/videos/youtube_SMRbZEbxepA.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import stages
from experiment.experiment3_stage0 import run_cv_preprocessing
from experiment.experiment3_stages123 import (
    GeminiStage1Timeline,
    GeminiStage2Detail,
    GeminiStage3Synthesis
)


class Experiment3Pipeline:
    """Main pipeline for Experiment 3."""
    
    def __init__(self, video_path: str, output_dir: str = None):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = self.video_path.stem
        self.output_dir = Path(output_dir or "outputs/experiment3") / f"{timestamp}_{video_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 3: Adaptive CV + Cached Gemini")
        print(f"{'='*80}")
        print(f"Video: {self.video_path.name}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")
        
        # Initialize tracking
        self.start_time = time.time()
        self.stage_times = {}
        self.stage_costs = {}
        self.total_cost = 0.0
    
    def run_stage0_cv(self) -> dict:
        """Stage 0: CV Preprocessing."""
        print(f"\n{'='*80}")
        print(f"STAGE 0: CV PREPROCESSING")
        print(f"{'='*80}")
        
        stage_start = time.time()
        
        # Run CV preprocessing (with caching)
        cv_data = run_cv_preprocessing(
            str(self.video_path),
            cache_dir=str(self.output_dir),
            force=False
        )
        
        # Save to output directory
        cv_cache_path = self.output_dir / "cv_cache.json"
        with open(cv_cache_path, 'w') as f:
            json.dump(cv_data, f, indent=2)
        
        stage_time = time.time() - stage_start
        self.stage_times['stage0_cv'] = stage_time
        self.stage_costs['stage0_cv'] = 0.0  # CV is free
        
        print(f"\n✓ Stage 0 complete ({stage_time:.1f}s, $0.00)")
        
        return cv_data
    
    def run_stage1_timeline(self, cv_data: dict) -> dict:
        """Stage 1: Position Timeline with Gemini."""
        print(f"\n{'='*80}")
        print(f"STAGE 1: POSITION TIMELINE (Gemini 2.5 Flash)")
        print(f"{'='*80}")
        
        stage_start = time.time()
        
        # Initialize Stage 1
        stage1 = GeminiStage1Timeline(
            video_path=str(self.video_path),
            cv_data=cv_data,
            output_dir=str(self.output_dir)
        )
        
        # Run analysis
        timeline_data = stage1.analyze()
        
        # Save outputs
        timeline_path = self.output_dir / "stage1_timeline.json"
        with open(timeline_path, 'w') as f:
            json.dump(timeline_data, f, indent=2)
        
        stage_time = time.time() - stage_start
        stage_cost = timeline_data.get('cost', {}).get('total', 0.0)
        
        self.stage_times['stage1_timeline'] = stage_time
        self.stage_costs['stage1_timeline'] = stage_cost
        self.total_cost += stage_cost
        
        print(f"\n✓ Stage 1 complete ({stage_time:.1f}s, ${stage_cost:.4f})")
        print(f"  Positions found: {len(timeline_data.get('positions', []))}")
        print(f"  Athletes identified: {len(timeline_data.get('athlete_profiles', {}))}")
        
        return timeline_data
    
    def run_stage2_detail(self, cv_data: dict, timeline_data: dict) -> dict:
        """Stage 2: Adaptive Detail Analysis with Gemini."""
        print(f"\n{'='*80}")
        print(f"STAGE 2: ADAPTIVE DETAIL ANALYSIS (Gemini 2.5 Flash)")
        print(f"{'='*80}")
        
        stage_start = time.time()
        
        # Initialize Stage 2
        stage2 = GeminiStage2Detail(
            video_path=str(self.video_path),
            cv_data=cv_data,
            timeline_data=timeline_data,
            output_dir=str(self.output_dir)
        )
        
        # Run analysis
        detail_data = stage2.analyze()
        
        # Save outputs
        detail_path = self.output_dir / "stage2_detail.json"
        with open(detail_path, 'w') as f:
            json.dump(detail_data, f, indent=2)
        
        stage_time = time.time() - stage_start
        stage_cost = detail_data.get('cost', {}).get('total', 0.0)
        
        self.stage_times['stage2_detail'] = stage_time
        self.stage_costs['stage2_detail'] = stage_cost
        self.total_cost += stage_cost
        
        print(f"\n✓ Stage 2 complete ({stage_time:.1f}s, ${stage_cost:.4f})")
        print(f"  Positions analyzed: {detail_data.get('positions_analyzed', 0)}")
        print(f"  Sub-segments found: {detail_data.get('total_subsegments', 0)}")
        
        return detail_data
    
    def run_stage3_synthesis(self, timeline_data: dict, detail_data: dict, cv_data: dict) -> dict:
        """Stage 3: Synthesis with Gemini."""
        print(f"\n{'='*80}")
        print(f"STAGE 3: SYNTHESIS (Gemini 2.5 Flash)")
        print(f"{'='*80}")
        
        stage_start = time.time()
        
        # Initialize Stage 3
        stage3 = GeminiStage3Synthesis(
            timeline_data=timeline_data,
            detail_data=detail_data,
            cv_data=cv_data,
            output_dir=str(self.output_dir)
        )
        
        # Run synthesis
        synthesis_data = stage3.synthesize()
        
        # Save outputs
        synthesis_path = self.output_dir / "stage3_synthesis.json"
        with open(synthesis_path, 'w') as f:
            json.dump(synthesis_data, f, indent=2)
        
        stage_time = time.time() - stage_start
        stage_cost = synthesis_data.get('cost', {}).get('total', 0.0)
        
        self.stage_times['stage3_synthesis'] = stage_time
        self.stage_costs['stage3_synthesis'] = stage_cost
        self.total_cost += stage_cost
        
        print(f"\n✓ Stage 3 complete ({stage_time:.1f}s, ${stage_cost:.4f})")
        
        return synthesis_data
    
    def generate_final_outputs(self, timeline_data: dict, detail_data: dict, synthesis_data: dict):
        """Generate final analysis.json and metadata.json."""
        print(f"\n{'='*80}")
        print(f"GENERATING FINAL OUTPUTS")
        print(f"{'='*80}")
        
        # Create HITL-compatible analysis.json
        analysis = {
            'video': str(self.video_path),
            'experiment': 'experiment3',
            'timestamp': datetime.now().isoformat(),
            'athlete_profiles': timeline_data.get('athlete_profiles', {}),
            'positions': detail_data.get('positions', []),
            'synthesis': synthesis_data.get('narrative', {}),
            'pace_analysis': synthesis_data.get('pace_analysis', {}),
            'coaching': synthesis_data.get('coaching', {})
        }
        
        analysis_path = self.output_dir / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Create metadata.json with cost tracking
        total_time = time.time() - self.start_time
        
        metadata = {
            'experiment': 'experiment3',
            'video': str(self.video_path),
            'video_name': self.video_path.name,
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': round(total_time, 2),
            'total_time_minutes': round(total_time / 60, 2),
            'total_cost': round(self.total_cost, 4),
            'stage_times': {k: round(v, 2) for k, v in self.stage_times.items()},
            'stage_costs': {k: round(v, 4) for k, v in self.stage_costs.items()},
            'positions_found': len(timeline_data.get('positions', [])),
            'subsegments_found': detail_data.get('total_subsegments', 0),
            'cost_per_position': round(self.total_cost / max(len(timeline_data.get('positions', [])), 1), 4),
            'cost_per_subsegment': round(self.total_cost / max(detail_data.get('total_subsegments', 1), 1), 4)
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Final outputs generated")
        print(f"  analysis.json: {analysis_path}")
        print(f"  metadata.json: {metadata_path}")
    
    def print_summary(self):
        """Print final summary."""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 3 COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
        print(f"Total cost: ${self.total_cost:.4f}")
        print(f"\nStage breakdown:")
        for stage, stage_time in self.stage_times.items():
            cost = self.stage_costs.get(stage, 0.0)
            print(f"  {stage}: {stage_time:.1f}s (${cost:.4f})")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def run(self):
        """Run complete pipeline."""
        try:
            # Stage 0: CV Preprocessing
            cv_data = self.run_stage0_cv()
            
            # Stage 1: Position Timeline
            timeline_data = self.run_stage1_timeline(cv_data)
            
            # Stage 2: Adaptive Detail
            detail_data = self.run_stage2_detail(cv_data, timeline_data)
            
            # Stage 3: Synthesis
            synthesis_data = self.run_stage3_synthesis(timeline_data, detail_data, cv_data)
            
            # Generate final outputs
            self.generate_final_outputs(timeline_data, detail_data, synthesis_data)
            
            # Print summary
            self.print_summary()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            return False
        except Exception as e:
            print(f"\n\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Adaptive CV + Cached Gemini')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', help='Output directory (default: outputs/experiment3)')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = Experiment3Pipeline(args.video, args.output)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
