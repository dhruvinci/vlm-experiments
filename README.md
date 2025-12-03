# Sensai - BJJ Video Analysis AI

Advanced AI system for analyzing Brazilian Jiu-Jitsu competition footage using vision-language models and computer vision.

## Current Baseline Performance

**Experiment 5 (Gemini 2.0 Flash):**
- Overall Score: **16.3/100**
- Position Quality: 27.1/100 (avg 1.35/5 stars)
- Detail Coverage: 0/100 (0 sub-segments vs 62 ground truth)

**Ground Truth Benchmark:**
- 32 macro positions
- 62 sub-segments (1.94 per position)
- 122 technique labels (1.97 per sub-segment)
- 42 unique BJJ techniques identified

## Project Structure

```
/Sensai/
├── prompts/              # Prompt engineering for VLM analysis
├── experiment/           # Experiment runner scripts
├── evaluation/           # Evaluation tools and ground truth data
├── hitl/                 # Human-in-the-loop annotation tool
├── results/              # Experiment outputs
└── data/                 # Test videos
```

## Quick Start

### Run Experiment 5 (Gemini Baseline)
```bash
export GOOGLE_API_KEY="your-api-key"
cd experiment
python bjj_video_analyzer_gemini.py \
  --video ../data/videos/youtube_SMRbZEbxepA.mp4 \
  --model flash \
  --output ../results/gemini_exp5/
```

### Evaluate Results
```bash
cd evaluation
python evaluate_detail_quality.py
```

### Run HITL Annotation Tool
```bash
# Terminal 1: Start evaluation server
cd evaluation
python evaluation_server.py

# Terminal 2: Start React frontend
cd hitl/frontend
npm install
npm start
```

## Next Experiments (Planned)

**Target: 70-90/100 overall score**

### Experiment 6A: Multi-Stage Gemini Pipeline
- Action segmentation at 3-5s granularity
- Technical classification with 42-technique vocabulary
- Hierarchical structure: position → sub_segments → labels

### Experiment 6B: Hybrid ByteTrack + Dual-Model
- ByteTrack for temporal segmentation
- Gemini 2.0 for position recognition
- Claude 3.5 Sonnet for technique analysis

### Experiment 6C: Single-Model Excellence
- Enhanced 42-technique ontology
- Three-pass analysis (chunk → group → enhance)
- Chain-of-thought reasoning

## Key Files

- `EXPERIMENTS_PLAN.md` - Detailed experiment roadmap
- `prompts/gemini_prompts.md` - Current prompts for Gemini models
- `prompts/bjj_ontology.md` - BJJ position/technique knowledge base
- `evaluation/ground_truth.json` - Human-annotated ground truth (32 positions, 62 sub-segments)
- `evaluation/evaluate_detail_quality.py` - Scoring script

## Test Video

**16-minute no-gi match:**
- Path: `data/videos/youtube_SMRbZEbxepA.mp4`
- Athletes: Andrew Tackett vs Micael Galvao
- Used for all experiments and evaluation

## Technology Stack

- **VLMs**: Gemini 2.0 Flash, Claude 3.5 Sonnet (planned)
- **CV Pipeline**: ByteTrack, MediaPipe, ST-GCN (planned)
- **HITL Tool**: React, Material-UI, FastAPI
- **Evaluation**: Python, custom scoring metrics

## Evaluation Metrics

```python
overall_score = (0.6 * rating_score) + (0.4 * detail_coverage)

rating_score = (avg_rating / 5.0) * 100  # Human ratings 1-5
detail_coverage = (ai_subsegments / gt_subsegments) * 100
```

## Contributing

See `EXPERIMENTS_PLAN.md` for planned experiments and implementation details.

## License

Research project - Internal use only
