# VLM Experiments for BJJ Video Analysis

Research experiments exploring Vision-Language Models (VLMs) for automated analysis of Brazilian Jiu-Jitsu competition footage. This repo documents the iterative development from a naive baseline (16% accuracy) to a production-ready architecture achieving 100% temporal coverage with rich micro-analysis.

## Key Findings

| Experiment | Approach | Coverage | Cost | Key Learning |
|------------|----------|----------|------|--------------|
| **Exp 1** | Single-pass Gemini | 16% | ~$0.01 | VLMs hallucinate without constraints |
| **Exp 2** | Multi-pass + Caching | 100% | ~$1.00 | Over-segmentation, hallucinations persist |
| **Exp 3** | CV Constraints + Gemini | 100% | ~$0.02 | Attention decay in long videos |
| **Exp 3.3** | Two-pass fix attempt | 46% | ~$0.04 | Reactive fixes don't work |
| **Exp 4** | Skeleton-Flesh Integrated | **100%** | **$0.013** | ✅ Best results |

**Best Result (Experiment 4):** 25 segments, 100% coverage, all micro-analysis fields, $0.013/video

## Project Structure

```
├── experiment/                 # Core analysis pipelines
│   ├── experiment4/           # Best performing architecture
│   │   ├── stage0.py          # CV preprocessing (YOLO + optical flow)
│   │   ├── stage1_v3.py       # Integrated skeleton-flesh analysis
│   │   └── stage2.py          # Detail enrichment (optional)
│   ├── experiment3_*.py       # CV + Gemini pipeline
│   └── bjj_video_analyzer_*.py # Earlier experiments
├── hitl/                      # Human-in-the-Loop evaluation tool
│   └── frontend/              # React app for reviewing results
├── evaluation/                # Accuracy metrics & ground truth
│   ├── evaluation_server.py   # FastAPI backend for HITL
│   └── ground_truth.json      # Human-annotated benchmark
├── outputs/                   # Experiment results (JSON + Markdown)
│   ├── experiment4.0/         # Latest outputs
│   └── experiment3/           # Earlier outputs
├── prompts/                   # Prompt engineering & BJJ ontology
└── design/                    # UI mockups and assets
```

## Experiments Overview

### Experiment 1: Naive Baseline
Single-pass Gemini 2.0 Flash with BJJ ontology prompt.
- **Result:** 16.3/100 accuracy, zero sub-segmentation
- **Failure:** Model hallucinated positions (standing → "armbar defense")

### Experiment 2: Multi-Pass Architecture  
4-pass pipeline with context caching (85% token reduction).
- **Result:** 100% coverage but 241 segments (7.5x over-segmentation)
- **Failure:** Hallucinations persisted, $1/video too expensive

### Experiment 3: CV Constraints
Added MediaPipe pose estimation to constrain VLM outputs.
- **Result:** 58 segments, $0.02/video, 100% coverage
- **Innovation:** Standing probability prevents ground position hallucinations
- **Failure:** Attention decay caused repetitive outputs mid-video

### Experiment 4: Skeleton-Flesh Integrated (Best)
Single-pass with 16-field output format combining skeleton + micro-analysis.
- **Result:** 25 segments, $0.013/video, 100% coverage, all fields complete
- **Innovation:** Dynamic segment range based on video duration
- **Output:** Position, control, action score, strategy, setup, execution, outcome, coaching

## HITL Evaluation Tool

React-based interface for reviewing AI predictions against ground truth.

**Features:**
- Segment-by-segment video review with synchronized playback
- 1-5 star rating system for AI predictions
- Sub-segment annotation with technique labels
- Progress tracking across all segments
- Export to JSON for training data

**Screenshot:** The tool displays video alongside AI predictions, allowing annotators to rate accuracy and add corrections.

## Sample Output (Experiment 4)

```json
{
  "time": "1:34-1:44",
  "position": "standing",
  "top": "Galvao",
  "control": "0.6",
  "action": "0.8",
  "notes": "Galvao hits an explosive uchi mata from double overhooks",
  "strategy": "Galvao uses double overhooks to control upper body, capitalizing on Tackett's forward pressure",
  "setup": "Galvao secures double overhooks as Tackett presses in",
  "execution": "With double overhooks, Galvao elevates hips, hooks leg for textbook uchi mata. Tackett immediately turtles to prevent back take.",
  "outcome": "Clean takedown, momentum shifts to Galvao",
  "coaching": "Beautiful takedown using opponent's momentum. Tackett's immediate turtle was good defensive reaction."
}
```

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+ (for HITL tool)
- Google AI API key (Gemini)

### Installation

```bash
# Clone the repo
git clone https://github.com/0xdhruva/vlm-experiments.git
cd vlm-experiments

# Install Python dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-gemini-api-key"
```

### Run Experiment 4 (Recommended)

```bash
# Stage 0: CV Preprocessing (generates action scores)
python experiment/experiment4/stage0.py \
  --video_path data/videos/your_video.mp4 \
  --output_dir outputs/your_experiment

# Stage 1: Integrated Analysis
python experiment/experiment4/stage1_v3.py \
  --video_path data/videos/your_video.mp4 \
  --output_dir outputs/your_experiment \
  --mode competition \
  --run 1
```

### Run HITL Evaluation Tool

```bash
# Terminal 1: Start backend
cd evaluation
python evaluation_server.py
# Server runs on http://localhost:5002

# Terminal 2: Start frontend
cd hitl/frontend
npm install
npm start
# App runs on http://localhost:3000
```

### Run Earlier Experiments

```bash
# Experiment 2 (multi-pass)
python experiment/bjj_video_analyzer_exp2.py \
  --video data/videos/your_video.mp4 \
  --output outputs/exp2

# Experiment 3 (CV + Gemini)
python experiment/experiment3_main.py \
  --video data/videos/your_video.mp4 \
  --output outputs/exp3
```

## Output Formats

All experiments output both JSON (structured) and Markdown (human-readable):

- `stage1v3_skeleton_competition_run1.json` - Parsed segments with all fields
- `stage1v3_skeleton_competition_run1.md` - Raw model output
- `stage0_cv_checkpoints_5s.json` - CV preprocessing data

## Key Technical Insights

1. **VLMs need constraints:** Without CV data, models hallucinate positions
2. **Attention decay is real:** Models get "lazy" after ~4 minutes of video
3. **Format matters:** Integrated output (skeleton + detail in one line) prevents task-switching fatigue
4. **Dynamic targets help:** Telling model "aim for 30-70 segments" produces better results than fixed limits
5. **Caching is essential:** Video caching reduces costs by 85%+

## Test Video

Experiments use a 16-minute no-gi match:
- **Athletes:** Andrew Tackett vs Micael Galvao
- **Event:** Who's Number One
- **Ground Truth:** 32 positions, 62 sub-segments, 122 technique labels

(Video not included in repo due to size - use your own BJJ footage)

## Technology Stack

- **VLM:** Gemini 2.5 Flash (with context caching)
- **CV:** YOLO11 pose detection, OpenCV optical flow
- **Backend:** FastAPI, Python
- **Frontend:** React, Material-UI
- **Evaluation:** Custom accuracy metrics, human annotation

## Cost Analysis

| Stage | Tokens | Cost |
|-------|--------|------|
| Video cache | ~286k | $0.007 |
| Stage 1 analysis | ~6k output | $0.006 |
| **Total** | ~292k | **$0.013** |

## Future Work

- [ ] Multi-video evaluation across different match types
- [ ] Real-time analysis for live streaming
- [ ] Technique-specific deep dives (Stage 2)
- [ ] Training data generation for fine-tuning

## License

MIT - Feel free to use for research and experimentation.

## Citation

If you use this work, please cite:
```
@misc{vlm-bjj-experiments,
  author = {Dhruva},
  title = {VLM Experiments for BJJ Video Analysis},
  year = {2025},
  url = {https://github.com/0xdhruva/vlm-experiments}
}
```
