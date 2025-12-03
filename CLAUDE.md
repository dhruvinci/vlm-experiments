# CLAUDE.md - Sensai Project Guide

**Last Updated**: 2025-10-18

---

## Project Overview

**Sensai**: BJJ video analysis system using Gemini 2.5 Flash + CV preprocessing

**Goal**: Achieve 70-90/100 accuracy (from baseline 16.3/100)

**Test Video**: Andrew Tackett vs Micael Galvao (16-min no-gi match)

**Ground Truth**: 62 sub-segments, 122 technique labels (human-annotated via HITL)

**Current Experiment**: Experiment 3 - Adaptive CV + Cached Gemini ✅ Complete

**Latest Results**: 58 segments, $0.02 cost, 2 min runtime, 100% coverage

## Key Technical Constraints

**Gemini 2.5 Flash Limits:**
- Input: 1M tokens (16-min video fits easily)
- Output: 65K tokens
- Rate limit: 1M tokens/min (Tier 1)

**Context Caching:**
- Cached tokens: $0.01875/1M (94% cheaper)
- Regular tokens: $0.30/1M input, $1.20/1M output
- TTL: 1 hour (video), 24 hours (ontology)

## Directory Structure

```
Sensai/
├── CLAUDE.md, README.md, EXPERIMENTS_PLAN.md
├── experiment/              # Experiment code
│   ├── experiment3_stage0.py       # CV preprocessing + smart segmentation
│   ├── experiment3_stages123.py    # Gemini Stage 1 (Timeline)
│   └── experiment3_main.py         # Main orchestrator
├── prompts/                 # BJJ ontology + prompts
├── evaluation/              # Evaluation framework + ground truth
├── hitl/frontend/           # React HITL tool (port 3000) - needs Exp3 support
├── outputs/experiment3/     # Experiment 3 outputs
│   ├── *_cv_cache.json            # Stage 0 CV data (cached forever)
│   ├── stage1_timeline.json       # Stage 1 segments
│   └── stage1_markdown.md         # Raw Gemini output
└── data/videos/             # Test videos
```

## HITL Tool (Human-in-the-Loop)

**Backend** (port 5002): `python3 evaluation/evaluation_server.py`

**Frontend** (port 3000): `cd hitl/frontend && npm start`

**Workflow**: Select experiment → Review/rate segments → Add labels → Save ground truth

## Experiment Results Summary

### Experiment 1: Baseline (16.3/100) ❌
**Architecture**: Single-pass Gemini

**Results**:
- Position Accuracy: 13.8%
- Transition Recall: 0%
- Sub-segments: 0 (vs 62 ground truth)
- Technique Labels: 0 (vs 122 ground truth)

**Key Failures**: No granularity, no technique vocabulary, fixed sampling

**Learning**: Single-pass insufficient, need multi-pass + adaptive sampling

---

### Experiment 2: Multi-Pass (52.9→70-80/100) ⚠️
**Architecture**: 4-pass (CV → Holistic → Detail → Synthesis)

**What Worked**:
- ✅ Context caching (85% token reduction)
- ✅ Multi-window approach (100% coverage)
- ✅ Rich metadata (labels, dominance, quality)

**Critical Failures**:
- ❌ Hallucinations: Standing (1:05) → misclassified as ground armbar
- ❌ Over-segmentation: 241 segments (7.5x more than ground truth)
- ❌ Position accuracy: Many fundamental errors

**Root Causes**:
1. Window boundary context loss (6-min windows)
2. No visual constraints (pure VLM hallucinates)
3. Prompt overload (11 columns simultaneously)
4. Optical flow doesn't align with semantic BJJ positions

**Key Learning**: 🔑 Need deterministic CV constraints (pose estimation) to prevent hallucinations

**Cost**: $1.00 per video, 31 minutes

---

### Experiment 3: Adaptive CV + Cached Gemini ✅ Complete
**Status**: Implementation complete, pending accuracy evaluation

**Architecture (Simplified)**:
- **Stage 0**: CV preprocessing with smart segmentation (43 suggested segments)
- **Stage 1**: Gemini timeline with unified segment format (58 final segments)
- **Stage 2 & 3**: Deferred to future optional analysis

**Results**:
- **Segments**: 58 (100% coverage, 0:00-16:10)
- **Cost**: $0.019 (~2 cents, 95% cheaper than Exp2)
- **Time**: 109s (~2 min, 71% faster than Exp2)
- **Output**: 9,577 tokens (62% more efficient than table format)
- **Action distribution**: 84.5% medium, 10.3% low, 5.2% high

**Key Innovations**:
1. Smart CV segmentation (action + transitions)
2. Unified segment format (all data together)
3. Dynamic narrative limits (150-350 chars by action)
4. Key actions indexing (point-scoring moves)
5. Token efficiency (9.6k vs 25k for full match)

**Hallucination Prevention**: CV standing probability + suggested segments guide Gemini

See EXPERIMENTS_PLAN.md for detailed results and analysis.

## BJJ Ontology

**18 Core Positions**: mount, side_control, back_control, knee_on_belly, north_south, guard, half_guard, standing, turtle, scramble, clinch, takedown, submission_attempt, passing_guard, sweep, single_leg, double_leg, unknown

**91 Technique Labels**: Full vocabulary in `prompts/bjj_ontology.md`

**ADCC Scoring**: Takedowns (2), Knee on belly (2), Guard passes (3), Mount/Back (4)

## Known Issues

**HITL Completed Status Bug**: ✅ Fixed (2025-10-09)
- Issue: Checkmark disappeared when navigating between segments
- Fix: Backend must run from Sensai root, reordered tabs, fixed `isLabeled` property

## Key Technical Learnings

### Context Caching (Breakthrough)
- **Problem**: Video re-uploaded every request (~80K tokens each)
- **Solution**: Cache video + ontology once, reuse for all positions
- **Impact**: 100K → 15K tokens per request (85% reduction)
- **Performance**: 16 positions/min (vs 2 positions/min without caching)

### Multi-Window Approach
- **Problem**: 16-min video too long for single pass
- **Solution**: 6-min windows with 15s overlap + deduplication
- **Result**: 100% coverage, 39 positions detected

### Rate Limiting
- **Free tier**: 250K tokens/min hard limit
- **Solution**: Batch processing (2 positions per batch, 60s wait)
- **Tier 1**: 1M tokens/min (no more waiting)

## Evaluation Metrics

**Composite Score (0-100):**
- Position Accuracy: 35% weight
- Transition Detection: 25% weight  
- Scoring Accuracy: 25% weight
- Temporal IoU: 15% weight

**Detail Quality:**
- Sub-segmentation granularity (3-5s segments)
- Technique label coverage (91-label ontology)
- Analytical depth (dominance, quality, coaching)

## Quick Commands

**Run Experiment:**
```bash
python3 experiment/bjj_video_analyzer_exp3.py --video data/youtube_SMRbZEbxepA.mp4
```

**Start HITL Tool:**
```bash
# Backend
python3 evaluation/evaluation_server.py

# Frontend (separate terminal)
cd hitl/frontend && npm start
```

**Evaluate:**
```bash
python3 evaluation/accuracy_calculator.py \
  evaluation/ground_truth.json \
  results/experiment3/*/analysis.json
```

## Dependencies

**Python**: google-generativeai, opencv-python, mediapipe, scenedetect, fastapi, numpy

**Node.js**: react, @mui/material, video.js

**Setup**: `pip install -r requirements.txt && cd hitl/frontend && npm install`

---

## Critical Insights

**From Experiment 2:**
1. **Coverage ≠ Accuracy**: 241 segments with many errors worse than 32 accurate segments
2. **Detail comes AFTER accuracy**: Rich labels useless if position is wrong
3. **Context window size matters**: 6-min too long (context loss), 15s too short (fragmentation)
4. **Prompt complexity has costs**: 11 columns simultaneously → hallucination pressure
5. **Multi-pass may be superior**: Separate position classification from detail extraction

**Path to 70-90/100:**
**Accuracy first, coverage second, detail third**

Exp2 optimized in reverse order. Exp3 reorders priorities.

---

**Document Status**: Active development  
**Last Updated**: 2025-10-18  
**Current Focus**: Experiment 3 implementation

For detailed experiment plans, see EXPERIMENTS_PLAN.md
