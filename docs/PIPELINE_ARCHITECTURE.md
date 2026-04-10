# Kakashi CV Pipeline Architecture: Layered Position Classification

> From VLM-only to a hybrid CV + VLM stack that delivers real-time position labels with async deep analysis.

## Context

Kakashi currently runs a VLM-only pipeline: upload video → compress → send to Gemini → get position labels + coaching analysis. This works but is slow (minutes per video), expensive (Gemini API costs per analysis), and fully asynchronous — no real-time capability.

The goal is to build a layered architecture where a lightweight CV model handles real-time position classification, and the VLM focuses on what it's actually good at: reasoning, coaching commentary, and tactical analysis.

## Core Insight: Measurement vs Reasoning

The fundamental architectural mistake competitors make (Aether, Wrestle AI) is conflating two different computational problems:

- **Measurement**: What position are the athletes in? How long have they been there? What transitions occurred? These are quantitative, objective, and need to be fast.
- **Reasoning**: Why did that sweep work? What should the athlete do differently? What's the tactical pattern? These are qualitative, subjective, and can be slow.

VLMs are good at reasoning but overkill for measurement. A CV classifier is fast at measurement but can't reason. The architecture separates these concerns.

## The Layered Architecture

Each layer works independently at its level of precision. Higher layers add refinement but are not required — the system degrades gracefully.

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: VLM REASONING (async, seconds-to-minutes)     │
│  Input: video clip + structured data from L0-L2          │
│  Output: coaching notes, tactical analysis, why not what │
│  Model: Gemini / Molmo 2 / self-hosted VLM               │
│  This is what Kakashi already has.                       │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: TEMPORAL RESOLVER (near-real-time, buffered)   │
│  Input: sequence of L0/L1 predictions over time          │
│  Output: smoothed labels, transition events, corrections │
│  Catches noise: "one frame said mount in a guard         │
│  sequence — that's an outlier, ignore it"                │
│  Model: simple temporal filter / HMM / small transformer │
├─────────────────────────────────────────────────────────┤
│  LAYER 1: JOINT-ENHANCED CLASSIFIER (real-time, optional)│
│  Input: 3D joint coordinates from pose estimation        │
│  Output: L1A + L1B position, higher precision            │
│  Only fires when pose confidence is high                 │
│  Model: ST-GCN or MLP on joint coordinates               │
│  Dependency: requires good pose detection (ViTPose etc)  │
├─────────────────────────────────────────────────────────┤
│  LAYER 0: VISUAL CLASSIFIER (real-time, always-on)       │
│  Input: 2-3 second video clips (raw frames)              │
│  Output: L1A position class + confidence                 │
│  NO dependency on pose detection                         │
│  Model: Video Swin Tiny / 3D ResNet-18 / X3D-S           │
│  This is the baseline — works even if everything else     │
│  fails.                                                  │
└─────────────────────────────────────────────────────────┘
```

### Why This Ordering Matters

**Layer 0 has zero external dependencies.** It takes raw video frames and outputs a position. No pose model, no VLM, no joint detection. If you only had Layer 0, you'd still have real-time position labels. This is the foundation.

**Layer 1 adds geometric precision** but depends on pose detection quality. In BJJ, pose models struggle with entangled athletes and heavy occlusion. When pose confidence is high (standing sequences, clear separation), Layer 1 refines the output. When confidence is low (deep guard, scrambles), it stays silent and Layer 0's prediction stands.

**Layer 2 adds temporal consistency.** Individual frame/clip predictions are noisy. A temporal resolver smooths them: if 20 consecutive predictions say "guard" and one says "mount," that one is noise. It also detects transitions — a sustained shift from "guard" to "side_control" is a guard pass event.

**Layer 3 is the existing VLM pipeline**, now focused only on reasoning. It receives structured position data from L0-L2 as context, so it doesn't need to waste attention on measurement. It focuses on: why did that transition happen? What should the athlete do differently? What's the strategic pattern over multiple rounds?

## Layer 0 Deep Dive: The PyTorch Project

Layer 0 is the first thing to build. It's a self-contained video clip classifier.

### Architecture

```
Input: 2-3 second clip (48-72 frames at 24fps)
  ↓
Temporal sampling: select 8-16 evenly spaced frames
  ↓
Resize to 224x224
  ↓
Pretrained video backbone (frozen or partially unfrozen):
  - Video Swin Transformer Tiny (~28M params)
  - or 3D ResNet-18 (~33M params)  
  - or X3D-S (~3.8M params, most memory efficient)
  ↓
Classification head: Linear(feature_dim → 7 classes)
  ↓
Output: L1A position probabilities
  [standing, guard, side_control, mount, back, turtle, leg_entanglement]
```

### Training Data Pipeline

**Source 1: Existing Kakashi analyses (219 videos, ~2,278 labeled segments)**
- Each segment has a time range `[MM:SS-MM:SS]` and an L1A label
- Extract the clip for each segment's time range from the compressed video
- ~43% of segments have null/empty L1A (idle/scramble) — exclude or treat as 8th/9th class

**Source 2: YouTube auto-labeling (scalable)**
- Curate competition match URLs (ADCC, IBJJF, CJI)
- Run through Kakashi's youtube-synopsis or training-labeller-v7 pipeline
- Generates the same structured labels as Source 1
- Priority: target underrepresented classes (mount: 84 segments, leg_entanglement: 75)

**Source 3: OutlierDB instructional clips (732k tagged clips)**
- Instructionals demonstrate positions clearly, often in isolation
- Already tagged with technique names that map to L1A positions
- Quality varies but volume is massive

### Label Quality Pipeline

Auto-labeled data from Gemini is noisy. The verification pipeline:

```
Step 1: Cross-verify with second model
  - Run disputed segments through a different VLM (Claude vision, Molmo 2, etc)
  - Where both models agree → high confidence, minimal human review
  - Where they disagree → flag for human review

Step 2: Human review (HITL)
  - Review tool shows: short clip + Gemini label + second model label
  - Human confirms, corrects, or rejects
  - Priority: review all disagreements first, then sample agreements
  - Estimated: ~300-500 segments need manual review out of 2,278

Step 3: Clean dataset
  - Remove rejected segments
  - Apply corrections
  - Output: (clip_path, verified_l1a_label, confidence) tuples
```

### VLM Distillation (Training Enhancement)

Instead of just hard labels ("guard"), use VLM knowledge to generate richer supervision:

**Soft labels**: Ask the VLM to output confidence across all 7 classes for each segment. Train with KL divergence loss against these soft targets. This teaches the model that "half guard is more like closed guard than it is like mount" — relationships between classes that hard labels don't capture.

**Visual feature extraction** (optional, for difficult cases): Run a video VLM (Molmo 2 4B or LLaVA-NeXT-Video 7B) on RunPod, extract intermediate visual features from the vision encoder. Use these as additional supervision signal or as a teacher for distillation. This is a one-time batch job.

### Class Imbalance Strategy

Current L1A distribution from 219 analyzed videos:

| Position | Segments | % |
|----------|----------|---|
| guard | 865 | 38.0% |
| side_control | 404 | 17.7% |
| turtle | 299 | 13.1% |
| standing | 290 | 12.7% |
| back | 261 | 11.5% |
| mount | 84 | 3.7% |
| leg_entanglement | 75 | 3.3% |

*Percentages recalculated excluding null/empty segments (2,278 total)*

Strategies:
1. **Weighted sampling**: Oversample minority classes during training (WeightedRandomSampler)
2. **YouTube targeting**: Specifically find ADCC/CJI matches with mount finishes and leg lock exchanges
3. **Augmentation**: Temporal jittering, spatial crops, color jitter — more aggressive on minority classes
4. **Focal loss**: Down-weights easy examples, focuses training on hard/rare cases

### Hardware Constraints

**Training: RTX 3060 (6GB VRAM)**
- Batch size: 2-4 clips (depending on model and frame count)
- Gradient accumulation: accumulate over 8-16 steps for effective batch size of 32
- Mixed precision (FP16): halves memory usage
- Model choice matters: X3D-S (~3.8M params) fits easily; Video Swin Tiny (~28M params) is tight
- **Critical**: release tensors explicitly, use `del` + `torch.cuda.empty_cache()`, never accumulate history in lists
- Pin memory in DataLoader, use num_workers=2-4 for async data loading
- Profile memory with `torch.cuda.max_memory_allocated()` before scaling up batch size

**Feature extraction: RunPod A100 (one-time batch job)**
- VLM feature extraction and soft label generation
- Run once, save features to disk, train locally

### Model Selection Rationale

| Model | Params | VRAM (train) | Accuracy potential | PyTorch learning value |
|-------|--------|-------------|-------------------|----------------------|
| X3D-S | 3.8M | ~2GB | Good baseline | Efficient 3D convolutions |
| 3D ResNet-18 | 33M | ~4GB | Good | Classic architecture, well understood |
| Video Swin Tiny | 28M | ~5GB | Best | Transformers, attention, modern arch |

**Recommendation**: Start with X3D-S (fits easily, fast iteration), graduate to Video Swin Tiny once the data pipeline is solid.

### Evaluation

**Metrics:**
- Per-class accuracy (critical given imbalance)
- Macro F1 score (treats all classes equally)
- Confusion matrix (which positions get confused with each other?)
- Temporal consistency (when run frame-by-frame on a full video, do predictions make sense?)

**Validation strategy:**
- Hold out ~20% of videos (not segments — entire videos) for validation
- Never train and validate on segments from the same video
- Test on completely unseen YouTube matches for generalization

## Layer 1: Joint-Enhanced Classifier (Future)

Once Layer 0 establishes a baseline, Layer 1 adds geometric precision.

### Architecture

```
Input: 3D joint coordinates
  - 2 athletes x 17 keypoints x 3 coordinates (x, y, z) = 102 values per frame
  - 8-16 frames temporal window
  ↓
Model options:
  - ST-GCN: treats skeleton as graph, learns spatial + temporal patterns
  - MLP + temporal pooling: simpler, treats joints as flat feature vector
  - Small transformer on keypoint sequence
  ↓
Output: L1A + L1B position, confidence score
```

### Key Challenge: Pose Detection in Grappling

Standard pose models (MediaPipe, ViTPose) degrade significantly when athletes are entangled:
- Limb assignment errors (which arm belongs to whom?)
- Occluded keypoint hallucination
- Tracking identity swaps

Potential solutions:
- SAM-Body4D: training-free 4D mesh recovery, occlusion-robust
- Fine-tuned ViTPose on grappling data (requires annotation effort)
- Confidence gating: only use Layer 1 when pose confidence > threshold

### Fusion with Layer 0

When both layers produce predictions:
- If Layer 1 confidence is high and agrees with Layer 0 → use Layer 1 (more precise)
- If Layer 1 confidence is high and disagrees → use Layer 1 (it has geometric evidence)
- If Layer 1 confidence is low → fall back to Layer 0
- Simple learned gating: small network that takes both predictions + confidences and outputs final label

## Layer 2: Temporal Resolver (Future)

### Purpose

Raw per-clip predictions are noisy. Layer 2 smooths them into coherent timelines.

### Approaches (increasing complexity)

1. **Median filter**: Sliding window majority vote over predictions. Simple, effective.
2. **Hidden Markov Model**: Models valid position transitions (you can't go from mount to guard without a sweep/escape). Encodes domain knowledge about what transitions are possible.
3. **Small transformer**: Learns temporal patterns from sequences of (prediction, confidence) pairs. Most flexible but needs training data.

### Transition Detection

A sustained change in Layer 0/1 predictions signals a transition event:
- guard → side_control = guard pass
- bottom mount → guard = escape
- standing → guard = takedown

These map directly to the transition events in Kakashi's existing L1A ontology.

## Layer 3: VLM Reasoning (Existing)

This is the current Kakashi pipeline (Gemini via training-labeller-v7), refocused:

### Current Role
The VLM does everything: position detection, transition detection, coaching analysis, commentary.

### Future Role
With L0-L2 handling measurement, the VLM receives structured position data as input context and focuses purely on reasoning:
- Why did that guard pass succeed? (tactical analysis)
- What should the athlete prioritize? (coaching)
- How does this match compare to their last 5? (longitudinal patterns)

This makes the VLM prompt simpler, cheaper (fewer output tokens), and more accurate (it's reasoning about verified data, not raw video).

## Implementation Roadmap

### Phase 1: Layer 0 — Video Clip Classifier (the PyTorch learning project)
1. Build dataset extraction pipeline (Kakashi DB → video clips + labels)
2. Label verification pipeline (cross-model + HITL)
3. YouTube auto-labeling for class balance (especially mount, leg_entanglement)
4. Train X3D-S baseline on verified dataset
5. Graduate to Video Swin Tiny
6. VLM soft label distillation for training enhancement
7. Evaluate on held-out videos

### Phase 2: Layer 2 — Temporal Resolver
8. Run Layer 0 on full videos, analyze prediction sequences
9. Implement temporal smoothing (median filter → HMM)
10. Add transition detection logic

### Phase 3: Layer 1 — Joint-Enhanced Classifier
11. Evaluate pose detection quality on BJJ footage (ViTPose, SAM-Body4D)
12. If viable: train joint-coordinate classifier
13. Implement confidence-gated fusion with Layer 0

### Phase 4: Refocus Layer 3
14. Modify VLM prompts to receive L0-L2 structured data as input
15. Benchmark: does pre-computed position data improve VLM reasoning quality?
16. Reduce VLM costs by eliminating redundant measurement work

## Open Questions

- **Model hosting for inference**: Where does Layer 0 run in production? On Railway alongside the API? On a dedicated GPU instance? Serverless GPU (Replicate, Modal)?
- **Latency target**: What's "real-time" — per-frame at 24fps (42ms budget) or per-clip every 2-3 seconds?
- **L1B granularity**: Should Layer 0 also predict L1B (closed guard vs half guard vs butterfly) or just L1A? More classes = harder with limited data.
- **Wrestling/MMA generalization**: Does the same model work for wrestling positions, or does it need sport-specific training?

## References

- [Video-Based Detection of Combat Positions and Automatic Scoring in Jiu-jitsu (ACM 2022)](https://dl.acm.org/doi/10.1145/3552437.3555707) — pose + visual + structural cues for grappling position classification
- [SAM-Body4D](https://github.com/gaomingqi/sam-body4d) — training-free occlusion-robust 3D human mesh recovery
- [GrappleMap](https://github.com/Eelis/GrappleMap) — graph of grappling positions and transitions (topology reference)
- [Molmo 2 (Allen AI)](https://allenai.org/blog/molmo2) — efficient open-weight VLM for video understanding
- Kakashi's training-labeller-v7 prompt — existing auto-labeling pipeline for generating training data
