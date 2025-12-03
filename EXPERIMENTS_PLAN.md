# BJJ AI Coach - Experiments Plan

**Goal**: Achieve 70-90/100 accuracy score (from baseline 16.3/100) for BJJ video analysis
**Test Video**: Andrew Tackett vs Micael Galvao (16-min no-gi match)
**Ground Truth**: 62 sub-segments, 122 technique labels (human-annotated via HITL)

---

## ✅ LATEST EXPERIMENT

**Currently Active**: Experiment 4 - Skeleton-Flesh Integrated Analysis

**Status**: ✅ Implementation complete, Run 2 validated

**Results**: 25 segments, $0.0134 cost, 92.5s runtime, 100% coverage, all micro-analysis fields present

---

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

---

## Experiment 1: Baseline (Simple Gemini)

### Intent
Establish baseline performance with minimal preprocessing.

### Architecture
- **Model**: Gemini 2.0 Flash
- **Approach**: Single-pass analysis with BJJ ontology
- **Sampling**: Fixed (1 frame per 10-15s)

### Results
**Score: 16.3/100** ❌

- Position Accuracy: 13.8%
- Transition Recall: 0%
- Sub-segments: 0 (vs 62 ground truth)
- Technique Labels: 0 (vs 122 ground truth)
- Temporal IoU: 50.1%

### Key Failures
1. **Zero granularity**: No sub-segmentation
2. **No technique vocabulary**: Missing all 91 technique labels
3. **Incorrect ontology usage**: Used "escape" and "sweep" as positions
4. **No analytical depth**: Missing dominance, quality, style analysis
5. **Fixed sampling**: Wastes tokens on stalling, under-samples action

### Learnings
- Single-pass insufficient for detail
- Need multi-pass architecture
- Require adaptive sampling for action vs stalling
- Must enforce technique vocabulary

**Status**: ✅ Complete (2025-10-07)

---

## Experiment 2: Multi-Pass Adaptive Analysis

### Intent
Achieve >100/100 by adding multi-pass architecture with adaptive sampling and analytical depth.

### Architecture
**4-Pass Pipeline:**
1. **Pass 0**: OpenCV optical flow → action/stalling detection
2. **Pass 1**: Gemini holistic context → macro timeline (33-40 positions)
3. **Pass 2**: Gemini adaptive detail → sub-segments + labels per position
4. **Pass 3**: Gemini synthesis → unified timeline + fighter profiles

**Key Innovations:**
- Multi-window Pass 1 (6-min windows, 15s overlap, deduplication)
- Context caching (video + ontology cached, 85% token reduction)
- Model upgrade to Gemini 2.5 Flash across all passes
- Rate limiting (6.7s delays between positions)

### Results
**Initial: 52.9/120** ❌ (architectural flaw)
**Revised: ~70-80/100** ⚠️ (improved but still issues)

**What Worked:**
- ✅ Temporal coverage: 103% (241 positions vs 32 ground truth)
- ✅ Rich metadata: Labels, dominance, quality scores
- ✅ Context caching: 85% token reduction
- ✅ Multi-window Pass 1: 100% video coverage

**Critical Failures:**
- ❌ **Hallucinations**: Standing grip fighting → misclassified as ground armbar defense
- ❌ **Over-segmentation**: 241 segments (7.5x more than ground truth)
- ❌ **Position accuracy**: Many fundamental misclassifications
- ❌ **Example (1:05)**: Athletes standing with collar ties → model said "leg_entanglement" + "armbar_defense"

### Root Causes
1. **Window boundary context loss**: 6-min windows lose temporal continuity
2. **No visual constraints**: Model hallucinates without physics-based validation
3. **Prompt overload**: Too much information, model confused
4. **Forced short segments**: 15s max segments too granular for some positions

### Key Learnings
- ✅ Context caching essential (16 positions/min vs 2 positions/min)
- ✅ Multi-window approach works for long videos
- ❌ Optical flow (Pass 0) doesn't align with semantic BJJ positions
- ❌ Pure VLM insufficient - needs CV constraints to prevent hallucination
- ❌ Over-segmentation worse than under-segmentation
- 🔑 **Critical insight**: Need deterministic CV constraints (pose estimation) to prevent hallucinations

### Cost & Time
- **Cost**: ~$1.00 per video
- **Time**: ~31 minutes
- **Token efficiency**: 85% reduction via caching

**Status**: ✅ Complete (2025-10-09)

---

## Experiment 3: Adaptive CV + Cached Gemini

### Intent
Fix Exp2 hallucinations by adding deterministic CV constraints while maintaining cost efficiency through caching and adaptive granularity.

### Core Innovation
**Continuous action scoring drives adaptive granularity:**
- CV computes action_score (0-1) per second using optical flow + pose changes
- action_score determines segmentation density (static: 30-60s, extreme: 2-4s)
- Cost scales with match complexity (slow match: $0.16, fast match: $0.52)

### Architecture
**4-Stage Pipeline:**

**Stage 0: CV Preprocessing** (Free, Cached Forever)
- Scene detection → macro boundaries
- MediaPipe pose → standing_probability per second
- Optical flow → motion magnitude
- **Action scoring**: `0.4×flow + 0.3×pose_change + 0.2×spatial + 0.1×complexity`
- **Output**: cv_cache.json with action scores + CV constraints

**Stage 1: Position Timeline** (Gemini 2.5 Flash, Cached)
- Input: Full video + CV constraints + action statistics
- Cache: Video (100K) + ontology (3K) + framework (5K)
- **CV constraints prevent hallucinations**: "1:05 STANDING (prob=0.92) → MUST use standing/clinch/takedown"
- Output: 20-120 positions (varies by match pace)
- Cost: $0.003-0.006

**Stage 2: Adaptive Detail** (Gemini 2.5 Flash, Cached Clips)
- Extract 30-60s clip per position
- Granularity driven by action_score:
  - Low action (0.2): 30-60s sub-segments
  - High action (0.7): 4-8s sub-segments
  - Extreme action (0.9): 2-4s sub-segments
- Output: Sub-segments with labels, scores, coaching
- Cost: $0.005 per position

**Stage 3: Synthesis** (Gemini 2.5 Flash, Text-Only)
- Input: Stage 1 + Stage 2 outputs (no video)
- Output: Match narrative + pace analytics + fighter profiles
- Cost: $0.005-0.015

### Hallucination Prevention
**CV Constraint Validation:**
- MediaPipe calculates hip_height, knee_angles → standing_probability
- If standing_prob > 0.8 → MUST use standing positions only
- If standing_prob < 0.2 → MUST use ground positions only
- **Fixes 1:05 bug**: CV says standing (0.92) → Gemini cannot say "armbar"
- Post-analysis validation flags violations

### Match Variance Handling
**Cost scales with complexity (not time):**

**Slow Positional Match:**
- Positions: 25-40 (long durations)
- Sub-segments: 75-120 (2-3 per position)
- Avg action: 0.25
- Cost: $0.16-0.20
- Time: 6-8 min

**Fast Scramble Match:**
- Positions: 80-120 (short durations)
- Sub-segments: 280-480 (4-6 per position)
- Avg action: 0.75
- Cost: $0.50-0.65
- Time: 13-16 min

### Output Format
**Token-optimized hybrid format:**
- **Stage 1**: Markdown table (1,050-3,325 tokens)
- **Stage 2**: Hybrid compact (structured + rich notes, 7,350-26,600 tokens)
- **Stage 3**: Structured prose (2,500-4,500 tokens)
- **Total output**: 11,400-34,425 tokens (49% reduction vs JSON)

### Token & Cost Tracking
**Built-in monitoring per stage:**
- Input tokens (new vs cached)
- Output tokens
- Cost breakdown (input/output/total)
- Time per stage
- Positions/sub-segments found
- Cost per position/sub-segment

**Enables:**
- Real-time cost monitoring
- Performance optimization
- Match complexity analysis

### Expected Results
**Target Accuracy:**
- Position detection: >70% (CV constraints + validation)
- Hallucination rate: <5% (CV prevents major errors)
- Segmentation: Variable (20-120 positions based on pace)
- Cost: $0.16-0.52 (scales with complexity)
- Time: 6-16 min (scales with complexity)

**vs Experiment 2:**
- Hallucinations: Fixed (CV constraints)
- Segmentation: Natural (not forced)
- Cost: 48-84% cheaper
- Time: 48-81% faster
- Token efficiency: 97% reduction (caching)

### Iterative Refinement
**Cached clips enable cheap follow-up analysis:**
- Initial analysis: $0.35
- Re-analyze with different focus (within 1hr): $0.03 (91% cheaper)
- Deep-dive on 3 positions: $0.001 (99.7% cheaper)
- Total session: $0.38 vs $1.05 without caching (64% savings)

### Key Innovations
1. **Continuous action scoring**: Drives granularity decisions automatically
2. **CV constraint validation**: Prevents hallucinations (standing vs ground)
3. **3-tier caching**: Global ontology, video, clips
4. **Adaptive segmentation**: No arbitrary limits (20-120 positions based on pace)
5. **Cost fairness**: Pay for complexity, not time
6. **Token optimization**: Compact output formats (49% reduction)

### Implementation Plan
**Phase 1: CV Preprocessing (3-4 hours)**
- Scene detection, pose estimation, optical flow
- Action score computation and granularity mapping

**Phase 2: Gemini Integration (3-4 hours)**
- 3-tier context caching setup
- Action-aware prompts for all stages
- Clip extraction and caching

**Phase 3: Validation & Tracking (2-3 hours)**
- CV constraint validation logic
- Token/cost tracking per stage
- Output format parsing

**Phase 4: Testing (1-2 hours)**
- Run on 16-min test video
- Verify 1:05 hallucination fixed
- Validate cost estimates

**Total: 8-12 hours**

**Status**: ✅ Complete (2025-10-18)

### Actual Results

**Implementation Complete!** Experiment 3 successfully addresses Exp2's hallucinations with a hybrid CV + Gemini approach.

**Final Architecture (Simplified):**
- **Stage 0**: CV Preprocessing with smart segmentation (43 suggested segments)
- **Stage 1**: Gemini Timeline Analysis with unified segment format
- **Stage 2 & 3**: Deferred to future optional analysis passes

**Performance Metrics:**
- **Segments**: 58 segments (16:10 match)
- **Coverage**: 100% (0:00 - 16:10)
- **Avg segment**: 16.7s (range: 8-59s)
- **Output tokens**: 9,577 tokens
- **Cost**: $0.019 (~2 cents per match)
- **Time**: 109s (~2 minutes)

**Action Distribution:**
- Low action (< 0.30): 6 segments (10.3%, avg 22.3s)
- Medium action (0.30-0.50): 49 segments (84.5%, avg 16.1s)
- High action (≥ 0.50): 3 segments (5.2%, avg 15.3s)

**Key Innovations Implemented:**
1. **Smart segmentation**: CV suggests 30-100 segments based on action + transitions
2. **Unified output format**: All segment data (position, athletes, actions, narrative) in one structure
3. **Dynamic narrative limits**: 150-350 chars based on action score
4. **Key actions indexing**: Point-scoring moves tagged for video navigation
5. **Token efficiency**: 62% more efficient than table format (9.6k vs 25k tokens)

**Format Comparison:**
- Old table format: Hit 25k token limit at 5:51 (36% coverage)
- New segment format: 9.6k tokens for full match (100% coverage)

**Hallucination Prevention:**
- CV standing probability guides position classification
- Suggested segments prevent arbitrary micro-segmentation
- Gemini refines CV hints with video analysis

**Cost Efficiency:**
- 95% cheaper than Exp2 ($0.02 vs $1.00)
- 71% faster than Exp2 (2 min vs 31 min)
- Scales with match complexity (not duration)

**Next Steps:**
- Accuracy evaluation pending manual review
- HITL tool development for review interface
- Stage 2 & 3 as optional deep-dive analysis

### Experiment 3.3: Two-Pass Architecture with Attention Cliff Detection (2025-10-22)

**Problem**: Model attention decay causing repetitive/lazy outputs in middle sections despite accurate CV data.

**Attempted Solutions (All Failed)**:
1. ❌ Frequency/presence penalties - Not supported on Gemini 2.5 Flash
2. ❌ Switch to Gemini 2.0 Flash - No caching support (breaks cost efficiency)
3. ❌ Higher temperature (0.7) + anti-repetition prompts - Made laziness worse (26 consecutive "back_control")
4. ❌ Compressed output format with pipes - Still hit MAX_TOKENS at 7:30 (46% coverage)

**Architecture Implemented**:
- **Stage 1**: Full video analysis (single pass)
- **Stage 2**: Lazy detection (5 signals) + window re-analysis
- **Stage 3**: Merge Pass 1 + Pass 2 segments

**Test Results**:
- **v3.3-pass1** (verbose format): 49 segments, 7:10 coverage (44%), $0.0166, truncated
- **v3.3-compressed2** (pipe format): 24 segments, 7:30 coverage (46%), $0.0109, truncated
- **Full pipeline**: 5 flagged segments, 4 windows, $0.0408 total (if completed)

**Critical Shortcomings Discovered**:

1. **MAX_TOKENS Hit Prematurely**
   - Set limit: 25,000 tokens
   - Actual output: 2,978-7,641 tokens before truncation
   - Coverage: Only 44-46% of 16-minute video
   - Root cause: Unknown (API issue or internal model behavior)

2. **Attention Decay Persists**
   - Attention cliff at 0:34 (model lazy after 34 seconds!)
   - Repetitive outputs despite CV constraints
   - Compressed format didn't solve core issue

3. **Two-Pass Overhead**
   - Stage 2 creates 4 windows for 5 flagged segments
   - Cost increase: 146% vs Stage 1 alone ($0.041 vs $0.017)
   - Complexity: 5 API calls vs 1
   - Merge issues: 1 gap, 3 overlaps in continuity

4. **Format Compression Insufficient**
   - Verbose format: ~500 chars/segment
   - Compressed format: ~400 chars/segment (20% reduction)
   - Still needs ~6,500 tokens for full match
   - Should fit in 25k limit but doesn't

**Key Learnings**:

✅ **What Worked**:
- Lazy detection algorithm (5 signals) successfully identified problematic segments
- Compressed format reduced cost by 34% ($0.0109 vs $0.0166)
- Parser successfully handled both verbose and compressed formats
- Window-based re-analysis architecture is sound

❌ **What Failed**:
- Single-pass full video analysis doesn't scale to 16+ minutes
- Gemini 2.5 Flash has attention decay issues regardless of format
- MAX_TOKENS limit appears to be enforced earlier than specified
- Two-pass "fix" approach is expensive and doesn't address root cause

**Root Cause Analysis**:

The fundamental issue is **architectural**: asking a VLM to analyze a 16-minute video in one pass exceeds its attention span, regardless of output format or token limits.

**Conclusion**: Experiment 3.3 proves that **reactive fixes (detecting and re-analyzing lazy segments) are insufficient**. Need **proactive architecture** that prevents attention decay from occurring.

**Status**: ❌ Failed - Architecture fundamentally flawed for long videos

---

## Experiment 4: Skeleton-Flesh Integrated Analysis (In Progress)

### Intent
Achieve 100% coverage with integrated skeleton + micro-analysis in single pass, avoiding attention decay through dynamic segmentation and interleaved output.

### Core Innovation
**Integrated format prevents attention decay:**
- Single **SEG** line contains all 16 fields (skeleton + micro-analysis)
- No task switching between skeleton and detail generation
- Dynamic segment_range based on video duration
- Model treats all fields equally (no differentiation)

### Architecture

**Stage 0: CV Preprocessing** (Cached)
- YOLO11 pose detection + optical flow
- Action scoring: `0.4×flow + 0.3×pose_change + 0.2×spatial + 0.1×complexity`
- Output: cv_cache.json with 5-second checkpoints

**Stage 1: Integrated Analysis (Single Pass)**
- **Input**: Full video (cached) + CV checkpoints + dynamic segment_range
- **Output**: 16-field segments (skeleton + micro-analysis combined)
- **Format**: `**SEG** | [time] | pos | top | ctrl | score | action | conf | reasons | focus | notes | strategy | setup | execution | outcome | coaching`
- **Segment Range (Dynamic)**:
  - 0-4 min: 10-30 segments
  - 4-8 min: 20-40 segments
  - 8-20+ min: 30-70 segments
- **Token cost**: ~7,500 tokens (estimated)
- **Cost**: ~$0.010-0.015
- **Why this works**: Single focused task, no context switching, natural rhythm

### 16-Field Output Format

**Skeleton Fields (10)**:
1. Time: [MM:SS-MM:SS]
2. Position: standing, guard, mount, side_control, back_control, half_guard, turtle, scramble
3. Top: athlete name, A1, A2, or - (neutral)
4. Control: 0.0-1.0 (dominance)
5. Score: X-Y format
6. Action: 0.0-1.0 (activity level)
7. Confidence: 0.0-1.0 (accuracy confidence)
8. Reasons: Brief tags (e.g., "sweep_to_mount")
9. Focus: Tags for deeper analysis
10. Notes: Brief description (100 char limit)

**Micro-Analysis Fields (6)**:
11. Strategy (100-150 chars): Game planning and tactical insights
12. Setup (100-150 chars): How position/technique was created
13. Execution (250-350 chars): Technical breakdown - grips, angles, pressure, timing
14. Outcome (100-150 chars): Result and impact on momentum
15. Coaching (150-200 chars): What was done well, what could improve

### Implementation Status

**Stage 1v1**: Initial skeleton-only pass
- **Result**: 22 segments, 100% coverage, 2,739 tokens, $0.0095, 59.7s
- **Issue**: No micro-analysis, attention decay in middle sections

**Stage 1v2**: Skeleton + separate detail lines
- **Result**: Interleaved **SEG** and **DETAIL** lines
- **Issue**: Parser complexity, model task-switching

**Stage 1v3 (Current)**: Integrated 16-field format
- **Run 1**: 15 segments, 100% coverage, 3,916 tokens, $0.0107, 71.6s
  - All micro-analysis fields present (100%)
  - Tokens per segment: 261.1
  - Fewer segments but richer detail
  
- **Run 2**: 25 segments, 100% coverage, 6,116 tokens, $0.0134, 92.5s
  - Dynamic segment_range: "30-70" (16:10 video)
  - All micro-analysis fields present (100%)
  - Tokens per segment: 244.6 (-6.3% vs Run 1)
  - +66.7% more segments with better granularity
  - Action distribution: 3 low, 8 medium, 14 high action

### Results Summary

| Metric | Run 1 | Run 2 | Change |
|--------|-------|-------|--------|
| Segments | 15 | 25 | +66.7% |
| Coverage | 100% | 100% | Same |
| Output tokens | 3,916 | 6,116 | +56.2% |
| Cost | $0.0107 | $0.0134 | +25.2% |
| Runtime | 71.6s | 92.5s | +29.2% |
| Tokens/segment | 261.1 | 244.6 | -6.3% |

### Key Findings

✅ **Successes**:
- Integrated format works (no task differentiation needed)
- Dynamic segment_range effective (model responded to "aim for 30-70")
- All micro-analysis fields complete (100% quality)
- Output tokens well within limit (6,116 << 25,000)
- Cost increase minimal (+$0.0027 for 66% more content)
- Better token efficiency with more segments

⚠️ **Trade-offs**:
- Runtime increased +29.2% (acceptable for 66% more content)
- Cost increased +25.2% (still very reasonable at $0.0134)

### Prompt Enhancements

**Dynamic Segment Range Calculation**:
```python
if duration_sec < 240:  # 0-4 minutes
    segment_range = "10-30"
elif duration_sec < 480:  # 4-8 minutes
    segment_range = "20-40"
else:  # 8-20+ minutes
    segment_range = "30-70"
```

**Critical Instructions**:
- "You MUST AIM for {segment_range} segments for the full match"
- "You MUST add as many relevant position tags as possible (limited to 10)"
- "Analyze COMPLETE video from 0:00 to {duration}"

### Parser Implementation

**Single-pass parsing** (simplified vs v2):
- Collects all **SEG** lines
- Extracts 16 pipe-separated fields per segment
- No merging needed (all data in one line)
- JSON output with interleaved skeleton + micro-analysis

### Cost & Performance Comparison

| Experiment | Approach | Segments | Tokens | Cost | Coverage | Time |
|------------|----------|----------|--------|------|----------|------|
| Exp 3 | CV + Gemini | 58 | 9,577 | $0.019 | 100% | 109s |
| Exp 3.3 | Two-pass | 24 | 7,641 | $0.041 | 46% ❌ | - |
| **Exp 4v3** | **Integrated** | **25** | **6,116** | **$0.0134** | **100%** | **92.5s** |

**vs Exp 3**:
- Segments: -57% (58 → 25)
- Tokens: -36% (9,577 → 6,116)
- Cost: -29% ($0.019 → $0.0134)
- Time: -15% (109s → 92.5s)
- **Advantage**: Micro-analysis included, cleaner format

### Status

**✅ Implementation Complete** - Experiment 4 successfully delivers integrated skeleton + micro-analysis with:
- Dynamic segment range based on video duration
- 16-field format (skeleton + 6 micro-analysis fields)
- 100% coverage maintained
- Improved cost efficiency vs Exp 3
- All micro-analysis fields present in all segments

**Next Steps**:
- Accuracy evaluation on test video
- Extend to multiple videos
- Optimize segment_range thresholds based on match type

---

## Future Experiments (Deprioritized)

These experiments were part of the original plan but are not currently prioritized:

- **Experiment 5**: LLaVA-Video-7B (native video understanding)
- **Experiment 6**: Alternative Gemini configurations
- **Experiment 7**: Hybrid ByteTrack + VLM architecture
- **Experiment 8**: Qwen2-VL-72B scale testing

---

**Document Status**: Active development
**Last Updated**: 2025-10-27
**Current Focus**: Experiment 4 validation and accuracy evaluation
