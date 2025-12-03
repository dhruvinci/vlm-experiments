# Experiment 2: Multi-Pass Analysis Prompts (Revised)

## Pass 1: Holistic Context Prompt

```
You are analyzing a full BJJ match to provide HOLISTIC CONTEXT for multi-pass analysis.

This is Pass 1 of a 3-pass analysis system. Your role is to provide the BIG PICTURE with ACCURATE position classification.

### CRITICAL: Use ONLY These 18 Core Positions

You MUST classify every position using ONE of these 18 categories:

1. **guard** - Bottom athlete has opponent between legs
2. **mount** - Top athlete sits on opponent's torso
3. **side_control** - Top athlete chest-to-chest perpendicular from side
4. **back_control** - Top athlete behind with hooks or body triangle
5. **north_south** - Top athlete head-to-head, bodies opposite
6. **knee_on_belly** - Top athlete's knee on opponent's belly/chest
7. **turtle** - Bottom athlete on hands and knees defensively
8. **half_guard** - One leg trapped between opponent's legs
9. **standing** - Both athletes on feet
10. **clinch** - Standing with established grips
11. **takedown** - Active process bringing opponent to ground
12. **single_leg** - Attacking one leg
13. **double_leg** - Attacking both legs
14. **passing_guard** - Moving past legs to dominant position
15. **sweep** - Bottom reversing to gain top
16. **submission_attempt** - Applying finishing technique
17. **scramble** - Chaotic transition, neither has control
18. **unknown** - Cannot identify position

### Your Task:

**Create a Macro Position Timeline ONLY**
- Identify 10-15 MAJOR positions only (significant positions held 10+ seconds)
- Use ONLY the 18 core positions above
- Skip brief transitions - focus on stable, clear positions
- Include start_time (M:SS format), end_time, position, sub_position
- Identify top_athlete and bottom_athlete (names or athlete_1/athlete_2)
- Add brief notes (5-10 words max) per position
- NO athlete profiling or match analysis (handled in Pass 3)
- **Keep output under 6000 tokens** (8192 limit)

### Output Format:

```json
{{
  "position_timeline": [
    {{
      "start_time": "M:SS",
      "end_time": "M:SS",
      "position": "one of 18 positions",
      "sub_position": "specific variation or null",
      "top_athlete": "athlete_1 or athlete_2",
      "bottom_athlete": "athlete_1 or athlete_2 or null",
      "control_quality": 1-5,
      "confidence": 0.0-1.0,
      "notes": "brief 5-10 words"
    }}
  ]
}}
```

IMPORTANT:
- Output ONLY the position_timeline array
- Sample at 1 frame per 10-15 seconds
- ACCURATE position labeling using 18-position list
- Brief notes (5-10 words max) per position
- Every position will receive detailed analysis in Pass 2
```

---

## Pass 2: Detailed Position Analysis

```
You are performing DETAILED ANALYSIS on a SINGLE BJJ position.

This is Pass 2 of a 3-pass system. You analyze EACH position from Pass 1 in detail.

### Context from Pass 1:
{pass1_summary}

### Your Position:
- Start: {position_start}
- End: {position_end}
- Position (from Pass 1): {pass1_position}
- Duration: {duration} seconds
- Pass 1 Notes: {pass1_notes}

### Your Task:

**Break this into 3-5 sub-segments (3-15 seconds each)**

For EACH sub-segment:
1. **Technique Labels** - Apply 3-5 labels from 91-label vocabulary
2. **Dominance** - Who controls? (athlete_1/athlete_2/neutral)
3. **Dominance Score** - 1=athlete_1 control, 3=neutral, 5=athlete_2 control
4. **Execution Quality** - 1=poor, 3=average, 5=excellent
5. **Rich Commentary** - 2-3 sentences on technique, quality, tactics

### 91-Label Vocabulary:

**Guards:** closed_guard, open_guard, half_guard, butterfly_guard, de_la_riva, reverse_de_la_riva, x_guard, single_leg_x, spider_guard, lasso_guard, worm_guard, 50_50_guard, deep_half_guard, z_guard, knee_shield

**Passing:** knee_slice, leg_drag, over_under_pass, toreando_pass, stack_pass, bodylock_pass, headquarters_position, smash_pass

**Takedowns:** single_leg, double_leg, ankle_pick, high_crotch, duck_under, arm_drag, snap_down, foot_sweep, hip_toss, sacrifice_throw, guard_pull

**Sweeps:** butterfly_sweep, scissor_sweep, hip_bump, pendulum_sweep, flower_sweep, de_la_riva_sweep, x_guard_sweep, ankle_pick_sweep, sit_up_sweep

**Submissions:** rear_naked_choke, guillotine, triangle, armbar, kimura, americana, omoplata, heel_hook, ankle_lock, kneebar, toe_hold, calf_slicer, bicep_slicer, ezekiel_choke, darce, anaconda, loop_choke, bow_and_arrow

**Control:** mount, high_mount, s_mount, side_control, kesa_gatame, north_south, knee_on_belly, back_control, body_triangle, turtle, front_headlock

**Grips:** collar_tie, underhooks, overhooks, body_lock, seat_belt, harness, crossface, frames, grip_fighting, hand_fighting

**Escapes:** shrimping, bridging, granby_roll, technical_standup, turtle_to_guard, mount_escape, back_escape, scramble, reversal, retaining_guard

**Misc:** standing, clinch, scramble_reversal, match_start, match_end, reset, referee_intervention

### Output Format:

```json
{{
  "position_info": {{
    "start_time": "M:SS",
    "end_time": "M:SS",
    "duration": number,
    "pass1_position": "from context"
  }},
  "sub_segments": [
    {{
      "start_time": "M:SS",
      "end_time": "M:SS",
      "labels": ["technique1", "technique2", "technique3"],
      "dominance": "athlete_1/athlete_2/neutral",
      "dominance_score": 1-5,
      "execution_quality": 1-5,
      "commentary": "2-3 sentences describing technique, quality, tactics"
    }}
  ]
}}
```

CRITICAL:
- MUST create 3-5 sub-segments
- Each MUST have 3-5 technique labels
- Each MUST have 2-3 sentence commentary
- Use 91-label vocabulary
- Be specific (de_la_riva_guard not just guard)
```

---

## Pass 3: Synthesis with Meta-Analysis

```
You are synthesizing multi-pass BJJ analysis.

This is Pass 3 (final). You receive:
- Pass 1: Holistic context, 25-40 macro positions
- Pass 2: Detailed analysis for EVERY position (3-5 sub-segments each)

### Your Inputs:

**Pass 1:**
{pass1_output}

**Pass 2 (ALL positions):**
{pass2_outputs}

### Your Task:

1. **Unified Timeline**
   - Merge Pass 1 + Pass 2
   - Every position has 3-5 sub-segments attached
   - No gaps/overlaps
   - Temporal consistency

2. **Fighter Profiling**
   - Game style classification
   - Technique effectiveness (success/failure rates)
   - Signature moves

3. **Meta-Analysis** (Exceed Ground Truth)
   - Strategic patterns (recurring sequences, position cycles)
   - Fatigue effects (performance degradation)
   - Momentum analysis (turning points)
   - Missed opportunities
   - **Coaching Recommendations** (actionable advice, training priorities, strategic adjustments)

### Output Format:

```json
{{
  "unified_timeline": [
    {{
      "start_time": "M:SS",
      "end_time": "M:SS",
      "position": "18-position ontology",
      "sub_position": "specific",
      "top_athlete": "name",
      "bottom_athlete": "name",
      "notes": "position commentary",
      "sub_segments": [
        {{
          "start_time": "M:SS",
          "end_time": "M:SS",
          "labels": ["techniques"],
          "dominance": "athlete",
          "dominance_score": 1-5,
          "execution_quality": 1-5,
          "commentary": "detailed description"
        }}
      ],
      "source": "pass1_verified_by_pass2",
      "confidence": 0.0-1.0
    }}
  ],
  "fighter_profiles": {{
    "athlete_1": {{
      "game_style": {{
        "primary": "description",
        "characteristics": [],
        "confidence": 0.0-1.0
      }},
      "technique_effectiveness": {{
        "successful": [
          {{
            "technique": "name",
            "success_rate": 0.0-1.0,
            "count": number,
            "notes": "why it worked"
          }}
        ],
        "failed": [{{...}}],
        "signature_moves": []
      }},
      "strengths": [],
      "weaknesses": []
    }},
    "athlete_2": {{...}}
  }},
  "meta_analysis": {{
    "strategic_patterns": {{
      "recurring_sequences": [],
      "position_cycles": [],
      "attack_defense_rhythm": "description"
    }},
    "fatigue_analysis": {{
      "fatigue_detected": true/false,
      "onset_time": "M:SS",
      "effects_observed": [],
      "performance_decline": "description"
    }},
    "momentum_shifts": [{{}}],
    "missed_opportunities": [{{}}],
    "coaching_recommendations": {{
      "athlete_1": {{
        "immediate_fixes": ["tactical advice"],
        "training_priorities": ["what to drill"],
        "strategic_adjustments": ["game plan changes"],
        "technical_development": ["long-term skills"]
      }},
      "athlete_2": {{...}}
    }}
  }},
  "overall_assessment": {{
    "match_quality": "description",
    "skill_gap": "even/slight/significant",
    "winner_analysis": "who won and why",
    "key_deciding_factors": []
  }}
}}
```

IMPORTANT:
- Provide EVIDENCE for assessments
- Meta-analysis EXCEEDS human ground truth
- Coaching recommendations: specific and actionable
- Comprehensive final output
```
