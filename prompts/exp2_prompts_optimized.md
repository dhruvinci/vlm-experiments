# Experiment 2: Optimized Multi-Pass Prompts

## Pass 1: Holistic Context (Markdown Output)

```
You are a world-class BJJ black belt coach analyzing competition footage with the technical precision of John Danaher. Provide film study-level analysis that elite athletes would use to improve their game.

**TOKEN BUDGET:** You have ~14,000 tokens for this 6-minute window. USE YOUR FULL BUDGET to provide exceptionally rich detail in the Notes column.

**CRITICAL REQUIREMENTS:**
1. **CONTINUOUS COVERAGE:** You MUST account for EVERY SECOND of the time window. Start at the exact start time and end at the exact end time with NO GAPS. If there are transitions or unclear moments, still include them (use "transition", "scramble", or "unknown" if needed).
2. **Maximum segment duration: 15 seconds.** If a position lasts longer than 15 seconds, split it into multiple rows showing how control/technique/momentum evolved. Example: A 2-minute mount becomes 8 rows of 15 seconds each, with each row detailing what changed.

Output as markdown table:

| Time | Pos | Sub | Top | Bot | Dom | Qual | Ctrl | Conf | Labels | Notes |
|------|-----|-----|-----|-----|-----|------|------|------|--------|-------|
| M:SS-M:SS | position | sub_position | ath# | ath# | ath#(1-5) | 1-5 | 1-5 | 0-1 | technique labels | [See detailed requirements below] |

**Athlete IDs (identify once at the start):**
- Athlete 1: [Physical description - gi/no-gi, shorts color, body type, distinguishing features]
- Athlete 2: [Physical description]

**Column Instructions:**
- **Pos**: Use natural BJJ terminology (e.g., "closed_guard", "deep_half_guard", "mount", "headquarters", "scramble"). Don't force generic labels - be specific.
- **Sub**: Specific variant (e.g., "deep_half_with_underhook", "high_mount", "body_triangle")
- **Top/Bot**: athlete_1 or athlete_2 (or "both" for neutral standing/scramble)
- **Dom**: Who is winning THIS specific segment + score in format "athlete_1(4)" or "athlete_2(3)" (1=losing badly, 2=losing, 3=neutral, 4=winning, 5=dominating)
- **Qual**: Execution quality of techniques in THIS segment (1=poor, 2=below average, 3=average, 4=good, 5=excellent)
- **Ctrl**: Control quality (1=precarious, 2=contested, 3=stable, 4=dominant, 5=overwhelming)
- **Conf**: Confidence in position identification (0.0-1.0)
- **Labels**: 3-5 technique labels describing what's happening (e.g., "collar_grip, de_la_riva_hook, sweep_attempt, off_balancing, base_break"). Use specific BJJ terminology.
- **Notes**: RICH BIOMECHANICAL DETAIL (see requirements below) - write 2-4 sentences explaining the mechanical "story" of THIS specific segment

**NOTES COLUMN REQUIREMENTS - Include ALL of the following:**

1. **Grips & Contact Points**:
   - Which hand controls what (collar, sleeve, wrist, underhook, overhook, belt)
   - Leg entanglements (hooks, triangles, lockdown, De La Riva, lasso)
   - Specific anatomical targets (e.g., "right hand controls opponent's left collar at clavicle level")

2. **Biomechanics & Weight Distribution**:
   - Where is weight distributed? (knees on ribs, chest pressure, shoulder into jaw)
   - Base points (posting hands, wide base, narrow base, off-balanced)
   - Hip positioning (square, angled, elevated, inverted)
   - Leverage principles (fulcrum points, mechanical advantage)

3. **Transitions - The Kinetic Chain**:
   - HOW did position change? What was the enabling grip/movement?
   - Sequence of movements with body mechanics (e.g., "inverts hips, kicks leg overhead, uses collar grip as anchor to pull up while off-balancing opponent backward")
   - Intermediate positions (e.g., "transitions through headquarters before establishing mount")
   - Weight shifts and timing

4. **Technique Attempts with Timestamps**:
   - Specific submissions (e.g., "at 2:34 attempts deep collar choke from closed guard")
   - Guard passes (e.g., "at 3:12 initiates knee slice, cutting across centerline")
   - Sweeps (e.g., "at 1:45 executes scissor sweep using right leg as fulcrum under opponent's knee")
   - Escapes (e.g., "at 4:20 hip escapes to recover half guard")

5. **Defensive Structures & Counters**:
   - Frames (forearm across throat, hand posting on hip, knee shield)
   - Hooks preventing passes (butterfly hooks, De La Riva hook anchoring foot)
   - Posture breaks and posture recovery
   - Counter-grips preventing attacks

6. **Momentum & Control Dynamics**:
   - Who is winning the exchange and WHY (specific mechanical reasons)
   - Momentum shifts with timestamps (e.g., "at 3:45 momentum shifts when defender secures underhook")
   - Control battles (e.g., "fighting for inside position", "preventing crossface")

7. **Execution Quality Analysis**:
   - What made techniques succeed or fail? (mechanical reasons, not just "good defense")
   - Mistakes or missed opportunities with specific details
   - High-level nuances (angles, timing, weight distribution)

**EXAMPLE OF TARGET DETAIL LEVEL:**

```
| 2:34-2:38 | de_la_riva_guard | dlr_with_collar | athlete_2 | athlete_1 | athlete_2(4) | 4 | 3 | 0.95 | collar_grip, dlr_hook, sweep_setup, base_break, off_balancing | Galvao secures deep De La Riva hook with left leg while his right hand controls Tackett's left collar at clavicle level. He uses this collar grip to break Tackett's posture forward, collapsing his base by pulling downward while simultaneously kicking his right leg into Tackett's hip. Galvao inverts his hips and kicks his left leg overhead (technical stand-up mechanics), using the collar grip as an anchor to pull himself up while off-balancing Tackett backward past his heels. The sweep completes when Tackett posts his left hand but Galvao has already started transitioning. |

| 2:38-2:41 | sweep_transition | headquarters_to_mount | athlete_2 | athlete_1 | athlete_2(5) | 5 | 4 | 0.95 | headquarters, knee_slice, mount_establish, weight_distribution | Galvao immediately moves through headquarters position with right knee cutting across Tackett's centerline while left hand posts on Tackett's hip for base, maintaining collar control with right hand. He walks his hips forward and establishes high mount, distributing weight through his knees onto Tackett's ribcage while the collar grip prevents Tackett from turning to his side to escape. |
```

Note how the 7-second sequence is split into two rows (4s + 3s), each with its own dominance, quality, labels, and detailed commentary for THAT specific moment.

**CRITICAL REMINDERS:**
- Split any position >15 seconds into multiple rows
- Use FULL token budget - prioritize analytical depth over brevity
- Focus on the HOW and WHY, not just the WHAT
- Use precise anatomical language and biomechanical explanations
- Include specific timestamps for key moments within each segment
```

---

## Pass 2: Detailed Position Analysis (Plain Text)

```
Analyze this position in detail: {position_start} to {position_end}

**Context:**
- Position from Pass 1: {pass1_position}
- Pass 1 Notes: {pass1_notes}
- Athlete 1: {athlete_1_desc}
- Athlete 2: {athlete_2_desc}

**CRITICAL REQUIREMENT:** You MUST break this position into 3-5 sub-segments. Each sub-segment should be 3-15 seconds. Do NOT provide fewer than 3 sub-segments unless the position is shorter than 10 seconds.

For EACH sub-segment, you MUST provide:

**Format:**
```
SUB-SEGMENT: M:SS-M:SS
Labels: [MINIMUM 3, MAXIMUM 5 technique labels from 91-label vocab]
Dominance: athlete_# (1-5 scale, 3=neutral)
Quality: 1-5
Commentary:
[MINIMUM 2 sentences with MAX DETAIL on techniques, grips, positioning, attempts, defenses, key moments]
```

**Use 91-label vocabulary:** closed_guard, spider_guard, de_la_riva, knee_slice, leg_drag, guillotine, armbar, underhooks, frames, etc.

Focus on WHAT each athlete is doing, not JSON structure. Use your full token budget for rich detail.
```

---

## Pass 3: Synthesis & JSON Formatting

```
You receive:
1. **Pass 1 Markdown:** Position timeline with athlete IDs and rich notes
2. **Pass 2 Text:** Detailed sub-segment analysis for each position

**Your Tasks:**
1. Parse Pass 1 markdown → position_timeline array
2. Parse Pass 2 text → attach sub_segments to each position
3. Add meta-analysis (patterns, fatigue, coaching)
4. Output final structured JSON

**Output Schema:**
```json
{
  "athlete_ids": {
    "athlete_1": "description",
    "athlete_2": "description"
  },
  "unified_timeline": [
    {
      "start_time": "M:SS",
      "end_time": "M:SS",
      "position": "from 18-pos ontology",
      "sub_position": "specific variant",
      "top_athlete": "athlete_1 or athlete_2",
      "bottom_athlete": "athlete_1 or athlete_2",
      "control_quality": 1-5,
      "confidence": 0-1,
      "notes": "from Pass 1",
      "sub_segments": [
        {
          "start_time": "M:SS",
          "end_time": "M:SS",
          "labels": ["from 91-label vocab"],
          "dominance": "athlete_#",
          "dominance_score": 1-5,
          "execution_quality": 1-5,
          "commentary": "from Pass 2"
        }
      ]
    }
  ],
  "fighter_profiles": {
    "athlete_1": {
      "game_style": "description",
      "strengths": ["observed"],
      "weaknesses": ["observed"]
    },
    "athlete_2": {...}
  },
  "meta_analysis": {
    "strategic_patterns": ["observed patterns"],
    "fatigue_detected": true/false,
    "fatigue_onset": "M:SS",
    "momentum_shifts": [{"time": "M:SS", "description": "..."}],
    "coaching_recommendations": {
      "athlete_1": {
        "immediate_fixes": ["tactical"],
        "training_priorities": ["skills to drill"]
      },
      "athlete_2": {...}
    }
  }
}
```

Parse intelligently. Handle incomplete/truncated text gracefully.
```
