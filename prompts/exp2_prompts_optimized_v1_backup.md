# Experiment 2: Optimized Multi-Pass Prompts

## Pass 1: Holistic Context (Markdown Output)

```
Analyze this BJJ match video. Identify 10-15 major positions (10+ seconds each).

Output as markdown table:

| Time | Pos | Sub | Top | Bot | Ctrl | Conf | Notes |
|------|-----|-----|-----|-----|------|------|-------|
| M:SS-M:SS | position | sub_position | ath# | ath# | 1-5 | 0-1 | Description (max detail, rich commentary on what's happening) |

**Athlete IDs (identify once):**
- Athlete 1: [Physical description - shorts color, distinguishing features]
- Athlete 2: [Physical description]

**Instructions:**
- Pos: Use 18-position ontology (guard, mount, side_control, back_control, etc.)
- Top/Bot: athlete_1 or athlete_2 (or "both" for neutral)
- Ctrl: Control quality (1=poor, 5=dominant)
- Conf: Confidence (0.0-1.0)
- Notes: MAX DETAIL - technique attempts, grips, body positioning, momentum shifts, key moments with timestamps

**Focus on content quality.** Use full 43K token budget for rich descriptions.
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
