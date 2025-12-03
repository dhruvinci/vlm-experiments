"""
BJJ Analysis Ontology

Comprehensive taxonomy for labeling BJJ positions, techniques, and transitions.
"""

BJJ_ONTOLOGY = {
    "positions": {
        "standing": {
            "label": "Standing",
            "sub_positions": [
                "neutral_stance",
                "grip_fighting",
                "collar_tie",
                "underhooks",
                "overhooks",
                "body_lock",
                "front_headlock",
                "pre_match",
                "reset"
            ]
        },
        "guard": {
            "label": "Guard",
            "sub_positions": [
                "closed_guard",
                "open_guard",
                "butterfly_guard",
                "spider_guard",
                "de_la_riva",
                "reverse_de_la_riva",
                "lasso_guard",
                "x_guard",
                "single_leg_x",
                "50_50_guard",
                "deep_half_guard"
            ]
        },
        "half_guard": {
            "label": "Half Guard",
            "sub_positions": [
                "half_guard_top",
                "half_guard_bottom",
                "deep_half",
                "knee_shield",
                "lockdown",
                "z_guard"
            ]
        },
        "mount": {
            "label": "Mount",
            "sub_positions": [
                "full_mount",
                "high_mount",
                "low_mount",
                "s_mount",
                "technical_mount"
            ]
        },
        "side_control": {
            "label": "Side Control",
            "sub_positions": [
                "standard_side_control",
                "reverse_side_control",
                "kesa_gatame",
                "modified_kesa_gatame",
                "gift_wrap"
            ]
        },
        "back_control": {
            "label": "Back Control",
            "sub_positions": [
                "hooks_in",
                "body_triangle",
                "one_hook_in",
                "seat_belt",
                "rear_mount"
            ]
        },
        "turtle": {
            "label": "Turtle",
            "sub_positions": [
                "defensive_turtle",
                "aggressive_turtle",
                "quarter_guard"
            ]
        },
        "north_south": {
            "label": "North South",
            "sub_positions": [
                "standard_north_south",
                "reverse_north_south"
            ]
        },
        "knee_on_belly": {
            "label": "Knee on Belly",
            "sub_positions": [
                "standard_knee_on_belly",
                "reverse_knee_on_belly"
            ]
        },
        "transitional": {
            "label": "Transitional",
            "sub_positions": [
                "scramble",
                "mid_transition",
                "unstable_position"
            ]
        }
    },

    "techniques": {
        "takedowns": [
            "single_leg",
            "double_leg",
            "ankle_pick",
            "high_crotch",
            "dump",
            "trip",
            "throw",
            "sacrifice_throw",
            "snap_down",
            "pull_guard"
        ],
        "sweeps": [
            "scissor_sweep",
            "hip_bump_sweep",
            "butterfly_sweep",
            "x_guard_sweep",
            "de_la_riva_sweep",
            "technical_standup"
        ],
        "passes": [
            "toreando_pass",
            "knee_slice",
            "leg_drag",
            "over_under_pass",
            "x_pass",
            "stack_pass",
            "long_step_pass",
            "headquarters_position"
        ],
        "submissions": [
            "rear_naked_choke",
            "armbar",
            "triangle",
            "guillotine",
            "kimura",
            "americana",
            "ezekiel",
            "bow_and_arrow",
            "heel_hook",
            "knee_bar",
            "toe_hold",
            "ankle_lock",
            "darce",
            "anaconda",
            "arm_triangle",
            "crucifix"
        ],
        "escapes": [
            "bridge_and_roll",
            "elbow_escape",
            "granby_roll",
            "sit_up_escape",
            "technical_standup"
        ],
        "reversals": [
            "sweep",
            "reversal",
            "scramble_reversal"
        ]
    },

    "transitions": [
        "standing_to_guard",
        "standing_to_mount",
        "standing_to_side_control",
        "standing_to_back_control",
        "guard_to_mount",
        "guard_to_side_control",
        "guard_to_back_control",
        "guard_to_half_guard",
        "half_guard_to_mount",
        "half_guard_to_side_control",
        "half_guard_to_back_control",
        "half_guard_to_guard",
        "side_control_to_mount",
        "side_control_to_north_south",
        "side_control_to_knee_on_belly",
        "mount_to_back_control",
        "back_control_to_mount",
        "turtle_to_back_control",
        "turtle_to_side_control"
    ],

    "scoring_events": [
        "takedown",
        "guard_pass",
        "knee_on_belly",
        "mount",
        "back_control",
        "sweep",
        "advantage",
        "penalty"
    ],

    "match_events": [
        "match_start",
        "match_end",
        "handshake",
        "bow",
        "referee_stop",
        "referee_restart",
        "out_of_bounds",
        "submission_attempt",
        "submission_success",
        "points_scored"
    ],

    "athlete_actions": [
        "attacking",
        "defending",
        "passing",
        "retaining_guard",
        "escaping",
        "controlling",
        "stalling",
        "grip_fighting"
    ]
}

# Rating scale definitions
RATING_SCALE = {
    0: {"label": "Completely Wrong", "description": "All aspects incorrect - position, timing, athletes"},
    1: {"label": "Mostly Wrong", "description": "Major errors in position classification or timing"},
    2: {"label": "Partially Wrong", "description": "Position correct but sub-position/details wrong"},
    3: {"label": "Mostly Correct", "description": "Position and timing correct, minor details missing"},
    4: {"label": "Almost Perfect", "description": "All major aspects correct, trivial details off"},
    5: {"label": "Perfect", "description": "Complete accuracy in all aspects"}
}

def get_all_labels():
    """Get flat list of all possible labels for autocomplete."""
    labels = []

    # Add positions and sub-positions
    for pos_key, pos_data in BJJ_ONTOLOGY["positions"].items():
        labels.append(pos_data["label"])
        labels.extend(pos_data["sub_positions"])

    # Add techniques
    for tech_category, tech_list in BJJ_ONTOLOGY["techniques"].items():
        labels.extend(tech_list)

    # Add other categories
    labels.extend(BJJ_ONTOLOGY["transitions"])
    labels.extend(BJJ_ONTOLOGY["scoring_events"])
    labels.extend(BJJ_ONTOLOGY["match_events"])
    labels.extend(BJJ_ONTOLOGY["athlete_actions"])

    return sorted(set(labels))

def get_position_hierarchy():
    """Get positions organized by hierarchy for dropdown."""
    hierarchy = []
    for pos_key, pos_data in BJJ_ONTOLOGY["positions"].items():
        hierarchy.append({
            "position": pos_data["label"],
            "key": pos_key,
            "sub_positions": pos_data["sub_positions"]
        })
    return hierarchy
