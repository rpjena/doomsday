"""
Aggregator: Raw Score → Clock Time
===================================
Physics framing: Think of this as a partition function.
Each component contributes a Boltzmann-like weight to the total "energy"
of the system. The mapping to "minutes" is our order parameter —
the macroscopic observable that tells us which phase we're in.

Zones (phase diagram):
  Safe     [0, 4)   → 23–21 min   (ordered phase)
  Normal   [4, 6)   → 20–11 min   (meta-stable)
  Elevated [6, 8)   → 10–6  min   (critical region)
  Critical [8, 10]  → 5–0   min   (phase transition / breakdown)
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal


# Default weights — fund manager tunes these
DEFAULT_WEIGHTS = {
    "A": 0.35,   # Factor Exposure
    "B": 0.30,   # Correlation Breakdown
    "C": 0.25,   # Crowding & Liquidity
    "D": 0.10,   # Macro Shock
}

# Zone definitions: (raw_score_min, raw_score_max, minutes_max, minutes_min)
ZONES = [
    (0.0, 4.0,  23.0, 21.0, "SAFE",     "🟢"),
    (4.0, 6.0,  20.0, 11.0, "NORMAL",   "🟡"),
    (6.0, 8.0,  10.0,  6.0, "ELEVATED", "🟠"),
    (8.0, 10.0,  5.0,  0.0, "CRITICAL", "🔴"),
]


@dataclass
class ClockReading:
    minutes_to_midnight: float
    raw_score: float
    zone: str
    zone_emoji: str
    component_scores: dict
    component_weights: dict
    weighted_contributions: dict
    sub_details: dict


def map_score_to_minutes(raw_score: float) -> tuple[float, str, str]:
    """
    Piecewise linear interpolation from raw score [0,10] → minutes [0,23].

    Pseudocode:
      for each zone (score_lo, score_hi, min_hi, min_lo):
        if score_lo <= raw_score <= score_hi:
          t = (raw_score - score_lo) / (score_hi - score_lo)
          minutes = min_hi - t * (min_hi - min_lo)   # inverse: more score = less time
          return minutes, zone_label
    """
    raw_score = float(np.clip(raw_score, 0.0, 10.0))
    for s_lo, s_hi, m_hi, m_lo, label, emoji in ZONES:
        if s_lo <= raw_score <= s_hi or (raw_score > 9.99 and label == "CRITICAL"):
            t = (raw_score - s_lo) / max(s_hi - s_lo, 1e-9)
            minutes = m_hi - t * (m_hi - m_lo)
            return round(minutes, 2), label, emoji
    return 0.0, "CRITICAL", "🔴"


def compute_clock(
    score_a: float,
    score_b: float,
    score_c: float,
    score_d: float,
    details_a: dict,
    details_b: dict,
    details_c: dict,
    details_d: dict,
    weights: dict = None,
) -> ClockReading:
    """
    Main aggregation function.

    Pseudocode:
      raw_score = Σ weight_i * score_i   (dot product)
      minutes, zone = piecewise_linear_map(raw_score)
      return ClockReading(...)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    scores = {"A": score_a, "B": score_b, "C": score_c, "D": score_d}
    weighted = {k: round(weights[k] * scores[k], 3) for k in scores}
    raw_score = sum(weighted.values())

    minutes, zone, emoji = map_score_to_minutes(raw_score)

    return ClockReading(
        minutes_to_midnight=minutes,
        raw_score=round(raw_score, 3),
        zone=zone,
        zone_emoji=emoji,
        component_scores=scores,
        component_weights=weights,
        weighted_contributions=weighted,
        sub_details={
            "A": details_a,
            "B": details_b,
            "C": details_c,
            "D": details_d,
        },
    )


def print_clock_report(reading: ClockReading):
    """Pretty-print the clock state to console."""
    mins = reading.minutes_to_midnight
    h = int(23 - mins // 60) if mins >= 60 else 23
    m = int(23 * 60 + 59 - mins) % 60  # always 11:XX PM equivalent

    print("=" * 60)
    print(f"  {reading.zone_emoji}  DOOMSDAY CLOCK — EMN RISK MONITOR")
    print("=" * 60)
    print(f"  Time to Midnight : {reading.minutes_to_midnight:.1f} minutes")
    print(f"  Clock Position   : 11:{int(60 - reading.minutes_to_midnight):02d} PM" if reading.minutes_to_midnight < 60 else "  Clock Position  : <1 hr to midnight")
    print(f"  Zone             : {reading.zone_emoji} {reading.zone}")
    print(f"  Raw Risk Score   : {reading.raw_score:.2f} / 10.0")
    print("-" * 60)
    print("  Component Scores (normalized 0–10):")
    labels = {
        "A": "Factor Exposure     (Quants Hand)",
        "B": "Correlation Breakdown (Structural)",
        "C": "Crowding & Liquidity (Mkt Struct.)",
        "D": "Macro Shock          (External)  ",
    }
    for k, label in labels.items():
        s = reading.component_scores[k]
        bar = "█" * int(s) + "░" * (10 - int(s))
        contrib = reading.weighted_contributions[k]
        print(f"  {label}  [{bar}] {s:5.2f}  → weighted {contrib:.3f}")
    print("=" * 60)
