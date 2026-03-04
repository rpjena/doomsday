"""
Component A: Factor Exposure Risk  — "The Quants' Hand"
========================================================
Physics framing: Think of this as measuring the "potential energy" stored
in concentrated factor bets. Like a spring compressed along one axis —
when it snaps back (factor reversal), kinetic energy is violent and sudden.

Metric: Herfindahl-Hirschman Index (HHI) of squared exposures.
  HHI = Σ e_i²

HHI = 1 means 100% concentration in one factor (maximum danger).
HHI → 0 means risk perfectly dispersed across all factors (safe).
"""

import numpy as np
import pandas as pd


# Historical calibration anchors
# These represent "what HHI looked like just before known quant crises"
# Aug 2007 Quant Meltdown: HHI ~0.45 (extreme momentum crowding)
# COVID shock 2020: HHI ~0.35
HISTORICAL_MAX_HHI = 0.50   # upper anchor → score of 10
HISTORICAL_MIN_HHI = 0.02   # lower anchor → score of 0


def compute_factor_hhi(factor_exposures: pd.Series) -> float:
    """
    Herfindahl concentration of factor loadings.

    Parameters
    ----------
    factor_exposures : pd.Series
        Portfolio's net factor loadings, e.g. from a Barra/Axioma model.
        Index = factor names, values = exposure (signed float).

    Returns
    -------
    hhi : float in [0, 1]
    """
    e = factor_exposures.values
    hhi = float(np.sum(e ** 2))
    return hhi


def normalize_component_a(
    factor_exposures: pd.Series,
    hist_max: float = HISTORICAL_MAX_HHI,
    hist_min: float = HISTORICAL_MIN_HHI,
) -> dict:
    """
    Compute Component A normalized score (0–10).

    Pseudocode:
      hhi = Σ e_i²
      score = clip((hhi - min) / (max - min) * 10, 0, 10)

    Returns dict with raw HHI, per-factor breakdown, and normalized score.
    """
    hhi = compute_factor_hhi(factor_exposures)
    score = float(np.clip((hhi - hist_min) / (hist_max - hist_min) * 10, 0.0, 10.0))

    # Per-factor contribution: how much does each factor "own" of the HHI?
    e2 = factor_exposures ** 2
    contributions = (e2 / e2.sum() * 10).round(2) if e2.sum() > 0 else e2 * 0

    return {
        "raw_hhi": round(hhi, 4),
        "normalized_score": round(score, 2),
        "factor_exposures": factor_exposures.round(4).to_dict(),
        "factor_contributions": contributions.to_dict(),
        "max_exposure_factor": factor_exposures.abs().idxmax(),
        "max_exposure_value": round(float(factor_exposures.abs().max()), 4),
    }
