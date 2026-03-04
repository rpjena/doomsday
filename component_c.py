"""
Component C: Crowding & Liquidity Risk — "The Market Structure Hand"
=====================================================================
Physics framing: Crowding is like increasing the density of a gas in a box.
At some critical density (pressure), small perturbations cause explosive
decompression — the "quant meltdown." Liquidity is the size of the escape valve.
When both are bad simultaneously, the system is critically unstable.

Three sub-signals:
  1. Portfolio crowding percentile (how popular are our positions)
  2. Portfolio liquidity score (bid-ask, volume, days-to-unwind)
  3. Market-wide liquidity indicator (VIX term structure, IG spread)

August 2007 analog: Everyone was long value/short momentum. When one fund
deleveraged, correlation spiked, others were forced to sell → cascade.
"""

import numpy as np
import pandas as pd


# Calibration anchors (pre-crisis peaks)
HIST_MAX_PORTFOLIO_CROWDING = 8.5
HIST_MAX_PORTFOLIO_LIQUIDITY = 7.5
HIST_MAX_MARKET_LIQ = 9.0

# VIX and IG spread calibration for market liquidity sub-score
VIX_LOW = 12.0    # benign regime
VIX_HIGH = 45.0   # crisis (COVID peak ~82, but practical EMN trigger ~45)
IG_SPREAD_LOW = 70.0    # bps, benign
IG_SPREAD_HIGH = 250.0  # bps, crisis


def market_liquidity_score(vix: float, ig_spread: float) -> float:
    """
    Composite market-wide liquidity score (0–10).
    Pseudocode:
      vix_score  = clip((vix - VIX_LOW) / (VIX_HIGH - VIX_LOW) * 10, 0, 10)
      ig_score   = clip((ig - IG_LOW) / (IG_HIGH - IG_LOW) * 10, 0, 10)
      market_liq = 0.6 * vix_score + 0.4 * ig_score
    """
    vix_score = float(np.clip((vix - VIX_LOW) / (VIX_HIGH - VIX_LOW) * 10, 0, 10))
    ig_score = float(np.clip((ig_spread - IG_SPREAD_LOW) / (IG_SPREAD_HIGH - IG_SPREAD_LOW) * 10, 0, 10))
    return 0.6 * vix_score + 0.4 * ig_score


def normalize_component_c(
    crowding_score: float,
    liquidity_score: float,
    vix_current: float,
    ig_spread_current: float,
    hist_max_crowding: float = HIST_MAX_PORTFOLIO_CROWDING,
    hist_max_liquidity: float = HIST_MAX_PORTFOLIO_LIQUIDITY,
) -> dict:
    """
    Compute Component C normalized score (0–10).

    Pseudocode:
      norm_crowding  = clip(crowding_score / hist_max * 10, 0, 10)
      norm_liquidity = clip(liquidity_score / hist_max * 10, 0, 10)
      mkt_liq        = market_liquidity_score(vix, ig_spread)
      final_score    = 0.35 * norm_crowding + 0.35 * norm_liquidity + 0.30 * mkt_liq
    """
    norm_crowding = float(np.clip(crowding_score / hist_max_crowding * 10, 0, 10))
    norm_liquidity = float(np.clip(liquidity_score / hist_max_liquidity * 10, 0, 10))
    mkt_liq = market_liquidity_score(vix_current, ig_spread_current)

    final_score = round(0.35 * norm_crowding + 0.35 * norm_liquidity + 0.30 * mkt_liq, 2)

    return {
        "portfolio_crowding_raw": round(crowding_score, 3),
        "portfolio_liquidity_raw": round(liquidity_score, 3),
        "vix_current": round(vix_current, 2),
        "ig_spread_current": round(ig_spread_current, 2),
        "score_crowding": round(norm_crowding, 2),
        "score_portfolio_liquidity": round(norm_liquidity, 2),
        "score_market_liquidity": round(mkt_liq, 2),
        "normalized_score": final_score,
    }
