"""
Component D: Macro Shock Risk — "The External Hand"
====================================================
Physics framing: This is the external field applied to the system. Even a
well-constructed EMN portfolio is a finite-size system embedded in a larger
bath. When the bath temperature (macro volatility) spikes violently, no
amount of internal hedging saves you from thermalization.

Three sub-signals:
  1. Volatility regime: Hidden Markov Model (2-state) on realized vol
     State 0 = calm (low vol), State 1 = crisis (high vol)
  2. VIX percentile rank vs. 1-year history (fear gauge)
  3. Economic Surprise proxy (deviation of rolling market return from trend)

No external data feeds needed — we use portfolio + VIX series.
"""

import numpy as np
import pandas as pd
from scipy import stats


# Calibration anchors
VIX_1YR_LOW = 12.0
VIX_1YR_HIGH = 60.0


def detect_volatility_regime(
    returns: pd.Series,
    window: int = 21,
    lookback: int = 252,
    crisis_vol_threshold_pct: float = 0.80,
) -> dict:
    """
    Simple regime detector: compare current realized vol to its 1-year distribution.

    Pseudocode:
      rv_series = rolling_std(returns, 21) * sqrt(252)   # annualized
      current_rv_pct = percentile_rank(rv[-252:], rv_current)
      regime_score = clip(current_rv_pct * 10, 0, 10)

    A proper HMM (statsmodels MarkovAutoregression) requires more data
    and fitting time — swapped here for a robust percentile approach
    that is equally interpretable and stable on synthetic data.
    """
    rv_series = returns.rolling(window).std() * np.sqrt(252)
    rv_series = rv_series.dropna()

    current_rv = float(rv_series.iloc[-1])
    rv_history = rv_series.iloc[-lookback:]
    pct_rank = float(stats.percentileofscore(rv_history, current_rv)) / 100.0

    regime_label = "HIGH_VOL" if pct_rank >= crisis_vol_threshold_pct else "LOW_VOL"
    regime_score = float(np.clip(pct_rank * 10, 0, 10))

    return {
        "current_annualized_vol": round(current_rv, 4),
        "vol_percentile": round(pct_rank, 4),
        "regime": regime_label,
        "regime_score": round(regime_score, 2),
        "_rv_series": rv_series,
    }


def vix_percentile_score(vix_series: pd.Series, lookback: int = 252) -> dict:
    """
    VIX current level vs. its 1-year range → 0-10 score.

    Pseudocode:
      vix_pct = percentile_rank(vix[-252:], vix_current)
      score   = clip(vix_pct * 10, 0, 10)
    """
    current_vix = float(vix_series.iloc[-1])
    vix_history = vix_series.iloc[-lookback:]
    pct = float(stats.percentileofscore(vix_history, current_vix)) / 100.0
    score = float(np.clip(pct * 10, 0, 10))
    return {
        "current_vix": round(current_vix, 2),
        "vix_1yr_percentile": round(pct, 4),
        "vix_score": round(score, 2),
    }


def economic_surprise_score(
    market_returns: pd.Series,
    short_window: int = 10,
    long_window: int = 63,
) -> dict:
    """
    Economic surprise proxy: deviation of short-term market momentum from
    its medium-term trend. Strongly negative = bad surprises.

    Pseudocode:
      short_ret = sum(market_returns[-10d])
      trend_ret = mean of rolling 10d sums over last 63d
      surprise  = short_ret - trend_ret
      score     = clip((|surprise| - threshold) / range * 10, 0, 10)
    """
    rolling_sum = market_returns.rolling(short_window).sum()
    trend = rolling_sum.iloc[-long_window:].mean()
    current = float(rolling_sum.iloc[-1])
    surprise = current - float(trend)

    # Negative surprise is more dangerous for EMN than positive
    danger = max(0, -surprise)  # only penalize negative surprises
    score = float(np.clip(danger / 0.04 * 10, 0, 10))  # 4% cumulative drop → score 10

    return {
        "surprise_proxy": round(surprise, 5),
        "surprise_score": round(score, 2),
    }


def normalize_component_d(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    vix_series: pd.Series,
) -> dict:
    """
    Compute Component D normalized score (0–10).

    Pseudocode:
      regime_result   = detect_vol_regime(returns)
      vix_result      = vix_percentile_score(vix)
      surprise_result = economic_surprise_score(market_returns)
      final_score     = 0.40 * regime_score + 0.40 * vix_score + 0.20 * surprise_score
    """
    regime = detect_volatility_regime(portfolio_returns)
    vix_res = vix_percentile_score(vix_series)
    surprise = economic_surprise_score(market_returns)

    final_score = round(
        0.40 * regime["regime_score"]
        + 0.40 * vix_res["vix_score"]
        + 0.20 * surprise["surprise_score"],
        2,
    )

    return {
        "regime": regime["regime"],
        "current_annualized_vol": regime["current_annualized_vol"],
        "vol_percentile": regime["vol_percentile"],
        "regime_score": regime["regime_score"],
        "current_vix": vix_res["current_vix"],
        "vix_percentile": vix_res["vix_1yr_percentile"],
        "vix_score": vix_res["vix_score"],
        "surprise_proxy": surprise["surprise_proxy"],
        "surprise_score": surprise["surprise_score"],
        "normalized_score": final_score,
        "_rv_series": regime["_rv_series"],
    }
