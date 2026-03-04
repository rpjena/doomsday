"""
Component B: Correlation Breakdown Risk — "The Structural Hand"
===============================================================
Physics framing: EMN is like a balanced dipole — equal and opposite charges.
Rolling correlation is the "alignment" of the dipole. When correlation drifts
toward zero, the dipole collapses into a monopole: a naked directional bet.

Two signals:
  1. Long/Short rolling correlation deviation from 1-yr mean
  2. Portfolio rolling beta to market (should be ~0)

Metric: z-score-like deviation of correlation from its 1-year mean,
combined with the absolute rolling beta.
"""

import numpy as np
import pandas as pd


# Calibration: large correlation shifts observed in historical crises
# Aug 2007: L/S corr swung from -0.65 to +0.10 overnight (Δ = 0.75)
# COVID: Δ ≈ 0.55 over 5 days
HISTORICAL_MAX_CORR_DEV = 0.80   # → score 10
HISTORICAL_MAX_BETA = 0.40       # portfolio beta of 0.40 → score 10


def rolling_long_short_correlation(
    long_returns: pd.Series,
    short_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    60-day rolling Pearson correlation between long book and short book.
    Ideal EMN: correlation hovers around -0.7 to -0.5.
    Red flag: drifts toward 0 or positive.
    """
    return long_returns.rolling(window).corr(short_returns)


def rolling_market_beta(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Rolling OLS beta: β = Cov(portfolio, market) / Var(market)
    Target: |β| ≈ 0. Drift away from zero = neutrality breakdown.
    """
    cov = portfolio_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def normalize_component_b(
    long_returns: pd.Series,
    short_returns: pd.Series,
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
    lookback_days: int = 252,
    hist_max_corr_dev: float = HISTORICAL_MAX_CORR_DEV,
    hist_max_beta: float = HISTORICAL_MAX_BETA,
) -> dict:
    """
    Compute Component B normalized score (0–10).

    Pseudocode:
      rolling_corr     = rolling(60d).corr(long, short)
      baseline_corr    = mean(rolling_corr[-252:])
      corr_deviation   = |current_corr - baseline_corr|
      score_corr       = clip(corr_deviation / max_dev * 10, 0, 10)

      rolling_beta     = cov(port, mkt) / var(mkt) over 60d
      score_beta       = clip(|current_beta| / max_beta * 10, 0, 10)

      final_score      = 0.6 * score_corr + 0.4 * score_beta
    """
    corr_series = rolling_long_short_correlation(long_returns, short_returns, window)
    beta_series = rolling_market_beta(portfolio_returns, market_returns, window)

    current_corr = float(corr_series.iloc[-1])
    baseline_corr = float(corr_series.iloc[-lookback_days:].mean())
    corr_dev = abs(current_corr - baseline_corr)

    current_beta = float(beta_series.iloc[-1])

    score_corr = float(np.clip(corr_dev / hist_max_corr_dev * 10, 0, 10))
    score_beta = float(np.clip(abs(current_beta) / hist_max_beta * 10, 0, 10))
    final_score = round(0.6 * score_corr + 0.4 * score_beta, 2)

    return {
        "current_long_short_corr": round(current_corr, 4),
        "baseline_corr_1yr": round(baseline_corr, 4),
        "corr_deviation": round(corr_dev, 4),
        "current_beta": round(current_beta, 4),
        "score_corr_breakdown": round(score_corr, 2),
        "score_beta_drift": round(score_beta, 2),
        "normalized_score": final_score,
        # Expose full series for notebook plots
        "_corr_series": corr_series,
        "_beta_series": beta_series,
    }
