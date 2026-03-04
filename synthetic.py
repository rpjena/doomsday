"""
Synthetic Data Generator
========================
Physics framing: We're generating a fake "universe" of market observables.
Think of this as a Monte Carlo sampler for a 5-factor equity world.
In production, replace with Bloomberg / FactSet / internal OMS feeds.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_market_data(n_days: int = 504, seed: int = 42) -> dict:
    """
    Generate 2 years of synthetic daily market data.

    Returns a dict of DataFrames:
      - factor_returns   : (n_days x 6) daily factor return matrix
      - portfolio_returns: (n_days,)    daily net portfolio P&L
      - long_returns     : (n_days,)    long book daily returns
      - short_returns    : (n_days,)    short book daily returns
      - market_returns   : (n_days,)    S&P 500 proxy returns
      - vix              : (n_days,)    VIX level
      - ig_spread        : (n_days,)    IG credit spread (bps)
      - factor_exposures : (6,)         current portfolio factor loadings
      - crowding_scores  : (n_days,)    average crowding percentile of positions
      - liquidity_scores : (n_days,)    average liquidity score (0=liquid, 10=illiquid)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=n_days)

    # --- Factor returns: correlated 6-factor model (Value, Mom, Size, Vol, Growth, Quality)
    factor_names = ["Value", "Momentum", "Size", "Volatility", "Growth", "Quality"]
    cov = np.array([
        [1.0,  0.1, -0.2,  0.3, -0.1,  0.0],
        [0.1,  1.0,  0.0,  0.2,  0.3, -0.1],
        [-0.2, 0.0,  1.0, -0.1,  0.1,  0.2],
        [0.3,  0.2, -0.1,  1.0, -0.2, -0.3],
        [-0.1, 0.3,  0.1, -0.2,  1.0,  0.1],
        [0.0, -0.1,  0.2, -0.3,  0.1,  1.0],
    ])
    factor_vols = np.array([0.008, 0.010, 0.006, 0.012, 0.009, 0.007])
    D = np.diag(factor_vols)
    full_cov = D @ cov @ D

    factor_rets = rng.multivariate_normal(np.zeros(6), full_cov, size=n_days)
    factor_df = pd.DataFrame(factor_rets, index=dates, columns=factor_names)

    # --- Market returns: S&P proxy with stochastic vol (GARCH-lite)
    market_vol = 0.012
    market_rets = []
    v = market_vol ** 2
    for _ in range(n_days):
        v = max(0.0001**2, 0.85 * v + 0.15 * (market_vol**2) + 0.1 * rng.standard_normal()**2 * market_vol**2)
        market_rets.append(rng.normal(0.0003, np.sqrt(v)))
    market_rets = pd.Series(market_rets, index=dates, name="market")

    # --- Long / Short book returns (EMN: mostly factor-hedged, small residual beta)
    # Long book: slight positive value tilt
    long_rets = (
        0.5 * factor_df["Value"]
        + 0.3 * factor_df["Quality"]
        - 0.2 * factor_df["Volatility"]
        + 0.02 * market_rets
        + rng.normal(0, 0.005, n_days)
    )
    # Short book: slight negative momentum tilt
    short_rets = (
        -0.4 * factor_df["Momentum"]
        - 0.3 * factor_df["Growth"]
        + 0.2 * factor_df["Volatility"]
        - 0.02 * market_rets
        + rng.normal(0, 0.005, n_days)
    )
    long_rets = pd.Series(long_rets, index=dates, name="long")
    short_rets = pd.Series(short_rets, index=dates, name="short")
    portfolio_rets = pd.Series(long_rets.values - short_rets.values, index=dates, name="portfolio")

    # --- VIX: mean-reverting (Ornstein-Uhlenbeck)
    vix = np.zeros(n_days)
    vix[0] = 18.0
    theta, mu, sigma_v = 0.05, 18.0, 1.5
    for i in range(1, n_days):
        vix[i] = max(9, vix[i-1] + theta * (mu - vix[i-1]) + sigma_v * rng.standard_normal())
    # Inject a crisis spike in the last 60 days
    vix[-60:-30] += rng.uniform(10, 25, 30)
    vix = pd.Series(vix, index=dates, name="vix")

    # --- IG Spread: correlated with VIX
    ig_spread = 80 + 3.5 * (vix - 18) + rng.normal(0, 5, n_days)
    ig_spread = pd.Series(np.maximum(ig_spread, 50), index=dates, name="ig_spread")

    # --- Crowding & Liquidity scores (0-10 scale, higher = worse)
    crowding = np.clip(
        5 + 1.5 * rng.standard_normal(n_days).cumsum() * 0.05 + 0.1 * (vix - 18) / 5,
        0, 10
    )
    crowding = pd.Series(crowding, index=dates, name="crowding")

    liquidity = np.clip(
        3 + 0.4 * (vix - 18) / 5 + rng.normal(0, 0.5, n_days),
        0, 10
    )
    liquidity = pd.Series(liquidity, index=dates, name="liquidity")

    # --- Current factor exposures (snapshot, 6-vector)
    # Simulate a slightly crowded value/momentum book
    factor_exposures = pd.Series(
        [0.45, 0.30, -0.15, 0.20, -0.10, 0.25],
        index=factor_names,
        name="exposures"
    )

    return {
        "dates": dates,
        "factor_returns": factor_df,
        "factor_exposures": factor_exposures,
        "portfolio_returns": portfolio_rets,
        "long_returns": long_rets,
        "short_returns": short_rets,
        "market_returns": market_rets,
        "vix": vix,
        "ig_spread": ig_spread,
        "crowding_scores": crowding,
        "liquidity_scores": liquidity,
    }
