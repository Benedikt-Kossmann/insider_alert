"""Leverage-specific feature computation.

Computes tracking error, estimated decay, underlying trend alignment, and
correlation between a leveraged ETF and its underlying index.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily percentage returns."""
    return close.pct_change().dropna()


def compute_leverage_features(
    etf_ohlcv: pd.DataFrame,
    underlying_ohlcv: pd.DataFrame,
    leverage: int = 3,
    direction: str = "long",
) -> dict:
    """Compute leverage-specific features.

    Parameters
    ----------
    etf_ohlcv : pd.DataFrame
        OHLCV data for the leveraged ETF.
    underlying_ohlcv : pd.DataFrame
        OHLCV data for the underlying index/ETF.
    leverage : int
        Leverage factor (e.g. 3 for 3x).
    direction : str
        ``'long'`` or ``'short'`` (inverse ETFs).

    Returns
    -------
    dict
        Leverage features including tracking error, decay, underlying
        trend, correlation, and alignment.
    """
    defaults = {
        "tracking_error_5d": 0.0,
        "estimated_daily_decay": 0.0,
        "underlying_return_5d": 0.0,
        "underlying_return_20d": 0.0,
        "underlying_vs_etf_corr_20d": 0.0,
        "underlying_trend_aligned": 0,
        "leverage_adjusted_atr_pct": 0.0,
    }

    etf_df = etf_ohlcv.copy()
    und_df = underlying_ohlcv.copy()
    etf_df.columns = [c.lower() for c in etf_df.columns]
    und_df.columns = [c.lower() for c in und_df.columns]

    if etf_df.empty or und_df.empty or "close" not in etf_df.columns or "close" not in und_df.columns:
        return defaults

    etf_close = etf_df["close"]
    und_close = und_df["close"]

    if len(etf_close) < 6 or len(und_close) < 6:
        return defaults

    etf_ret = _daily_returns(etf_close)
    und_ret = _daily_returns(und_close)

    # Align on common dates
    common_idx = etf_ret.index.intersection(und_ret.index)
    if len(common_idx) < 5:
        return defaults
    etf_ret = etf_ret.loc[common_idx]
    und_ret = und_ret.loc[common_idx]

    # Direction multiplier: long=+1, short=-1
    dir_mult = -1.0 if direction == "short" else 1.0
    expected_ret = dir_mult * leverage * und_ret

    # Tracking error (last 5 days) — RMSE of actual vs expected daily returns
    last_5_etf = etf_ret.iloc[-5:]
    last_5_exp = expected_ret.iloc[-5:]
    tracking_error_5d = float(np.sqrt(((last_5_etf - last_5_exp) ** 2).mean()))

    # Estimated daily decay ≈ L² × σ² / 2
    und_var = float(und_ret.iloc[-20:].var()) if len(und_ret) >= 20 else float(und_ret.var())
    estimated_daily_decay = (leverage ** 2) * und_var / 2.0

    # Underlying returns
    underlying_return_5d = float(und_close.iloc[-1] / und_close.iloc[-6] - 1.0) if len(und_close) >= 6 else 0.0
    underlying_return_20d = float(und_close.iloc[-1] / und_close.iloc[-21] - 1.0) if len(und_close) >= 21 else 0.0

    # Correlation between ETF returns and expected leveraged returns (last 20 days)
    if len(common_idx) >= 20:
        corr_data = etf_ret.iloc[-20:]
        exp_data = expected_ret.iloc[-20:]
        corr = float(corr_data.corr(exp_data))
        corr = corr if np.isfinite(corr) else 0.0
    else:
        corr = float(etf_ret.corr(expected_ret))
        corr = corr if np.isfinite(corr) else 0.0

    # Underlying trend alignment
    # For long ETFs: underlying_return_5d > 0 = aligned
    # For short ETFs: underlying_return_5d < 0 = aligned
    if direction == "short":
        trend_aligned = 1 if underlying_return_5d < 0 else 0
    else:
        trend_aligned = 1 if underlying_return_5d > 0 else 0

    # Leverage-adjusted ATR%
    if "high" in etf_df.columns and "low" in etf_df.columns:
        tr = pd.concat([
            etf_df["high"] - etf_df["low"],
            (etf_df["high"] - etf_df["close"].shift(1)).abs(),
            (etf_df["low"] - etf_df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        price = float(etf_close.iloc[-1])
        leverage_adjusted_atr_pct = atr / (price + 1e-9)
    else:
        leverage_adjusted_atr_pct = 0.0

    return {
        "tracking_error_5d": round(tracking_error_5d, 6),
        "estimated_daily_decay": round(estimated_daily_decay, 6),
        "underlying_return_5d": round(underlying_return_5d, 6),
        "underlying_return_20d": round(underlying_return_20d, 6),
        "underlying_vs_etf_corr_20d": round(corr, 4),
        "underlying_trend_aligned": trend_aligned,
        "leverage_adjusted_atr_pct": round(leverage_adjusted_atr_pct, 6),
    }
