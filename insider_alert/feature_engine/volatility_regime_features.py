"""Volatility regime feature computation (VIX, Bollinger Bands, ATR regime)."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_volatility_regime_features(
    ohlcv: pd.DataFrame,
    vix_data: pd.DataFrame | None = None,
    *,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0,
    atr_regime_window: int = 20,
    vix_high: float = 30.0,
    vix_low: float = 15.0,
) -> dict:
    """Compute volatility-regime features.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV data for the ETF or stock.
    vix_data : pd.DataFrame | None
        OHLCV data for VIX (ticker ``^VIX``).  Only ``close`` is used.
    bollinger_period : int
        Lookback window for Bollinger Bands (default 20).
    bollinger_std : float
        Standard deviation multiplier for Bollinger Bands (default 2.0).
    atr_regime_window : int
        Lookback window for ATR regime classification (default 20).
    vix_high : float
        VIX level above which regime is "high" (default 30).
    vix_low : float
        VIX level below which regime is "low" (default 15).

    Returns
    -------
    dict
        Volatility regime features.
    """
    defaults = {
        "vix_current": 0.0,
        "vix_sma_20": 0.0,
        "vix_regime": "unknown",
        "bollinger_upper": 0.0,
        "bollinger_lower": 0.0,
        "bollinger_pct_b": 0.5,
        "atr_regime": "normal",
        "realized_vol_20d": 0.0,
    }

    d = ohlcv.copy()
    d.columns = [c.lower() for c in d.columns]
    if d.empty or "close" not in d.columns:
        return defaults

    close = d["close"]

    # --- VIX ---
    vix_current = 0.0
    vix_sma_20 = 0.0
    vix_regime = "unknown"
    if vix_data is not None and not vix_data.empty:
        vd = vix_data.copy()
        vd.columns = [c.lower() for c in vd.columns]
        if "close" in vd.columns:
            vix_current = float(vd["close"].iloc[-1])
            vix_sma_20 = float(vd["close"].rolling(20, min_periods=1).mean().iloc[-1])
            if vix_current >= vix_high:
                vix_regime = "high"
            elif vix_current <= vix_low:
                vix_regime = "low"
            else:
                vix_regime = "normal"

    # --- Bollinger Bands ---
    if len(close) >= bollinger_period:
        sma = close.rolling(bollinger_period).mean()
        std = close.rolling(bollinger_period).std()
        upper = sma + bollinger_std * std
        lower = sma - bollinger_std * std
        bollinger_upper = float(upper.iloc[-1])
        bollinger_lower = float(lower.iloc[-1])
        band_width = bollinger_upper - bollinger_lower
        if band_width > 0:
            bollinger_pct_b = float((close.iloc[-1] - bollinger_lower) / band_width)
            bollinger_pct_b = max(0.0, min(1.0, bollinger_pct_b))
        else:
            bollinger_pct_b = 0.5
    else:
        bollinger_upper = float(close.iloc[-1]) if len(close) > 0 else 0.0
        bollinger_lower = float(close.iloc[-1]) if len(close) > 0 else 0.0
        bollinger_pct_b = 0.5

    # --- ATR Regime ---
    atr_regime = "normal"
    if len(d) >= atr_regime_window + 5 and {"high", "low"}.issubset(d.columns):
        prev_close = close.shift(1)
        tr = pd.concat([
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(14, min_periods=1).mean()
        recent_atr = atr_series.iloc[-5:].mean()
        prior_atr = atr_series.iloc[-atr_regime_window:-5].mean()
        if prior_atr > 0:
            atr_change = (recent_atr - prior_atr) / prior_atr
            if atr_change > 0.15:
                atr_regime = "expanding"
            elif atr_change < -0.15:
                atr_regime = "contracting"

    # --- Realized Volatility (20d annualised) ---
    returns = close.pct_change().dropna()
    if len(returns) >= 20:
        realized_vol_20d = float(returns.iloc[-20:].std() * np.sqrt(252))
    elif len(returns) >= 2:
        realized_vol_20d = float(returns.std() * np.sqrt(252))
    else:
        realized_vol_20d = 0.0

    return {
        "vix_current": round(vix_current, 2),
        "vix_sma_20": round(vix_sma_20, 2),
        "vix_regime": vix_regime,
        "bollinger_upper": round(bollinger_upper, 4),
        "bollinger_lower": round(bollinger_lower, 4),
        "bollinger_pct_b": round(bollinger_pct_b, 4),
        "atr_regime": atr_regime,
        "realized_vol_20d": round(realized_vol_20d, 4),
    }
