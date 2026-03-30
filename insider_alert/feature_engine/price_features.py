"""Price-based feature computation."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum relative gap (open vs prior close) to count as a gap-up/gap-down day
GAP_THRESHOLD = 0.005  # 0.5%
# ATR rolling window (days)
ATR_WINDOW = 14


def compute_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> float:
    """Compute the Average True Range for the most recent *window* bars.

    Requires columns ``high``, ``low``, ``close`` (case-insensitive).
    Returns 0.0 when the DataFrame is too short or columns are missing.
    """
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    required = {"high", "low", "close"}
    if not required.issubset(d.columns):
        return 0.0
    if len(d) < 2:
        return 0.0

    prev_close = d["close"].shift(1)
    true_range = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = true_range.rolling(window, min_periods=1).mean()
    return float(atr_series.iloc[-1]) if not atr_series.empty else 0.0


def compute_price_features(ohlcv: pd.DataFrame) -> dict:
    """Compute price-based features from OHLCV data."""
    defaults = {
        "return_1d": 0.0,
        "return_3d": 0.0,
        "return_5d": 0.0,
        "return_10d": 0.0,
        "daily_return_zscore": 0.0,
        "gap_up_count_5d": 0,
        "gap_down_count_5d": 0,
        "atr_14": 0.0,
        "atr_pct": 0.0,
    }
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 2:
        return defaults

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        return defaults

    closes = df["close"].dropna()
    if len(closes) < 2:
        return defaults

    def pct_return(n: int) -> float:
        if len(closes) <= n:
            return 0.0
        return float((closes.iloc[-1] - closes.iloc[-1 - n]) / (closes.iloc[-1 - n] + 1e-9))

    return_1d = pct_return(1)
    return_3d = pct_return(3)
    return_5d = pct_return(5)
    return_10d = pct_return(10)

    daily_returns = closes.pct_change().dropna()
    if len(daily_returns) >= 2:
        rolling_mean = daily_returns.rolling(20, min_periods=2).mean()
        rolling_std = daily_returns.rolling(20, min_periods=2).std()
        last_return = daily_returns.iloc[-1]
        mean_val = rolling_mean.iloc[-1] if not np.isnan(rolling_mean.iloc[-1]) else daily_returns.mean()
        std_val = rolling_std.iloc[-1] if not np.isnan(rolling_std.iloc[-1]) else daily_returns.std()
        daily_return_zscore = float((last_return - mean_val) / (std_val + 1e-9))
    else:
        daily_return_zscore = 0.0

    gap_up_count_5d = 0
    gap_down_count_5d = 0
    if "open" in df.columns:
        opens = df["open"].dropna()
        aligned_close = closes.reindex(opens.index)
        prior_close = aligned_close.shift(1)
        opens_5d = opens.iloc[-5:]
        prior_close_5d = prior_close.iloc[-5:]
        for o, pc in zip(opens_5d, prior_close_5d):
            if np.isnan(o) or np.isnan(pc) or pc == 0:
                continue
            gap = (o - pc) / pc
            if gap > GAP_THRESHOLD:
                gap_up_count_5d += 1
            elif gap < -GAP_THRESHOLD:
                gap_down_count_5d += 1

    atr_14 = compute_atr(df, window=ATR_WINDOW)
    current_price = float(closes.iloc[-1])
    atr_pct = atr_14 / (current_price + 1e-9)

    return {
        "return_1d": return_1d,
        "return_3d": return_3d,
        "return_5d": return_5d,
        "return_10d": return_10d,
        "daily_return_zscore": daily_return_zscore,
        "gap_up_count_5d": gap_up_count_5d,
        "gap_down_count_5d": gap_down_count_5d,
        "atr_14": atr_14,
        "atr_pct": atr_pct,
    }
