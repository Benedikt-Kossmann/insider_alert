"""Price-based feature computation."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
            if gap > 0.005:
                gap_up_count_5d += 1
            elif gap < -0.005:
                gap_down_count_5d += 1

    return {
        "return_1d": return_1d,
        "return_3d": return_3d,
        "return_5d": return_5d,
        "return_10d": return_10d,
        "daily_return_zscore": daily_return_zscore,
        "gap_up_count_5d": gap_up_count_5d,
        "gap_down_count_5d": gap_down_count_5d,
    }
