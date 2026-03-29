"""Accumulation pattern feature computation."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_accumulation_features(ohlcv: pd.DataFrame) -> dict:
    """Compute accumulation pattern features using last 10 rows of OHLCV data."""
    defaults = {
        "range_compression_score": 0.0,
        "higher_lows_score": 0.0,
        "volume_under_resistance_score": 0.0,
        "wyckoff_accumulation_score": 0.0,
    }
    if ohlcv is None or ohlcv.empty:
        return defaults

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        return defaults

    df10 = df.tail(10)
    if len(df10) < 2:
        return defaults

    range_compression_score = 0.0
    df5 = df10.tail(5)
    if len(df5) >= 1 and len(df10) >= 2:
        range_10d = ((df10["high"] - df10["low"]) / (df10["close"] + 1e-9)).mean()
        range_5d = ((df5["high"] - df5["low"]) / (df5["close"] + 1e-9)).mean()
        if range_10d > 1e-9:
            range_compression_score = float(np.clip(1.0 - range_5d / range_10d, 0.0, 1.0))

    higher_lows_score = 0.0
    if len(df5) >= 2:
        lows = df5["low"].values
        higher_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
        higher_lows_score = float(higher_count / (len(lows) - 1))

    volume_under_resistance_score = 0.0
    if "volume" in df10.columns:
        df3 = df10.tail(3)
        vols = df3["volume"].values.astype(float)
        high_10d = float(df10["high"].max())
        last_close = float(df10["close"].iloc[-1])
        if len(vols) >= 2:
            vol_trend_positive = vols[-1] > vols[0]
            near_resistance = abs(last_close - high_10d) / (high_10d + 1e-9) < 0.02
            if vol_trend_positive and near_resistance:
                volume_under_resistance_score = 0.7

    wyckoff_accumulation_score = float(
        np.mean([range_compression_score, higher_lows_score, volume_under_resistance_score])
    )

    return {
        "range_compression_score": range_compression_score,
        "higher_lows_score": higher_lows_score,
        "volume_under_resistance_score": volume_under_resistance_score,
        "wyckoff_accumulation_score": wyckoff_accumulation_score,
    }
