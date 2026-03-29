"""Volume-based feature computation."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_volume_features(ohlcv: pd.DataFrame) -> dict:
    """Compute volume-based features from OHLCV data."""
    defaults = {
        "volume_rvol_20d": 1.0,
        "volume_zscore_20d": 0.0,
        "close_volume_ratio": 0.0,
        "intraday_volume_acceleration": 0.0,
        "tight_range_high_volume_flag": 0,
    }
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 2:
        return defaults

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"close", "volume"}
    if not required.issubset(df.columns):
        return defaults

    volumes = df["volume"].dropna().astype(float)
    closes = df["close"].dropna()
    if len(volumes) < 2:
        return defaults

    current_vol = float(volumes.iloc[-1])
    avg_vol_20d = float(volumes.iloc[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
    std_vol_20d = float(volumes.iloc[-20:].std()) if len(volumes) >= 20 else float(volumes.std())

    volume_rvol_20d = current_vol / (avg_vol_20d + 1e-9)
    volume_zscore_20d = (current_vol - avg_vol_20d) / (std_vol_20d + 1e-9)

    current_close = float(closes.iloc[-1])
    close_volume_ratio = current_close / (current_vol + 1e-9)

    vol_5d = volumes.iloc[-5:].values if len(volumes) >= 5 else volumes.values
    if len(vol_5d) >= 2:
        x = np.arange(len(vol_5d), dtype=float)
        slope = float(np.polyfit(x, vol_5d, 1)[0])
        intraday_volume_acceleration = slope / (avg_vol_20d + 1e-9)
    else:
        intraday_volume_acceleration = 0.0

    tight_range_high_volume_flag = 0
    if "high" in df.columns and "low" in df.columns:
        highs = df["high"].dropna()
        lows = df["low"].dropna()
        if len(highs) > 0 and len(closes) > 0:
            h = float(highs.iloc[-1])
            lo = float(lows.iloc[-1])
            c = float(closes.iloc[-1])
            range_ratio = (h - lo) / (c + 1e-9)
            if range_ratio < 0.01 and volume_rvol_20d > 1.5:
                tight_range_high_volume_flag = 1

    return {
        "volume_rvol_20d": volume_rvol_20d,
        "volume_zscore_20d": volume_zscore_20d,
        "close_volume_ratio": close_volume_ratio,
        "intraday_volume_acceleration": intraday_volume_acceleration,
        "tight_range_high_volume_flag": tight_range_high_volume_flag,
    }
