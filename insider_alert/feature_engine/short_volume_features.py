"""Short volume features from FINRA RegSHO data."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_short_volume_features(short_df: pd.DataFrame) -> dict:
    """Compute features from FINRA short volume data.

    Parameters
    ----------
    short_df : pd.DataFrame
        Output of :func:`~insider_alert.data_ingestion.short_volume_data.fetch_short_volume`.

    Returns
    -------
    dict
        short_ratio_current : float — latest short ratio (0–1)
        short_ratio_zscore : float — Z-score vs trailing mean (clipped ±3)
        short_ratio_trend_5d : float — 5-day change in short ratio
        short_squeeze_score : float — combined score (0–1): high short + rising + volume spike
    """
    defaults = {
        "short_ratio_current": 0.0,
        "short_ratio_zscore": 0.0,
        "short_ratio_trend_5d": 0.0,
        "short_squeeze_score": 0.0,
    }

    if short_df is None or short_df.empty or len(short_df) < 5:
        return defaults

    ratios = short_df["ShortRatio"].values.astype(float)

    # 1. Current short ratio
    current = float(ratios[-1])

    # 2. Z-score vs full window mean
    mean_ = float(np.mean(ratios))
    std_ = float(np.std(ratios))
    zscore = float((current - mean_) / max(std_, 0.01))

    # 3. 5-day trend
    trend = float(ratios[-1] - ratios[-5]) if len(ratios) >= 5 else 0.0

    # 4. Short squeeze score
    # Fires when: short ratio > 50% AND rising AND volume above average
    volumes = short_df["TotalVolume"].values.astype(float)
    vol_mean = float(np.mean(volumes[:-1])) if len(volumes) > 1 else float(volumes[-1])
    vol_ratio = float(volumes[-1]) / max(vol_mean, 1.0)

    squeeze = 0.0
    if current > 0.5 and trend > 0.05 and vol_ratio > 1.2:
        squeeze = float(np.clip(
            (current - 0.4) * 2.0 + trend * 5.0 + (vol_ratio - 1.0) * 0.5,
            0.0, 1.0,
        ))

    return {
        "short_ratio_current": round(current, 3),
        "short_ratio_zscore": round(float(np.clip(zscore, -3.0, 3.0)), 2),
        "short_ratio_trend_5d": round(trend, 4),
        "short_squeeze_score": round(squeeze, 3),
    }
