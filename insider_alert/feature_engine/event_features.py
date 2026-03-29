"""Corporate event feature computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_event_features(
    days_to_earnings: int | None,
    price_features: dict,
    volume_features: dict,
    options_features: dict,
) -> dict:
    """Compute event proximity features."""
    dte = 999 if days_to_earnings is None else int(days_to_earnings)

    if dte <= 10:
        pre_event_return_score = float(np.clip(price_features.get("return_5d", 0.0) / 5.0, 0.0, 1.0))
        pre_event_volume_score = float(min(volume_features.get("volume_rvol_20d", 1.0) / 3.0, 1.0))
        raw_call_zscore = options_features.get("call_volume_zscore", 0.0)
        pre_event_options_score = float(np.clip(raw_call_zscore / 3.0, 0.0, 1.0))
    else:
        pre_event_return_score = 0.0
        pre_event_volume_score = 0.0
        pre_event_options_score = 0.0

    return {
        "days_to_earnings": dte,
        "days_to_corporate_event": dte,
        "pre_event_return_score": pre_event_return_score,
        "pre_event_volume_score": pre_event_volume_score,
        "pre_event_options_score": pre_event_options_score,
    }
