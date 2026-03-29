"""Corporate event feature computation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# How many days before earnings the pre-event scoring window is active
PRE_EVENT_WINDOW_DAYS = 10
# Divisor to normalise a 5-day return into a [0, 1] score
# (a 500 % move would score 1.0 – this is an intentionally conservative cap)
PRE_EVENT_RETURN_DIVISOR = 5.0
# RVOL divisor: a 3× relative-volume spike produces a score of 1.0
PRE_EVENT_RVOL_DIVISOR = 3.0
# Call-volume z-score divisor for the pre-event options score
PRE_EVENT_CALL_ZSCORE_DIVISOR = 3.0


def compute_event_features(
    days_to_earnings: int | None,
    price_features: dict,
    volume_features: dict,
    options_features: dict,
    days_to_corporate_event: int | None = None,
) -> dict:
    """Compute event proximity features.

    Parameters
    ----------
    days_to_earnings:
        Calendar days until the next earnings date (from ``days_to_next_earnings``).
        Pass ``None`` if unknown.
    price_features, volume_features, options_features:
        Feature dicts from the corresponding feature-engine modules.
    days_to_corporate_event:
        Calendar days since the most recent material 8-K filing (negative means
        the event already occurred, positive means it's upcoming).  Pass ``None``
        if no recent 8-K was found.
    """
    dte = 999 if days_to_earnings is None else int(days_to_earnings)
    dtce = 999 if days_to_corporate_event is None else int(days_to_corporate_event)
    # Use whichever event is closest (earnings or 8-K corporate event)
    effective_dte = min(dte, dtce)

    if effective_dte <= PRE_EVENT_WINDOW_DAYS:
        pre_event_return_score = float(np.clip(price_features.get("return_5d", 0.0) / PRE_EVENT_RETURN_DIVISOR, 0.0, 1.0))
        pre_event_volume_score = float(min(volume_features.get("volume_rvol_20d", 1.0) / PRE_EVENT_RVOL_DIVISOR, 1.0))
        raw_call_zscore = options_features.get("call_volume_zscore", 0.0)
        pre_event_options_score = float(np.clip(raw_call_zscore / PRE_EVENT_CALL_ZSCORE_DIVISOR, 0.0, 1.0))
    else:
        pre_event_return_score = 0.0
        pre_event_volume_score = 0.0
        pre_event_options_score = 0.0

    return {
        "days_to_earnings": dte,
        "days_to_corporate_event": dtce,
        "pre_event_return_score": pre_event_return_score,
        "pre_event_volume_score": pre_event_volume_score,
        "pre_event_options_score": pre_event_options_score,
    }
