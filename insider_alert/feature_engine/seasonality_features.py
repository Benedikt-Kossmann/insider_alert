"""Seasonality features based on well-documented calendar effects."""
import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Historical average monthly returns for the S&P 500 (1950–2023, approximate).
_MONTHLY_BIAS: dict[int, float] = {
    1:  0.012,   # January: +1.2% avg (January Effect)
    2:  0.001,   # February: flat
    3:  0.011,   # March: slightly positive
    4:  0.015,   # April: strong (Tax-Refund Season)
    5:  0.002,   # May: weak
    6:  0.001,   # June: flat
    7:  0.010,   # July: pre-earnings rally
    8: -0.003,   # August: weak
    9: -0.005,   # September: worst month historically
    10:  0.008,  # October: turnaround month
    11:  0.015,  # November: strong
    12:  0.013,  # December: Santa Rally
}

# Intraweek bias (Monday=0, Friday=4).
_WEEKDAY_BIAS: dict[int, float] = {
    0: -0.0005,  # Monday: slightly negative
    1:  0.0002,  # Tuesday: neutral
    2:  0.0003,  # Wednesday: slightly positive
    3:  0.0002,  # Thursday: neutral
    4:  0.0003,  # Friday: slight closing-rally effect
}


def _is_quad_witching_week(d: date) -> bool:
    """Return True if *d* falls within the Quad-Witching week.

    Quad-Witching occurs on the 3rd Friday of March, June, September and
    December.  The entire calendar week (Mon–Fri) of that Friday is
    considered volatile.
    """
    if d.month not in (3, 6, 9, 12):
        return False

    first_day = date(d.year, d.month, 1)
    # Move to first Friday of the month
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday_day = first_day.day + days_to_friday
    third_friday_day = first_friday_day + 14

    third_friday = date(d.year, d.month, third_friday_day)
    # Monday of that week
    week_start_ord = third_friday.toordinal() - third_friday.weekday()
    week_end_ord = week_start_ord + 4

    return week_start_ord <= d.toordinal() <= week_end_ord


def _compute_historical_month_strength(
    ohlcv: Optional[pd.DataFrame],
    current_month: int,
    years_back: int = 5,
) -> float:
    """Return the ticker-specific average return for *current_month* over the last *years_back* years.

    Falls back to the generic ``_MONTHLY_BIAS`` value when less than 2 years
    of monthly data are available.

    Returns value in roughly [-0.15, 0.15] range.
    """
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 252:
        return _MONTHLY_BIAS.get(current_month, 0.0)

    close_col = "close" if "close" in ohlcv.columns else (
        "Close" if "Close" in ohlcv.columns else None
    )
    if close_col is None:
        return _MONTHLY_BIAS.get(current_month, 0.0)

    try:
        close = ohlcv[close_col]
        if not hasattr(close.index, "month"):
            close.index = pd.to_datetime(close.index)
        monthly = close.resample("ME").last().pct_change().dropna()
        same_month = monthly[monthly.index.month == current_month]
        if len(same_month) < 2:
            return _MONTHLY_BIAS.get(current_month, 0.0)
        avg_return = float(same_month.tail(years_back).mean())
        return float(np.clip(avg_return, -0.15, 0.15))
    except Exception:
        return _MONTHLY_BIAS.get(current_month, 0.0)


def compute_seasonality_features(
    ohlcv: Optional[pd.DataFrame] = None,
    current_date: Optional[date] = None,
) -> dict:
    """Compute calendar-based seasonality features.

    Parameters
    ----------
    ohlcv :
        Optional OHLCV DataFrame for ticker-specific monthly strength.
    current_date :
        Reference date (default: ``date.today()``).

    Returns
    -------
    dict with keys:
        ``monthly_bias``       – generic S&P monthly return tendency
        ``weekday_bias``       – intraweek return tendency
        ``quad_witching``      – True during Quad-Witching week
        ``sell_in_may_active`` – True May–October
        ``seasonal_score``     – combined [0, 1] (0 = bearish, 0.5 = neutral, 1 = bullish)
        ``month_strength``     – ticker-specific historical month return
    """
    if current_date is None:
        current_date = date.today()

    month = current_date.month
    weekday = current_date.weekday()

    monthly_bias: float = _MONTHLY_BIAS.get(month, 0.0)
    weekday_bias: float = _WEEKDAY_BIAS.get(weekday, 0.0)
    quad_witching: bool = _is_quad_witching_week(current_date)
    sell_in_may: bool = 5 <= month <= 10
    month_strength: float = _compute_historical_month_strength(ohlcv, month)

    # Combined score [0, 1]
    score_components = [
        float(np.clip(monthly_bias * 50 + 0.5, 0.0, 1.0)),   # monthly tendency
        0.3 if sell_in_may else 0.7,                           # sell-in-may penalty
        0.3 if quad_witching else 0.6,                         # quad-witching volatility penalty
        float(np.clip(weekday_bias * 1_000 + 0.5, 0.0, 1.0)), # weekday tendency
    ]
    seasonal_score = float(np.mean(score_components))

    return {
        "monthly_bias": round(monthly_bias, 4),
        "weekday_bias": round(weekday_bias, 5),
        "quad_witching": quad_witching,
        "sell_in_may_active": sell_in_may,
        "seasonal_score": round(seasonal_score, 3),
        "month_strength": round(month_strength, 4),
    }
