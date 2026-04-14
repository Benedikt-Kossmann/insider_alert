"""GARCH(1,1) volatility forecasting."""
import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_OBSERVATIONS = 252   # at least 1 year of daily returns
_FORECAST_HORIZONS = [1, 5, 10]  # days

# In-memory GARCH cache: ticker → (date_str, result_dict)
# Avoids re-fitting every run; invalidated daily or on large return shocks.
_garch_cache: dict[str, tuple[str, dict]] = {}
_REFIT_INTERVAL_DAYS = 7
_SHOCK_THRESHOLD = 0.05   # abs daily return > 5% → force refit


def _fit_garch(returns: np.ndarray):
    """Fit GARCH(1,1) on a return series.

    Returns arch.univariate.ARCHModelResult or None on failure.
    """
    try:
        from arch import arch_model  # lazy import — optional dependency
        # Rescale to % for numerical stability
        am = arch_model(returns * 100, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        result = am.fit(disp="off", show_warning=False)
        return result
    except ImportError:
        logger.debug("arch library not installed — GARCH forecasting disabled")
        return None
    except Exception as exc:
        logger.warning("GARCH fit failed: %s", exc)
        return None


def _extract_forecasts(result, returns: np.ndarray) -> dict:
    """Extract feature dict from a fitted GARCH result."""
    defaults = {
        "garch_forecast_1d": 0.0,
        "garch_forecast_5d": 0.0,
        "garch_forecast_10d": 0.0,
        "vol_surprise_ratio": 1.0,
        "vol_regime_forecast": "stable",
        "vol_of_vol": 0.0,
    }
    try:
        horizon = max(_FORECAST_HORIZONS)
        forecasts = result.forecast(horizon=horizon)
        # arch returns variance in (%)^2 — convert: sqrt(var)/100 → decimal daily vol
        # annualise with sqrt(252)
        var_fc = forecasts.variance.iloc[-1].values  # shape: (horizon,)

        vol_1d = float(np.sqrt(var_fc[0]) / 100 * np.sqrt(252))
        vol_5d = float(np.sqrt(np.mean(var_fc[:5])) / 100 * np.sqrt(252))
        vol_10d = float(np.sqrt(np.mean(var_fc[:10])) / 100 * np.sqrt(252))

        # Realised volatility over last 20 days (annualised)
        realized_vol = float(np.std(returns[-20:]) * np.sqrt(252))

        # Vol surprise ratio: forecast / realized
        vol_surprise = vol_1d / max(realized_vol, 0.01)

        if vol_surprise > 1.3:
            vol_regime = "expanding"
        elif vol_surprise < 0.7:
            vol_regime = "contracting"
        else:
            vol_regime = "stable"

        # Vol-of-Vol: coefficient of variation of recent conditional volatility
        cond_vol = result.conditional_volatility[-60:].values
        if len(cond_vol) > 10 and np.mean(cond_vol) > 0:
            vol_of_vol = float(np.std(cond_vol) / np.mean(cond_vol))
        else:
            vol_of_vol = 0.0

        return {
            "garch_forecast_1d": round(vol_1d, 4),
            "garch_forecast_5d": round(vol_5d, 4),
            "garch_forecast_10d": round(vol_10d, 4),
            "vol_surprise_ratio": round(vol_surprise, 3),
            "vol_regime_forecast": vol_regime,
            "vol_of_vol": round(vol_of_vol, 4),
        }
    except Exception as exc:
        logger.warning("GARCH forecast extraction failed: %s", exc)
        return defaults


def compute_volatility_forecast(
    ohlcv: pd.DataFrame,
    ticker: str = "",
    refit_interval_days: int = _REFIT_INTERVAL_DAYS,
) -> dict:
    """Compute GARCH-based volatility forecast features.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV data with at least a ``close`` column and >= 252 rows.
    ticker : str
        Optional ticker symbol used as cache key.  Pass ``""`` to skip caching.
    refit_interval_days : int
        Days before forcing a cache refit (default 7).

    Returns
    -------
    dict
        garch_forecast_1d  : 1-day annualised volatility forecast
        garch_forecast_5d  : 5-day average annualised forecast
        garch_forecast_10d : 10-day average annualised forecast
        vol_surprise_ratio : forecast / realized  (>1.3 = expanding regime)
        vol_regime_forecast: "expanding" | "contracting" | "stable"
        vol_of_vol         : coefficient of variation of recent conditional vol
    """
    defaults = {
        "garch_forecast_1d": 0.0,
        "garch_forecast_5d": 0.0,
        "garch_forecast_10d": 0.0,
        "vol_surprise_ratio": 1.0,
        "vol_regime_forecast": "stable",
        "vol_of_vol": 0.0,
    }

    if ohlcv is None or ohlcv.empty:
        return defaults

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        return defaults

    close = df["close"].dropna()
    if len(close) < _MIN_OBSERVATIONS:
        return defaults

    returns = close.pct_change().dropna().values
    if len(returns) < _MIN_OBSERVATIONS:
        return defaults

    today_str = str(date.today())

    # --- Cache check ---
    if ticker:
        cached = _garch_cache.get(ticker)
        if cached is not None:
            cached_date, cached_result = cached
            age_days = (date.today() - date.fromisoformat(cached_date)).days
            latest_return = abs(float(returns[-1])) if len(returns) else 0.0
            force_refit = age_days >= refit_interval_days or latest_return > _SHOCK_THRESHOLD
            if not force_refit:
                logger.debug("GARCH cache hit for %s (age %dd)", ticker, age_days)
                return cached_result

    result = _fit_garch(returns)
    if result is None:
        return defaults

    features = _extract_forecasts(result, returns)

    # --- Update cache ---
    if ticker:
        _garch_cache[ticker] = (today_str, features)
        logger.debug("GARCH cache updated for %s", ticker)

    return features
