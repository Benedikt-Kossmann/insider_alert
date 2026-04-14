"""Cross-asset correlation and regime detection."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Core asset universe for correlation analysis
_CORE_ASSETS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "GLD": "Gold",
    "TLT": "20Y Treasury",
    "UUP": "US Dollar",
}

_CORRELATION_WINDOW = 20  # trading days

# Historically stable normal correlations
_NORMAL_CORRELATIONS: dict[tuple[str, str], float] = {
    ("SPY", "QQQ"): 0.90,   # normally high
    ("SPY", "GLD"): -0.05,  # slight negative
    ("SPY", "TLT"): -0.30,  # risk-on/off
    ("GLD", "TLT"):  0.20,  # both safe-haven
}


def _fetch_returns(tickers: list[str], lookback_days: int = 60) -> pd.DataFrame | None:
    """Fetch daily returns for multiple tickers via yfinance."""
    try:
        import yfinance as yf
        data = yf.download(tickers, period=f"{lookback_days}d", progress=False, auto_adjust=True)
        if data.empty:
            return None
        close = data["Close"] if "Close" in data.columns else data
        returns = close.pct_change().dropna()
        if len(returns) < _CORRELATION_WINDOW // 2:
            return None
        return returns
    except Exception as exc:
        logger.warning("Cross-asset data fetch failed: %s", exc)
        return None


def _safe_corr(corr_matrix: pd.DataFrame, a: str, b: str) -> float:
    """Return correlation value from matrix, 0.0 on missing key."""
    try:
        return float(corr_matrix.loc[a, b])
    except (KeyError, ValueError):
        return 0.0


def compute_cross_asset_features() -> dict:
    """Compute cross-asset correlation features.

    Fetches daily returns for SPY, QQQ, GLD, TLT, UUP and builds a
    20-day rolling correlation matrix.

    Returns
    -------
    dict
        equity_correlation_regime  : "normal" | "decorrelation" | "panic"
        spy_qqq_correlation        : float, current SPY/QQQ 20d correlation
        spy_gld_correlation        : float, current SPY/Gold correlation
        spy_tlt_correlation        : float, current SPY/Bond correlation
        correlation_anomaly_score  : float in [0, 1], deviation from normal
    """
    defaults = {
        "equity_correlation_regime": "normal",
        "spy_qqq_correlation": 0.9,
        "spy_gld_correlation": 0.0,
        "spy_tlt_correlation": -0.3,
        "correlation_anomaly_score": 0.0,
    }

    tickers = list(_CORE_ASSETS.keys())
    returns = _fetch_returns(tickers)
    if returns is None:
        return defaults

    recent = returns.tail(_CORRELATION_WINDOW)
    if len(recent) < _CORRELATION_WINDOW // 2:
        return defaults

    corr = recent.corr()

    spy_qqq = _safe_corr(corr, "SPY", "QQQ")
    spy_gld = _safe_corr(corr, "SPY", "GLD")
    spy_tlt = _safe_corr(corr, "SPY", "TLT")

    # Anomaly score: mean absolute deviation from historical norms
    deviations = []
    for (a, b), normal_val in _NORMAL_CORRELATIONS.items():
        actual = _safe_corr(corr, a, b)
        deviations.append(abs(actual - normal_val))
    anomaly = float(np.mean(deviations)) if deviations else 0.0

    # Regime classification
    if spy_qqq < 0.7 and anomaly > 0.3:
        regime = "panic"           # everything decorrelates → stress
    elif spy_qqq > 0.95 and spy_tlt > 0:
        regime = "panic"           # everything rises together → liquidity rally
    elif anomaly > 0.2:
        regime = "decorrelation"
    else:
        regime = "normal"

    return {
        "equity_correlation_regime": regime,
        "spy_qqq_correlation": round(spy_qqq, 3),
        "spy_gld_correlation": round(spy_gld, 3),
        "spy_tlt_correlation": round(spy_tlt, 3),
        "correlation_anomaly_score": round(float(np.clip(anomaly, 0.0, 1.0)), 3),
    }
